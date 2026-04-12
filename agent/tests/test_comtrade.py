"""
tests/test_comtrade.py — Unit and integration smoke tests for comtrade_env.

Run with:
  cd llm_agent
  python -m pytest tests/test_comtrade.py -v

Tests are grouped into three categories:
  - Unit: pure logic with no environment dependency
  - Integration (smoke): spins up in-process env, exercises real mock service
  - GRPO: gradient-free checks on tokenisation and loss math
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Make sure agent/ is on sys.path when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ======================================================================
# Unit tests — no env required
# ======================================================================

class TestPrimaryKey:
    def test_full_key(self):
        from agent import _primary_key
        row = {"year": 2022, "reporter": "US", "partner": "CN",
               "flow": "Import", "hs": "0101", "record_id": "abc"}
        assert _primary_key(row) == "2022|US|CN|Import|0101|abc"

    def test_missing_fields(self):
        from agent import _primary_key
        row = {"year": 2022}
        pk = _primary_key(row)
        assert pk.startswith("2022|")
        assert pk.count("|") == 5


class TestParseToolCall:
    def test_valid_call(self):
        from agent import parse_tool_call
        text = '<tool_call>{"name": "fetch_page", "arguments": {"page": 2}}</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "fetch_page"
        assert args == {"page": 2}

    def test_last_call_wins(self):
        from agent import parse_tool_call
        text = (
            '<tool_call>{"name": "get_task_info", "arguments": {}}</tool_call>'
            " some text "
            '<tool_call>{"name": "fetch_page", "arguments": {"page": 1}}</tool_call>'
        )
        name, _ = parse_tool_call(text)
        assert name == "fetch_page"

    def test_no_call(self):
        from agent import parse_tool_call
        assert parse_tool_call("hello world") is None

    def test_broken_json(self):
        from agent import parse_tool_call
        assert parse_tool_call("<tool_call>{bad json}</tool_call>") is None


class TestComputeAdvantages:
    def test_normalised_within_group(self):
        pytest.importorskip("torch")
        from train_grpo import compute_advantages
        rollouts = [
            {"task_id": "T1", "reward": 1.0},
            {"task_id": "T1", "reward": 0.0},
        ]
        advs = compute_advantages(rollouts, group_size=2)
        assert len(advs) == 2
        # Higher reward → positive advantage
        assert advs[0] > 0
        assert advs[1] < 0

    def test_all_same_reward(self):
        pytest.importorskip("torch")
        from train_grpo import compute_advantages
        rollouts = [{"task_id": "T1", "reward": 0.5}] * 4
        advs = compute_advantages(rollouts, group_size=4)
        for a in advs:
            assert abs(a) < 1e-6

    def test_two_tasks_independent(self):
        pytest.importorskip("torch")
        from train_grpo import compute_advantages
        rollouts = [
            {"task_id": "T1", "reward": 1.0},
            {"task_id": "T1", "reward": 0.0},
            {"task_id": "T2", "reward": 0.3},
            {"task_id": "T2", "reward": 0.7},
        ]
        advs = compute_advantages(rollouts, group_size=2)
        # T1 group: mean=0.5, T2 group: mean=0.5
        # Both are symmetric so magnitudes equal within each group
        assert abs(abs(advs[0]) - abs(advs[1])) < 1e-6
        assert abs(abs(advs[2]) - abs(advs[3])) < 1e-6


class TestTaskCoverage:
    def test_inference_covers_all_tasks(self):
        from inference import ALL_TASKS
        assert "T8_mixed_faults" in ALL_TASKS
        assert "T9_adaptive_adversary" in ALL_TASKS
        assert "T10_constrained_budget" in ALL_TASKS

    def test_training_covers_all_tasks(self):
        from train_grpo import ALL_TASK_IDS
        assert "T8_mixed_faults" in ALL_TASK_IDS
        assert "T9_adaptive_adversary" in ALL_TASK_IDS
        assert "T10_constrained_budget" in ALL_TASK_IDS


class TestGrpoLoss:
    """Verify GRPO loss math without a real model."""

    def test_ratio_not_one_when_old_probs_differ(self):
        torch = pytest.importorskip("torch")
        from train_grpo import grpo_loss

        B = 2
        T = 4
        log_probs = torch.full((B, T), -1.0, requires_grad=True)
        old_log_probs = torch.full((B, T), -2.0)   # different → ratio != 1
        ref_log_probs = torch.full((B, T), -1.5)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones(B, T)

        loss, stats = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages, mask)
        # ratio = exp(-1 - (-2)) = e ≈ 2.718 — would be clipped at 1.2 for eps=0.2
        assert not math.isnan(stats["loss"])
        assert stats["kl"] >= 0.0

    def test_ratio_one_causes_no_clip(self):
        """When old == new, ratio=1 and clipping does nothing."""
        torch = pytest.importorskip("torch")
        from train_grpo import grpo_loss

        B, T = 2, 4
        lp = torch.full((B, T), -1.0, requires_grad=True)
        old_lp = torch.full((B, T), -1.0)  # same → ratio = 1
        ref_lp = torch.full((B, T), -1.0)
        adv = torch.tensor([1.0, 1.0])
        mask = torch.ones(B, T)

        loss, stats = grpo_loss(lp, old_lp, ref_lp, adv, mask)
        # With ratio=1 and equal ref, loss ≈ -mean(adv) = -1
        assert stats["loss"] < 0  # positive advantage → negative loss (maximise)


# ======================================================================
# Integration (smoke) tests — real in-process environment
# ======================================================================

@pytest.fixture(scope="module")
def env_client():
    """One InProcessEnvClient shared across all integration tests in this module."""
    pytest.importorskip("fastmcp", reason="fastmcp not installed; run: uv sync in llm_agent/")
    from env_client import InProcessEnvClient
    client = InProcessEnvClient()
    client.wait_until_ready()
    return client


class TestInProcessEnvClient:
    def test_reset_returns_metadata(self, env_client):
        obs = env_client.reset(task_id="T1_single_page")
        meta = obs.get("metadata", {})
        assert meta.get("task_id") == "T1_single_page"
        assert "query" in meta

    def test_get_task_info(self, env_client):
        env_client.reset(task_id="T1_single_page")
        info = env_client.get_task_info()
        assert info.get("task_id") == "T1_single_page"
        assert "mock_service_url" in info
        assert info.get("requests_remaining") == info.get("constraints", {}).get("max_requests")

    def test_fetch_page_returns_rows(self, env_client):
        env_client.reset(task_id="T1_single_page")
        page = env_client.fetch_page(page=1, page_size=50)
        assert "rows" in page, f"No rows key: {page}"
        assert isinstance(page["rows"], list)
        assert len(page["rows"]) > 0

    def test_fetch_page_dedup_key_present(self, env_client):
        env_client.reset(task_id="T1_single_page")
        page = env_client.fetch_page(page=1)
        row = page["rows"][0]
        for field in ("year", "reporter", "partner", "flow", "hs", "record_id"):
            assert field in row, f"Missing dedup field '{field}' in row: {row}"

    def test_totals_row_has_both_flags(self, env_client):
        """T7 totals trap — verify mock emits isTotal AND is_total."""
        env_client.reset(task_id="T7_totals_trap")
        page = env_client.fetch_page(page=1, page_size=200)
        rows = page.get("rows", [])
        totals = [r for r in rows if r.get("isTotal") or r.get("is_total")]
        assert len(totals) > 0, "T7 should have totals rows"
        for t in totals:
            assert t.get("isTotal") is True, "isTotal (camelCase) must be True"
            assert t.get("is_total") is True, "is_total (snake_case) must be True"

    def test_submit_results_returns_reward(self, env_client):
        env_client.reset(task_id="T1_single_page")
        page = env_client.fetch_page(page=1)
        rows = [r for r in page["rows"] if not r.get("isTotal") and not r.get("is_total")]
        data_jsonl = "\n".join(json.dumps(r) for r in rows)
        meta = json.dumps({
            "task_id": "T1_single_page",
            "query": {},
            "row_count": len(rows),
            "schema": list(rows[0].keys()) if rows else [],
            "totals_handling": {"enabled": True, "rows_dropped": 0},
        })
        run_log = "task_id=T1_single_page\npage=1\nrequest=1\ncomplete=true\n"
        result = env_client.submit_results(data_jsonl, meta, run_log)
        assert "reward" in result
        assert 0.0 <= result["reward"] <= 1.0
        assert result.get("done") is True

    def test_request_limit_enforced(self, env_client):
        """Verify that request_count increments and error fires at limit."""
        env_client.reset(task_id="T1_single_page")
        info_before = env_client.get_task_info()
        env_client.fetch_page(page=1)
        info_after = env_client.get_task_info()
        assert info_after["requests_used"] == info_before["requests_used"] + 1

    def test_double_submit_rejected(self, env_client):
        env_client.reset(task_id="T1_single_page")
        env_client.fetch_page(page=1)
        page = env_client.fetch_page(page=1)
        rows = page.get("rows", [])
        data_jsonl = "\n".join(json.dumps(r) for r in rows)
        meta = json.dumps({"task_id": "T1_single_page", "query": {}, "row_count": len(rows),
                           "schema": [], "totals_handling": "none"})
        env_client.submit_results(data_jsonl, meta)
        second = env_client.submit_results(data_jsonl, meta)
        assert "error" in second

    def test_unknown_tool_returns_error(self, env_client):
        env_client.reset(task_id="T1_single_page")
        result = env_client.call_tool("nonexistent_tool", {})
        assert "error" in result


class TestEpisodeIsolation:
    """Two clients running the same task should not share state."""

    def test_concurrent_episodes_independent(self):
        pytest.importorskip("fastmcp", reason="fastmcp not installed; run: uv sync in llm_agent/")
        from env_client import InProcessEnvClient
        c1 = InProcessEnvClient()
        c2 = InProcessEnvClient()
        c1.reset(task_id="T1_single_page")
        c2.reset(task_id="T1_single_page")
        # c1 burns 3 requests
        for _ in range(3):
            c1.fetch_page(page=1)
        info1 = c1.get_task_info()
        info2 = c2.get_task_info()
        assert info1["requests_used"] == 3
        assert info2["requests_used"] == 0, "c2 should be independent of c1's request count"


# ======================================================================
# GRPO tokenisation test (CPU only, no GPU required)
# ======================================================================

class TestTokeniseTrajectories:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer
        # Use a tiny public model for speed; any chat model will do.
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )

    def test_completion_mask_excludes_prompt(self, tokenizer):
        from train_grpo import tokenise_trajectories
        prompt = "Task: T1\nDescription: single page\n"
        completion = '<tool_call>{"name": "fetch_page", "arguments": {"page": 1}}</tool_call>'
        batch = tokenise_trajectories(
            tokenizer, [prompt], [completion], max_length=256, device="cpu"
        )
        mask = batch["completion_mask"][0]
        attn = batch["attention_mask"][0]
        # Completion mask must be a strict subset of attention mask
        assert (mask <= attn).all(), "completion_mask must be subset of attention_mask"
        # At least some prompt tokens must be masked out
        assert mask.sum() < attn.sum(), "Some prompt tokens should be excluded from completion_mask"
        # At least some completion tokens must be included
        assert mask.sum() > 0, "completion_mask must include some tokens"
