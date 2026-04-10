"""
agent.py — LLM-powered comtrade agent.

The agent runs an agentic loop:
  1. Receive task description from the environment
  2. LLM decides which MCP tool to call next
  3. Execute the tool, feed the result back to the LLM
  4. Repeat until submit_results is called or max_steps reached
  5. Return (full_trajectory_text, reward)

The trajectory is the raw text the model generated (tool calls only —
tool results are observations, not model outputs). This trajectory is
later used as the "completion" in GRPO.

Tool call format understood by the model:
  <tool_call>{"name": "fetch_page", "arguments": {"page": 1}}</tool_call>

Tool result injected back into context:
  <tool_result>{"rows": [...], "has_more": true, "total_pages": 5}</tool_result>
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a data-fetching agent for the UN Comtrade API benchmark.

You have three tools:
  1. get_task_info()            — view task details (query params, constraints)
  2. fetch_page(page, page_size) — fetch one page of trade records
  3. submit_results(data_jsonl, metadata_json, run_log) — submit final answer

Rules:
- Fetch ALL pages until has_more=False.
- Deduplicate records by primary key: (year, reporter, partner, flow, hs, record_id).
- Drop any row where is_total=true OR isTotal=true (totals trap — both field names exist).
- If you get HTTP 429 or 500, retry that page (up to 3 times with a short wait).
- Call submit_results exactly once when done.

Tool call format — you MUST use this exact XML:
  <tool_call>{"name": "TOOL_NAME", "arguments": {ARG_JSON}}</tool_call>

Tool results will be shown as:
  <tool_result>RESULT_JSON</tool_result>

Think briefly before each tool call. Be efficient.
""".strip()


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
@dataclass
class AgentStep:
    """One turn in the agentic loop."""
    model_text: str       # raw text the model generated (thinking + tool_call tag)
    tool_name: str
    tool_args: dict
    tool_result: dict
    is_submit: bool = False


@dataclass
class Episode:
    """Complete episode trajectory."""
    task_info: dict
    steps: list[AgentStep] = field(default_factory=list)
    reward: float = 0.0
    score: float = 0.0
    breakdown: dict = field(default_factory=dict)
    error: Optional[str] = None

    def model_text_only(self) -> str:
        """All text the model generated (tool_calls only, no results)."""
        return "\n".join(s.model_text for s in self.steps)

    def full_conversation(self) -> str:
        """Full trajectory: model outputs + tool results interleaved."""
        parts = []
        for step in self.steps:
            parts.append(step.model_text)
            result_str = json.dumps(step.tool_result, ensure_ascii=False)
            # Truncate large results so they don't blow up the context
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "...(truncated)"
            parts.append(f"<tool_result>{result_str}</tool_result>")
        return "\n".join(parts)


# ------------------------------------------------------------------
# Tool call parsing
# ------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> Optional[tuple[str, dict]]:
    """
    Extract the LAST <tool_call>...</tool_call> block from the model output.
    Returns (tool_name, arguments) or None.
    """
    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        return None
    raw_json = matches[-1]
    try:
        obj = json.loads(raw_json)
        return obj.get("name", ""), obj.get("arguments", {})
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------
# LLM wrapper (supports HuggingFace transformers pipeline or OpenAI-style API)
# ------------------------------------------------------------------
class LLMBackend:
    """
    Wraps either a HuggingFace text-generation pipeline or an
    OpenAI-compatible API (for Qwen, DeepSeek, etc.).

    Usage:
      backend = LLMBackend.from_hf("Qwen/Qwen2.5-7B-Instruct")
      backend = LLMBackend.from_api("http://localhost:11434/v1", "qwen2.5:7b")
    """

    def __init__(self):
        self._pipe = None       # HuggingFace pipeline
        self._client = None     # OpenAI client
        self._model_name = ""

    @classmethod
    def from_hf(cls, model_name: str, device: str = "auto", **kwargs) -> "LLMBackend":
        """Load a HuggingFace model for local inference."""
        from transformers import pipeline as hf_pipeline

        backend = cls()
        backend._model_name = model_name
        backend._pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            device_map=device,
            trust_remote_code=True,
            **kwargs,
        )
        logger.info(f"Loaded HF model: {model_name}")
        return backend

    @classmethod
    def from_api(cls, base_url: str, model: str, api_key: str = "none") -> "LLMBackend":
        """Use an OpenAI-compatible API endpoint (vLLM, Ollama, etc.)."""
        from openai import OpenAI

        backend = cls()
        backend._model_name = model
        backend._client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"Using API backend: {base_url} model={model}")
        return backend

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a response given a chat messages list."""
        if self._pipe is not None:
            out = self._pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
            )
            return out[0]["generated_text"].strip()

        if self._client is not None:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            )
            return resp.choices[0].message.content.strip()

        raise RuntimeError("No backend initialized. Call from_hf() or from_api() first.")


# ------------------------------------------------------------------
# Core agent loop
# ------------------------------------------------------------------
class ComtradeAgent:
    """
    Runs a full episode on comtrade_env using an LLM backend.

    The agent calls tools in a loop:
      LLM → tool_call → env executes → tool_result → LLM → ...
    until submit_results is called or max_steps is reached.
    """

    def __init__(
        self,
        llm: LLMBackend,
        env_client,                   # ComtradeEnvClient
        max_steps: int = 30,
        max_tokens_per_step: int = 512,
        retry_limit: int = 3,
        temperature: float = 0.7,
    ):
        self.llm = llm
        self.env = env_client
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.retry_limit = retry_limit
        self.temperature = temperature

    def run_episode(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> Episode:
        """
        Run one full episode.

        Returns an Episode with the trajectory and final reward.
        """
        # Reset environment
        obs = self.env.reset(task_id=task_id, seed=seed)
        task_meta = obs.get("metadata", {})

        episode = Episode(task_info=task_meta)

        # Build initial user message
        task_desc = self._format_task_description(task_meta)

        # Conversation history for multi-turn LLM calls
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_desc},
        ]

        collected_rows: dict[str, dict] = {}  # deduplicated by primary key
        totals_dropped: int = 0  # count of is_total rows filtered
        dedup_key_fields = task_meta.get("dedup_key",
                           ["year", "reporter", "partner", "flow", "hs", "record_id"])
        ep_start_time = time.time()
        run_log_lines: list[str] = []
        run_log_lines.append(f"task_id={task_meta.get('task_id', 'unknown')}")

        for step_idx in range(self.max_steps):
            # ---- LLM generates next tool call ----
            model_output = self.llm.generate(
                messages=messages,
                max_new_tokens=self.max_tokens_per_step,
                temperature=self.temperature,
                stop=["</tool_call>"],
            )
            # Ensure the closing tag is present
            if "<tool_call>" in model_output and "</tool_call>" not in model_output:
                model_output += "</tool_call>"

            logger.debug(f"Step {step_idx+1} model output:\n{model_output}")

            parsed = parse_tool_call(model_output)
            if parsed is None:
                # Model did not generate a valid tool call — prompt it
                messages.append({"role": "assistant", "content": model_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call a tool. Use the format:\n"
                        '<tool_call>{"name": "TOOL", "arguments": {}}</tool_call>'
                    ),
                })
                continue

            tool_name, tool_args = parsed
            logger.info(f"Step {step_idx+1}: {tool_name}({tool_args})")
            run_log_lines.append(f"request={step_idx+1} tool={tool_name}")

            # ---- Execute tool ----
            retry_count = 0
            tool_result: dict = {}
            while retry_count <= self.retry_limit:
                try:
                    tool_result = self.env.call_tool(tool_name, tool_args)
                    break
                except Exception as exc:
                    retry_count += 1
                    logger.warning(f"Tool call failed ({retry_count}/{self.retry_limit}): {exc}")
                    time.sleep(1)
                    if retry_count > self.retry_limit:
                        tool_result = {"error": str(exc)}

            # Handle rate-limit / server-error retry at the tool level
            if tool_result.get("status") in (429, 500) or tool_result.get("retry"):
                wait = 2 * (retry_count + 1)
                logger.info(f"HTTP {tool_result.get('status')}, retrying in {wait}s")
                time.sleep(wait)
                tool_result = self.env.call_tool(tool_name, tool_args)

            # ---- Collect rows if this was fetch_page ----
            if tool_name == "fetch_page" and "rows" in tool_result:
                page_num = tool_result.get("page", step_idx + 1)
                run_log_lines.append(f"page={page_num}")
                for row in tool_result["rows"]:
                    if row.get("isTotal") or row.get("is_total"):
                        totals_dropped += 1
                        continue  # drop totals trap (mock returns isTotal, spec says is_total)
                    pk = _primary_key(row, dedup_key_fields)
                    collected_rows[pk] = row

            # ---- Record step ----
            step = AgentStep(
                model_text=model_output,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                is_submit=(tool_name == "submit_results"),
            )
            episode.steps.append(step)

            # ---- Append to conversation ----
            messages.append({"role": "assistant", "content": model_output})
            result_str = json.dumps(tool_result, ensure_ascii=False)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "...(truncated)"
            messages.append({
                "role": "user",
                "content": f"<tool_result>{result_str}</tool_result>",
            })

            # ---- Check for submission ----
            if tool_name == "submit_results":
                reward = tool_result.get("reward", 0.0)
                episode.reward = reward
                episode.score = tool_result.get("score", 0.0)
                episode.breakdown = tool_result.get("breakdown", {})
                run_log_lines.append(f"complete=true reward={reward:.4f}")
                logger.info(f"Episode done. reward={reward:.4f} score={episode.score:.1f}")
                return episode

            # ---- Auto-submit when all pages fetched ----
            if not tool_result.get("has_more", False) and tool_name == "fetch_page" and collected_rows:
                # All pages fetched — submit automatically (LLM can't generate
                # large JSONL payloads within its token limit, so we handle it)
                logger.info(f"All pages fetched. Auto-submitting {len(collected_rows)} rows.")
                data_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in collected_rows.values())
                metadata = json.dumps({
                    "task_id": task_meta.get("task_id"),
                    "query": task_meta.get("query", {}),
                    "row_count": len(collected_rows),
                    "schema": list(next(iter(collected_rows.values())).keys()) if collected_rows else [],
                    "dedup_key": dedup_key_fields,
                    "totals_handling": {"enabled": True, "rows_dropped": totals_dropped},
                    "execution_time_seconds": round(time.time() - ep_start_time, 2),
                })
                run_log = "\n".join(run_log_lines + ["complete=true"])
                submit_result = self.env.submit_results(data_jsonl, metadata, run_log)
                submit_step = AgentStep(
                    model_text="[auto-submit after all pages fetched]",
                    tool_name="submit_results",
                    tool_args={"row_count": len(collected_rows)},
                    tool_result=submit_result,
                    is_submit=True,
                )
                episode.steps.append(submit_step)
                episode.reward = submit_result.get("reward", 0.0)
                episode.score = submit_result.get("score", 0.0)
                episode.breakdown = submit_result.get("breakdown", {})
                logger.info(f"Episode done. reward={episode.reward:.4f} score={episode.score:.1f}")
                return episode

        # Max steps reached without submission — submit whatever we have
        logger.warning("Max steps reached. Force-submitting collected rows.")
        run_log_lines.append("complete=false")
        if collected_rows:
            data_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in collected_rows.values())
            metadata = {
                "task_id": task_meta.get("task_id"),
                "query": task_meta.get("query", {}),
                "row_count": len(collected_rows),
                "schema": list(next(iter(collected_rows.values())).keys()) if collected_rows else [],
                "dedup_key": dedup_key_fields,
                "totals_handling": {"enabled": True, "rows_dropped": totals_dropped},
                "execution_time_seconds": round(time.time() - ep_start_time, 2),
            }
            run_log = "\n".join(run_log_lines)
            try:
                result = self.env.submit_results(data_jsonl, json.dumps(metadata), run_log)
                episode.reward = result.get("reward", 0.0)
                episode.score = result.get("score", 0.0)
                episode.breakdown = result.get("breakdown", {})
            except Exception as exc:
                episode.error = str(exc)

        return episode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_task_description(meta: dict) -> str:
        task_id = meta.get("task_id", "unknown")
        description = meta.get("description", "")
        query = meta.get("query", {})
        constraints = meta.get("constraints", {})
        return (
            f"Task: {task_id}\n"
            f"Description: {description}\n\n"
            f"Query parameters:\n{json.dumps(query, indent=2)}\n\n"
            f"Constraints:\n{json.dumps(constraints, indent=2)}\n\n"
            "Instructions:\n"
            "1. Call get_task_info() to confirm parameters.\n"
            "2. Fetch all pages with fetch_page(page=1), fetch_page(page=2), ...\n"
            "3. Deduplicate records. Drop rows with is_total=true.\n"
            "4. Call submit_results() once with all clean records."
        )


_DEFAULT_DEDUP_FIELDS = ("year", "reporter", "partner", "flow", "hs", "record_id")


def _primary_key(row: dict, fields: tuple | list = _DEFAULT_DEDUP_FIELDS) -> str:
    return "|".join(str(row.get(k, "")) for k in fields)
