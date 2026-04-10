"""
smoke_test.py — verify that one full episode returns a non-zero reward.

Uses a rule-based deterministic agent (no LLM required).
Runs in-process: auto-discovers OpenEnv, instantiates ComtradeEnvironment
directly, and hits the embedded mock service. No external HTTP server needed.

Expected layout:
  comtrade_env/
    agent/smoke_test.py   ← this file
    server/               ← environment + mock service

Usage:
  cd comtrade_env
  python agent/smoke_test.py --task T1_single_page

  # Run all tasks:
  for t in T1_single_page T2_multi_page T3_duplicates \
            T4_rate_limit_429 T5_server_error_500 T6_page_drift T7_totals_trap; do
      python agent/smoke_test.py --task $t
  done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import os
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("smoke_test")

MOCK_PORT = int(os.environ.get("MOCK_SERVICE_PORT", "7654"))


def _setup_openenv_path() -> None:
    """Add OpenEnv source paths, searching relative to this file.

    Layout: comtrade_env/agent/smoke_test.py
            comtrade_env/server/
            OpenEnv/src/  (two levels up from comtrade_env/)
    """
    repo_root = Path(__file__).resolve().parent.parent  # comtrade_env/
    candidates = [
        repo_root.parent.parent,               # OpenEnv/envs/comtrade_env → OpenEnv
        repo_root.parent / "OpenEnv",
        repo_root.parent.parent / "OpenEnv",
    ]
    openenv_root = next((p for p in candidates if (p / "src").exists()), None)
    if openenv_root is None:
        raise RuntimeError(
            "Cannot find OpenEnv/src/. Searched:\n"
            + "\n".join(f"  {c}" for c in candidates)
        )
    for sub_path in [
        openenv_root / "src",
        openenv_root / "envs",
        repo_root,
    ]:
        s = str(sub_path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _fetch_page(task_id: str, query: dict, page: int, page_size: int = 500) -> dict:
    """Fetch one page from the embedded mock service via HTTP."""
    params = urllib.parse.urlencode({
        "task_id": task_id,
        "page": page,
        "page_size": page_size,
        **query,
    })
    url = f"http://localhost:{MOCK_PORT}/api/data?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "retry": exc.code in (429, 500), "rows": []}
    except Exception as exc:
        return {"error": str(exc), "rows": []}


def run_rule_based_episode(task_id: str) -> dict:
    """
    Deterministic baseline agent:
      1. Fetch pages until has_more=False
      2. Deduplicate by primary key
      3. Drop is_total rows
      4. Submit via judge
    """
    _setup_openenv_path()
    from comtrade_env.server.comtrade_env_environment import ComtradeEnvironment

    env = ComtradeEnvironment()
    obs = env.reset(task_id=task_id)
    meta = obs.metadata
    task = env._current_task

    logger.info(f"Task: {meta.get('task_id')}  desc: {meta.get('description')}")

    collected: dict[str, dict] = {}
    page = 1
    run_log_lines = [f"task_id={task.task_id}"]

    while True:
        result = _fetch_page(task.task_id, task.query, page=page, page_size=500)

        # Retry on transient faults
        if result.get("retry") or result.get("status") in (429, 500):
            logger.warning(f"HTTP {result.get('status')} on page {page}, retrying in 3s...")
            time.sleep(3)
            result = _fetch_page(task.task_id, task.query, page=page, page_size=500)

        if "error" in result and not result.get("rows"):
            logger.error(f"Fetch error on page {page}: {result}")
            break

        rows = result.get("rows", [])
        run_log_lines.append(f"page={page}")

        for row in rows:
            # Drop totals trap rows — check both field names for compatibility
            if row.get("isTotal") or row.get("is_total"):
                continue
            pk = "|".join(str(row.get(k, "")) for k in
                         ("year", "reporter", "partner", "flow", "hs", "record_id"))
            collected[pk] = row

        has_more = result.get("has_more", False)
        total_pages = result.get("total_pages", 1)
        logger.info(f"Page {page}/{total_pages}: {len(rows)} rows returned, "
                    f"{len(collected)} unique collected, has_more={has_more}")

        if not has_more:
            break
        page += 1

    logger.info(f"Collected {len(collected)} unique rows total.")
    run_log_lines.append(f"request={page}")
    run_log_lines.append("complete=true")

    # Build submission payload
    data_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in collected.values())
    first_row = next(iter(collected.values()), {})
    metadata = {
        "task_id": task.task_id,
        "query": task.query,
        "row_count": len(collected),
        "schema": list(first_row.keys()) if first_row else [],
        "dedup_key": ["year", "reporter", "partner", "flow", "hs", "record_id"],
        "totals_handling": {"enabled": True, "rows_dropped": 0},
    }
    run_log = "\n".join(run_log_lines)

    # Write output and score via judge
    out_base = tempfile.mkdtemp(prefix="smoke_test_")
    out_dir = Path(out_base) / task.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.jsonl").write_text(data_jsonl, encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (out_dir / "run.log").write_text(run_log, encoding="utf-8")

    score_result = env._judge_mod.score_task(
        task_id=task.task_id,
        output_dir=out_base,
        task=task,
    )
    reward = round(min(score_result.total / 100.0, 1.0), 4)
    return {
        "reward": reward,
        "score": score_result.total,
        "breakdown": score_result.breakdown,
        "errors": score_result.errors,
    }


def main():
    p = argparse.ArgumentParser(description="Rule-based smoke test for comtrade_env")
    p.add_argument("--task", default="T1_single_page",
                   help="Task ID to run (default: T1_single_page)")
    args = p.parse_args()

    result = run_rule_based_episode(task_id=args.task)
    reward = result.get("reward", 0.0)
    score = result.get("score", 0.0)
    breakdown = result.get("breakdown", {})
    errors = result.get("errors", [])

    logger.info(f"\n{'='*50}")
    logger.info(f"reward = {reward:.4f}  score = {score:.1f}")
    logger.info(f"breakdown = {json.dumps(breakdown, indent=2)}")
    if errors:
        logger.warning(f"errors = {errors}")

    if reward > 0:
        logger.info("SMOKE TEST PASSED")
        sys.exit(0)
    else:
        logger.error("SMOKE TEST FAILED: reward=0")
        sys.exit(1)


if __name__ == "__main__":
    main()
