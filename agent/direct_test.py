"""
direct_test.py — test ComtradeEnvironment directly (no HTTP server).

Instantiates the environment in-process; faster than smoke_test.py.
Auto-discovers OpenEnv relative to this file — no PYTHONPATH needed.

Expected directory layout:
  comtrade_env/
    agent/direct_test.py   ← this file
    server/                ← environment code

Usage:
  cd comtrade_env
  python agent/direct_test.py --task T1_single_page
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("direct_test")


def _setup_openenv_path():
    """Add OpenEnv source paths, searching relative to this file's location.

    Layout: comtrade_env/agent/direct_test.py  →  comtrade_env/server/
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


def run_direct_episode(task_id: str) -> dict:
    """Instantiate ComtradeEnvironment and run one full episode."""
    _setup_openenv_path()

    # Import directly from file to avoid comtrade_env/__init__.py → gradio chain
    import importlib.util as _ilu
    _env_file = Path(__file__).resolve().parent.parent / "server" / "comtrade_env_environment.py"
    _spec = _ilu.spec_from_file_location("comtrade_env_environment", _env_file)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ComtradeEnvironment = _mod.ComtradeEnvironment

    env = ComtradeEnvironment()

    # Reset
    obs = env.reset(task_id=task_id)
    meta = obs.metadata
    logger.info(f"Task: {meta.get('task_id')}  desc: {meta.get('description')}")

    # Fetch all pages via the MCP tools directly
    collected: dict[str, dict] = {}
    page = 1
    run_log_lines = [f"task_id={meta.get('task_id')}"]

    # Access MCP tools directly (they are closures in __init__)
    # We'll call them via the env's internal _mcp tools
    # Instead, call the environment's mock service via HTTP (it's already started)
    import urllib.request
    import urllib.parse
    import os

    MOCK_PORT = int(os.environ.get("MOCK_SERVICE_PORT", "7654"))
    task = env._current_task

    while True:
        params = urllib.parse.urlencode({
            "task_id": task.task_id,
            "page": page,
            "page_size": 500,
            **task.query,
        })
        url = f"http://localhost:{MOCK_PORT}/api/data?{params}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception as exc:
            # Handle 429/500
            import urllib.error
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 500):
                logger.warning(f"HTTP {exc.code} on page {page}, retrying in 3s...")
                time.sleep(3)
                with urllib.request.urlopen(url, timeout=10) as resp:
                    result = json.loads(resp.read().decode())
            else:
                logger.error(f"Fetch error page {page}: {exc}")
                break

        rows = result.get("rows", [])
        run_log_lines.append(f"page={page}")
        for row in rows:
            if row.get("isTotal") or row.get("is_total"):
                continue
            pk = "|".join(str(row.get(k, "")) for k in
                         ("year", "reporter", "partner", "flow", "hs", "record_id"))
            collected[pk] = row

        has_more = result.get("has_more", False)
        total_pages = result.get("total_pages", 1)
        logger.info(f"Page {page}/{total_pages}: {len(rows)} rows, has_more={has_more}")

        if not has_more:
            break
        page += 1

    logger.info(f"Collected {len(collected)} unique rows")
    run_log_lines.append(f"request={page}")
    run_log_lines.append("complete=true")

    # Build submission
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

    # Submit via judge directly
    import tempfile
    out_base = tempfile.mkdtemp(prefix="comtrade_test_")
    env._output_dir = out_base  # override so judge writes here
    env._submitted = False      # allow submission

    result = env._judge_mod.score_task(
        task_id=task.task_id,
        output_dir=out_base,
        task=task,
    )

    # Actually write the files and score
    from pathlib import Path as P
    out_dir = P(out_base) / task.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.jsonl").write_text(data_jsonl)
    (out_dir / "metadata.json").write_text(json.dumps(metadata))
    (out_dir / "run.log").write_text(run_log)

    result = env._judge_mod.score_task(
        task_id=task.task_id,
        output_dir=out_base,
        task=task,
    )
    reward = round(min(result.total / 100.0, 1.0), 4)
    return {
        "reward": reward,
        "score": result.total,
        "breakdown": result.breakdown,
        "errors": result.errors,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="T1_single_page")
    args = p.parse_args()

    result = run_direct_episode(task_id=args.task)
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
        logger.info("TEST PASSED")
        sys.exit(0)
    else:
        logger.error("TEST FAILED: reward=0")
        sys.exit(1)


if __name__ == "__main__":
    main()
