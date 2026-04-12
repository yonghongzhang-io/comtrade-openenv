"""
run_baseline_grpo_sim.py — Run rule-based agent through GRPO-style rollouts.

Demonstrates the full training pipeline end-to-end without requiring a GPU
or LLM API. Uses the rule-based baseline as the "policy" with slight
randomisation (page_size variation) to create reward variance across rollouts.

This produces real metrics.jsonl that plot_training.py can visualise.

Usage:
    cd comtrade_env
    python agent/run_baseline_grpo_sim.py --iterations 10
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grpo_sim")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _setup():
    from env_client import _setup_openenv_path, InProcessEnvClient
    _setup_openenv_path()
    return InProcessEnvClient


def run_one_episode(EnvClient, task_id: str, page_size: int = 500) -> dict:
    """Run one rule-based episode with given page_size."""
    import json as _json
    import urllib.parse, urllib.request, urllib.error, os

    env = EnvClient()
    obs = env.reset(task_id=task_id)
    meta = obs.get("metadata", {})
    task_info = env.get_task_info()

    collected = {}
    totals_dropped = 0
    page = 1
    request_count = 0
    retry_total = 0
    retry_429 = 0
    retry_500 = 0
    run_log = [f"task_id={task_id}"]

    while True:
        result = env.fetch_page(page=page, page_size=page_size)
        request_count += 1
        run_log.append(f"request={request_count} page={page}")

        if result.get("status") in (429, 500) or result.get("retry"):
            status = result.get("status")
            retry_total += 1
            if status == 429:
                retry_429 += 1
            if status == 500:
                retry_500 += 1
            run_log.append(f"retry status={status} attempt=1 wait_s=0.1")
            time.sleep(0.1)
            result = env.fetch_page(page=page, page_size=page_size)
            request_count += 1
            run_log.append(f"request={request_count} page={page}")

        for row in result.get("rows", []):
            if row.get("isTotal") or row.get("is_total"):
                totals_dropped += 1
                continue
            pk = "|".join(str(row.get(k, "")) for k in
                         ("year", "reporter", "partner", "flow", "hs", "record_id"))
            collected[pk] = row

        if not result.get("has_more", False):
            break
        page += 1

    run_log.append(f"retry_count={retry_total}")
    run_log.append("complete=true")
    data_jsonl = "\n".join(_json.dumps(r, ensure_ascii=False) for r in collected.values())
    metadata = _json.dumps({
        "task_id": task_id,
        "query": task_info.get("query", {}),
        "row_count": len(collected),
        "schema": list(next(iter(collected.values())).keys()) if collected else [],
        "dedup_key": ["year", "reporter", "partner", "flow", "hs", "record_id"],
        "totals_handling": {"enabled": True, "rows_dropped": totals_dropped},
        "request_count": request_count,
        "request_budget": task_info.get("constraints", {}).get("max_requests", 100),
        "request_stats": {
            "retries_total": retry_total,
            "retries_429": retry_429,
            "retries_500": retry_500,
        },
        "execution_time_seconds": 0.5,
    })
    submit = env.submit_results(data_jsonl, metadata, "\n".join(run_log))

    return {
        "task_id": task_id,
        "reward": submit.get("reward", 0.0),
        "score": submit.get("score", 0.0),
        "breakdown": submit.get("breakdown", {}),
        "page_size": page_size,
        "requests": request_count,
        "rows_collected": len(collected),
        "totals_dropped": totals_dropped,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--output", type=str, default="agent/grpo_sim_metrics.jsonl")
    args = p.parse_args()

    EnvClient = _setup()

    ALL_TASKS = [
        "T1_single_page", "T2_multi_page", "T3_duplicates",
        "T4_rate_limit_429", "T5_server_error_500", "T6_page_drift",
        "T7_totals_trap", "T8_mixed_faults", "T9_adaptive_adversary",
        "T10_constrained_budget",
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running {args.iterations} iterations, group_size={args.group_size}")

    with open(out_path, "w") as f:
        for iteration in range(1, args.iterations + 1):
            t_start = time.time()
            batch_tasks = random.choices(ALL_TASKS, k=2)
            all_rewards = []
            task_rewards = defaultdict(list)

            for task_id in batch_tasks:
                for g in range(args.group_size):
                    # Vary page_size to create reward variance (simulates LLM stochasticity)
                    ps = random.choice([100, 250, 500, 500, 500])
                    ep = run_one_episode(EnvClient, task_id, page_size=ps)
                    all_rewards.append(ep["reward"])
                    task_rewards[task_id].append(ep["reward"])

            mean_r = sum(all_rewards) / len(all_rewards)
            max_r = max(all_rewards)
            elapsed = time.time() - t_start

            record = {
                "iteration": iteration,
                "mean_reward": round(mean_r, 4),
                "max_reward": round(max_r, 4),
                "n_rollouts": len(all_rewards),
                "elapsed_s": round(elapsed, 2),
                "task_rewards": {k: round(sum(v)/len(v), 4) for k, v in task_rewards.items()},
                "tasks": batch_tasks,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            logger.info(
                f"Iter {iteration}/{args.iterations}  "
                f"mean={mean_r:.4f}  max={max_r:.4f}  "
                f"tasks={batch_tasks}  elapsed={elapsed:.1f}s"
            )

    logger.info(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
