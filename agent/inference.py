"""
inference.py — Evaluate a trained or API-based LLM agent on all comtrade_env tasks.

Runs all benchmark tasks (T1-T10), collects rewards and scores, prints a summary table,
and writes results to a JSON file.

Usage:
  # Rule-based baseline (no LLM):
  python inference.py --mode rule-based

  # API-based LLM (Ollama / vLLM / any OpenAI-compatible):
  python inference.py --mode llm --api-url http://localhost:11434/v1 --api-model qwen2.5:7b

  # HuggingFace local model:
  python inference.py --mode llm --hf-model Qwen/Qwen2.5-7B-Instruct

  # Trained checkpoint:
  python inference.py --mode llm --hf-model ./grpo_output/final
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference")

sys.path.insert(0, str(Path(__file__).parent))

from env_client import InProcessEnvClient
from agent import ComtradeAgent, LLMBackend
from smoke_test import run_rule_based_episode

ALL_TASKS = [
    "T1_single_page",
    "T2_multi_page",
    "T3_duplicates",
    "T4_rate_limit_429",
    "T5_server_error_500",
    "T6_page_drift",
    "T7_totals_trap",
    "T8_mixed_faults",
    "T9_adaptive_adversary",
    "T10_multi_agent_coop",
]


def run_llm_episode(llm: LLMBackend, task_id: str, seed: int = 42, max_steps: int = 30) -> dict:
    env = InProcessEnvClient()
    agent = ComtradeAgent(llm=llm, env_client=env, max_steps=max_steps, temperature=0.0)
    ep = agent.run_episode(task_id=task_id, seed=seed)
    return {
        "task_id": task_id,
        "reward": ep.reward,
        "score": ep.score,
        "breakdown": ep.breakdown,
        "error": ep.error,
        "steps": len(ep.steps),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate comtrade agent on all tasks")
    p.add_argument("--mode", choices=["rule-based", "llm"], default="rule-based",
                   help="Agent mode: rule-based (no LLM) or llm")
    p.add_argument("--hf-model", type=str, default=None,
                   help="HuggingFace model name or local checkpoint path")
    p.add_argument("--api-url", type=str, default=None,
                   help="OpenAI-compatible API base URL")
    p.add_argument("--api-model", type=str, default="qwen2.5:7b",
                   help="Model name for API backend")
    p.add_argument("--max-steps", type=int, default=30,
                   help="Max agent steps per episode (default: 30)")
    p.add_argument("--output", type=str, default="inference_results.json",
                   help="Output file for results (default: inference_results.json)")
    args = p.parse_args()

    llm = None
    if args.mode == "llm":
        if args.hf_model:
            llm = LLMBackend.from_hf(args.hf_model)
        elif args.api_url:
            llm = LLMBackend.from_api(args.api_url, args.api_model)
        else:
            p.error("--mode llm requires --hf-model or --api-url")

    results = []
    total_reward = 0.0
    total_score = 0.0

    print(f"\n{'='*70}")
    print(f"{'Task':<25} {'Score':>8} {'Reward':>8} {'Steps':>6}  Breakdown")
    print(f"{'-'*70}")

    for task_id in ALL_TASKS:
        t0 = time.time()
        if args.mode == "rule-based":
            r = run_rule_based_episode(task_id)
            r["task_id"] = task_id
            r["steps"] = 0
        else:
            r = run_llm_episode(llm, task_id, max_steps=args.max_steps)

        elapsed = time.time() - t0
        r["elapsed_s"] = round(elapsed, 1)
        results.append(r)

        total_reward += r["reward"]
        total_score += r["score"]

        bd = r.get("breakdown", {})
        bd_str = " ".join(f"{k[:4]}={v}" for k, v in bd.items()) if bd else ""
        err = f" ERR: {r['error']}" if r.get("error") else ""
        print(f"{task_id:<25} {r['score']:>8.1f} {r['reward']:>8.4f} {r.get('steps',0):>6}  {bd_str}{err}")

    n = len(ALL_TASKS)
    print(f"{'-'*70}")
    print(f"{'TOTAL':<25} {total_score:>8.1f} {total_reward:>8.4f}")
    print(f"{'AVERAGE':<25} {total_score/n:>8.1f} {total_reward/n:>8.4f}")
    print(f"{'='*70}\n")

    summary = {
        "mode": args.mode,
        "model": args.hf_model or args.api_model or "rule-based",
        "total_score": round(total_score, 1),
        "average_score": round(total_score / n, 1),
        "total_reward": round(total_reward, 4),
        "average_reward": round(total_reward / n, 4),
        "tasks": results,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
