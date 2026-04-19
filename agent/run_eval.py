"""
run_eval.py — Generic multi-task LLM evaluation runner for ComtradeBench.

Supports:
  * Any OpenAI-compatible endpoint (Moonshot, Nebius, Groq, local Ollama, etc.)
  * Any subset of tasks (--tasks T1 T4 T5 | --all)
  * Optional custom system prompt (--prompt-file path/to/prompt.txt | --prompt "...")
  * Timestamped JSON outputs in the current directory (never overwrites)
  * Auto-loads comtrade_env/.env for API keys

Examples:
  # Kimi-128k apples-to-apples rerun on T1-T8
  python agent/run_eval.py \
      --api-url https://api.moonshot.cn/v1 \
      --api-model moonshot-v1-128k \
      --env-key KIMI_API_KEY \
      --tasks T1 T2 T3 T4 T5 T6 T7 T8 \
      --label kimi_128k_t1_t8

  # DeepSeek V3 via Nebius, all 10 tasks
  python agent/run_eval.py \
      --api-url https://api.studio.nebius.ai/v1 \
      --api-model deepseek-ai/DeepSeek-V3-0324 \
      --env-key NEBIUS_API_KEY \
      --all \
      --label deepseek_v3

  # T4/T5 EVENTS scratchpad ablation (enhanced prompt)
  python agent/run_eval.py \
      --api-url https://api.moonshot.cn/v1 \
      --api-model moonshot-v1-128k \
      --env-key KIMI_API_KEY \
      --tasks T4 T5 \
      --prompt-file agent/prompts/enhanced_events.txt \
      --label kimi_ablation_events
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _load_dotenv() -> None:
    """Minimal .env loader (no python-dotenv dep). Existing env vars win."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

from env_client import InProcessEnvClient
from agent import ComtradeAgent, LLMBackend


TASK_ID_MAP = {
    "T1": "T1_single_page",
    "T2": "T2_multi_page",
    "T3": "T3_duplicates",
    "T4": "T4_rate_limit_429",
    "T5": "T5_server_error_500",
    "T6": "T6_page_drift",
    "T7": "T7_totals_trap",
    "T8": "T8_mixed_faults",
    "T9": "T9_adaptive_adversary",
    "T10": "T10_constrained_budget",
}
ALL_TASKS = list(TASK_ID_MAP.values())


def run_one(
    llm: LLMBackend,
    task_id: str,
    max_steps: int,
    seed: int,
    system_prompt: str | None,
) -> dict:
    env = InProcessEnvClient()
    agent = ComtradeAgent(
        llm=llm,
        env_client=env,
        max_steps=max_steps,
        temperature=0.0,
        system_prompt=system_prompt,
    )
    t0 = time.time()
    try:
        ep = agent.run_episode(task_id=task_id, seed=seed)
        return {
            "task_id": task_id,
            "reward": ep.reward,
            "score": ep.score,
            "breakdown": ep.breakdown,
            "error": ep.error,
            "steps": len(ep.steps),
            "elapsed_s": round(time.time() - t0, 1),
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n--- traceback for {task_id} ---\n{tb}--- end ---\n", file=sys.stderr)
        return {
            "task_id": task_id,
            "reward": 0.0,
            "score": 0.0,
            "breakdown": {},
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb,
            "steps": 0,
            "elapsed_s": round(time.time() - t0, 1),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-url", required=True, help="OpenAI-compatible API base URL")
    p.add_argument("--api-model", required=True, help="Model name")
    p.add_argument("--env-key", default="KIMI_API_KEY",
                   help="Env var name for API key (default: KIMI_API_KEY). Falls back to common names.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Task IDs to run (e.g. T1 T4 T5) or full IDs")
    p.add_argument("--all", action="store_true", help="Run all 10 tasks")
    p.add_argument("--prompt-file", default=None, help="Path to custom system prompt")
    p.add_argument("--prompt", default=None, help="Inline custom system prompt string")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="run", help="Label for output filename")
    args = p.parse_args()

    # Resolve API key
    api_key = (
        os.environ.get(args.env_key)
        or os.environ.get("MOONSHOT_API_KEY")
        or os.environ.get("KIMI_API_KEY")
        or os.environ.get("NEBIUS_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        sys.exit(f"ERROR: no API key in env. Tried {args.env_key} + common fallbacks.")

    # Resolve task list
    if args.all:
        tasks = ALL_TASKS
    elif args.tasks:
        tasks = [TASK_ID_MAP.get(t, t) for t in args.tasks]
        for t in tasks:
            if t not in ALL_TASKS:
                sys.exit(f"ERROR: unknown task id: {t}")
    else:
        sys.exit("ERROR: must pass --tasks or --all")

    # Resolve system prompt override
    system_prompt = None
    prompt_label = "default"
    if args.prompt_file:
        system_prompt = Path(args.prompt_file).read_text()
        prompt_label = f"file:{args.prompt_file}"
    elif args.prompt:
        system_prompt = args.prompt
        prompt_label = "inline"

    llm = LLMBackend.from_api(args.api_url, args.api_model, api_key=api_key)

    print(f"\n{'='*78}")
    print(f"Model: {args.api_model}  |  Prompt: {prompt_label}  |  Tasks: {len(tasks)}")
    print(f"{'='*78}\n")
    print(f"{'Task':<26} {'Score':>8} {'Reward':>8} {'Steps':>6} {'Time':>6}  Breakdown")
    print(f"{'-'*78}")

    results: list[dict] = []
    total_score = 0.0
    total_reward = 0.0
    for task_id in tasks:
        r = run_one(llm, task_id, args.max_steps, args.seed, system_prompt)
        results.append(r)
        total_score += r["score"]
        total_reward += r["reward"]
        bd = r.get("breakdown", {})
        bd_str = " ".join(f"{k[:4]}={v}" for k, v in bd.items()) if bd else ""
        err = f"  ERR: {r['error']}" if r.get("error") else ""
        print(f"{r['task_id']:<26} {r['score']:>8.1f} {r['reward']:>8.4f} "
              f"{r.get('steps', 0):>6} {r['elapsed_s']:>5.1f}s  {bd_str}{err}")

    n = len(tasks) or 1
    avg_score = total_score / n
    avg_reward = total_reward / n
    print(f"{'-'*78}")
    print(f"{'AVERAGE':<26} {avg_score:>8.1f} {avg_reward:>8.4f}")
    print(f"{'='*78}\n")

    out = {
        "label": args.label,
        "model": args.api_model,
        "api_url": args.api_url,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "prompt_variant": prompt_label,
        "tasks_run": tasks,
        "average_score": round(avg_score, 2),
        "average_reward": round(avg_reward, 4),
        "results": results,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"eval_{args.label}_{ts}.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
