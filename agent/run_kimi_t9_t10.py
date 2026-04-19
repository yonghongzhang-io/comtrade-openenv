"""
One-off: run the Kimi LLM agent on T9 and T10 only, to see what it looks like
before deciding whether to publish. Does NOT overwrite llm_results_kimi.json.

Usage:
  export OPENAI_API_KEY=<your-moonshot-key>

  # Moonshot international (recommended):
  python agent/run_kimi_t9_t10.py \
      --api-url https://api.moonshot.ai/v1 \
      --api-model moonshot-v1-32k

  # Moonshot China:
  python agent/run_kimi_t9_t10.py \
      --api-url https://api.moonshot.cn/v1 \
      --api-model moonshot-v1-32k

Why 32k not 8k: T9/T10 can have longer episodes; the original 8k run likely
truncated on these two, which is why they were omitted from the published
snapshot. 32k gives breathing room. If you want to reproduce the exact 8k
conditions, pass --api-model moonshot-v1-8k.

Output goes to a timestamped file in the current directory so nothing gets
silently overwritten:
  kimi_t9_t10_<timestamp>.json
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
    """Load KEY=value pairs from comtrade_env/.env into os.environ if present.
    Minimal parser, no python-dotenv dependency. Existing env vars win."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

from env_client import InProcessEnvClient
from agent import ComtradeAgent, LLMBackend


TASKS = ["T9_adaptive_adversary", "T10_constrained_budget"]


def run_one(llm: LLMBackend, task_id: str, max_steps: int, seed: int = 42) -> dict:
    env = InProcessEnvClient()
    agent = ComtradeAgent(llm=llm, env_client=env, max_steps=max_steps, temperature=0.0)
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
        print(f"\n--- full traceback for {task_id} ---\n{tb}--- end traceback ---\n",
              file=sys.stderr)
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
    p.add_argument("--api-url", default="https://api.moonshot.ai/v1",
                   help="Moonshot API base URL (default: https://api.moonshot.ai/v1)")
    p.add_argument("--api-model", default="moonshot-v1-32k",
                   help="Model name (default: moonshot-v1-32k)")
    p.add_argument("--api-key", default=None,
                   help="API key (default: read from OPENAI_API_KEY env)")
    p.add_argument("--max-steps", type=int, default=30,
                   help="Max agent steps per episode (default: 30)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("MOONSHOT_API_KEY")
        or os.environ.get("KIMI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        sys.exit("ERROR: set MOONSHOT_API_KEY (or KIMI_API_KEY / OPENAI_API_KEY) env or pass --api-key")

    llm = LLMBackend.from_api(args.api_url, args.api_model, api_key=api_key)

    print(f"\n{'='*70}")
    print(f"Running Kimi ({args.api_model}) on T9 + T10 (private test)")
    print(f"{'='*70}\n")
    print(f"{'Task':<25} {'Score':>8} {'Reward':>8} {'Steps':>6} {'Time':>6}  Breakdown")
    print(f"{'-'*70}")

    results: list[dict] = []
    for task_id in TASKS:
        r = run_one(llm, task_id, max_steps=args.max_steps, seed=args.seed)
        results.append(r)
        bd = r.get("breakdown", {})
        bd_str = " ".join(f"{k[:4]}={v}" for k, v in bd.items()) if bd else ""
        err = f"  ERR: {r['error']}" if r.get("error") else ""
        print(f"{r['task_id']:<25} {r['score']:>8.1f} {r['reward']:>8.4f} "
              f"{r.get('steps', 0):>6} {r['elapsed_s']:>5.1f}s  {bd_str}{err}")

    print(f"{'-'*70}\n")

    out = {
        "note": "Private probe on T9+T10, not merged into llm_results_kimi.json",
        "model": args.api_model,
        "api_url": args.api_url,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "results": results,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"kimi_t9_t10_{ts}.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
