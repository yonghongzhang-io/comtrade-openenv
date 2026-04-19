#!/bin/bash
# Multi-seed eval for variance quantification.
# Usage: bash agent/run_multiseed.sh <backend> <task_short> <label>
#   backend: kimi | anthropic | groq | openai
#   task_short: T4 | T5 | T9 | T10 (short ID from run_eval.py TASK_ID_MAP)
#   label: string label (e.g. kimi_t9)
#
# Runs seeds {42, 137, 2024, 7, 31} sequentially and writes:
#   eval_<label>_seed<N>_<timestamp>.json (five files)
#   multiseed_<label>_summary.json        (aggregated mean/std)

set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${SRC}/.."

BACKEND="${1:?backend required: kimi|anthropic|groq|openai}"
TASK="${2:?task required: T1..T10}"
LABEL="${3:?label required}"

case "$BACKEND" in
  kimi)
    API_URL="https://api.moonshot.cn/v1"
    API_MODEL="moonshot-v1-128k"
    ENV_KEY="KIMI_API_KEY"
    ;;
  anthropic)
    API_URL="https://api.anthropic.com/v1"
    API_MODEL="claude-sonnet-4-6"
    ENV_KEY="ANTHROPIC_API_KEY"
    ;;
  groq)
    API_URL="https://api.groq.com/openai/v1"
    API_MODEL="llama-3.3-70b-versatile"
    ENV_KEY="GROQ_API_KEY"
    ;;
  openai)
    API_URL="https://api.openai.com/v1"
    API_MODEL="${OPENAI_MODEL:-gpt-5}"
    ENV_KEY="OPENAI_API_KEY"
    ;;
  *)
    echo "ERROR: unknown backend: $BACKEND"
    exit 1
    ;;
esac

SEEDS=(42 137 2024 7 31)
echo "=== Multi-seed run: backend=$BACKEND task=$TASK label=$LABEL ==="
echo "Seeds: ${SEEDS[*]}"
echo ""

for SEED in "${SEEDS[@]}"; do
  echo ">>> Seed $SEED <<<"
  python agent/run_eval.py \
    --api-url "$API_URL" \
    --api-model "$API_MODEL" \
    --env-key "$ENV_KEY" \
    --tasks "$TASK" \
    --label "${LABEL}_seed${SEED}" \
    --seed "$SEED"
done

# Aggregate
python - <<PY
import json, glob, statistics, time
label = "${LABEL}"
files = sorted(glob.glob(f"eval_{label}_seed*.json"))
scores = []
per_seed = []
for f in files:
    d = json.loads(open(f).read())
    r = d["results"][0]
    scores.append(r["score"])
    per_seed.append({"seed": d["seed"], "score": r["score"], "breakdown": r.get("breakdown", {})})
out = {
    "label": label,
    "backend": "${BACKEND}",
    "model": "${API_MODEL}",
    "task": "${TASK}",
    "n_seeds": len(scores),
    "seeds": [p["seed"] for p in per_seed],
    "mean_score": round(statistics.mean(scores), 2),
    "std_score": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
    "min_score": round(min(scores), 2),
    "max_score": round(max(scores), 2),
    "per_seed": per_seed,
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
path = f"multiseed_{label}_summary.json"
open(path, "w").write(json.dumps(out, indent=2))
print(f"\n=== Summary: {label} ===")
print(f"  mean={out['mean_score']}  std={out['std_score']}  range=[{out['min_score']}, {out['max_score']}]")
print(f"  seeds={out['seeds']}")
print(f"  scores={scores}")
print(f"Wrote: {path}")
PY
