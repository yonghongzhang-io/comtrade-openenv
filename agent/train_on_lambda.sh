#!/bin/bash
# ============================================================================
# Lambda Labs GRPO training runbook — Qwen2.5-3B on H100 80GB
# ============================================================================
#
# Prerequisite on Lambda instance (H100 80GB recommended):
#   - Ubuntu 22.04 + CUDA 12.x
#   - Python 3.11+
#   - Git + curl
#
# One-shot usage:
#   ssh ubuntu@<lambda-ip>
#   git clone https://github.com/yonghongzhang-io/comtrade-openenv
#   cd comtrade-openenv
#   bash agent/train_on_lambda.sh
#
# Expected wall time: 50 iterations × ~45s = ~40 min (Qwen2.5-3B on H100 80GB).
# Expected cost on Lambda: ~$1-3 at $1.29/hr A100 80GB or $2.49/hr H100.
#
# After training completes, download artifacts back to your laptop:
#   scp ubuntu@<lambda-ip>:comtrade-openenv/grpo_gradient_training/metrics.jsonl .
#   scp ubuntu@<lambda-ip>:comtrade-openenv/grpo_gradient_training_summary.json .
# ============================================================================

set -euo pipefail

# --- 0. Sanity: must be on a GPU box ----------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi
nvidia-smi | head -20

# --- 1. Install deps (idempotent) -------------------------------------------
echo "=== Installing Python deps ==="
pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    transformers \
    accelerate \
    peft \
    trl \
    openai \
    requests \
    fastmcp \
    fastapi \
    uvicorn \
    pydantic \
    openenv-core

# Install the env package itself so `import server`, `import models` work.
pip install --quiet -e .

# --- 2. Training config -----------------------------------------------------
MODEL="${TRAIN_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
ITERS="${TRAIN_ITERS:-50}"
BATCH="${TRAIN_BATCH:-4}"
GROUP="${TRAIN_GROUP:-4}"
LR="${TRAIN_LR:-1e-5}"
MAX_STEPS="${TRAIN_MAX_STEPS:-30}"
SEQLEN="${TRAIN_SEQLEN:-2048}"
OUT_DIR="${TRAIN_OUT_DIR:-grpo_gradient_training}"

echo ""
echo "=== GRPO gradient training on Lambda ==="
echo "model=${MODEL}  iters=${ITERS}  batch=${BATCH}  group=${GROUP}  lr=${LR}"
echo "max_steps=${MAX_STEPS}  seqlen=${SEQLEN}  out=${OUT_DIR}"
echo ""

# --- 3. Launch training -----------------------------------------------------
python agent/train_grpo.py \
    --hf-model "${MODEL}" \
    --num-iterations "${ITERS}" \
    --batch-size "${BATCH}" \
    --group-size "${GROUP}" \
    --lr "${LR}" \
    --max-steps "${MAX_STEPS}" \
    --max-seq-length "${SEQLEN}" \
    --output-dir "${OUT_DIR}" \
    --save-every 25 \
    --curriculum-warmup-iters 5 \
    --temperature 0.7 \
    2>&1 | tee "${OUT_DIR}.log"

# --- 4. Post-process: summary JSON for easy integration ---------------------
echo ""
echo "=== Post-processing ==="
python - <<PY
import json, time
from pathlib import Path
from statistics import mean, stdev

out_dir = Path("${OUT_DIR}")
metrics_path = out_dir / "metrics.jsonl"
if not metrics_path.exists():
    print(f"ERROR: {metrics_path} not found — training may have failed before iter 1")
    raise SystemExit(1)

metrics = [json.loads(l) for l in metrics_path.read_text().strip().split("\n") if l.strip()]
mean_rewards = [m["mean_reward"] for m in metrics]
losses = [m.get("loss") for m in metrics if "loss" in m]
kls = [m.get("kl") for m in metrics if "kl" in m]
first_10 = mean_rewards[:10]
last_10 = mean_rewards[-10:]

summary = {
    "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": "${MODEL}",
    "num_iterations": len(metrics),
    "config": {
        "batch_size": ${BATCH},
        "group_size": ${GROUP},
        "lr": ${LR},
        "max_steps": ${MAX_STEPS},
        "seq_len": ${SEQLEN},
    },
    "reward_first_10_mean": round(mean(first_10), 4) if first_10 else None,
    "reward_last_10_mean": round(mean(last_10), 4) if last_10 else None,
    "reward_improvement": round(mean(last_10) - mean(first_10), 4) if (first_10 and last_10) else None,
    "reward_overall_mean": round(mean(mean_rewards), 4),
    "reward_overall_std": round(stdev(mean_rewards), 4) if len(mean_rewards) > 1 else 0.0,
    "loss_final": losses[-1] if losses else None,
    "kl_final": kls[-1] if kls else None,
    "gradient_steps_taken": len(losses),
}
open("grpo_gradient_training_summary.json", "w").write(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"\nSaved: grpo_gradient_training_summary.json")
PY

echo ""
echo "=== Done! Artifacts to download ==="
echo "  ${OUT_DIR}/metrics.jsonl         — per-iter training metrics"
echo "  ${OUT_DIR}.log                    — full stdout"
echo "  grpo_gradient_training_summary.json  — aggregated summary"
echo ""
echo "From your laptop:"
echo "  scp ubuntu@<LAMBDA_IP>:comtrade-openenv/${OUT_DIR}/metrics.jsonl ."
echo "  scp ubuntu@<LAMBDA_IP>:comtrade-openenv/grpo_gradient_training_summary.json ."
