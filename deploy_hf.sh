#!/bin/bash
# Deploy comtrade_env to Hugging Face Spaces.
# Uses `hf upload` (handles LFS automatically — no git-lfs required).
# Prerequisites: hf auth login (run once to set token).
#
# Usage: bash /absolute/path/to/deploy_hf.sh

set -e

SPACE_REPO="yonghongzhang/comtrade-env"
SPACE_URL="https://huggingface.co/spaces/${SPACE_REPO}"
SRC="$(cd "$(dirname "$0")" && pwd)"
STAGE_DIR="/tmp/hf-comtrade-env-stage"

echo "=== Deploying comtrade_env to HF Space: ${SPACE_REPO} ==="

if ! hf auth whoami >/dev/null 2>&1; then
    echo "ERROR: Not logged in to HF. Run: hf auth login"
    exit 1
fi

# Stage a clean copy so we never upload __pycache__ / .git junk.
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

cp "${SRC}/README.md"                     "${STAGE_DIR}/"
cp "${SRC}/Dockerfile"                    "${STAGE_DIR}/"
cp "${SRC}/.dockerignore"                 "${STAGE_DIR}/"
cp "${SRC}/pyproject.toml"                "${STAGE_DIR}/"
cp "${SRC}/uv.lock"                       "${STAGE_DIR}/"
cp "${SRC}/openenv.yaml"                  "${STAGE_DIR}/"
cp "${SRC}/__init__.py"                   "${STAGE_DIR}/"
cp "${SRC}/client.py"                     "${STAGE_DIR}/"
cp "${SRC}/models.py"                     "${STAGE_DIR}/"
cp "${SRC}/blog_post.md"                  "${STAGE_DIR}/"
cp "${SRC}/llm_results_kimi.json"         "${STAGE_DIR}/"
cp "${SRC}/llm_results_claude.json"       "${STAGE_DIR}/"
cp "${SRC}/llm_results_llama.json"        "${STAGE_DIR}/"
cp "${SRC}/ablation_context_vs_prompt.json" "${STAGE_DIR}/"
cp "${SRC}/inference_results_baseline.json" "${STAGE_DIR}/"
cp "${SRC}/banner.png"                    "${STAGE_DIR}/"
cp "${SRC}/benchmark_results.png"         "${STAGE_DIR}/"
cp "${SRC}/training_curve.png"            "${STAGE_DIR}/"
cp -r "${SRC}/server"                     "${STAGE_DIR}/"
cp -r "${SRC}/green"                      "${STAGE_DIR}/"

# Scrub python bytecode anywhere under the staging tree.
find "${STAGE_DIR}" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${STAGE_DIR}" -type f -name '*.pyc' -delete

hf upload "${SPACE_REPO}" "${STAGE_DIR}" . \
    --repo-type=space \
    --commit-message="Deploy comtrade_env: green consistency + scope clarifications + landing page"

echo ""
echo "=== Deployed! ==="
echo "Space URL: ${SPACE_URL}"
echo "It may take 2-5 minutes for the Space to build and start."
