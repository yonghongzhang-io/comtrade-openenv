#!/bin/bash
# Deploy comtrade_env to Hugging Face Spaces
# Prerequisites: hf auth login (run once to set token)
#
# Usage: bash deploy_hf.sh

set -e

SPACE_REPO="yonghongzhang/comtrade-env"
SPACE_URL="https://huggingface.co/spaces/${SPACE_REPO}"

echo "=== Deploying comtrade_env to HF Space: ${SPACE_REPO} ==="

# Check auth
if ! hf auth whoami >/dev/null 2>&1; then
    echo "ERROR: Not logged in to HF. Run: hf auth login"
    exit 1
fi

# Clone the HF Space repo (or update if exists)
DEPLOY_DIR="/tmp/hf-comtrade-env-deploy"
rm -rf "${DEPLOY_DIR}"
git clone "https://huggingface.co/spaces/${SPACE_REPO}" "${DEPLOY_DIR}" 2>/dev/null || {
    echo "Cloning fresh..."
    mkdir -p "${DEPLOY_DIR}"
    cd "${DEPLOY_DIR}"
    git init
    git remote add origin "https://huggingface.co/spaces/${SPACE_REPO}"
}

cd "${DEPLOY_DIR}"

# Copy all environment files
SRC="$(dirname "$0")"
cp "${SRC}/README.md" .
cp "${SRC}/Dockerfile" .
cp "${SRC}/.dockerignore" .
cp "${SRC}/pyproject.toml" .
cp "${SRC}/uv.lock" .
cp "${SRC}/openenv.yaml" .
cp "${SRC}/__init__.py" .
cp "${SRC}/client.py" .
cp "${SRC}/models.py" .
cp -r "${SRC}/server" .

# Git add and push
git add -A
git commit -m "Deploy comtrade_env environment to HF Space" 2>/dev/null || echo "No changes to commit"
git push origin main --force

echo ""
echo "=== Deployed! ==="
echo "Space URL: ${SPACE_URL}"
echo "It may take 2-5 minutes for the Space to build and start."
