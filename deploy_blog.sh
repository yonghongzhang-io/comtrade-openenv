#!/bin/bash
# Deploy the ComtradeBench blog to a dedicated static HF Space.
# Uses `hf upload` (handles LFS automatically — no git-lfs required).
# Prerequisites: hf auth login (run once to set token).
#
# Usage: bash /absolute/path/to/deploy_blog.sh

set -e

BLOG_REPO="yonghongzhang/comtrade-bench-blog"
BLOG_URL="https://huggingface.co/spaces/${BLOG_REPO}"
SRC="$(cd "$(dirname "$0")" && pwd)/blog_space"

echo "=== Deploying ComtradeBench blog to HF Space: ${BLOG_REPO} ==="

if ! hf auth whoami >/dev/null 2>&1; then
    echo "ERROR: Not logged in to HF. Run: hf auth login"
    exit 1
fi

# Ensure the Space exists (no-op if already created). Static SDK.
hf repo create "${BLOG_REPO}" --repo-type=space --space_sdk=static -y 2>/dev/null || true

hf upload "${BLOG_REPO}" "${SRC}" . \
    --repo-type=space \
    --commit-message="Deploy ComtradeBench blog (static)"

echo ""
echo "=== Deployed! ==="
echo "Blog URL: ${BLOG_URL}"
