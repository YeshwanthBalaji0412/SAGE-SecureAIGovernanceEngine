#!/usr/bin/env bash
# Deploy the current `main` to the Hugging Face Space.
#
# HF Spaces reject the binary PDFs/docx in Documentation/ (they require LFS),
# so we push an orphan branch containing everything EXCEPT that folder. This
# leaves the GitHub repo and its history untouched.
#
# Usage:
#   HF_TOKEN=hf_xxx ./deploy-hf.sh
# (Create a write token at https://huggingface.co/settings/tokens)

set -euo pipefail

HF_USER="yeshwanthbalaji"
HF_SPACE="sage-compliance-assistant"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: set HF_TOKEN to a Hugging Face write token." >&2
  echo "  HF_TOKEN=hf_xxx ./deploy-hf.sh" >&2
  exit 1
fi

# Make sure we start clean and on main.
START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git checkout main

# Build a single-commit orphan branch with Documentation/ excluded.
git branch -D hf-deploy 2>/dev/null || true
git checkout --orphan hf-deploy
git rm -r --cached Documentation >/dev/null 2>&1 || true
git commit -m "HF Spaces deployment ($(date -u +%Y-%m-%dT%H:%MZ))"

# Push it as the Space's main branch.
git push "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}" \
  hf-deploy:main --force

# Return to where we started; force is safe because Documentation/ on disk
# is identical to main (we only unstaged it, never edited it).
git checkout -f "$START_BRANCH"
git branch -D hf-deploy

echo
echo "Deployed. Live at: https://${HF_USER}-${HF_SPACE}.hf.space"
