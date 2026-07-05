#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One command to tear the whole AWS demo DOWN (back to ~$0).
# Run this as soon as you finish recording/demoing.
#
# Usage (from anywhere):  ./scripts/demo_down.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$REPO_ROOT/terraform/aws"

echo "→ terraform destroy (removing all AWS resources)..."
terraform -chdir="$TF_DIR" destroy -auto-approve

echo ""
echo "✅ Destroyed. Nothing is billing now."
echo "   Your Hugging Face demo stays live as the always-on link."
