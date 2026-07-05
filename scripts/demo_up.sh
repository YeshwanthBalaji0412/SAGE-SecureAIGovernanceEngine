#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One command to bring the whole AWS demo UP and wait until it's live.
#   terraform apply  ->  build & push image  ->  wait for health  ->  print URL
#
# Prereqs (one-time): terraform + aws cli configured, Docker installed,
# and terraform/aws/terraform.tfvars filled in (OpenAI key + email).
#
# Usage (from anywhere):  ./scripts/demo_up.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$REPO_ROOT/terraform/aws"

# ── Make sure Docker is running (needed to build the image) ───────────────────
if ! docker info >/dev/null 2>&1; then
  echo "→ Docker isn't running. Starting Docker Desktop..."
  open -a Docker 2>/dev/null || open -a "Docker Desktop" 2>/dev/null || true
  for i in $(seq 1 30); do
    docker info >/dev/null 2>&1 && break
    sleep 5
  done
  docker info >/dev/null 2>&1 || { echo "✗ Docker still not running. Start Docker Desktop and retry."; exit 1; }
fi
echo "✓ Docker is running"

# ── Sanity: tfvars must exist ─────────────────────────────────────────────────
if [ ! -f "$TF_DIR/terraform.tfvars" ]; then
  echo "✗ $TF_DIR/terraform.tfvars not found."
  echo "  cp $TF_DIR/terraform.tfvars.example $TF_DIR/terraform.tfvars  and fill in your OpenAI key + email."
  exit 1
fi

# ── 1. Create/refresh the infrastructure ──────────────────────────────────────
echo "→ [1/3] terraform apply (creating AWS infrastructure)..."
terraform -chdir="$TF_DIR" init -input=false >/dev/null
terraform -chdir="$TF_DIR" apply -auto-approve

# ── 2. Build & push the image (also forces a fresh deployment) ────────────────
echo "→ [2/3] building & pushing the app image..."
"$REPO_ROOT/scripts/build_push_ecr.sh"

# ── 3. Wait for the app to report healthy ─────────────────────────────────────
URL="$(terraform -chdir="$TF_DIR" output -raw app_url)"
echo "→ [3/3] waiting for the app to become healthy at $URL ..."
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$URL/_stcore/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo ""
    echo "✅ LIVE:  $URL"
    echo "   (remember: run ./scripts/demo_down.sh when you're done to stop billing)"
    exit 0
  fi
  printf "."
  sleep 15
done

echo ""
echo "⚠️  App not healthy yet. Check logs:  aws logs tail /ecs/sage --follow"
echo "   URL (may just need another minute):  $URL"
exit 1
