#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the AWS container image and push it to ECR.
#
# Run this AFTER `terraform apply` has created the ECR repo (so the repo URL
# exists), and again whenever you change the app and want to redeploy.
#
# Usage (from repo root):
#   ./scripts/build_push_ecr.sh
#
# It reads the ECR repo URL and region straight from terraform outputs, logs in,
# builds Dockerfile.aws for the right CPU architecture, pushes, then tells ECS to
# pull the new image.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$REPO_ROOT/terraform/aws"
IMAGE_TAG="${1:-latest}"

# ── Pull values from terraform outputs ────────────────────────────────────────
echo "→ Reading terraform outputs..."
ECR_URL="$(terraform -chdir="$TF_DIR" output -raw ecr_repository_url)"
REGION="$(terraform -chdir="$TF_DIR" output -raw aws_region)"
CLUSTER="$(terraform -chdir="$TF_DIR" output -raw ecs_cluster_name)"
SERVICE="$(terraform -chdir="$TF_DIR" output -raw ecs_service_name)"
REGISTRY="${ECR_URL%%/*}" # account.dkr.ecr.region.amazonaws.com

# ── Match the architecture Terraform deploys (default ARM64/Graviton) ─────────
ARCH="$(terraform -chdir="$TF_DIR" output -raw cpu_architecture 2>/dev/null || echo ARM64)"
if [ "$ARCH" = "ARM64" ]; then PLATFORM="linux/arm64"; else PLATFORM="linux/amd64"; fi

echo "→ ECR:      $ECR_URL"
echo "→ Region:   $REGION"
echo "→ Platform: $PLATFORM (tag: $IMAGE_TAG)"

# ── Authenticate Docker to ECR ────────────────────────────────────────────────
echo "→ Logging in to ECR..."
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

# ── Build (from repo root, using Dockerfile.aws) and push ─────────────────────
echo "→ Building image..."
docker buildx build \
  --platform "$PLATFORM" \
  -f "$REPO_ROOT/Dockerfile.aws" \
  -t "$ECR_URL:$IMAGE_TAG" \
  --push \
  "$REPO_ROOT"

# ── Roll the service to pick up the new image (no-op on first deploy) ─────────
echo "→ Forcing ECS to pull the new image..."
aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --force-new-deployment \
  --region "$REGION" >/dev/null

echo "✓ Done. Image pushed and service updated."
