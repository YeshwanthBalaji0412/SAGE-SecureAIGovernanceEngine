# ─────────────────────────────────────────────────────────────────────────────
# Secrets Manager — securely stores the OpenAI API key.
# ECS injects this into the task's OPENAI_API_KEY env var at runtime (see ecs.tf),
# so the key is never baked into the image or committed to git.
# ─────────────────────────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "openai" {
  name        = "${var.project_name}/openai-api-key"
  description = "OpenAI API key for the SAGE app on ECS Fargate"

  # recovery_window_in_days = 0 deletes the secret immediately on destroy instead
  # of the default 30-day retention — important so a clean re-apply can reuse the
  # same secret name, and so nothing lingers after teardown.
  recovery_window_in_days = 0

  tags = { Name = "${var.project_name}-openai-secret" }
}

resource "aws_secretsmanager_secret_version" "openai" {
  secret_id     = aws_secretsmanager_secret.openai.id
  secret_string = var.openai_api_key
}
