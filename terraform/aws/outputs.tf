# ─────────────────────────────────────────────────────────────────────────────
# Outputs — printed after `terraform apply`. Your quick reference for the demo.
# ─────────────────────────────────────────────────────────────────────────────

output "app_url" {
  description = "Open this in your browser (and in the demo video)."
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "Push your image here (used by scripts/build_push_ecr.sh)."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "Cluster name — used to pause the app (set desired_count = 0)."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Service name — used to check status or pause the app."
  value       = aws_ecs_service.app.name
}

output "aws_region" {
  description = "Region everything is deployed in."
  value       = var.aws_region
}

output "secret_name" {
  description = "Secrets Manager entry holding the OpenAI key."
  value       = aws_secretsmanager_secret.openai.name
}

output "cpu_architecture" {
  description = "Architecture the image must be built for (used by build_push_ecr.sh)."
  value       = var.cpu_architecture
}
