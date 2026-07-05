# ─────────────────────────────────────────────────────────────────────────────
# ECR — private registry that stores the container image ECS pulls and runs.
# ─────────────────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE" # allows re-pushing the "latest" tag during iteration

  image_scanning_configuration {
    scan_on_push = true # free vulnerability scan on push — nice SAA/security touch
  }

  # force_delete lets `terraform destroy` remove the repo even if images remain,
  # so teardown is clean and never leaves an orphaned repo blocking re-create.
  force_delete = true

  tags = { Name = "${var.project_name}-ecr" }
}

# Keep storage tiny: expire all but the 3 most recent images automatically.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the last 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}
