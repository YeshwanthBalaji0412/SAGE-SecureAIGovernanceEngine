# ─────────────────────────────────────────────────────────────────────────────
# IAM — least-privilege roles for the ECS task.
#
# Two distinct roles (an SAA best-practice separation):
#   • execution role — used by the ECS agent to START the task: pull the image
#     from ECR, write logs to CloudWatch, and read the OpenAI secret to inject it.
#   • task role      — the identity of the RUNNING app. Our app only calls the
#     external OpenAI API (no AWS calls), so it needs no AWS permissions.
# ─────────────────────────────────────────────────────────────────────────────

# Trust policy: only ECS tasks may assume these roles.
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# ── Execution role ────────────────────────────────────────────────────────────
resource "aws_iam_role" "execution" {
  name               = "${var.project_name}-ecs-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = { Name = "${var.project_name}-ecs-execution-role" }
}

# AWS-managed policy: ECR pull + CloudWatch Logs create/put. Standard for Fargate.
resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Inline policy: allow reading ONLY this one secret (scoped to its ARN).
data "aws_iam_policy_document" "read_secret" {
  statement {
    sid       = "ReadOpenAISecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.openai.arn]
  }
}

resource "aws_iam_role_policy" "execution_read_secret" {
  name   = "${var.project_name}-read-openai-secret"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secret.json
}

# ── Task role ───────────────────────────────────────────────────────────────
# The running app makes no AWS API calls (OpenAI is external), so this role has
# no attached permissions. It exists for correct task identity and easy future
# extension (e.g., if you later add S3 or DynamoDB access).
resource "aws_iam_role" "task" {
  name               = "${var.project_name}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = { Name = "${var.project_name}-ecs-task-role" }
}
