# ─────────────────────────────────────────────────────────────────────────────
# ECS on Fargate — the compute that actually runs the container.
#   cluster         : logical grouping for the service
#   task definition : the spec (image, cpu/mem, secret injection, logging)
#   service         : keeps `desired_count` copies running behind the ALB
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  # Container Insights is left OFF: it publishes extra CloudWatch metrics that
  # cost money. Not needed for a short demo. (Flip to "enabled" for production.)
  setting {
    name  = "containerInsights"
    value = "disabled"
  }

  tags = { Name = "${var.project_name}-cluster" }
}

resource "aws_ecs_task_definition" "app" {
  family                   = var.project_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn # used to START the task
  task_role_arn            = aws_iam_role.task.arn      # identity of the running app

  # Must match the architecture the image is built for (see scripts/build_push_ecr.sh).
  # ARM64 = Graviton: cheaper and native to Apple Silicon builds.
  runtime_platform {
    cpu_architecture        = var.cpu_architecture
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([
    {
      name      = var.project_name
      image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      essential = true

      portMappings = [
        { containerPort = var.container_port, protocol = "tcp" }
      ]

      # THE KEY LINE: inject the OpenAI key from Secrets Manager into the exact
      # env var the app already reads. No code change, no key in the image.
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai.arn }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}/_stcore/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60 # give Streamlit time to boot before health checks count
      }
    }
  ])

  tags = { Name = "${var.project_name}-taskdef" }
}

resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  # Streamlit is slow to start; don't let the ALB kill the task before it's up.
  health_check_grace_period_seconds = 120

  network_configuration {
    subnets          = local.task_subnet_ids
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = local.assign_public_ip # true in the cheap public-subnet mode
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.project_name
    container_port   = var.container_port
  }

  # The listener must exist before the service registers targets.
  depends_on = [aws_lb_listener.http]

  tags = { Name = "${var.project_name}-service" }
}
