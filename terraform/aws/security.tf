# ─────────────────────────────────────────────────────────────────────────────
# Security groups — the firewalls.
#
# The chain enforces least-privilege exposure:
#   internet --(80)--> ALB SG --(8501)--> Task SG
# The app is NEVER directly reachable from the internet; only the ALB can talk
# to it, and only on the container port. This is a key SAA talking point.
# ─────────────────────────────────────────────────────────────────────────────

# ── ALB security group: public entry point ────────────────────────────────────
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "Allow inbound HTTP to the load balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP from allowed clients"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ingress_cidr]
  }

  egress {
    description = "Allow all outbound to the task"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-alb-sg" }
}

# ── Task security group: only the ALB may reach the app ───────────────────────
resource "aws_security_group" "task" {
  name        = "${var.project_name}-task-sg"
  description = "Allow inbound only from the ALB on the container port"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "App port from the ALB only"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id] # source = ALB SG, not a CIDR
  }

  # Egress open so the task can reach OpenAI's API, pull the image from ECR,
  # and read the secret from Secrets Manager.
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-task-sg" }
}
