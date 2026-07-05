# ─────────────────────────────────────────────────────────────────────────────
# Application Load Balancer — the public entry point.
#   ALB           : lives in the public subnets, gets a public DNS name
#   target group  : health-checks the Fargate task; sticky sessions for Streamlit
#   listener      : port 80 -> forward to the app
#
# HTTPS note: for a throwaway demo we serve HTTP on :80. Adding HTTPS needs an
# ACM certificate + a domain (Route 53), which is out of scope for the budget
# demo but a documented next step in the README.
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  # Streamlit uses a long-lived WebSocket; raise the idle timeout so the
  # connection isn't dropped during a quiet moment in the demo.
  idle_timeout = 300

  enable_deletion_protection = false # must be false so `terraform destroy` works

  tags = { Name = "${var.project_name}-alb" }
}

resource "aws_lb_target_group" "app" {
  name        = "${var.project_name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # Fargate awsvpc tasks register by IP, not instance

  health_check {
    enabled             = true
    path                = "/_stcore/health" # Streamlit's built-in health endpoint
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  # Sticky sessions: pin each visitor to the same task. Important for Streamlit's
  # session state + WebSocket, especially if you ever scale beyond one task.
  stickiness {
    type            = "lb_cookie"
    enabled         = true
    cookie_duration = 86400
  }

  tags = { Name = "${var.project_name}-tg" }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
