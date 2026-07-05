# ─────────────────────────────────────────────────────────────────────────────
# Input variables — the "knobs" for the whole stack.
# You set real values in terraform.tfvars (copied from terraform.tfvars.example).
# ─────────────────────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region. us-east-1 has the cheapest pricing and the billing metric CloudWatch alarm requires it."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short name used as a prefix for all resource names."
  type        = string
  default     = "sage"
}

variable "environment" {
  description = "Environment label (used in tags/names). This is a throwaway demo."
  type        = string
  default     = "demo"
}

# ── Container / app ───────────────────────────────────────────────────────────

variable "image_tag" {
  description = "The container image tag to deploy (must already be pushed to ECR)."
  type        = string
  default     = "latest"
}

variable "container_port" {
  description = "Port Streamlit listens on inside the container."
  type        = number
  default     = 8501
}

variable "task_cpu" {
  description = "Fargate task CPU units. 1024 = 1 vCPU. 1 vCPU comfortably fits chromadb + langchain + streamlit."
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate task memory (MiB). 2048 = 2 GB. Must be a valid CPU/memory combo for Fargate."
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of running tasks. 1 for the demo. Set to 0 to PAUSE billing without destroying the stack."
  type        = number
  default     = 1
}

variable "cpu_architecture" {
  description = "Fargate CPU architecture. ARM64 = AWS Graviton (~20% cheaper, builds natively on Apple Silicon). X86_64 = Intel/AMD. Must match the architecture you build the image for."
  type        = string
  default     = "ARM64"

  validation {
    condition     = contains(["ARM64", "X86_64"], var.cpu_architecture)
    error_message = "cpu_architecture must be ARM64 or X86_64."
  }
}

# ── Secrets ───────────────────────────────────────────────────────────────────

variable "openai_api_key" {
  description = "Your OpenAI API key. Stored in AWS Secrets Manager and injected into the task at runtime — NEVER baked into the image or committed."
  type        = string
  sensitive   = true
}

# ── Networking ──────────────────────────────────────────────────────────────

variable "az_count" {
  description = "Number of Availability Zones. ALB requires subnets in at least 2 AZs."
  type        = number
  default     = 2
}

variable "enable_private_networking" {
  description = "false = cheap demo (task in public subnet, no NAT Gateway). true = SAA-canonical (private subnets + NAT Gateway, ~$32/mo while running). Keep false for the budget demo."
  type        = bool
  default     = false
}

variable "allowed_ingress_cidr" {
  description = "CIDR allowed to reach the load balancer on port 80. Default is open to the internet; set to 'YOUR.IP.ADDR.ESS/32' to lock the demo to just you."
  type        = string
  default     = "0.0.0.0/0"
}

# ── Monitoring / cost safety ────────────────────────────────────────────────

variable "alert_email" {
  description = "Email address to receive CloudWatch alarm + AWS Budget notifications. You must confirm the SNS subscription email after apply."
  type        = string
}

variable "monthly_budget_usd" {
  description = "AWS Budget ceiling in USD. Alerts fire at 50/80/100%. Your safety net against forgetting to destroy."
  type        = number
  default     = 5
}
