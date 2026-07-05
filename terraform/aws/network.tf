# ─────────────────────────────────────────────────────────────────────────────
# Networking — the private network the app lives in.
#
# Two topologies, chosen by var.enable_private_networking:
#   false (default, cheap demo): task runs in a PUBLIC subnet with a public IP.
#                                Egress to the internet (OpenAI, ECR, Secrets)
#                                goes straight out the Internet Gateway — NO NAT.
#   true  (SAA-canonical):       task runs in PRIVATE subnets; egress goes through
#                                a NAT Gateway (~$32/mo while running).
#
# The ALB always lives in the public subnets (it is the public entry point).
# ─────────────────────────────────────────────────────────────────────────────

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, var.az_count)

  # Where the ECS task runs, and whether it gets a public IP, depends on topology.
  task_subnet_ids  = var.enable_private_networking ? aws_subnet.private[*].id : aws_subnet.public[*].id
  assign_public_ip = var.enable_private_networking ? false : true
}

# ── VPC ───────────────────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = local.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project_name}-vpc" }
}

# ── Internet Gateway (public egress + inbound to the ALB) ──────────────────────
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

# ── Public subnets (one per AZ; ALB requires >= 2 AZs) ────────────────────────
resource "aws_subnet" "public" {
  count                   = var.az_count
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(local.vpc_cidr, 8, count.index) # 10.0.0.0/24, 10.0.1.0/24
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-public-${local.azs[count.index]}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = var.az_count
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── Private subnets + NAT (only when enable_private_networking = true) ─────────
resource "aws_subnet" "private" {
  count             = var.enable_private_networking ? var.az_count : 0
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(local.vpc_cidr, 8, count.index + 100) # 10.0.100.0/24, ...
  availability_zone = local.azs[count.index]

  tags = { Name = "${var.project_name}-private-${local.azs[count.index]}" }
}

resource "aws_eip" "nat" {
  count  = var.enable_private_networking ? 1 : 0
  domain = "vpc"
  tags   = { Name = "${var.project_name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  count         = var.enable_private_networking ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id # NAT lives in a public subnet
  tags          = { Name = "${var.project_name}-nat" }

  depends_on = [aws_internet_gateway.main]
}

resource "aws_route_table" "private" {
  count  = var.enable_private_networking ? 1 : 0
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[0].id
  }
  tags = { Name = "${var.project_name}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = var.enable_private_networking ? var.az_count : 0
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[0].id
}
