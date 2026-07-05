# AWS provider configuration.
provider "aws" {
  region = var.aws_region

  # default_tags are applied to every taggable resource this provider creates.
  # This makes it trivial to find (and, if ever needed, clean up) everything
  # belonging to this project in the AWS console or Cost Explorer.
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Repo        = "SAGE-SecureAIGovernanceEngine"
    }
  }
}

# AWS billing metrics are published ONLY to us-east-1. This aliased provider lets
# the billing alarm work even if you deploy the app in a different region.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Repo        = "SAGE-SecureAIGovernanceEngine"
    }
  }
}
