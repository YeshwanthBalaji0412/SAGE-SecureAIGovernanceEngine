# Terraform + provider version pins.
# Pinning avoids "works on my machine" drift and is an IaC best practice.
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # For this throwaway demo we use LOCAL state (a terraform.tfstate file on your
  # machine). That is fine for a single operator who tears down after recording.
  #
  # The SAA-canonical pattern is a remote backend (S3 for state + DynamoDB for
  # locking). Uncomment and fill in to use it:
  #
  # backend "s3" {
  #   bucket         = "your-tfstate-bucket"
  #   key            = "sage/aws/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "your-tf-lock-table"
  #   encrypt        = true
  # }
}
