# ─────────────────────────────────────────────────────────────────────────────
# Cost safety net — your protection against forgetting to `terraform destroy`.
#   1. AWS Budget: emails you at 50% / 80% / 100% of the monthly ceiling, and
#      also on a FORECASTED overrun. Region-agnostic.
#   2. Billing CloudWatch alarm: fires when estimated charges cross the ceiling.
#      (Requires "Receive Billing Alerts" enabled in the Billing console, and
#       the billing metric only exists in us-east-1 — hence the aliased provider.)
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_budgets_budget" "monthly" {
  name         = "${var.project_name}-monthly-budget"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

resource "aws_cloudwatch_metric_alarm" "billing" {
  provider            = aws.us_east_1 # billing metric only lives in us-east-1
  alarm_name          = "${var.project_name}-estimated-charges"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600 # 6h; billing metric updates a few times per day
  statistic           = "Maximum"
  threshold           = var.monthly_budget_usd
  alarm_description   = "Estimated AWS charges exceeded the budget ceiling"
  treat_missing_data  = "notBreaching"

  dimensions = {
    Currency = "USD"
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
}
