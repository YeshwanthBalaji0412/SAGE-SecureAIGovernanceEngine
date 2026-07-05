# SAGE on AWS — Deploy Runbook

Step-by-step guide to stand up the AWS (v2) deployment, verify it, record your
demo, and tear it down to ~$0. Same app and model as v1 (OpenAI GPT-4o) — only
the hosting platform is AWS.

> **Golden rule:** when you're done demoing, run **`terraform destroy`**. That is
> what keeps this a $5 project instead of a $36/month one.

---

## 0. One-time AWS account setup

Do these once, in order.

1. **Secure the root user**: AWS console → IAM → enable **MFA** on the root account. Stop using root for daily work.
2. **Create an admin IAM user** (for the CLI):
   - IAM → Users → Create user `sage-admin` → attach `AdministratorAccess`
   - Enable MFA on it
   - Create an **access key** (type: CLI) and save the key + secret
3. **Enable billing alerts** (required for the billing alarm to have data):
   - Billing console → **Billing preferences** → check **Receive CloudWatch billing alerts** → save
4. **(Recommended) An OpenAI key with a spend cap**: create/limit a key at platform.openai.com so a runaway demo can't overspend on the OpenAI side.

## 1. Install tools (local, one-time)

```bash
# macOS (Homebrew)
brew install awscli terraform
# Docker Desktop must be installed and running (you already have Docker)

# Verify
aws --version        # aws-cli/2.x
terraform version    # >= 1.5
docker version       # daemon running
```

Configure the CLI with the access key from step 0.2:

```bash
aws configure
#   AWS Access Key ID:     <from step 0.2>
#   AWS Secret Access Key: <from step 0.2>
#   Default region name:   us-east-1
#   Default output format: json

aws sts get-caller-identity   # confirms you're authenticated
```

## 2. Configure your variables

```bash
cd terraform/aws
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and set at least:
- `openai_api_key` — your OpenAI key (goes into Secrets Manager)
- `alert_email`    — where budget/alarm emails go

`terraform.tfvars` is git-ignored — it holds your secret and is never committed.

## 3. Deploy the infrastructure

```bash
# still in terraform/aws
terraform init       # downloads the AWS provider (no resources created)
terraform validate   # checks the config is well-formed
terraform plan        # shows EXACTLY what will be created — review it
terraform apply       # type "yes" to create the resources (~5–10 min)
```

When it finishes, note the outputs (especially `app_url`). Re-print anytime:

```bash
terraform output
```

**Confirm the SNS email:** AWS sends a "Subscription Confirmation" email to
`alert_email`. Click the link, or you won't receive alarm notifications.

## 4. Build & push the app image

The ECR repo now exists, so package and upload the app (run from repo root):

```bash
cd ../..                      # back to repo root
./scripts/build_push_ecr.sh   # builds Dockerfile.aws (arm64) + pushes + redeploys
```

The ECS service pulls the image and starts the task. First start takes ~1–2 min.

## 5. Verify

```bash
cd terraform/aws
open "$(terraform output -raw app_url)"      # macOS; or paste the URL in a browser
```

Checklist:
- [ ] Page loads (if you see a 502/503, the task is still starting — wait ~1 min)
- [ ] Load a demo org (e.g. TechNova) — the app uses the injected OpenAI key automatically
- [ ] Ask a sample compliance question → you get a grounded answer with citations
- [ ] Behavior matches your Hugging Face / v1 version (it's the same code + model)

Watch logs / status if needed:

```bash
aws ecs describe-services --cluster "$(terraform output -raw ecs_cluster_name)" \
  --services "$(terraform output -raw ecs_service_name)" \
  --query 'services[0].{running:runningCount,desired:desiredCount,status:status}'

aws logs tail /ecs/sage --follow    # live app logs
```

## 6. Record your demo 🎥

Open the app URL, walk through the features, and record. If you want to show the
SAA architecture, also screen-record the AWS console (ECS service, ALB, CloudWatch
logs) and the Terraform files.

---

## 7. Pause vs. Destroy

**Pause** (stop Fargate billing, keep everything else, fast to resume):

```bash
terraform apply -var="desired_count=0"   # stops the task
# ...later...
terraform apply -var="desired_count=1"   # brings it back (no rebuild needed)
```
> Note: the ALB still bills ~$0.02/hr while paused. For gaps longer than a day,
> full destroy is cheaper.

**Destroy** (remove everything, back to ~$0):

```bash
cd terraform/aws
terraform destroy    # type "yes"  (~5 min)
```

After destroy: no Fargate, no ALB, no secret — nothing bills. Your Hugging Face
demo keeps running free as the always-on interview link. Re-deploy anytime by
repeating steps 3–4.

---

## 8. Cost summary

| State | Cost |
|---|---|
| Running (ALB + 1 Fargate task, ARM64) | ~$0.07/hr → ~$1–2 for a 15-hr demo window |
| Paused (`desired_count=0`) | ~$0.02/hr (ALB only) |
| Destroyed | ~$0 (ECR/S3/logs pennies; Bedrock not used) |
| OpenAI tokens | billed separately to your OpenAI account (small for a demo) |

The `$5` AWS Budget + billing alarm email you long before anything gets scary.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| ALB URL returns 503 | Task still starting, or failing health checks. `aws logs tail /ecs/sage` to see why. |
| Task keeps restarting | App crash on boot — check logs. Often a bad `OPENAI_API_KEY` value in the secret. |
| `exec format error` in logs | Image architecture ≠ task architecture. Ensure the image was built for the same `cpu_architecture` (default ARM64) — rerun `build_push_ecr.sh`. |
| No alarm/budget emails | You didn't confirm the SNS subscription email (step 3), or billing alerts aren't enabled (step 0.3). |
| `terraform destroy` blocked | Rare. Ensure `desired_count` isn't stuck; retry destroy. ECR `force_delete` and secret `recovery_window_in_days=0` are set to avoid this. |
