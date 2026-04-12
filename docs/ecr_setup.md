# ECR Setup for MediBot v2

## 1. Create ECR Repositories

Run once in your AWS account:

```bash
aws ecr create-repository --repository-name medibot-v2-backend --region us-east-1
aws ecr create-repository --repository-name medibot-v2-frontend --region us-east-1
```

Note the registry URI output — format: `<account-id>.dkr.ecr.us-east-1.amazonaws.com`

## 2. Create IAM User for GitHub Actions

Create an IAM user with programmatic access and attach this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    }
  ]
}
```

## 3. Add GitHub Secrets

Go to your GitHub repo → Settings → Secrets and variables → Actions → New secret:

| Secret name             | Value                                      |
|-------------------------|--------------------------------------------|
| `AWS_ACCESS_KEY_ID`     | IAM user access key ID                     |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret access key                 |
| `OPENAI_API_KEY`        | Your OpenAI API key                        |
| `PINECONE_API_KEY`      | Your Pinecone API key                      |

## 4. Update CI Region (if not us-east-1)

Edit `.github/workflows/ci.yml` line 7:
```yaml
AWS_REGION: us-east-1   # change to your region
```

## 5. How CI/CD Works

| Event | What happens |
|---|---|
| Push to any branch / PR | Lint + unit tests + Docker build (no push) |
| Push to `main` | Lint + unit tests + Docker build + **push to ECR** |
| Manual trigger | RAGAS regression suite runs against live Pinecone + OpenAI |

Images are tagged with both the **commit SHA** and **latest**:
- `<registry>/medibot-v2-backend:abc1234`
- `<registry>/medibot-v2-backend:latest`
