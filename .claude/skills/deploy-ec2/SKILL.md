---
name: deploy-ec2
description: Deploy MediBot v2 to AWS EC2 via SSH. Pulls latest ECR images, runs docker compose up, verifies all 8 containers healthy, checks /api/health, and tails logs for 30s to confirm clean startup. Use for hotfix deploys outside normal GitHub Actions CI/CD.
user-invocable: true
disable-model-invocation: true
---

# EC2 Deploy Skill

Manual deploy to production EC2. Normally GitHub Actions handles this on push to `main` (see `.github/workflows/ci.yml`). This skill is for hotfixes or when CI/CD is down.

## Pre-flight checks (BEFORE deploy)

1. Confirm all tests pass locally: `pytest tests/ --ignore=tests/ragas_regression/`
2. Confirm ECR images exist for the target SHA: 
```bash
   aws ecr describe-images --repository-name medibot-v2-backend --image-ids imageTag=<sha>
   aws ecr describe-images --repository-name medibot-v2-frontend --image-ids imageTag=<sha>
```
3. Confirm SSH key at `~/.ssh/medibot-ec2.pem` (0400 permissions)
4. Confirm EC2 instance state: 
```bash
   aws ec2 describe-instances --filters "Name=tag:Name,Values=medibot-v2-prod" --query "Reservations[].Instances[].State.Name"
```

## Deploy steps

1. SSH to EC2:
```bash
   ssh -i ~/.ssh/medibot-ec2.pem ubuntu@<ELASTIC_IP>
```
2. Navigate to deployment directory:
```bash
   cd /opt/medibot-v2
```
3. Pull latest compose config:
```bash
   git pull origin main
```
4. Log in to ECR:
```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
```
5. Pull new images:
```bash
   docker compose -f deploy/docker-compose.prod.yml pull
```
6. Restart with new images:
```bash
   docker compose -f deploy/docker-compose.prod.yml up -d --remove-orphans
```
7. Wait 30s for services to come up
8. Health check (must return 200):
```bash
   curl -f http://localhost/api/health
```
9. Tail backend logs for 30s to watch for startup errors:
```bash
   docker compose -f deploy/docker-compose.prod.yml logs -f --tail 100 fastapi-backend
```
10. Smoke test: send one test query to `/api/chat/stream` — should stream response

## Post-flight checks

1. Prometheus scraping all targets:
```bash
   curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health}'
```
2. Grafana dashboard reachable: `curl http://localhost/grafana/api/health`
3. Celery worker running: `docker compose ps celery-worker`
4. Redis queue accessible: `docker exec medibot-redis redis-cli -n 1 LLEN celery`

## Rollback

If deploy fails or metrics regress:

```bash
docker compose -f deploy/docker-compose.prod.yml down
# Update image tags in compose file to previous SHA
docker compose -f deploy/docker-compose.prod.yml pull
docker compose -f deploy/docker-compose.prod.yml up -d
```

## Known gotchas

- **t3.medium RAM tight:** backend image is ~2.5GB; with Pinecone client + PubMedBERT + cross-encoder loaded, can approach 5–6GB. If OOM killed, upgrade to t3.large or lazy-load the cross-encoder
- **SSE through Nginx:** make sure `proxy_buffering off` is set for `/api/chat/stream` location — otherwise SSE events batch and the frontend sees no streaming
- **Port 5433 inside container:** the Postgres container exposes 5432 internally (standard). The 5433 host mapping is only for dev. In `docker-compose.prod.yml`, internal container-to-container uses 5432.