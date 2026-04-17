---
name: ci-gate
description: Run the full CI gate locally — ruff format, ruff check, pytest unit tests, Docker build smoke test. Mirrors what `.github/workflows/ci.yml` runs on PR. Use before pushing to catch issues before CI.
user-invocable: true
---

# CI Gate Skill

Local mirror of GitHub Actions CI pipeline. Run before `git push` to avoid red CI.

## What CI does

The pipeline in `.github/workflows/ci.yml` runs on push to `main` and PRs:

1. **Lint:** `ruff check backend/`
2. **Unit tests:** `pytest tests/ --ignore=tests/ragas_regression/`
3. **Docker build:** backend + frontend images
4. **Push to ECR** (main branch only): tag with commit SHA + latest
5. **Deploy to EC2** (main branch only): SSH, pull, `docker compose up -d`, health check

## Local mirror steps

1. **Format Python:**
```bash
   ruff format backend/
```
2. **Lint:**
```bash
   ruff check backend/ --fix
```
3. **Unit tests:**
```bash
   pytest tests/ --ignore=tests/ragas_regression/ -v
```
4. **Docker build smoke (optional):**
```bash
   docker build -f docker/Dockerfile.backend -t medibot-v2-backend:check .
   docker build -f frontend/Dockerfile.frontend -t medibot-v2-frontend:check frontend/
```
5. **Full stack smoke (optional):**
```bash
   docker compose -f docker/docker-compose.yml up -d
   sleep 30
   curl -f http://localhost/api/health
   docker compose -f docker/docker-compose.yml down
```

## Quality gates

- ruff check exit code 0
- pytest exit code 0 (all tests pass)
- Docker builds succeed without errors

## What this doesn't cover

- RAGAS regression — separate manual workflow (use `ragas-regression` skill)
- ECR push — only runs on `main` branch in CI
- EC2 deploy — only runs on `main` branch in CI

So: local `ci-gate` passing = your PR will pass GitHub Actions CI, modulo RAGAS regression which you should run separately before any pipeline-touching merge.