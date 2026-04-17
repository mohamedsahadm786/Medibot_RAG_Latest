# Terminal command safety

Dev machine: Windows, 8GB RAM. Long-running processes in Claude Code's terminal block the session and sometimes hang.

**Never run these in Claude Code's terminal. Instruct the user to run them in a separate PowerShell/CMD/VS Code terminal:**

- `docker compose up` / `docker compose build` / any long-running docker command
- `pip install -r requirements.txt` / `pip install <any heavy package>`
- `npm install` / `npm run dev` / `npm run build`
- `celery -A ...` / any celery worker
- `uvicorn backend.main:app --reload` / any server start
- `alembic upgrade head` / any DB-mutating command
- `python run_ingestion.py` (long-running, costs OpenAI tokens)

**Safe to run in Claude Code's terminal:**

- `ruff check backend/` / `ruff format backend/`
- `pytest tests/ --ignore=tests/ragas_regression/`
- `git status` / `git diff` / `git log`
- `ls` / `cat` / `grep` — read-only inspection
- Short `python -c "..."` snippets that finish in seconds

When in doubt: ask the user to run it themselves and paste the output back.