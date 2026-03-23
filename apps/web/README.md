# Crabspy web (`crabspy_web`)

Local FastAPI application with Jinja2 templates and HTMX. See the repository root [`docs/crabspy-rebuild-plan.md`](../../docs/crabspy-rebuild-plan.md).

## Run (development)

From the repository root, with `apps/web` on `PYTHONPATH`:

```bash
cd apps/web
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn crabspy_web.app:app --reload --app-dir .
```

Or from repo root:

```bash
PYTHONPATH=apps/web uvicorn crabspy_web.app:app --reload --app-dir apps/web
```

Environment variables (optional):

- `CRABSPY_DATA_DIR` — root for `uploads/`, `db/`, `exports/`, `cache/` (default: `<repo>/data`).
- `CRABSPY_DATABASE_URL` — SQLAlchemy URL (default: SQLite under `CRABSPY_DATA_DIR/db/project.sqlite`).

## Docker

From the repository root:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8000` and `http://127.0.0.1:8000/api/health`.

## Alembic

Configuration lives under `apps/web/alembic.ini`. After installing the package:

```bash
cd apps/web
alembic revision --autogenerate -m "init"
alembic upgrade head
```

(Autogenerate requires SQLAlchemy models to be wired in `alembic/env.py`.)
