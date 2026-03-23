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
- `CRABSPY_DATABASE_URL` — SQLAlchemy URL (default: SQLite under `CRABSPY_DATA_DIR/db/project.sqlite`). For PostgreSQL, use e.g. `postgresql+psycopg://user:pass@host:5432/dbname` after `pip install -e ".[postgres]"`.

## Docker

From the repository root:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8000` and `http://127.0.0.1:8000/api/health`.

## Alembic

Configuration lives under `apps/web/alembic.ini`. Migrations are in `apps/web/alembic/versions/`.

After installing the package, apply migrations (creates/updates tables — required before relying on the DB):

```bash
cd apps/web
export CRABSPY_DATABASE_URL=sqlite:////absolute/path/to/project.sqlite   # optional override
alembic upgrade head
```

To generate a new revision after changing models:

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```
