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

## Phase 3b — annotations and backup

Annotations are stored relationally (`annotation`, `annotation_point`) with normalized coordinates. The web UI supports video point placement; polylines and image viewers can use the same JSON API.

**Backup (recommended before migrations or risky edits):**

1. **Annotations CSV** — Download from the [Media](http://127.0.0.1:8000/media/) page (“Annotations backup”) or open `GET /media/export_annotations.csv` (flattened rows, UTF-8). This is portable and version-control friendly.
2. **Full database copy** — With the app **stopped** (and no other process holding the file), copy the active SQLite file (default: `CRABSPY_DATA_DIR/db/project.sqlite`, plus `-wal`/`-shm` if present) to a safe location. Restores the whole project, including media rows and uploads paths.

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
