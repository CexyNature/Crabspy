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

- `CRABSPY_DATA_DIR` — root for `uploads/`, `db/`, `exports/`, `cache/`, and `config/` (default: `<repo>/data`).
- `CRABSPY_DATABASE_URL` — SQLAlchemy URL. If unset, the app uses `data/config/database_url` (one line) when present, otherwise SQLite at `CRABSPY_DATA_DIR/db/project.sqlite`. When this env var **is** set, it always wins and the Settings → Database form is read-only.

**Project databases (Postgres-ready):** use one SQLite file per study or campaign, e.g. `data/db/coastal_site_2025.sqlite`, with URL `sqlite:////absolute/path/to/data/db/coastal_site_2025.sqlite` (four slashes after `sqlite:` for absolute paths on Unix). Switch in the UI under **Database** (`/settings/database`) or by editing `data/config/database_url`. For PostgreSQL: `postgresql+psycopg://user:pass@host:5432/dbname` after `pip install -e ".[postgres]"`.

**Web UI:** register draft media and export CSV from [`/media/`](http://127.0.0.1:8000/media/) (see nav). On startup the app applies Alembic migrations to the active database automatically.

**Bulk import:** [`/media/import`](http://127.0.0.1:8000/media/import) accepts a UTF-8 CSV (headers in row 1). Required: a path column (`storage_path`, `path`, or `video_path`). Optional: `collected_at` / `date_collected` / `date`, `sample_code`, `site_name`, `location_name`, `notes`, `original_filename`. Dates accept ISO-8601 or `YYYY-MM-DD` (and a few common formats). Rows with path, date, sample, site, and location all set are stored as `ready_for_processing`; others as `draft`. Duplicate `storage_path` values already in the database are skipped.

## Docker

From the repository root:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8000` and `http://127.0.0.1:8000/api/health`.

## Alembic

Configuration lives under `apps/web/alembic.ini`. Migrations are in `apps/web/alembic/versions/`.

The running app calls `alembic upgrade head` on startup for the resolved database URL (so a fresh clone usually only needs `pip install` and `uvicorn`). You can still run migrations manually:

```bash
cd apps/web
export CRABSPY_DATABASE_URL=sqlite:////absolute/path/to/project.sqlite   # optional override
alembic upgrade head
```

To generate a new revision after changing models:

```bash
cd apps/web
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```
