"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from crabspy_web.config import get_settings
from crabspy_web.db.migrate import run_alembic_upgrade_head
from crabspy_web.db.session import init_engine
from crabspy_web.routers import health, media, pages, settings as settings_router

PACKAGE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    settings.db_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.config_dir.mkdir(parents=True, exist_ok=True)

    url = settings.resolve_database_url()
    # Run Alembic before creating the SQLAlchemy engine so SQLite is not contending
    # with an open pool (and so schema exists before ORM connects).
    run_alembic_upgrade_head(url)
    init_engine(url)

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Crabspy Web",
        version="0.1.0",
        description="Local web UI for Crabspy (see docs/crabspy-rebuild-plan.md).",
        lifespan=lifespan,
    )
    app.state.settings = settings

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.state.templates = templates

    static_dir = PACKAGE_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(pages.router)
    app.include_router(health.router)
    app.include_router(media.router)
    app.include_router(settings_router.router)

    return app


app = create_app()
