"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from crabspy_web.config import get_settings
from crabspy_web.routers import health, pages

PACKAGE_DIR = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Crabspy Web",
        version="0.1.0",
        description="Local web UI for Crabspy (see docs/crabspy-rebuild-plan.md).",
    )
    app.state.settings = settings

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.state.templates = templates

    static_dir = PACKAGE_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(pages.router)
    app.include_router(health.router)

    return app


app = create_app()
