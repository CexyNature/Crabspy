"""Project database URL (switch SQLite file / Postgres) without restarting the process."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from crabspy_web.config import Settings, persist_database_url
from crabspy_web.db.migrate import run_alembic_upgrade_head
from crabspy_web.db.session import init_engine

router = APIRouter(prefix="/settings", tags=["settings"])


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/database", response_class=HTMLResponse)
def settings_database_page(
    request: Request,
    app_settings: Annotated[Settings, Depends(_get_settings)],
) -> HTMLResponse:
    templates = request.app.state.templates
    saved = request.query_params.get("saved") == "1"
    return templates.TemplateResponse(
        request,
        "settings/database.html",
        {
            "title": "Database",
            "current_url": app_settings.resolve_database_url(),
            "env_locked": app_settings.database_url_from_env,
            "saved": saved,
        },
    )


@router.post("/database")
def settings_database_post(
    request: Request,
    app_settings: Annotated[Settings, Depends(_get_settings)],
    database_url: Annotated[str, Form()],
) -> Response:
    """Persist URL and reinit engine (ignored when ``CRABSPY_DATABASE_URL`` is set)."""
    templates = request.app.state.templates

    if app_settings.database_url_from_env:
        if request.headers.get("hx-request") == "true":
            return templates.TemplateResponse(
                request,
                "partials/settings_database_body.html",
                {
                    "current_url": app_settings.resolve_database_url(),
                    "env_locked": True,
                    "saved": False,
                },
            )
        return RedirectResponse(url="/settings/database", status_code=303)

    url = database_url.strip()
    if not url:
        return RedirectResponse(url="/settings/database", status_code=303)

    persist_database_url(app_settings, url)
    init_engine(url)
    run_alembic_upgrade_head(url)

    if request.headers.get("hx-request") == "true":
        return templates.TemplateResponse(
            request,
            "partials/settings_database_body.html",
            {
                "current_url": app_settings.resolve_database_url(),
                "env_locked": False,
                "saved": True,
            },
        )

    return RedirectResponse(url="/settings/database?saved=1", status_code=303)
