"""HTML pages and HTMX-friendly fragments."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    # Starlette API: TemplateResponse(request, name, context) — request must be first.
    return templates.TemplateResponse(
        request,
        "pages/index.html",
        {"title": "Crabspy"},
    )


@router.get("/fragments/status", response_class=HTMLResponse)
async def fragment_status(request: Request) -> HTMLResponse:
    """Example partial for HTMX swaps (see base layout)."""
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "partials/status.html", {})
