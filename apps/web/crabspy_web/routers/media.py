"""Media list, create draft, and CSV export."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from crabspy_web.db.session import get_db
from crabspy_web.models.media import Media, MediaKind, MediaProcessingStatus
from crabspy_web.services.csv_export import media_rows_to_csv_bytes

router = APIRouter(prefix="/media", tags=["media"])


def _htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def _validate_storage_path(raw: str) -> str:
    p = raw.strip()
    if not p:
        raise HTTPException(status_code=400, detail="Storage path is required.")
    if ".." in p:
        raise HTTPException(status_code=400, detail="Storage path must not contain '..'")
    return p


def _parse_media_kind(raw: str) -> MediaKind:
    try:
        return MediaKind(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid media kind.") from exc


@router.get("/", response_class=HTMLResponse)
def media_list_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    templates = request.app.state.templates
    rows = list(db.scalars(select(Media).order_by(Media.created_at.desc())).all())
    return templates.TemplateResponse(
        request,
        "media/list.html",
        {
            "title": "Media",
            "rows": rows,
            "error": None,
        },
    )


@router.post("/", response_class=HTMLResponse)
def media_create_draft(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    storage_path: Annotated[str, Form()],
    original_filename: Annotated[str | None, Form()] = None,
    media_kind: Annotated[str, Form()] = "unknown",
) -> Response:
    path = _validate_storage_path(storage_path)
    kind = _parse_media_kind(media_kind)

    row = Media(
        storage_path=path,
        original_filename=original_filename.strip() if original_filename else None,
        media_kind=kind,
        processing_status=MediaProcessingStatus.draft,
    )
    db.add(row)

    templates = request.app.state.templates
    rows = list(db.scalars(select(Media).order_by(Media.created_at.desc())).all())

    if _htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/media_list_block.html",
            {"rows": rows, "error": None},
        )

    return RedirectResponse(url="/media/", status_code=303)


@router.get("/export.csv")
def media_export_csv(db: Annotated[Session, Depends(get_db)]) -> Response:
    rows = list(db.scalars(select(Media).order_by(Media.created_at.desc())).all())
    body = media_rows_to_csv_bytes(rows)
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="media_export.csv"'},
    )
