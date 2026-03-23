"""Media list, create draft, CSV import/export, detail/edit, delete."""

from __future__ import annotations

import mimetypes
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from crabspy_web.db.session import get_db
from crabspy_web.models.annotation import Annotation
from crabspy_web.models.calibration import Calibration
from crabspy_web.models.media import Media, MediaKind, MediaMeasurementMode, MediaProcessingStatus
from crabspy_web.schemas.annotation import AnnotationCreate, AnnotationOut
from crabspy_web.schemas.calibration import CalibrationCreate, CalibrationOut
from crabspy_web.services.annotation import (
    annotation_to_out,
    build_annotation_row,
    validate_annotation_for_media,
)
from crabspy_web.services.calibration_measure import path_length_mm_for_polyline
from crabspy_web.services.calibration_service import calibration_to_out, create_calibration
from crabspy_web.services.csv_export import annotation_rows_to_csv_bytes, media_rows_to_csv_bytes
from crabspy_web.services.media_csv_import import decode_uploaded_csv, parse_collected_at, parse_media_import_csv
from crabspy_web.services.media_files import resolve_storage_path_to_file
from crabspy_web.services.media_form_utils import empty_to_none, parse_optional_float, parse_optional_int
from crabspy_web.services.media_readiness import core_metadata_ready_for_processing

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


@router.get("/export_annotations.csv")
def annotations_export_csv(db: Annotated[Session, Depends(get_db)]) -> Response:
    rows = list(
        db.scalars(
            select(Annotation)
            .options(selectinload(Annotation.points))
            .order_by(Annotation.media_id, Annotation.created_at)
        ).all()
    )
    body = annotation_rows_to_csv_bytes(rows)
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="annotations_export.csv"'},
    )


@router.get("/{media_id:uuid}/view", response_class=HTMLResponse)
def media_view_page(
    request: Request,
    media_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    row = db.scalars(
        select(Media).where(Media.id == media_id).options(selectinload(Media.active_calibration))
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Media not found")
    settings = request.app.state.settings
    try:
        resolve_storage_path_to_file(settings, row.storage_path)
        file_on_disk = True
    except (FileNotFoundError, ValueError):
        file_on_disk = False
    templates = request.app.state.templates
    annotations: list[Annotation] = []
    if file_on_disk:
        annotations = list(
            db.scalars(
                select(Annotation)
                .where(Annotation.media_id == row.id)
                .options(selectinload(Annotation.points))
                .order_by(Annotation.created_at.asc())
            ).all()
        )
    all_calibrations = list(db.scalars(select(Calibration).order_by(Calibration.created_at.desc())).all())
    annotation_path_mm: dict[str, float | None] = {}
    if file_on_disk and annotations:
        cal = row.active_calibration
        for a in annotations:
            annotation_path_mm[str(a.id)] = path_length_mm_for_polyline(a, row, cal)
    return templates.TemplateResponse(
        request,
        "media/view.html",
        {
            "title": "View media",
            "media": row,
            "file_on_disk": file_on_disk,
            "annotations": annotations,
            "all_calibrations": all_calibrations,
            "annotation_path_mm": annotation_path_mm,
        },
    )


@router.post(
    "/{media_id:uuid}/annotations",
    status_code=201,
    response_model=AnnotationOut,
)
def annotation_create(
    media_id: UUID,
    body: AnnotationCreate,
    db: Annotated[Session, Depends(get_db)],
) -> AnnotationOut:
    row = db.get(Media, media_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Media not found")
    validate_annotation_for_media(row, body)
    ann = build_annotation_row(media_id, body)
    db.add(ann)
    db.flush()
    aid = ann.id
    db.commit()
    media_row = db.scalars(
        select(Media).where(Media.id == media_id).options(selectinload(Media.active_calibration))
    ).one()
    ann = db.scalars(
        select(Annotation)
        .where(Annotation.id == aid)
        .options(selectinload(Annotation.points))
    ).one()
    return annotation_to_out(ann, media=media_row)


@router.post(
    "/{media_id:uuid}/calibration",
    status_code=201,
    response_model=CalibrationOut,
)
def calibration_create_endpoint(
    media_id: UUID,
    body: CalibrationCreate,
    db: Annotated[Session, Depends(get_db)],
) -> CalibrationOut:
    row = db.scalars(select(Media).where(Media.id == media_id)).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Media not found")
    cal = create_calibration(db, row, body, set_active_on_this_media=True)
    return calibration_to_out(cal)


@router.post("/{media_id:uuid}/annotations/{annotation_id:uuid}/delete")
def annotation_delete(
    media_id: UUID,
    annotation_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, bool]:
    ann = db.get(Annotation, annotation_id)
    if ann is None or ann.media_id != media_id:
        raise HTTPException(status_code=404, detail="Annotation not found.")
    db.delete(ann)
    db.commit()
    return {"ok": True}


@router.get("/{media_id:uuid}/file")
def media_serve_file(
    request: Request,
    media_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> FileResponse:
    row = db.get(Media, media_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Media not found")
    settings = request.app.state.settings
    try:
        path = resolve_storage_path_to_file(settings, row.storage_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found on disk.") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_type, _ = mimetypes.guess_type(path.name)
    if not media_type:
        media_type = "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


def _import_page_response(request: Request, templates, *, result: dict | None) -> HTMLResponse:
    """Full page for normal requests; fragment for HTMX (same content as inside ``<main>``)."""
    ctx = {"title": "Import media from CSV", "result": result}
    name = "partials/media_import_body.html" if _htmx(request) else "media/import.html"
    return templates.TemplateResponse(request, name, ctx)


@router.get("/import", response_class=HTMLResponse)
def media_import_form(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return _import_page_response(request, templates, result=None)


@router.post("/import", response_class=HTMLResponse)
async def media_import_upload(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File(..., description="UTF-8 CSV with media columns")],
) -> HTMLResponse:
    templates = request.app.state.templates
    raw = await file.read()
    if not raw:
        return _import_page_response(
            request,
            templates,
            result={
                "created": 0,
                "skipped_duplicate": [],
                "parse_errors": [(0, "Empty file.")],
            },
        )

    try:
        text = decode_uploaded_csv(raw)
    except UnicodeDecodeError:
        return _import_page_response(
            request,
            templates,
            result={
                "created": 0,
                "skipped_duplicate": [],
                "parse_errors": [(0, "File is not valid UTF-8.")],
            },
        )

    parsed, row_errors = parse_media_import_csv(text)
    existing = set(db.scalars(select(Media.storage_path)).all())
    created = 0
    skipped_duplicate: list[tuple[int, str]] = []
    for item in parsed:
        path = item.kwargs["storage_path"]
        if path in existing:
            skipped_duplicate.append((item.line_no, path))
            continue
        db.add(Media(**item.kwargs))
        existing.add(path)
        created += 1

    return _import_page_response(
        request,
        templates,
        result={
            "created": created,
            "skipped_duplicate": skipped_duplicate,
            "parse_errors": row_errors,
        },
    )


def _media_detail_context(
    *,
    media: Media,
    all_calibrations: list[Calibration],
    form: dict[str, str | None] | None = None,
    form_error: str | None = None,
) -> dict:
    """Template context: ``form`` overrides display when re-rendering after validation error."""
    return {
        "title": "Edit media",
        "media": media,
        "all_calibrations": all_calibrations,
        "form": form,
        "form_error": form_error,
    }


@router.get("/{media_id:uuid}", response_class=HTMLResponse)
def media_detail(
    request: Request,
    media_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    row = db.scalars(
        select(Media).where(Media.id == media_id).options(selectinload(Media.active_calibration))
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Media not found")
    all_calibrations = list(db.scalars(select(Calibration).order_by(Calibration.created_at.desc())).all())
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "media/detail.html",
        _media_detail_context(media=row, all_calibrations=all_calibrations),
    )


@router.post("/{media_id:uuid}", response_class=HTMLResponse)
def media_update(
    request: Request,
    media_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    storage_path: Annotated[str, Form()],
    processing_status: Annotated[str, Form()],
    media_kind: Annotated[str, Form()],
    collected_at: Annotated[str, Form()] = "",
    sample_code: Annotated[str, Form()] = "",
    site_name: Annotated[str, Form()] = "",
    location_name: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
    camera_id: Annotated[str, Form()] = "",
    deployment_time: Annotated[str, Form()] = "",
    deployment_type: Annotated[str, Form()] = "",
    latitude: Annotated[str, Form()] = "",
    longitude: Annotated[str, Form()] = "",
    original_filename: Annotated[str, Form()] = "",
    mime_type: Annotated[str, Form()] = "",
    checksum_sha256: Annotated[str, Form()] = "",
    width_px: Annotated[str, Form()] = "",
    height_px: Annotated[str, Form()] = "",
    duration_seconds: Annotated[str, Form()] = "",
    frame_rate: Annotated[str, Form()] = "",
    active_calibration_id: Annotated[str, Form()] = "",
    measurement_mode: Annotated[str, Form()] = "homography",
) -> Response:
    row = db.scalars(
        select(Media).where(Media.id == media_id).options(selectinload(Media.active_calibration))
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Media not found")

    templates = request.app.state.templates

    form_snapshot = {
        "storage_path": storage_path,
        "processing_status": processing_status,
        "media_kind": media_kind,
        "collected_at": collected_at,
        "sample_code": sample_code,
        "site_name": site_name,
        "location_name": location_name,
        "notes": notes,
        "camera_id": camera_id,
        "deployment_time": deployment_time,
        "deployment_type": deployment_type,
        "latitude": latitude,
        "longitude": longitude,
        "original_filename": original_filename,
        "mime_type": mime_type,
        "checksum_sha256": checksum_sha256,
        "width_px": width_px,
        "height_px": height_px,
        "duration_seconds": duration_seconds,
        "frame_rate": frame_rate,
        "active_calibration_id": active_calibration_id,
        "measurement_mode": measurement_mode,
    }

    try:
        path = _validate_storage_path(storage_path)
        status = MediaProcessingStatus(processing_status)
        kind = _parse_media_kind(media_kind)
        collected = parse_collected_at(empty_to_none(collected_at))
        dep_time = parse_collected_at(empty_to_none(deployment_time))
        lat = parse_optional_float(empty_to_none(latitude))
        lon = parse_optional_float(empty_to_none(longitude))
        w = parse_optional_int(empty_to_none(width_px))
        h = parse_optional_int(empty_to_none(height_px))
        dur = parse_optional_float(empty_to_none(duration_seconds))
        fps = parse_optional_float(empty_to_none(frame_rate))
        mode = MediaMeasurementMode(measurement_mode)
        acid_raw = empty_to_none(active_calibration_id.strip())
        active_cal: UUID | None = None
        if acid_raw:
            try:
                active_cal = UUID(acid_raw)
            except ValueError as exc:
                raise ValueError("Active calibration must be a valid UUID or empty.") from exc
    except HTTPException as exc:
        msg = exc.detail
        if isinstance(msg, list):
            msg = "; ".join(str(x) for x in msg)
        else:
            msg = str(msg)
        all_calibrations_err = list(db.scalars(select(Calibration).order_by(Calibration.created_at.desc())).all())
        return templates.TemplateResponse(
            request,
            "media/detail.html",
            _media_detail_context(
                media=row,
                all_calibrations=all_calibrations_err,
                form=form_snapshot,
                form_error=msg,
            ),
        )
    except ValueError as exc:
        all_calibrations_err = list(db.scalars(select(Calibration).order_by(Calibration.created_at.desc())).all())
        return templates.TemplateResponse(
            request,
            "media/detail.html",
            _media_detail_context(
                media=row,
                all_calibrations=all_calibrations_err,
                form=form_snapshot,
                form_error=str(exc),
            ),
        )

    if status == MediaProcessingStatus.ready_for_processing and not core_metadata_ready_for_processing(
        collected,
        empty_to_none(sample_code),
        empty_to_none(site_name),
        empty_to_none(location_name),
    ):
        all_calibrations_rf = list(db.scalars(select(Calibration).order_by(Calibration.created_at.desc())).all())
        return templates.TemplateResponse(
            request,
            "media/detail.html",
            _media_detail_context(
                media=row,
                all_calibrations=all_calibrations_rf,
                form=form_snapshot,
                form_error="Cannot set status to “ready for processing” until date collected, sample code, site name, and location name are all filled in.",
            ),
        )

    row.storage_path = path
    row.processing_status = status
    row.media_kind = kind
    row.collected_at = collected
    row.sample_code = empty_to_none(sample_code)
    row.site_name = empty_to_none(site_name)
    row.location_name = empty_to_none(location_name)
    row.notes = empty_to_none(notes)
    row.camera_id = empty_to_none(camera_id)
    row.deployment_time = dep_time
    row.deployment_type = empty_to_none(deployment_type)
    row.latitude = lat
    row.longitude = lon
    row.original_filename = empty_to_none(original_filename)
    row.mime_type = empty_to_none(mime_type)
    row.checksum_sha256 = empty_to_none(checksum_sha256)
    row.width_px = w
    row.height_px = h
    row.duration_seconds = dur
    row.frame_rate = fps
    row.measurement_mode = mode

    if active_cal is None:
        row.active_calibration_id = None
    else:
        cal_row = db.get(Calibration, active_cal)
        if cal_row is None:
            all_calibrations_nf = list(db.scalars(select(Calibration).order_by(Calibration.created_at.desc())).all())
            return templates.TemplateResponse(
                request,
                "media/detail.html",
                _media_detail_context(
                    media=row,
                    all_calibrations=all_calibrations_nf,
                    form=form_snapshot,
                    form_error="Selected calibration was not found.",
                ),
            )
        row.active_calibration_id = cal_row.id

    return RedirectResponse(url=f"/media/{media_id}", status_code=303)


@router.post("/{media_id:uuid}/delete")
def media_delete(
    media_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> RedirectResponse:
    row = db.get(Media, media_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Media not found")
    db.delete(row)
    return RedirectResponse(url="/media/", status_code=303)
