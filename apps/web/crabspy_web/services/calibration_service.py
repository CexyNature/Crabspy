"""Create and serialize calibration rows."""

from __future__ import annotations

import json
import math
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from crabspy_web.models.calibration import Calibration
from crabspy_web.models.media import Media, MediaKind
from crabspy_web.schemas.calibration import CalibrationCornerIn, CalibrationCreate, CalibrationOut
from crabspy_web.services.annotation import effective_fps_for_frame_index
from crabspy_web.services.calibration_math import mm_per_px_from_quadrat


def calibration_to_out(row: Calibration) -> CalibrationOut:
    corners_raw = json.loads(row.corners_json)
    corners = [CalibrationCornerIn(**c) for c in corners_raw]
    return CalibrationOut(
        id=str(row.id),
        source_media_id=str(row.source_media_id),
        frame_index=row.frame_index,
        time_seconds=row.time_seconds,
        corners=corners,
        reference_edge_index=row.reference_edge_index,
        reference_length_mm=row.reference_length_mm,
        ref_width_px=row.ref_width_px,
        ref_height_px=row.ref_height_px,
        mm_per_px=row.mm_per_px,
        label=row.label,
    )


def infer_calibration_frame_index(media: Media, body: CalibrationCreate) -> int | None:
    """Prefer client frame_index; else derive from time_seconds × FPS (media.frame_rate or default)."""
    if body.frame_index is not None:
        return body.frame_index
    if media.media_kind == MediaKind.image:
        return None
    if body.time_seconds is None:
        return None
    fps = effective_fps_for_frame_index(media)
    if fps is None:
        return None
    return int(math.floor(body.time_seconds * fps))


def create_calibration(
    db: Session,
    media: Media,
    body: CalibrationCreate,
    *,
    set_active_on_this_media: bool = True,
) -> Calibration:
    """Persist a quadrat; optionally set ``media.active_calibration_id`` to this row."""
    if body.ref_width_px is None or body.ref_height_px is None:
        raise HTTPException(status_code=400, detail="ref_width_px and ref_height_px are required.")

    if media.media_kind == MediaKind.image:
        if body.frame_index is not None or body.time_seconds is not None:
            raise HTTPException(
                status_code=400,
                detail="Image calibration must not set frame_index or time_seconds.",
            )
    elif media.media_kind in (MediaKind.video, MediaKind.unknown):
        if body.frame_index is None and body.time_seconds is None:
            raise HTTPException(
                status_code=400,
                detail="Video calibration requires frame_index and/or time_seconds.",
            )

    coords = [(c.x_norm, c.y_norm) for c in body.corners]
    try:
        mm_px = mm_per_px_from_quadrat(
            coords,
            body.reference_edge_index,
            body.reference_length_mm,
            body.ref_width_px,
            body.ref_height_px,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row = Calibration(
        source_media_id=media.id,
        frame_index=infer_calibration_frame_index(media, body),
        time_seconds=body.time_seconds,
        corners_json=json.dumps([{"x_norm": c.x_norm, "y_norm": c.y_norm} for c in body.corners]),
        reference_edge_index=body.reference_edge_index,
        reference_length_mm=body.reference_length_mm,
        ref_width_px=body.ref_width_px,
        ref_height_px=body.ref_height_px,
        mm_per_px=mm_px,
        label=body.label.strip() if body.label and body.label.strip() else None,
    )
    db.add(row)
    db.flush()

    if set_active_on_this_media:
        media.active_calibration_id = row.id

    db.refresh(row)
    return row


def set_media_active_calibration(
    db: Session,
    media: Media,
    calibration_id: UUID | None,
) -> None:
    if calibration_id is None:
        media.active_calibration_id = None
        return
    cal = db.get(Calibration, calibration_id)
    if cal is None:
        raise HTTPException(status_code=404, detail="Calibration not found.")
    media.active_calibration_id = cal.id
