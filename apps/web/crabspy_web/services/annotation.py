"""Create / validate annotations against media kind (image vs video)."""

from __future__ import annotations

import math

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from crabspy_web.models.annotation import Annotation, AnnotationKind, AnnotationPoint
from crabspy_web.models.calibration import Calibration
from crabspy_web.models.media import Media, MediaKind
from crabspy_web.schemas.annotation import AnnotationCreate, AnnotationOut, AnnotationPointOut
from crabspy_web.services.calibration_measure import path_length_mm_and_mode_for_polyline

# When media.frame_rate is unknown, derive frame_index from time_seconds using this FPS (documented in UI).
DEFAULT_VIDEO_FPS_FOR_FRAME_INDEX = 30.0


def effective_fps_for_frame_index(media: Media) -> float | None:
    """FPS used with time_seconds to derive frame_index; None for images."""
    if media.media_kind == MediaKind.image:
        return None
    if media.frame_rate is not None:
        return float(media.frame_rate)
    return DEFAULT_VIDEO_FPS_FOR_FRAME_INDEX


def validate_annotation_for_media(media: Media, body: AnnotationCreate) -> None:
    """Enforce time/frame rules from rebuild plan (video vs image)."""
    if media.media_kind == MediaKind.image:
        if body.time_seconds is not None or body.frame_index is not None:
            raise HTTPException(
                status_code=400,
                detail="Image annotations must not set time_seconds or frame_index.",
            )
    elif media.media_kind in (MediaKind.video, MediaKind.unknown):
        if body.time_seconds is None and body.frame_index is None:
            raise HTTPException(
                status_code=400,
                detail="Video annotations require time_seconds and/or frame_index.",
            )


def infer_video_frame_index(media: Media, body: AnnotationCreate) -> int | None:
    """Use client frame_index, or derive from time_seconds × FPS (media.frame_rate or default)."""
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


def apply_polyline_measurement_fields(
    ann: Annotation,
    media: Media,
    calibration: Calibration | None,
) -> None:
    """Persist mm length (one decimal) and effective measurement mode for polylines when calibration allows."""
    mm, mode = path_length_mm_and_mode_for_polyline(ann, media, calibration)
    ann.path_length_mm = round(mm, 1) if mm is not None else None
    ann.measurement_mode_used = mode


def refresh_media_annotation_derived_fields(db: Session, media: Media) -> None:
    """Recompute frame_index for video annotations and mm/mode for polylines (e.g. after calibration/mode change)."""
    cal: Calibration | None = None
    if media.active_calibration_id is not None:
        cal = db.get(Calibration, media.active_calibration_id)
    anns = db.scalars(
        select(Annotation)
        .where(Annotation.media_id == media.id)
        .options(selectinload(Annotation.points))
    ).all()
    fps = effective_fps_for_frame_index(media)
    for ann in anns:
        if media.media_kind != MediaKind.image and ann.time_seconds is not None and fps is not None:
            ann.frame_index = int(math.floor(ann.time_seconds * fps))
        if ann.kind == AnnotationKind.polyline:
            apply_polyline_measurement_fields(ann, media, cal)


def build_annotation_row(media: Media, body: AnnotationCreate) -> Annotation:
    kind = AnnotationKind(body.kind)
    ann = Annotation(
        media_id=media.id,
        kind=kind,
        label=body.label.strip() if body.label and body.label.strip() else None,
        frame_index=infer_video_frame_index(media, body),
        time_seconds=body.time_seconds,
        ref_width_px=body.ref_width_px,
        ref_height_px=body.ref_height_px,
    )
    for i, p in enumerate(body.points):
        ann.points.append(
            AnnotationPoint(
                order_index=i,
                x_norm=p.x_norm,
                y_norm=p.y_norm,
            )
        )
    return ann


def annotation_to_out(
    ann: Annotation,
    *,
    media: Media | None = None,
    calibration: Calibration | None = None,
) -> AnnotationOut:
    path_mm: float | None = ann.path_length_mm
    mode_used = ann.measurement_mode_used
    if media is not None and ann.kind == AnnotationKind.polyline:
        cal = calibration if calibration is not None else getattr(media, "active_calibration", None)
        if path_mm is None or mode_used is None:
            cmm, cmode = path_length_mm_and_mode_for_polyline(ann, media, cal)
            if path_mm is None:
                path_mm = cmm
            if mode_used is None:
                mode_used = cmode
    if path_mm is not None:
        path_mm = round(float(path_mm), 1)
    return AnnotationOut(
        id=str(ann.id),
        kind=ann.kind.value,
        points=[AnnotationPointOut(x_norm=p.x_norm, y_norm=p.y_norm) for p in ann.points],
        label=ann.label,
        time_seconds=ann.time_seconds,
        frame_index=ann.frame_index,
        ref_width_px=ann.ref_width_px,
        ref_height_px=ann.ref_height_px,
        path_length_mm=path_mm,
        measurement_mode_used=mode_used.value if mode_used is not None else None,
    )
