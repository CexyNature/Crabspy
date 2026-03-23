"""Create / validate annotations against media kind (image vs video)."""

from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException

from crabspy_web.models.annotation import Annotation, AnnotationKind, AnnotationPoint
from crabspy_web.models.media import Media, MediaKind
from crabspy_web.schemas.annotation import AnnotationCreate, AnnotationOut, AnnotationPointOut


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


def build_annotation_row(media_id: UUID, body: AnnotationCreate) -> Annotation:
    kind = AnnotationKind(body.kind)
    ann = Annotation(
        media_id=media_id,
        kind=kind,
        label=body.label.strip() if body.label and body.label.strip() else None,
        frame_index=body.frame_index,
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


def annotation_to_out(ann: Annotation) -> AnnotationOut:
    return AnnotationOut(
        id=str(ann.id),
        kind=ann.kind.value,
        points=[AnnotationPointOut(x_norm=p.x_norm, y_norm=p.y_norm) for p in ann.points],
        label=ann.label,
        time_seconds=ann.time_seconds,
        frame_index=ann.frame_index,
        ref_width_px=ann.ref_width_px,
        ref_height_px=ann.ref_height_px,
    )
