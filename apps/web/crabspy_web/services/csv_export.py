"""CSV export helpers for media rows."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

from crabspy_web.models.annotation import Annotation, AnnotationPoint
from crabspy_web.models.media import Media


def media_rows_to_csv_bytes(rows: Iterable[Media]) -> bytes:
    """UTF-8 CSV with header; suitable for ``Content-Disposition`` download."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "processing_status",
            "media_kind",
            "collected_at",
            "sample_code",
            "site_name",
            "location_name",
            "notes",
            "camera_id",
            "deployment_time",
            "deployment_type",
            "latitude",
            "longitude",
            "storage_path",
            "original_filename",
            "mime_type",
            "width_px",
            "height_px",
            "duration_seconds",
            "frame_rate",
            "checksum_sha256",
            "created_at",
            "updated_at",
        ]
    )
    for m in rows:
        writer.writerow(
            [
                str(m.id),
                m.processing_status.value,
                m.media_kind.value,
                m.collected_at.isoformat() if m.collected_at else "",
                m.sample_code or "",
                m.site_name or "",
                m.location_name or "",
                m.notes or "",
                m.camera_id or "",
                m.deployment_time.isoformat() if m.deployment_time else "",
                m.deployment_type or "",
                m.latitude if m.latitude is not None else "",
                m.longitude if m.longitude is not None else "",
                m.storage_path,
                m.original_filename or "",
                m.mime_type or "",
                m.width_px if m.width_px is not None else "",
                m.height_px if m.height_px is not None else "",
                m.duration_seconds if m.duration_seconds is not None else "",
                m.frame_rate if m.frame_rate is not None else "",
                m.checksum_sha256 or "",
                m.created_at.isoformat() if m.created_at else "",
                m.updated_at.isoformat() if m.updated_at else "",
            ]
        )
    return buffer.getvalue().encode("utf-8")


def annotation_rows_to_csv_bytes(annotations: Iterable[Annotation]) -> bytes:
    """One row per vertex; suitable for spreadsheets and GIS joins."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "annotation_id",
            "media_id",
            "kind",
            "label",
            "time_seconds",
            "frame_index",
            "ref_width_px",
            "ref_height_px",
            "point_order",
            "x_norm",
            "y_norm",
            "annotation_created_at",
            "annotation_updated_at",
        ]
    )
    for ann in annotations:
        points: list[AnnotationPoint] = sorted(ann.points, key=lambda p: p.order_index)
        for p in points:
            writer.writerow(
                [
                    str(ann.id),
                    str(ann.media_id),
                    ann.kind.value,
                    ann.label or "",
                    ann.time_seconds if ann.time_seconds is not None else "",
                    ann.frame_index if ann.frame_index is not None else "",
                    ann.ref_width_px if ann.ref_width_px is not None else "",
                    ann.ref_height_px if ann.ref_height_px is not None else "",
                    p.order_index,
                    p.x_norm,
                    p.y_norm,
                    ann.created_at.isoformat() if ann.created_at else "",
                    ann.updated_at.isoformat() if ann.updated_at else "",
                ]
            )
    return buffer.getvalue().encode("utf-8")
