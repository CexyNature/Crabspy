"""CSV export helpers for media rows."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

from crabspy_web.models.annotation import Annotation, AnnotationPoint
from crabspy_web.models.media import Media
from crabspy_web.services.annotation_geometry import annotation_path_lengths, edge_norm_and_px


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
            "path_length_norm",
            "path_length_px",
            "point_order",
            "x_norm",
            "y_norm",
            "edge_length_norm",
            "edge_length_px",
            "annotation_created_at",
            "annotation_updated_at",
        ]
    )
    for ann in annotations:
        points: list[AnnotationPoint] = sorted(ann.points, key=lambda p: p.order_index)
        coords = [(p.x_norm, p.y_norm) for p in points]
        path_n, path_px = annotation_path_lengths(ann, coords)
        rw = ann.ref_width_px
        rh = ann.ref_height_px
        for i, p in enumerate(points):
            edge_n: float | str = ""
            edge_px_out: float | str = ""
            if i > 0:
                prev = points[i - 1]
                en, epx = edge_norm_and_px(
                    prev.x_norm, prev.y_norm, p.x_norm, p.y_norm, rw, rh
                )
                edge_n = en
                if epx is not None:
                    edge_px_out = epx
            writer.writerow(
                [
                    str(ann.id),
                    str(ann.media_id),
                    ann.kind.value,
                    ann.label or "",
                    ann.time_seconds if ann.time_seconds is not None else "",
                    ann.frame_index if ann.frame_index is not None else "",
                    rw if rw is not None else "",
                    rh if rh is not None else "",
                    path_n,
                    "" if path_px is None else path_px,
                    p.order_index,
                    p.x_norm,
                    p.y_norm,
                    edge_n,
                    edge_px_out,
                    ann.created_at.isoformat() if ann.created_at else "",
                    ann.updated_at.isoformat() if ann.updated_at else "",
                ]
            )
    return buffer.getvalue().encode("utf-8")
