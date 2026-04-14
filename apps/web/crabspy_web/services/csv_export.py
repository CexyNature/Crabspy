"""CSV export helpers for media rows."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from uuid import UUID

from crabspy_web.models.annotation import Annotation, AnnotationPoint
from crabspy_web.models.calibration import Calibration
from crabspy_web.models.media import Media
from crabspy_web.services.annotation_geometry import annotation_path_lengths, edge_norm_and_px
from crabspy_web.services.calibration_measure import path_length_mm_and_mode_for_polyline


def _media_denormalized_cells(m: Media | None) -> list[object]:
    """One flat row of media metadata (same information as media export)."""
    if m is None:
        return [""] * len(MEDIA_DENORMALIZED_HEADER)
    return [
        m.processing_status.value,
        m.media_kind.value,
        m.measurement_mode.value,
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


def _active_calibration_denormalized_cells(cal: Calibration | None) -> list[object]:
    """Active calibration used for measurement on this media (empty if none)."""
    if cal is None:
        return ["", "", "", "", "", "", "", "", "", "", "", "", ""]
    return [
        str(cal.id),
        str(cal.source_media_id),
        cal.frame_index if cal.frame_index is not None else "",
        cal.time_seconds if cal.time_seconds is not None else "",
        cal.corners_json,
        cal.reference_edge_index,
        cal.reference_length_mm,
        cal.ref_width_px if cal.ref_width_px is not None else "",
        cal.ref_height_px if cal.ref_height_px is not None else "",
        cal.mm_per_px,
        cal.label or "",
        cal.created_at.isoformat() if cal.created_at else "",
        cal.updated_at.isoformat() if cal.updated_at else "",
    ]


MEDIA_DENORMALIZED_HEADER = [
    "media_processing_status",
    "media_kind",
    "media_measurement_mode",
    "media_collected_at",
    "media_sample_code",
    "media_site_name",
    "media_location_name",
    "media_notes",
    "media_camera_id",
    "media_deployment_time",
    "media_deployment_type",
    "media_latitude",
    "media_longitude",
    "media_storage_path",
    "media_original_filename",
    "media_mime_type",
    "media_width_px",
    "media_height_px",
    "media_duration_seconds",
    "media_frame_rate",
    "media_checksum_sha256",
    "media_created_at",
    "media_updated_at",
]

CALIBRATION_DENORMALIZED_HEADER = [
    "active_calibration_id",
    "calibration_source_media_id",
    "calibration_frame_index",
    "calibration_time_seconds",
    "calibration_corners_json",
    "calibration_reference_edge_index",
    "calibration_reference_length_mm",
    "calibration_ref_width_px",
    "calibration_ref_height_px",
    "calibration_mm_per_px",
    "calibration_label",
    "calibration_created_at",
    "calibration_updated_at",
]


def _mm_mode_for_csv_row(ann: Annotation, media: Media | None) -> tuple[float | str, str]:
    """Stored polyline mm/mode, or compute from media + calibration; empty strings if N/A."""
    if ann.path_length_mm is not None:
        mode = ann.measurement_mode_used.value if ann.measurement_mode_used else ""
        return round(float(ann.path_length_mm), 1), mode
    if media is None:
        return "", ""
    cal = getattr(media, "active_calibration", None)
    mm, mode_enum = path_length_mm_and_mode_for_polyline(ann, media, cal)
    if mm is None:
        return "", ""
    return round(float(mm), 1), mode_enum.value if mode_enum else ""


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


def annotation_rows_to_csv_bytes(
    annotations: Iterable[Annotation],
    *,
    media_by_id: dict[UUID, Media] | None = None,
) -> bytes:
    """One row per vertex; includes denormalized media + active calibration for a single spreadsheet."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "annotation_id",
            "media_id",
            *MEDIA_DENORMALIZED_HEADER,
            *CALIBRATION_DENORMALIZED_HEADER,
            "kind",
            "label",
            "time_seconds",
            "frame_index",
            "ref_width_px",
            "ref_height_px",
            "path_length_norm",
            "path_length_px",
            "path_length_mm",
            "measurement_mode_used",
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
        media = media_by_id.get(ann.media_id) if media_by_id else None
        mm_out, mode_out = _mm_mode_for_csv_row(ann, media)
        active_cal = getattr(media, "active_calibration", None) if media is not None else None
        media_cells = _media_denormalized_cells(media)
        cal_cells = _active_calibration_denormalized_cells(active_cal)
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
                    *media_cells,
                    *cal_cells,
                    ann.kind.value,
                    ann.label or "",
                    ann.time_seconds if ann.time_seconds is not None else "",
                    ann.frame_index if ann.frame_index is not None else "",
                    rw if rw is not None else "",
                    rh if rh is not None else "",
                    path_n,
                    "" if path_px is None else path_px,
                    mm_out,
                    mode_out,
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
