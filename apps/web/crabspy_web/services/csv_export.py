"""CSV export helpers for media rows."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

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
