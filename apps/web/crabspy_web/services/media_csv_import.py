"""Parse bulk-import CSV files into ``Media`` field dicts (see ``/media/import``)."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath

from crabspy_web.models.media import MediaKind, MediaProcessingStatus
from crabspy_web.services.media_readiness import core_metadata_ready_for_processing

# First row = headers. Aliases normalize to canonical names (lower, spaces → underscores).
_HEADER_ALIASES: dict[str, str] = {
    "storage_path": "storage_path",
    "path": "storage_path",
    "video_path": "storage_path",
    "file_path": "storage_path",
    "filepath": "storage_path",
    "date_collected": "collected_at",
    "collected_at": "collected_at",
    "date": "collected_at",
    "sample_code": "sample_code",
    "site_name": "site_name",
    "site": "site_name",
    "location_name": "location_name",
    "location": "location_name",
    "notes": "notes",
    "original_filename": "original_filename",
    "filename": "original_filename",
    "camera_id": "camera_id",
    "camera": "camera_id",
    "deployment_time": "deployment_time",
    "deployment_type": "deployment_type",
    "deployment": "deployment_type",
    "latitude": "latitude",
    "lat": "latitude",
    "longitude": "longitude",
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
}

_CANONICAL = frozenset(
    {
        "storage_path",
        "collected_at",
        "sample_code",
        "site_name",
        "location_name",
        "notes",
        "original_filename",
        "camera_id",
        "deployment_time",
        "deployment_type",
        "latitude",
        "longitude",
    }
)

_VIDEO_SUFFIX = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".MP4", ".MOV", ".AVI"}
_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".gif", ".JPG", ".JPEG", ".PNG"}


def _normalize_header(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return _HEADER_ALIASES.get(s, s)


def _validate_storage_path(raw: str) -> str:
    p = raw.strip()
    if not p:
        raise ValueError("storage path is empty")
    if ".." in p:
        raise ValueError("storage path must not contain '..'")
    return p


def parse_collected_at(raw: str | None) -> datetime | None:
    """Parse date/time from CSV; returns timezone-aware UTC when possible."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        s_clean = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().replace(",", ".")
    return float(s)


def infer_media_kind(storage_path: str) -> MediaKind:
    suffix = PurePosixPath(storage_path).suffix
    if suffix in _VIDEO_SUFFIX:
        return MediaKind.video
    if suffix in _IMAGE_SUFFIX:
        return MediaKind.image
    return MediaKind.unknown


@dataclass(frozen=True)
class MediaImportRow:
    """Line number (1-based, file lines; data row) and kwargs for ``Media``."""

    line_no: int
    kwargs: dict


def parse_media_import_csv(text: str) -> tuple[list[MediaImportRow], list[tuple[int, str]]]:
    """
    Parse CSV text. Expected columns (aliases allowed), minimum **path** column:

    - ``storage_path`` (or ``path``, ``video_path``, …)
    - ``collected_at`` (or ``date_collected``, ``date``, …)
    - ``sample_code``, ``site_name``, ``location_name``, ``notes``
    - optional ``original_filename``
    - optional ``camera_id``, ``deployment_time``, ``deployment_type``, ``latitude``, ``longitude``

    Rows with path + date + sample + site + location all set become ``ready_for_processing``;
    otherwise ``draft``. Empty optional fields (including camera/deployment/geo) never block readiness.
    """
    stream = io.StringIO(text)
    reader = csv.DictReader(stream)
    if not reader.fieldnames:
        return [], [(1, "CSV has no header row.")]

    norm_to_raw: dict[str, str] = {}
    for raw in reader.fieldnames:
        if raw is None:
            continue
        norm = _normalize_header(raw)
        if norm in _CANONICAL:
            # First wins if duplicate canonical columns
            if norm not in norm_to_raw:
                norm_to_raw[norm] = raw

    if "storage_path" not in norm_to_raw:
        return [], [(1, "Missing a path column (e.g. storage_path, path, or video_path).")]

    def cell(row: dict[str, str | None], norm: str) -> str | None:
        raw = norm_to_raw.get(norm)
        if raw is None or raw not in row:
            return None
        v = row.get(raw)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    ok: list[MediaImportRow] = []
    errors: list[tuple[int, str]] = []
    line_no = 1
    for row in reader:
        line_no += 1
        path_raw = cell(row, "storage_path")
        try:
            path = _validate_storage_path(path_raw or "")
        except ValueError as e:
            errors.append((line_no, str(e)))
            continue

        collected = parse_collected_at(cell(row, "collected_at"))
        sample = cell(row, "sample_code")
        site = cell(row, "site_name")
        loc = cell(row, "location_name")
        notes = cell(row, "notes")
        orig = cell(row, "original_filename")
        camera_id = cell(row, "camera_id")
        deployment_type = cell(row, "deployment_type")
        deployment_time = parse_collected_at(cell(row, "deployment_time"))

        try:
            latitude = _parse_optional_float(cell(row, "latitude"))
            longitude = _parse_optional_float(cell(row, "longitude"))
        except ValueError:
            errors.append((line_no, "invalid latitude or longitude"))
            continue

        status = (
            MediaProcessingStatus.ready_for_processing
            if core_metadata_ready_for_processing(collected, sample, site, loc)
            else MediaProcessingStatus.draft
        )

        kwargs = {
            "storage_path": path,
            "collected_at": collected,
            "sample_code": sample,
            "site_name": site,
            "location_name": loc,
            "notes": notes,
            "original_filename": orig,
            "camera_id": camera_id,
            "deployment_time": deployment_time,
            "deployment_type": deployment_type,
            "latitude": latitude,
            "longitude": longitude,
            "media_kind": infer_media_kind(path),
            "processing_status": status,
        }
        ok.append(MediaImportRow(line_no=line_no, kwargs=kwargs))

    return ok, errors


def decode_uploaded_csv(raw: bytes) -> str:
    """Decode file body; strip UTF-8 BOM if present."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8")
