"""Resolve ``Media.storage_path`` to a readable file on disk (Phase 3a viewer)."""

from __future__ import annotations

from pathlib import Path

from crabspy_web.config import Settings


def resolve_storage_path_to_file(settings: Settings, storage_path: str) -> Path:
    """
    - **Relative** paths are resolved under ``CRABSPY_DATA_DIR`` (must stay within it after resolve).
    - **Absolute** paths (e.g. external SSD) are used as-is after ``resolve()``.

    Raises:
        ValueError: empty path, ``..``, or relative path escaping ``data_dir``.
        FileNotFoundError: path is not a regular file.
    """
    raw = storage_path.strip()
    if not raw or ".." in raw:
        raise ValueError("Invalid storage path.")

    p = Path(raw)
    data_root = settings.data_dir.resolve()

    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (settings.data_dir / p).resolve()
        try:
            resolved.relative_to(data_root)
        except ValueError:
            raise ValueError("Resolved path would escape the data directory.") from None

    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))

    return resolved
