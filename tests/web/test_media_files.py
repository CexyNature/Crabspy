"""Unit tests for ``resolve_storage_path_to_file``."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from crabspy_web.config import Settings
from crabspy_web.services.media_files import resolve_storage_path_to_file


def test_resolve_relative_under_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRABSPY_DATA_DIR", str(tmp_path))
    (tmp_path / "uploads" / "a").mkdir(parents=True)
    f = tmp_path / "uploads" / "a" / "x.mp4"
    f.write_bytes(b"x")
    settings = Settings()
    out = resolve_storage_path_to_file(settings, "uploads/a/x.mp4")
    assert out == f.resolve()


def test_resolve_absolute_outside_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRABSPY_DATA_DIR", str(tmp_path))
    outside = tmp_path.parent / f"crabspy_abs_{uuid.uuid4().hex}"
    outside.mkdir(parents=True)
    other = outside / "f.jpg"
    other.write_bytes(b"img")
    settings = Settings()
    out = resolve_storage_path_to_file(settings, str(other))
    assert out == other.resolve()


def test_reject_dotdot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CRABSPY_DATA_DIR", str(tmp_path))
    settings = Settings()
    with pytest.raises(ValueError, match="Invalid"):
        resolve_storage_path_to_file(settings, "uploads/../etc/passwd")


def test_reject_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRABSPY_DATA_DIR", str(tmp_path))
    settings = Settings()
    with pytest.raises(FileNotFoundError):
        resolve_storage_path_to_file(settings, "uploads/nope.mp4")
