"""Isolate Crabspy web data dir and DB per test so the repo ``data/`` tree is not touched."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_crabspy_web_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``CRABSPY_DATA_DIR`` and SQLite at a temporary directory."""
    (tmp_path / "db").mkdir(parents=True)
    dbfile = tmp_path / "db" / "test.sqlite"
    monkeypatch.setenv("CRABSPY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CRABSPY_DATABASE_URL", f"sqlite:///{dbfile.resolve().as_posix()}")
