"""Run Alembic migrations programmatically (same revisions as CLI ``alembic upgrade head``)."""

from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def alembic_ini_path() -> Path:
    """``apps/web/alembic.ini`` (two levels above ``crabspy_web/db/``)."""
    return Path(__file__).resolve().parents[2] / "alembic.ini"


def run_alembic_upgrade_head(database_url: str) -> None:
    """Apply migrations to ``database_url`` (sets ``CRABSPY_DATABASE_URL`` only for the upgrade call)."""
    prev = os.environ.get("CRABSPY_DATABASE_URL")
    ini = alembic_ini_path()
    try:
        os.environ["CRABSPY_DATABASE_URL"] = database_url
        cfg = Config(str(ini))
        # ``script_location`` in alembic.ini is relative; resolve so CLI and pytest (any cwd) work.
        cfg.set_main_option("script_location", str(ini.parent / "alembic"))
        cfg.set_main_option("prepend_sys_path", str(ini.parent))
        command.upgrade(cfg, "head")
    finally:
        if prev is None:
            os.environ.pop("CRABSPY_DATABASE_URL", None)
        else:
            os.environ["CRABSPY_DATABASE_URL"] = prev
