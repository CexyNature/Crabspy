"""Application settings (override with environment variables)."""

import os
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


class Settings:
    """Load settings from ``CRABSPY_*`` environment variables."""

    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("CRABSPY_DATA_DIR", _default_data_dir())).resolve()
        default_sqlite = (self.data_dir / "db" / "project.sqlite").resolve()
        self.database_url = os.environ.get(
            "CRABSPY_DATABASE_URL",
            f"sqlite:///{default_sqlite.as_posix()}",
        )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


def get_settings() -> Settings:
    return Settings()
