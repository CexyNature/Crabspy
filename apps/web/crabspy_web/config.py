"""Application settings (override with environment variables)."""

import os
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


class Settings:
    """Load settings from ``CRABSPY_*`` environment variables and optional persisted config."""

    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("CRABSPY_DATA_DIR", _default_data_dir())).resolve()

    @property
    def config_dir(self) -> Path:
        return self.data_dir / "config"

    def resolve_database_url(self) -> str:
        """Active DB URL: ``CRABSPY_DATABASE_URL`` env wins; else ``data/config/database_url``; else default SQLite."""
        env = os.environ.get("CRABSPY_DATABASE_URL")
        if env and env.strip():
            return env.strip()
        path = self.config_dir / "database_url"
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        default_sqlite = (self.data_dir / "db" / "project.sqlite").resolve()
        return f"sqlite:///{default_sqlite.as_posix()}"

    @property
    def database_url(self) -> str:
        return self.resolve_database_url()

    @property
    def database_url_from_env(self) -> bool:
        """If True, ``CRABSPY_DATABASE_URL`` is set and overrides persisted / default URLs."""
        return bool(os.environ.get("CRABSPY_DATABASE_URL", "").strip())

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


def persist_database_url(settings: Settings, url: str) -> None:
    """Write ``data/config/database_url`` (one line). Used when switching projects from the UI."""
    settings.config_dir.mkdir(parents=True, exist_ok=True)
    path = settings.config_dir / "database_url"
    path.write_text(url.strip() + "\n", encoding="utf-8")


def get_settings() -> Settings:
    return Settings()
