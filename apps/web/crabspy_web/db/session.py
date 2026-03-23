"""Engine and session factory (PostgreSQL or SQLite via ``CRABSPY_DATABASE_URL``)."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def create_engine_from_url(database_url: str) -> Engine:
    kwargs: dict = {"future": True}
    if database_url.startswith("sqlite"):
        # timeout seconds: wait on SQLITE_BUSY instead of failing immediately (e.g. reload + migration).
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30.0}
    return create_engine(database_url, **kwargs)


def init_engine(database_url: str) -> None:
    """Create (or replace) the global engine and session factory."""
    global _engine, _session_factory
    _engine = create_engine_from_url(database_url)
    _session_factory = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Database engine not initialized; call init_engine() during app startup.")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized; call init_engine() during app startup.")
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request, always closed."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
