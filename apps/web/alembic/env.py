"""Alembic migration environment (wire SQLAlchemy metadata here)."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from crabspy_web.models import Base

target_metadata = Base.metadata


def get_url() -> str:
    return os.environ.get("CRABSPY_DATABASE_URL") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = get_url()
    # SQLite: engine_from_config does not apply app connect_args; set timeout here so
    # migrations wait on SQLITE_BUSY instead of hanging indefinitely when the file is busy.
    if url.startswith("sqlite"):
        connectable = create_engine(
            url,
            poolclass=pool.NullPool,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30.0},
        )
    else:
        configuration = config.get_section(config.config_ini_section) or {}
        configuration["sqlalchemy.url"] = url
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
