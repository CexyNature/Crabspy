"""Declarative base with cross-dialect naming suited for PostgreSQL."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


# Explicit names for constraints and indexes (matches typical PostgreSQL conventions).
_NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for Crabspy web."""

    metadata = MetaData(naming_convention=_NAMING_CONVENTION)
