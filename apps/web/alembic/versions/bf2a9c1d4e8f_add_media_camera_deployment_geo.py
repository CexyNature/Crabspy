"""add camera deployment and geo fields to media

Revision ID: bf2a9c1d4e8f
Revises: 44abc7f92d9d
Create Date: 2026-03-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "bf2a9c1d4e8f"
down_revision: Union[str, None] = "44abc7f92d9d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("media", sa.Column("camera_id", sa.String(length=255), nullable=True))
    op.add_column(
        "media",
        sa.Column("deployment_time", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "media",
        sa.Column("deployment_type", sa.String(length=255), nullable=True),
    )
    op.add_column("media", sa.Column("latitude", sa.Float(), nullable=True))
    op.add_column("media", sa.Column("longitude", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("media", "longitude")
    op.drop_column("media", "latitude")
    op.drop_column("media", "deployment_type")
    op.drop_column("media", "deployment_time")
    op.drop_column("media", "camera_id")
