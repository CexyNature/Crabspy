"""add media measurement_mode

Revision ID: 9a2b7c1d4e6f
Revises: f8e9a1b2c3d4
Create Date: 2026-03-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "9a2b7c1d4e6f"
down_revision: Union[str, None] = "f8e9a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("media") as batch:
        batch.add_column(
            sa.Column(
                "measurement_mode",
                sa.Enum("isotropic", "homography", name="mediameasurementmode", native_enum=False),
                nullable=False,
                server_default="homography",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("media") as batch:
        batch.drop_column("measurement_mode")

