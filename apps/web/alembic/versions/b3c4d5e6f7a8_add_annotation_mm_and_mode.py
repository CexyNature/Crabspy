"""add annotation path_length_mm and measurement_mode_used

Revision ID: b3c4d5e6f7a8
Revises: 9a2b7c1d4e6f
Create Date: 2026-03-27

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3c4d5e6f7a8"
down_revision: Union[str, None] = "9a2b7c1d4e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("annotation") as batch:
        batch.add_column(sa.Column("path_length_mm", sa.Float(), nullable=True))
        batch.add_column(
            sa.Column(
                "measurement_mode_used",
                sa.Enum(
                    "isotropic",
                    "homography",
                    name="annotationmeasurementmodeused",
                    native_enum=False,
                ),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("annotation") as batch:
        batch.drop_column("measurement_mode_used")
        batch.drop_column("path_length_mm")
