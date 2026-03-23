"""add calibration table; media.active_calibration_id

Revision ID: f8e9a1b2c3d4
Revises: e2f3a4b5c6d7
Create Date: 2026-03-23

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f8e9a1b2c3d4"
down_revision: Union[str, None] = "e2f3a4b5c6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "calibration",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_media_id", sa.Uuid(), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("time_seconds", sa.Float(), nullable=True),
        sa.Column("corners_json", sa.Text(), nullable=False),
        sa.Column("reference_edge_index", sa.Integer(), nullable=False),
        sa.Column("reference_length_mm", sa.Float(), nullable=False),
        sa.Column("ref_width_px", sa.Integer(), nullable=True),
        sa.Column("ref_height_px", sa.Integer(), nullable=True),
        sa.Column("mm_per_px", sa.Float(), nullable=False),
        sa.Column("label", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_media_id"],
            ["media.id"],
            name=op.f("fk_calibration_source_media_id_media"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_calibration")),
    )
    op.create_index(op.f("ix_calibration_source_media_id"), "calibration", ["source_media_id"], unique=False)

    # SQLite cannot ALTER TABLE ADD CONSTRAINT; keep column + FK + index in one batch_alter_table.
    with op.batch_alter_table("media") as batch:
        batch.add_column(
            sa.Column(
                "active_calibration_id",
                sa.Uuid(),
                nullable=True,
            )
        )
        batch.create_foreign_key(
            op.f("fk_media_active_calibration_id_calibration"),
            "calibration",
            ["active_calibration_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index(
            op.f("ix_media_active_calibration_id"),
            ["active_calibration_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("media") as batch:
        batch.drop_constraint(op.f("fk_media_active_calibration_id_calibration"), type_="foreignkey")
        batch.drop_index(op.f("ix_media_active_calibration_id"))
        batch.drop_column("active_calibration_id")

    op.drop_index(op.f("ix_calibration_source_media_id"), table_name="calibration")
    op.drop_table("calibration")
