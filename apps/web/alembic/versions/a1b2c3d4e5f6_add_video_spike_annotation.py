"""add video_spike_annotation table (thin annotation spike)

Revision ID: a1b2c3d4e5f6
Revises: bf2a9c1d4e8f
Create Date: 2026-03-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "bf2a9c1d4e8f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "video_spike_annotation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("media_id", sa.Uuid(), nullable=False),
        sa.Column("x_norm", sa.Float(), nullable=False),
        sa.Column("y_norm", sa.Float(), nullable=False),
        sa.Column("time_seconds", sa.Float(), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(
            ["media_id"],
            ["media.id"],
            name=op.f("fk_video_spike_annotation_media_id_media"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_video_spike_annotation")),
    )
    op.create_index(
        op.f("ix_video_spike_annotation_media_id"),
        "video_spike_annotation",
        ["media_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_video_spike_annotation_media_id"), table_name="video_spike_annotation")
    op.drop_table("video_spike_annotation")
