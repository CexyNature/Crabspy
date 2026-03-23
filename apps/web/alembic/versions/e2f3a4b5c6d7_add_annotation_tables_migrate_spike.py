"""add annotation + annotation_point; migrate video_spike_annotation; drop spike

Revision ID: e2f3a4b5c6d7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-23

"""
from __future__ import annotations

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "annotation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("media_id", sa.Uuid(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum("point", "polyline", name="annotationkind", native_enum=False),
            nullable=False,
        ),
        sa.Column("label", sa.String(length=512), nullable=True),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("time_seconds", sa.Float(), nullable=True),
        sa.Column("ref_width_px", sa.Integer(), nullable=True),
        sa.Column("ref_height_px", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(
            ["media_id"],
            ["media.id"],
            name=op.f("fk_annotation_media_id_media"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_annotation")),
    )
    op.create_index(op.f("ix_annotation_media_id"), "annotation", ["media_id"], unique=False)

    op.create_table(
        "annotation_point",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("annotation_id", sa.Uuid(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("x_norm", sa.Float(), nullable=False),
        sa.Column("y_norm", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["annotation_id"],
            ["annotation.id"],
            name=op.f("fk_annotation_point_annotation_id_annotation"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_annotation_point")),
    )
    op.create_index(
        op.f("ix_annotation_point_annotation_id"),
        "annotation_point",
        ["annotation_id"],
        unique=False,
    )

    # One statement copies all spike rows into annotation (faster + fewer locks than row-by-row).
    op.execute(
        sa.text(
            """
            INSERT INTO annotation (
                id, media_id, kind, label, frame_index, time_seconds,
                ref_width_px, ref_height_px, created_at, updated_at
            )
            SELECT
                id,
                media_id,
                'point',
                NULL,
                frame_index,
                time_seconds,
                NULL,
                NULL,
                created_at,
                created_at
            FROM video_spike_annotation
            """
        )
    )

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, x_norm, y_norm FROM video_spike_annotation")
    ).mappings().all()

    for r in rows:
        pid = uuid.uuid4()
        conn.execute(
            sa.text(
                "INSERT INTO annotation_point (id, annotation_id, order_index, x_norm, y_norm) "
                "VALUES (:pid, :aid, 0, :x, :y)"
            ),
            {
                "pid": str(pid),
                "aid": str(r["id"]),
                "x": r["x_norm"],
                "y": r["y_norm"],
            },
        )

    op.drop_index(op.f("ix_video_spike_annotation_media_id"), table_name="video_spike_annotation")
    op.drop_table("video_spike_annotation")


def downgrade() -> None:
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

    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT a.id, a.media_id, a.time_seconds, a.frame_index, a.created_at, "
            "p.x_norm, p.y_norm "
            "FROM annotation a "
            "JOIN annotation_point p ON p.annotation_id = a.id AND p.order_index = 0 "
            "WHERE a.kind = 'point'"
        )
    ).mappings().all()
    for r in rows:
        conn.execute(
            sa.text(
                "INSERT INTO video_spike_annotation "
                "(id, media_id, x_norm, y_norm, time_seconds, frame_index, created_at) "
                "VALUES (:id, :mid, :x, :y, :ts, :fi, :ca)"
            ),
            {
                "id": str(r["id"]),
                "mid": str(r["media_id"]),
                "x": r["x_norm"],
                "y": r["y_norm"],
                "ts": r["time_seconds"],
                "fi": r["frame_index"],
                "ca": r["created_at"],
            },
        )

    op.drop_index(op.f("ix_annotation_point_annotation_id"), table_name="annotation_point")
    op.drop_table("annotation_point")
    op.drop_index(op.f("ix_annotation_media_id"), table_name="annotation")
    op.drop_table("annotation")
