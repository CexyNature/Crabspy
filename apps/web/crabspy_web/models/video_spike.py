"""Thin vertical-slice: reference points on video (Phase 3 spike; evolves into full annotations in 3b)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from crabspy_web.models.base import Base


class VideoSpikeAnnotation(Base):
    """One click-placed point on a video frame, normalized to the visible picture area."""

    __tablename__ = "video_spike_annotation"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    media_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    x_norm: Mapped[float] = mapped_column(Float, nullable=False)
    y_norm: Mapped[float] = mapped_column(Float, nullable=False)
    time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
