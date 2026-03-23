"""Scale calibration from a reference rectangle (quadrat) on one video frame or image."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from crabspy_web.models.base import Base


class Calibration(Base):
    """Known-length edge of a quadrilateral in the image plane; defines mm per pixel for this project."""

    __tablename__ = "calibration"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # Media on which the quadrat was drawn (frame_index / time apply to this clip).
    source_media_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # JSON array of 4 {"x_norm","y_norm"} in [0,1], clockwise order
    corners_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Which edge (0–3): edge i connects corner i to corner (i+1)%4 and has length reference_length_mm.
    reference_edge_index: Mapped[int] = mapped_column(Integer, nullable=False)

    reference_length_mm: Mapped[float] = mapped_column(Float, nullable=False)

    ref_width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ref_height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # reference_length_mm / edge_length_px on that reference edge (isotropic scale).
    mm_per_px: Mapped[float] = mapped_column(Float, nullable=False)

    label: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    source_media: Mapped["Media"] = relationship(
        "Media",
        foreign_keys=[source_media_id],
        back_populates="calibrations_defined",
    )
