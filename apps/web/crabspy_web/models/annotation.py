"""Annotations: reference points and line measurements (normalized coordinates)."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from crabspy_web.models.base import Base


class AnnotationKind(str, enum.Enum):
    point = "point"
    polyline = "polyline"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class Annotation(Base):
    """One annotation (point or polyline) on a media item."""

    __tablename__ = "annotation"

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
    kind: Mapped[AnnotationKind] = mapped_column(
        Enum(AnnotationKind, values_callable=_enum_values, native_enum=False),
        nullable=False,
    )
    label: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Video: at least one of time_seconds / frame_index should be set; images: both null.
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    ref_width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ref_height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)

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

    points: Mapped[list["AnnotationPoint"]] = relationship(
        back_populates="annotation",
        cascade="all, delete-orphan",
        order_by="AnnotationPoint.order_index",
    )


class AnnotationPoint(Base):
    """Ordered vertex in normalized 0–1 space (relative to picture at annotation time)."""

    __tablename__ = "annotation_point"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    annotation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("annotation.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    x_norm: Mapped[float] = mapped_column(Float, nullable=False)
    y_norm: Mapped[float] = mapped_column(Float, nullable=False)

    annotation: Mapped["Annotation"] = relationship(back_populates="points")
