"""Data model.

Storage follows a **long format**: one row per (site, provider, variable,
timestamp). This is more verbose than a wide format, but necessary here,
since providers expose neither the same variables nor the same time step;
a wide schema would force a migration on every new source.

Time convention
----------------
Every instant is stored **in UTC**. Conversions to local time (or local
solar time, NASA POWER's default convention) happen on output, never in
the database.

`reference_time` distinguishes a measurement from a forecast:

* for historical data, ``reference_time == timestamp``;
* for a forecast, ``reference_time`` is the emission time of the run.

The column is non-nullable so the uniqueness constraint stays effective —
in SQL, two NULLs are not considered equal.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from solarcast.core.types import Provider, RunStatus, Variable

__all__ = ["Base", "Location", "Observation", "IngestionRun"]


class Base(DeclarativeBase):
    """Shared declarative base."""


class Location(Base):
    """Geographic site, instrumented or simulated."""

    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    altitude_m: Mapped[float | None] = mapped_column(Float)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    observations: Mapped[list["Observation"]] = relationship(
        back_populates="location", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("latitude BETWEEN -90 AND 90", name="ck_location_latitude"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="ck_location_longitude"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<Location {self.name} ({self.latitude:.4f}, {self.longitude:.4f})>"


class Observation(Base):
    """A time-series point, either a measurement or a forecast."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[Provider] = mapped_column(
        SAEnum(Provider, native_enum=False, length=32), nullable=False
    )
    variable: Mapped[Variable] = mapped_column(
        SAEnum(Variable, native_enum=False, length=32), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reference_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset: Mapped[str | None] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    location: Mapped[Location] = relationship(back_populates="observations")

    __table_args__ = (
        UniqueConstraint(
            "location_id",
            "provider",
            "variable",
            "timestamp",
            "reference_time",
            name="uq_observation_point",
        ),
        Index("ix_observation_lookup", "location_id", "variable", "timestamp"),
        Index("ix_observation_provider_ts", "provider", "timestamp"),
    )

    @property
    def is_forecast(self) -> bool:
        return self.reference_time < self.timestamp

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Observation {self.provider.value}/{self.variable.value} "
            f"@{self.timestamp.isoformat()} = {self.value}>"
        )


class IngestionRun(Base):
    """Execution trace of an ingestion job.

    Useful both for debugging and for resuming: a ``FAILED`` run points to
    the window that needs replaying, without having to query the entire
    observations table.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[Provider] = mapped_column(
        SAEnum(Provider, native_enum=False, length=32), nullable=False
    )
    location_id: Mapped[int | None] = mapped_column(
        ForeignKey("locations.id", ondelete="SET NULL")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[RunStatus] = mapped_column(
        SAEnum(RunStatus, native_enum=False, length=16),
        default=RunStatus.RUNNING,
        nullable=False,
    )
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rows_written: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_run_provider_started", "provider", "started_at"),)
