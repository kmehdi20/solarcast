"""Data access layer.

Two principles shape this module:

**Idempotence.** Writes go through an ``INSERT ... ON CONFLICT DO UPDATE``
on the ``uq_observation_point`` constraint. Replaying an already-ingested
window updates the values instead of duplicating rows — essential since
providers commonly revise their data after the fact, as reanalyses do.

**Pandas output.** The feature pipeline consumes pivoted DataFrames (time
index, one column per variable), not ORM objects. That conversion happens
here, once, rather than being scattered across downstream modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from solarcast.core.exceptions import StorageError
from solarcast.core.logging import get_logger
from solarcast.core.types import ObservationPoint, Provider, RunStatus, Variable
from solarcast.storage.models import IngestionRun, Location, Observation

logger = get_logger(__name__)

#: Insert batch size. A tradeoff between round-trip count and the bound
#: parameter limit (999 by default on older SQLite).
DEFAULT_CHUNK_SIZE = 500


class LocationRepository:
    """Site management."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_name(self, name: str) -> Location | None:
        stmt = select(Location).where(Location.name == name)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_or_create(
        self,
        name: str,
        latitude: float,
        longitude: float,
        altitude_m: float | None = None,
        tz: str = "UTC",
    ) -> Location:
        """Fetch the site or create it. Coordinates are never overwritten."""
        existing = await self.get_by_name(name)
        if existing is not None:
            return existing

        location = Location(
            name=name,
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude_m,
            timezone=tz,
        )
        self._session.add(location)
        await self._session.flush()
        logger.info(
            "site created",
            extra={"context": {"site": name, "lat": latitude, "lon": longitude}},
        )
        return location

    async def list_all(self) -> Sequence[Location]:
        stmt = select(Location).order_by(Location.name)
        return (await self._session.execute(stmt)).scalars().all()


class ObservationRepository:
    """Reading and writing time series."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ write

    def _upsert_statement(self, rows: list[dict[str, object]]):
        """Build the upsert statement for the current dialect."""
        dialect = self._session.bind.dialect.name if self._session.bind else "sqlite"

        if dialect == "postgresql":
            stmt = pg_insert(Observation).values(rows)
        elif dialect == "sqlite":
            stmt = sqlite_insert(Observation).values(rows)
        else:
            raise StorageError(
                f"upsert not implemented for dialect '{dialect}' "
                "(add a branch in _upsert_statement)"
            )

        return stmt.on_conflict_do_update(
            index_elements=[
                Observation.location_id,
                Observation.provider,
                Observation.variable,
                Observation.timestamp,
                Observation.reference_time,
            ],
            set_={
                "value": stmt.excluded.value,
                "unit": stmt.excluded.unit,
                "dataset": stmt.excluded.dataset,
            },
        )

    async def upsert_many(
        self,
        location_id: int,
        points: Iterable[ObservationPoint],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> int:
        """Write a batch of points idempotently.

        Returns the number of points submitted (not the number of rows
        actually created: an upsert doesn't portably distinguish insert
        from update).
        """
        rows = [
            {
                "location_id": location_id,
                "provider": p.provider,
                "variable": p.variable,
                "timestamp": p.timestamp,
                "reference_time": p.reference_time,
                "value": p.value,
                "unit": p.unit or "",
                "dataset": p.dataset,
            }
            for p in points
        ]
        if not rows:
            return 0

        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            await self._session.execute(self._upsert_statement(chunk))

        logger.info(
            "points written",
            extra={"context": {"location_id": location_id, "count": len(rows)}},
        )
        return len(rows)

    # ------------------------------------------------------------------- read

    async def fetch(
        self,
        location_id: int,
        variables: Sequence[Variable] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        provider: Provider | None = None,
        forecasts: bool = False,
    ) -> Sequence[Observation]:
        """Return the raw observations matching the filter.

        By default, only historical data is returned
        (``reference_time == timestamp``).
        """
        stmt = select(Observation).where(Observation.location_id == location_id)

        if variables:
            stmt = stmt.where(Observation.variable.in_(list(variables)))
        if start is not None:
            stmt = stmt.where(Observation.timestamp >= start)
        if end is not None:
            stmt = stmt.where(Observation.timestamp <= end)
        if provider is not None:
            stmt = stmt.where(Observation.provider == provider)
        if not forecasts:
            stmt = stmt.where(Observation.reference_time == Observation.timestamp)

        stmt = stmt.order_by(Observation.timestamp)
        return (await self._session.execute(stmt)).scalars().all()

    async def to_frame(
        self,
        location_id: int,
        variables: Sequence[Variable] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        provider: Provider | None = None,
    ) -> pd.DataFrame:
        """Return a pivoted DataFrame ready for the feature pipeline.

        Index: `timestamp` (UTC, sorted). Columns: one per variable. An
        empty DataFrame is returned if the filter matches nothing — that's
        a normal case, not an error.
        """
        observations = await self.fetch(
            location_id, variables=variables, start=start, end=end, provider=provider
        )
        if not observations:
            return pd.DataFrame()

        frame = pd.DataFrame(
            {
                "timestamp": [o.timestamp for o in observations],
                "variable": [o.variable.value for o in observations],
                "value": [o.value for o in observations],
            }
        )
        pivoted = frame.pivot_table(
            index="timestamp", columns="variable", values="value", aggfunc="last"
        )
        pivoted.columns.name = None
        return pivoted.sort_index()

    async def coverage(
        self, location_id: int, provider: Provider, variable: Variable
    ) -> tuple[datetime | None, datetime | None]:
        """First and last instant available for a series.

        Lets an incremental ingestion job request only the missing window
        instead of re-downloading the whole history.
        """
        from sqlalchemy import func as sa_func

        stmt = select(
            sa_func.min(Observation.timestamp), sa_func.max(Observation.timestamp)
        ).where(
            Observation.location_id == location_id,
            Observation.provider == provider,
            Observation.variable == variable,
            Observation.reference_time == Observation.timestamp,
        )
        first, last = (await self._session.execute(stmt)).one()
        return first, last


class IngestionRunRepository:
    """Log of ingestion runs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start(
        self,
        provider: Provider,
        location_id: int | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> IngestionRun:
        run = IngestionRun(
            provider=provider,
            location_id=location_id,
            window_start=window_start,
            window_end=window_end,
            status=RunStatus.RUNNING,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def finish(
        self,
        run: IngestionRun,
        status: RunStatus,
        rows_written: int = 0,
        error: str | None = None,
    ) -> None:
        run.status = status
        run.rows_written = rows_written
        run.error = error
        run.finished_at = datetime.now(timezone.utc)
        await self._session.flush()
        logger.info(
            "run finished",
            extra={
                "context": {
                    "provider": run.provider.value,
                    "status": status.value,
                    "rows": rows_written,
                }
            },
        )
