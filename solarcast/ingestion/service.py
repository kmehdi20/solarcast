"""Ingestion orchestration.

This module bridges the client layer (which knows how to talk to APIs)
and the repository layer (which knows how to write). It owns three
responsibilities that neither of the others should carry:

* **run logging** — every attempt leaves a trace in the database, success
  or failure, which makes resuming possible;
* **failure isolation** — one provider's failure doesn't cancel the others;
* **parallelism** — sites and sources are queried concurrently, with rate
  regulation still enforced by each client.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from solarcast.core.config import LocationConfig, ProviderConfig, Settings
from solarcast.core.exceptions import ConfigError, SolarCastError
from solarcast.core.logging import get_logger
from solarcast.core.types import Provider, RunStatus, Variable
from solarcast.ingestion.base import BaseProviderClient
from solarcast.ingestion.openmeteo import OpenMeteoClient
from solarcast.ingestion.pvgis import PVGISClient
from solarcast.storage.repository import (
    IngestionRunRepository,
    LocationRepository,
    ObservationRepository,
)
from solarcast.storage.session import session_scope

logger = get_logger(__name__)

#: Registry of available clients, indexed by provider name.
CLIENT_REGISTRY: dict[str, type[BaseProviderClient]] = {
    Provider.OPEN_METEO.value: OpenMeteoClient,
    Provider.PVGIS.value: PVGISClient,
}


def build_client(name: str, config: ProviderConfig) -> BaseProviderClient:
    """Instantiate the client matching the provider name."""
    try:
        client_cls = CLIENT_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(CLIENT_REGISTRY))
        raise ConfigError(
            f"no client registered for '{name}' (available: {known})"
        ) from exc
    return client_cls(config)


@dataclass
class IngestionResult:
    """Report of a single ingestion run."""

    provider: str
    location: str
    status: RunStatus
    points: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.SUCCESS


async def ingest_historical(
    provider_name: str,
    provider_config: ProviderConfig,
    location: LocationConfig,
    start: datetime,
    end: datetime,
    variables: list[Variable] | None = None,
) -> IngestionResult:
    """Ingest a provider's history for one site.

    Never raises: every error is caught, logged to the database, and
    returned in the result. This is what allows running several ingestions
    in parallel without a single failure interrupting the whole campaign.
    """
    provider = Provider(provider_name)

    async with session_scope() as session:
        location_row = await LocationRepository(session).get_or_create(
            name=location.name,
            latitude=location.latitude,
            longitude=location.longitude,
            altitude_m=location.altitude_m,
            tz=location.timezone,
        )
        location_id = location_row.id
        run = await IngestionRunRepository(session).start(
            provider=provider,
            location_id=location_id,
            window_start=start,
            window_end=end,
        )
        run_id = run.id

    try:
        client = build_client(provider_name, provider_config)
        async with client:
            points = await client.fetch_historical(
                latitude=location.latitude,
                longitude=location.longitude,
                start=start,
                end=end,
                variables=variables,
            )

        async with session_scope() as session:
            written = await ObservationRepository(session).upsert_many(
                location_id, points
            )
            run = await session.get(
                type(run), run_id
            )  # reload in the new session
            await IngestionRunRepository(session).finish(
                run, RunStatus.SUCCESS, rows_written=written
            )

        return IngestionResult(
            provider=provider_name,
            location=location.name,
            status=RunStatus.SUCCESS,
            points=written,
        )

    except (SolarCastError, asyncio.TimeoutError) as exc:
        message = str(exc)
        logger.error(
            "ingestion failed",
            extra={
                "context": {
                    "provider": provider_name,
                    "site": location.name,
                    "error": message,
                }
            },
        )
        async with session_scope() as session:
            from solarcast.storage.models import IngestionRun

            run = await session.get(IngestionRun, run_id)
            if run is not None:
                await IngestionRunRepository(session).finish(
                    run, RunStatus.FAILED, error=message[:2000]
                )
        return IngestionResult(
            provider=provider_name,
            location=location.name,
            status=RunStatus.FAILED,
            error=message,
        )


async def ingest_all(
    settings: Settings,
    start: datetime,
    end: datetime,
    providers: list[str] | None = None,
    variables: list[Variable] | None = None,
) -> list[IngestionResult]:
    """Ingest every (active provider x site) combination in parallel."""
    selected = providers or [
        name for name, cfg in settings.providers.items() if cfg.enabled
    ]

    tasks = [
        ingest_historical(
            provider_name=name,
            provider_config=settings.provider(name),
            location=location,
            start=start,
            end=end,
            variables=variables,
        )
        for name in selected
        for location in settings.locations
    ]

    if not tasks:
        logger.warning("no ingestion tasks: check sites and providers")
        return []

    results = await asyncio.gather(*tasks)
    succeeded = sum(1 for r in results if r.ok)
    logger.info(
        "ingestion campaign finished",
        extra={
            "context": {
                "tasks": len(results),
                "succeeded": succeeded,
                "failed": len(results) - succeeded,
                "points": sum(r.points for r in results),
            }
        },
    )
    return list(results)
