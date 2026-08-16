"""Open-Meteo client.

The only source in the framework covering both the past and forecasts,
hence two distinct hosts:

* ``https://archive-api.open-meteo.com`` — reanalysis archive;
* ``https://api.open-meteo.com``        — few-day-ahead forecast.

The ``timezone=UTC`` parameter is always enforced. This is the most
sensitive point in the whole ingestion module: letting the API return
local time would silently misalign series when joined with PVGIS or NASA
POWER, and a one-hour offset in an irradiance series ruins a forecasting
model without ever raising an error.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from solarcast.core.config import ProviderConfig
from solarcast.core.exceptions import ProviderError, ValidationError
from solarcast.core.logging import get_logger
from solarcast.core.types import ObservationPoint, Provider, Variable
from solarcast.ingestion.base import BaseProviderClient

logger = get_logger(__name__)

#: Canonical variable -> Open-Meteo hourly field mapping.
FIELD_MAP: dict[Variable, str] = {
    Variable.GHI: "shortwave_radiation",
    Variable.DNI: "direct_normal_irradiance",
    Variable.DHI: "diffuse_radiation",
    Variable.TEMP_AIR: "temperature_2m",
    Variable.WIND_SPEED: "wind_speed_10m",
    Variable.RELATIVE_HUMIDITY: "relative_humidity_2m",
    Variable.CLOUD_COVER: "cloud_cover",
    Variable.PRECIPITATION: "precipitation",
}

REVERSE_FIELD_MAP: dict[str, Variable] = {v: k for k, v in FIELD_MAP.items()}

DEFAULT_VARIABLES: list[Variable] = [
    Variable.GHI,
    Variable.DNI,
    Variable.DHI,
    Variable.TEMP_AIR,
    Variable.WIND_SPEED,
    Variable.RELATIVE_HUMIDITY,
    Variable.CLOUD_COVER,
]


class OpenMeteoClient(BaseProviderClient):
    """Weather and irradiance acquisition, historical and forecast."""

    provider = Provider.OPEN_METEO
    supports_forecast = True

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        # `base_url` holds the forecast host; the archive lives on a
        # separate host, declared in the provider's options.
        self._archive_url: str = config.options.get(
            "archive_url", "https://archive-api.open-meteo.com"
        )
        self._wind_speed_unit: str = config.options.get("wind_speed_unit", "ms")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _resolve(variables: list[Variable] | None) -> list[Variable]:
        selected = variables or DEFAULT_VARIABLES
        unknown = [v for v in selected if v not in FIELD_MAP]
        if unknown:
            names = ", ".join(v.value for v in unknown)
            raise ValidationError(f"variables not served by Open-Meteo: {names}")
        return selected

    def _parse(
        self,
        payload: dict[str, Any],
        variables: list[Variable],
        reference_time: datetime | None,
        dataset: str,
    ) -> list[ObservationPoint]:
        """Convert the response into canonical points."""
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise ProviderError(self.provider.value, "'hourly' block missing from response")

        raw_times = hourly.get("time")
        if not raw_times:
            logger.warning(
                "response has no timestamps",
                extra={"context": {"provider": self.provider.value}},
            )
            return []

        # With timezone=UTC, Open-Meteo returns naive ISO strings that must
        # be explicitly localized.
        timestamps = [
            datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in raw_times
        ]

        points: list[ObservationPoint] = []
        for variable in variables:
            field = FIELD_MAP[variable]
            series = hourly.get(field)
            if series is None:
                logger.warning(
                    "field missing from response",
                    extra={"context": {"field": field, "variable": variable.value}},
                )
                continue
            if len(series) != len(timestamps):
                raise ValidationError(
                    f"inconsistent length for '{field}': "
                    f"{len(series)} values for {len(timestamps)} timestamps"
                )

            for ts, value in zip(timestamps, series):
                points.append(
                    ObservationPoint(
                        provider=self.provider,
                        variable=variable,
                        timestamp=ts,
                        value=None if value is None else float(value),
                        reference_time=reference_time,
                        dataset=dataset,
                    )
                )
        return points

    # -------------------------------------------------------------- historical

    async def fetch_historical(
        self,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
        variables: list[Variable] | None = None,
    ) -> list[ObservationPoint]:
        """Query the reanalysis archive over `[start, end]`."""
        if end < start:
            raise ValidationError("window end precedes start")

        selected = self._resolve(variables)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(FIELD_MAP[v] for v in selected),
            "timezone": "UTC",
            "wind_speed_unit": self._wind_speed_unit,
        }

        # The archive lives on a different host: pass an absolute URL, so
        # httpx ignores the client's base_url.
        payload = await self.get_json(f"{self._archive_url}/v1/archive", params)
        points = self._parse(payload, selected, reference_time=None, dataset="archive")

        logger.info(
            "archive fetched",
            extra={
                "context": {
                    "provider": self.provider.value,
                    "points": len(points),
                    "start": params["start_date"],
                    "end": params["end_date"],
                }
            },
        )
        return points

    # ---------------------------------------------------------------- forecast

    async def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        horizon_days: int = 3,
        variables: list[Variable] | None = None,
    ) -> list[ObservationPoint]:
        """Fetch the hourly forecast for `horizon_days` days.

        Every point carries the same `reference_time` — the call instant —
        which later allows evaluating error as a function of horizon.
        """
        if not 1 <= horizon_days <= 16:
            raise ValidationError("horizon_days must be between 1 and 16")

        selected = self._resolve(variables)
        run_time = datetime.now(timezone.utc).replace(microsecond=0)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "forecast_days": horizon_days,
            "hourly": ",".join(FIELD_MAP[v] for v in selected),
            "timezone": "UTC",
            "wind_speed_unit": self._wind_speed_unit,
        }

        payload = await self.get_json("/v1/forecast", params)
        points = self._parse(
            payload, selected, reference_time=run_time, dataset="forecast"
        )

        logger.info(
            "forecast fetched",
            extra={
                "context": {
                    "provider": self.provider.value,
                    "points": len(points),
                    "run_time": run_time.isoformat(),
                    "horizon_days": horizon_days,
                }
            },
        )
        return points
