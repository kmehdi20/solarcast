"""PVGIS client (Joint Research Centre).

Endpoint used: ``seriescalc``, the non-interactive service of the "Hourly
data" tool. The API version is pinned in the configuration (``v5_2`` by
default) because the JRC has changed default behaviors between versions
before, notably the time coverage of non-interactive calls and the default
value of ``pvtechchoice``.

Three quirks worth knowing
---------------------------
**Annual granularity.** The window is specified in years (``startyear`` /
``endyear``), not dates. A three-month request returns the entire year,
which is then trimmed client-side.

**Timestamps.** Values are dated in ``YYYYMMDD:HHMM`` format. Depending on
the chosen database, minutes aren't necessarily zero: satellite-based
SARAH databases provide instantaneous values, while ERA5 provides hourly
averages. Timestamps are therefore floored to the start of the hour to
stay joinable with other sources, and the source database is kept in the
`dataset` field so this choice remains traceable.

**Tilt plane.** With ``angle=0``, ``G(i)`` is the global horizontal
irradiance; as soon as the plane is tilted, the same key denotes
plane-of-array irradiance. The translation to the canonical vocabulary
accounts for this.
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

#: PVGIS fields that are independent of tilt.
STATIC_FIELD_MAP: dict[str, Variable] = {
    "Gb(n)": Variable.DNI,
    "Gd(i)": Variable.DHI,
    "T2m": Variable.TEMP_AIR,
    "WS10m": Variable.WIND_SPEED,
    "P": Variable.PV_POWER,
}


class PVGISClient(BaseProviderClient):
    """Long-term irradiance history and modeled PV production."""

    provider = Provider.PVGIS
    supports_forecast = False  # PVGIS only publishes historical data

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        options = config.options
        self._api_version: str = options.get("api_version", "v5_2")
        self._raddatabase: str | None = options.get("raddatabase")
        self._surface_tilt: float = float(options.get("surface_tilt", 0.0))
        self._surface_azimuth: float = float(options.get("surface_azimuth", 0.0))
        self._pv_calculation: bool = bool(options.get("pv_calculation", False))
        self._peak_power_kw: float | None = options.get("peak_power_kw")
        self._system_loss: float = float(options.get("system_loss", 14.0))

    # ------------------------------------------------------------------ utils

    @property
    def _is_tilted(self) -> bool:
        return abs(self._surface_tilt) > 1e-9

    def _field_map(self) -> dict[str, Variable]:
        """PVGIS field -> canonical variable mapping."""
        mapping = dict(STATIC_FIELD_MAP)
        mapping["G(i)"] = Variable.POA_GLOBAL if self._is_tilted else Variable.GHI
        return mapping

    @staticmethod
    def _parse_timestamp(raw: str) -> datetime:
        """Convert ``YYYYMMDD:HHMM`` into a UTC instant floored to the hour."""
        try:
            date_part, time_part = raw.split(":")
            parsed = datetime.strptime(f"{date_part}{time_part[:2]}", "%Y%m%d%H")
        except (ValueError, IndexError) as exc:
            raise ValidationError(f"unreadable PVGIS timestamp: '{raw}'") from exc
        return parsed.replace(tzinfo=timezone.utc)

    def _parse(self, payload: dict[str, Any]) -> list[ObservationPoint]:
        outputs = payload.get("outputs", {})
        hourly = outputs.get("hourly")
        if not isinstance(hourly, list):
            raise ProviderError(
                self.provider.value, "'outputs.hourly' block missing from response"
            )

        dataset = (
            payload.get("inputs", {})
            .get("meteo_data", {})
            .get("radiation_db", self._raddatabase or "default")
        )
        field_map = self._field_map()
        points: list[ObservationPoint] = []

        for record in hourly:
            raw_time = record.get("time")
            if raw_time is None:
                continue
            ts = self._parse_timestamp(raw_time)

            for field, variable in field_map.items():
                if field not in record:
                    continue
                value = record[field]
                points.append(
                    ObservationPoint(
                        provider=self.provider,
                        variable=variable,
                        timestamp=ts,
                        value=None if value is None else float(value),
                        dataset=str(dataset),
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
        """Fetch hourly series for the years covering `[start, end]`.

        `variables` filters the output client-side: PVGIS returns a fixed
        set of fields and doesn't accept server-side selection.
        """
        if end < start:
            raise ValidationError("window end precedes start")

        params: dict[str, Any] = {
            "lat": latitude,
            "lon": longitude,
            "startyear": start.year,
            "endyear": end.year,
            "outputformat": "json",
            "components": 1,
            "angle": self._surface_tilt,
            "aspect": self._surface_azimuth,
            "loss": self._system_loss,
        }
        if self._raddatabase:
            params["raddatabase"] = self._raddatabase
        if self._pv_calculation:
            if self._peak_power_kw is None:
                raise ValidationError(
                    "pv_calculation enabled without peak_power_kw: PVGIS "
                    "requires the peak power to model production"
                )
            params["pvcalculation"] = 1
            params["peakpower"] = self._peak_power_kw
            params["pvtechchoice"] = self.config.options.get("pv_tech", "crystSi")
            params["mountingplace"] = self.config.options.get("mounting", "free")

        payload = await self.get_json(f"/{self._api_version}/seriescalc", params)
        points = self._parse(payload)

        # Fine trimming: the API can't bound below a full year.
        points = [p for p in points if start <= p.timestamp <= end]
        if variables:
            wanted = set(variables)
            points = [p for p in points if p.variable in wanted]

        logger.info(
            "PVGIS series fetched",
            extra={
                "context": {
                    "provider": self.provider.value,
                    "points": len(points),
                    "years": f"{start.year}-{end.year}",
                    "tilt": self._surface_tilt,
                }
            },
        )
        return points
