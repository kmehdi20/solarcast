"""Domain types, shared across all layers.

These definitions live in `core` rather than `storage` so the ingestion
layer never has to know about the ORM: it produces `ObservationPoint`
objects, and the repository is responsible for translating them into rows.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone


class Provider(str, enum.Enum):
    """Supported data providers."""

    OPEN_METEO = "open-meteo"
    PVGIS = "pvgis"
    NASA_POWER = "nasa-power"
    SYNTHETIC = "synthetic"


class Variable(str, enum.Enum):
    """Internal canonical vocabulary.

    Each ingestion client translates its own field names into this
    vocabulary, which lets the feature pipeline stay completely agnostic
    to where the data came from.
    """

    GHI = "ghi"  # global horizontal irradiance, W/m2
    DNI = "dni"  # direct normal irradiance, W/m2
    DHI = "dhi"  # diffuse horizontal irradiance, W/m2
    POA_GLOBAL = "poa_global"  # plane-of-array irradiance, W/m2
    TEMP_AIR = "temp_air"  # air temperature at 2 m, degC
    WIND_SPEED = "wind_speed"  # wind speed, m/s
    RELATIVE_HUMIDITY = "relative_humidity"  # %
    CLOUD_COVER = "cloud_cover"  # %
    PRECIPITATION = "precipitation"  # mm
    PV_POWER = "pv_power"  # PV power, W
    LOAD = "load"  # demanded power, W


#: Canonical unit associated with each variable.
UNITS: dict[Variable, str] = {
    Variable.GHI: "W/m2",
    Variable.DNI: "W/m2",
    Variable.DHI: "W/m2",
    Variable.POA_GLOBAL: "W/m2",
    Variable.TEMP_AIR: "degC",
    Variable.WIND_SPEED: "m/s",
    Variable.RELATIVE_HUMIDITY: "percent",
    Variable.CLOUD_COVER: "percent",
    Variable.PRECIPITATION: "mm",
    Variable.PV_POWER: "W",
    Variable.LOAD: "W",
}


class RunStatus(str, enum.Enum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class ObservationPoint:
    """A single point produced by an ingestion client.

    `reference_time` equals `timestamp` for historical data, and the
    emission time of the run for a forecast.
    """

    provider: Provider
    variable: Variable
    timestamp: datetime
    value: float | None
    reference_time: datetime | None = None
    unit: str | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC expected)")
        object.__setattr__(
            self, "timestamp", self.timestamp.astimezone(timezone.utc)
        )
        ref = self.reference_time or self.timestamp
        if ref.tzinfo is None:
            raise ValueError("reference_time must be timezone-aware")
        object.__setattr__(self, "reference_time", ref.astimezone(timezone.utc))
        if self.unit is None:
            object.__setattr__(self, "unit", UNITS.get(self.variable, ""))

    @property
    def is_forecast(self) -> bool:
        assert self.reference_time is not None
        return self.reference_time < self.timestamp
