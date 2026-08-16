"""Feature pipeline: solar geometry, clear sky, lags, calendar."""

from solarcast.features.pipeline import build_features, feature_names
from solarcast.features.solar import add_clear_sky, add_solar_geometry
from solarcast.features.temporal import add_calendar, add_lags, add_rolling

__all__ = [
    "build_features",
    "feature_names",
    "add_solar_geometry",
    "add_clear_sky",
    "add_lags",
    "add_rolling",
    "add_calendar",
]
