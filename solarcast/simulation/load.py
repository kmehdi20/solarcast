"""Synthetic residential load profile.

No open API publishes Moroccan residential load curves (unlike
irradiance, served by PVGIS/Open-Meteo) — this is already documented in
the project README. This module therefore provides a synthetic
generator, explicitly presented as such: it exists to exercise the
dispatch engine, not to represent measured consumption. For a real study,
replace it with a measured load curve or a published standardized
profile.

Profile shape
--------------
Two Gaussian peaks — morning (getting ready, leaving for work) and
evening (coming home, cooking, lighting) — layered on top of a constant
base load (standby, refrigerator). The profile's integral is calibrated
to match `daily_kwh` on average over the period covered by the index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_residential_load(
    index: pd.DatetimeIndex,
    daily_kwh: float = 10.0,
    morning_peak_hour: float = 8.0,
    evening_peak_hour: float = 20.0,
    peak_width_h: float = 1.5,
    base_load_fraction: float = 0.25,
) -> pd.Series:
    """Generate a synthetic hourly load profile.

    Parameters
    ----------
    index:
        Hourly DatetimeIndex. The profile follows the index's hour as-is
        (no timezone conversion is applied here).
    daily_kwh:
        Target daily consumption, kWh/day, averaged over `index`.
    morning_peak_hour, evening_peak_hour:
        Hours of the two consumption peaks (0-24).
    peak_width_h:
        Standard deviation of the Gaussian peaks, hours. Smaller = sharper peaks.
    base_load_fraction:
        Share of daily consumption covered by the constant base load; the
        rest (1 - base_load_fraction) forms the two peaks.

    Returns
    -------
    pd.Series
        Demanded power in kW, same index, always positive.
    """
    if daily_kwh <= 0:
        raise ValueError("daily_kwh must be positive.")
    if not 0 <= base_load_fraction <= 1:
        raise ValueError("base_load_fraction must be between 0 and 1.")
    if len(index) < 2:
        raise ValueError("index must contain at least two timestamps.")

    hours = index.hour + index.minute / 60.0
    dt_h = (index[1] - index[0]).total_seconds() / 3600.0

    def gaussian_bump(h: np.ndarray, center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((h - center) / width) ** 2)

    peaks_shape = gaussian_bump(hours.values, morning_peak_hour, peak_width_h) + gaussian_bump(
        hours.values, evening_peak_hour, peak_width_h
    )

    total_hours = len(index) * dt_h
    n_days = total_hours / 24.0
    base_kwh_total = daily_kwh * base_load_fraction * n_days
    peaks_kwh_total = daily_kwh * (1 - base_load_fraction) * n_days

    shape_integral = peaks_shape.sum() * dt_h
    scale = peaks_kwh_total / shape_integral if shape_integral > 0 else 0.0

    base_kw = base_kwh_total / total_hours if total_hours > 0 else 0.0
    load_kw = base_kw + peaks_shape * scale

    return pd.Series(load_kw, index=index, name="load_kw").clip(lower=0.0)
