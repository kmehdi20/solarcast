"""Temporal features: lags, rolling aggregates, calendar variables.

These features are the backbone of any time-series-based PV production
forecasting model: they give the model explicit access to recent history
and cyclical patterns.

Naming convention
-------------------
``{variable}_lag_{n}h``    — value of `variable` n hours ago
``{variable}_roll_{n}h``   — rolling mean over the last n hours
``{variable}_std_{n}h``    — rolling standard deviation (measures variability)
``hour``, ``month``, ``dayofyear``, ``weekday`` — calendar variables
``sin_hour``, ``cos_hour`` — cyclic hour encoding (avoids the 23->0 jump)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lags(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add time-shifted columns.

    Parameters
    ----------
    df:
        DataFrame with a sorted, hourly DatetimeIndex.
    columns:
        Names of the columns to shift.
    lags:
        List of shifts in hours. Defaults to [1, 2, 3, 6, 12, 24].
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]

    existing = [c for c in columns if c in df.columns]
    for col in existing:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] | None = None,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling statistics.

    Parameters
    ----------
    windows:
        Rolling windows in hours. Defaults to [3, 6, 24].
    stats:
        Statistics to compute: ``mean``, ``std``, ``max``. Defaults to mean.
    """
    if windows is None:
        windows = [3, 6, 24]
    if stats is None:
        stats = ["mean"]

    existing = [c for c in columns if c in df.columns]
    for col in existing:
        for window in windows:
            series = df[col]
            if "mean" in stats:
                df[f"{col}_roll_{window}h"] = (
                    series.rolling(window, min_periods=1).mean()
                )
            if "std" in stats:
                df[f"{col}_std_{window}h"] = (
                    series.rolling(window, min_periods=2).std()
                )
            if "max" in stats:
                df[f"{col}_max_{window}h"] = (
                    series.rolling(window, min_periods=1).max()
                )
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar variables and their cyclic encoding.

    Columns added
    --------------
    hour : int
        UTC hour (0-23).
    month : int
        Month (1-12).
    dayofyear : int
        Day of year (1-366).
    weekday : int
        Day of week (0=Monday, 6=Sunday).
    sin_hour, cos_hour : float
        Cyclic hour encoding — prevents the model from treating 23:00 and
        00:00 as distant values when they're actually adjacent.
    sin_dayofyear, cos_dayofyear : float
        Cyclic day-of-year encoding — captures seasonality.
    is_weekend : int
        1 if Saturday or Sunday, 0 otherwise.
    """
    idx = df.index
    df["hour"] = idx.hour
    df["month"] = idx.month
    df["dayofyear"] = idx.dayofyear
    df["weekday"] = idx.weekday

    # Cyclic encoding — sine/cosine turns the hour into a continuous circle.
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df
