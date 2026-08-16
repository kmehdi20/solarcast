"""Tests du pipeline de features."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from solarcast.features.pipeline import build_features
from solarcast.features.solar import add_clear_sky, add_solar_geometry
from solarcast.features.temporal import add_calendar, add_lags, add_rolling

# Meknes
LAT, LON, ALT = 33.8935, -5.5473, 550


def _hourly_frame(days: int = 10, with_ghi: bool = True) -> pd.DataFrame:
    """Synthetic DataFrame with an hourly UTC index."""
    index = pd.date_range(
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        periods=days * 24,
        freq="h",
    )
    data: dict[str, list[float]] = {}
    if with_ghi:
        data["ghi"] = [
            max(0.0, 800 * math.sin(math.pi * (h % 24 - 6) / 12))
            if 6 <= (h % 24) <= 18 else 0.0
            for h in range(len(index))
        ]
    data["temp_air"] = [20.0 + 5.0 * math.sin(math.pi * (h % 24) / 24) for h in range(len(index))]
    return pd.DataFrame(data, index=index)


# ------------------------------------------------------------------ solar


def test_solar_geometry_adds_columns():
    df = _hourly_frame()
    result = add_solar_geometry(df, LAT, LON, ALT)
    for col in ("solar_zenith", "solar_azimuth", "solar_elevation"):
        assert col in result.columns, f"missing {col}"


def test_solar_zenith_range():
    df = _hourly_frame()
    result = add_solar_geometry(df, LAT, LON, ALT)
    assert result["solar_zenith"].between(0, 180).all()


def test_solar_elevation_complement():
    df = _hourly_frame()
    result = add_solar_geometry(df, LAT, LON, ALT)
    diff = (result["solar_zenith"] + result["solar_elevation"] - 90).abs()
    assert (diff < 1e-6).all()


def test_clear_sky_non_negative():
    df = _hourly_frame()
    result = add_clear_sky(df, LAT, LON, ALT)
    assert (result["clear_sky_ghi"] >= 0).all()


def test_clear_sky_index_capped():
    df = _hourly_frame()
    result = add_clear_sky(df, LAT, LON, ALT)
    assert (result["clear_sky_index"].dropna() <= 1.5).all()


def test_clear_sky_zero_at_night():
    df = _hourly_frame()
    result = add_solar_geometry(df, LAT, LON, ALT)
    result = add_clear_sky(result, LAT, LON, ALT)
    night = result[result["solar_elevation"] < 0]
    assert (night["clear_sky_ghi"] < 1.0).all()


# ---------------------------------------------------------------- temporal


def test_lags_create_columns():
    df = _hourly_frame()
    result = add_lags(df, ["ghi"], lags=[1, 3])
    assert "ghi_lag_1h" in result.columns
    assert "ghi_lag_3h" in result.columns


def test_lag_values_correct():
    df = _hourly_frame()
    result = add_lags(df, ["ghi"], lags=[1])
    # The value at t is the GHI at t-1.
    assert result["ghi_lag_1h"].iloc[5] == pytest.approx(result["ghi"].iloc[4])


def test_rolling_mean_columns():
    df = _hourly_frame()
    result = add_rolling(df, ["ghi"], windows=[3])
    assert "ghi_roll_3h" in result.columns


def test_rolling_mean_values():
    df = _hourly_frame()
    result = add_rolling(df, ["ghi"], windows=[3], stats=["mean"])
    expected = df["ghi"].rolling(3, min_periods=1).mean().iloc[10]
    assert result["ghi_roll_3h"].iloc[10] == pytest.approx(expected)


def test_calendar_columns():
    df = _hourly_frame()
    result = add_calendar(df)
    for col in ("hour", "month", "dayofyear", "sin_hour", "cos_hour",
                "sin_dayofyear", "cos_dayofyear", "is_weekend"):
        assert col in result.columns


def test_cyclic_encoding_unit_circle():
    df = _hourly_frame()
    result = add_calendar(df)
    norm = (result["sin_hour"] ** 2 + result["cos_hour"] ** 2)
    assert (norm - 1.0).abs().max() < 1e-10


def test_is_weekend_binary():
    df = _hourly_frame()
    result = add_calendar(df)
    assert set(result["is_weekend"].unique()).issubset({0, 1})


# --------------------------------------------------------------- pipeline


def test_build_features_returns_correct_shapes():
    df = _hourly_frame(days=30)
    X, y = build_features(df, LAT, LON, ALT)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 10


def test_no_nan_in_output():
    df = _hourly_frame(days=30)
    X, y = build_features(df, LAT, LON, ALT)
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_ghi_excluded_from_X():
    df = _hourly_frame(days=30)
    X, _ = build_features(df, LAT, LON, ALT)
    assert "ghi" not in X.columns


def test_target_is_future():
    """y[t] doit valoir ghi[t+1]."""
    df = _hourly_frame(days=30)
    X, y = build_features(df, LAT, LON, ALT, horizon_h=1)
    # y's first index corresponds to t; ghi_lag_1h in X is ghi[t].
    # Check that y[t] = ghi[t+1] by comparing against the reconstructed lag.
    ghi_original = df["ghi"]
    for ts in y.index[:5]:
        next_ts = ts + pd.Timedelta(hours=1)
        if next_ts in ghi_original.index:
            assert y[ts] == pytest.approx(ghi_original[next_ts])


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        build_features(pd.DataFrame(), LAT, LON, ALT)


def test_missing_target_raises():
    df = _hourly_frame(with_ghi=False)
    with pytest.raises(ValueError, match="Target"):
        build_features(df, LAT, LON, ALT, target_col="ghi")
