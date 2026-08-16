"""Feature pipeline — single entry point.

Usage
-----
>>> from solarcast.features.pipeline import build_features
>>> X, y = build_features(df, latitude=34.26, longitude=-6.58)

`build_features` takes the pivoted DataFrame produced by
`ObservationRepository.to_frame()` and returns:

* ``X`` — feature matrix, no NaN, ready for scikit-learn
* ``y`` — target series (next-hour GHI by default)

The target is defined as ``ghi_lag_{-1}`` — i.e. the GHI one hour into the
future relative to each row. It's built with `shift(-1)`, then the last
row (where it's NaN) is dropped.

Step order
-----------
1. Solar geometry (pvlib)
2. Clear sky + clear-sky index
3. Time lags on GHI, temperature, clear_sky_index
4. Rolling aggregates
5. Calendar variables
6. Drop rows with NaN (unavoidable in the first few hours)
7. Split X / y
"""

from __future__ import annotations

import pandas as pd

from solarcast.core.logging import get_logger
from solarcast.features.solar import add_clear_sky, add_solar_geometry
from solarcast.features.temporal import add_calendar, add_lags, add_rolling

logger = get_logger(__name__)

#: Columns never used as features (identifiers, raw targets).
_DROP_FROM_X = {"ghi", "pv_power"}

#: Columns to compute lags on.
_LAG_COLS = ["ghi", "temp_air", "clear_sky_index", "cloud_cover"]

#: Columns to compute rolling aggregates on.
_ROLL_COLS = ["ghi", "clear_sky_index"]


def build_features(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
    tz: str = "UTC",
    target_col: str = "ghi",
    horizon_h: int = 1,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix and the target series.

    Parameters
    ----------
    df:
        Pivoted DataFrame (UTC index, columns = canonical variables).
    latitude, longitude, altitude_m:
        Site coordinates — required by pvlib.
    tz:
        Timezone for pvlib (doesn't affect storage, always UTC).
    target_col:
        Variable to predict (``ghi`` by default).
    horizon_h:
        Forecast horizon in hours (1 = next hour).
    lags:
        Lags to compute. If None, uses the temporal module's defaults.
    windows:
        Rolling windows. If None, uses the temporal module's defaults.

    Returns
    -------
    X : pd.DataFrame
        Features with no NaN, index aligned with y.
    y : pd.Series
        Future target values, same index as X.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty — nothing to process.")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' is missing from the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(
        "building features",
        extra={
            "context": {
                "rows": len(df),
                "columns": list(df.columns),
                "target": target_col,
                "horizon_h": horizon_h,
            }
        },
    )

    frame = df.copy()

    # 1. Solar geometry
    frame = add_solar_geometry(frame, latitude, longitude, altitude_m, tz)

    # 2. Clear sky
    frame = add_clear_sky(frame, latitude, longitude, altitude_m, tz)

    # 3. Lags
    lag_cols = [c for c in _LAG_COLS if c in frame.columns]
    frame = add_lags(frame, lag_cols, lags)

    # 4. Rolling aggregates
    roll_cols = [c for c in _ROLL_COLS if c in frame.columns]
    frame = add_rolling(frame, roll_cols, windows, stats=["mean", "std"])

    # 5. Calendar variables
    frame = add_calendar(frame)

    # 6. Target: future value
    frame["__target__"] = frame[target_col].shift(-horizon_h)

    # 7. Drop NaN
    before = len(frame)
    frame = frame.dropna()
    dropped = before - len(frame)
    if dropped:
        logger.info(
            "rows dropped (NaN)",
            extra={"context": {"dropped": dropped, "remaining": len(frame)}},
        )

    if frame.empty:
        raise ValueError(
            "No usable rows left after dropping NaN. "
            "Check the data's time coverage."
        )

    # 8. Split X / y
    drop_cols = (_DROP_FROM_X | {"__target__"}) & set(frame.columns)
    X = frame.drop(columns=list(drop_cols))
    y = frame["__target__"].rename(f"{target_col}_h{horizon_h}")

    logger.info(
        "features ready",
        extra={
            "context": {
                "X_shape": f"{X.shape[0]}x{X.shape[1]}",
                "features": list(X.columns),
            }
        },
    )
    return X, y


def feature_names(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
    tz: str = "UTC",
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> list[str]:
    """Return the list of features without building the full matrix.

    Useful for inspecting the pipeline before training a model.
    """
    # Use one week of data to keep this fast.
    sample = df.head(24 * 7).copy()
    X, _ = build_features(
        sample, latitude, longitude, altitude_m, tz,
        lags=lags, windows=windows
    )
    return list(X.columns)
