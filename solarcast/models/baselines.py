"""Baseline models.

Every forecasting model must be compared against these two references. If
an ML model doesn't beat persistence, it isn't adding value.

Persistence
-----------
The simplest possible prediction: the next value will equal the current
one. Works well over short horizons (< 30 min) but degrades quickly.
This is the floor to beat.

Clear sky
---------
Uses the pvlib clear-sky value directly as the prediction. Amounts to
assuming the sky will always be clear — good in summer in arid regions,
poor in cloudy weather. Serves as a physical reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class PersistenceModel(BaseEstimator, RegressorMixin):
    """Predicts y[t+h] = y[t] (last known value).

    Conforms to the scikit-learn API: usable in pipelines and with
    cross_val_score, even though fit() does nothing.

    Parameters
    ----------
    horizon_h:
        Forecast horizon in hours. Uses the matching lag in X if
        available, otherwise falls back to ghi_lag_1h.
    """

    def __init__(self, horizon_h: int = 1) -> None:
        self.horizon_h = horizon_h

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceModel":
        # Nothing to learn: just remember which column name to use.
        lag_col = f"ghi_lag_{self.horizon_h}h"
        fallback = "ghi_lag_1h"
        cols = list(X.columns)
        if lag_col in cols:
            self._col = lag_col
        elif fallback in cols:
            self._col = fallback
        else:
            raise ValueError(
                f"Neither '{lag_col}' nor '{fallback}' found in X. "
                f"Available columns: {cols[:10]}..."
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self._col].values


class ClearSkyModel(BaseEstimator, RegressorMixin):
    """Predicts y[t+h] = clear_sky_ghi[t+h] (assumes an always-clear sky).

    In practice, clear_sky_ghi at step t+h isn't available at forecast
    time, but since it's computed from solar geometry (deterministic), it
    can be calculated for any future instant. Here we simply use the
    value already present in X, which corresponds to the predicted
    instant — solar geometry is computed for the target hour by the
    feature pipeline.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClearSkyModel":
        if "clear_sky_ghi" not in X.columns:
            raise ValueError("'clear_sky_ghi' missing from X — check the feature pipeline.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["clear_sky_ghi"].values
