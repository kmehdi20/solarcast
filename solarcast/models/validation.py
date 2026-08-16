"""Walk-forward (out-of-sample) validation.

Standard cross-validation (KFold) is **forbidden** on time series: it
leaks the future into training — a test fold can fall before a training
fold — which produces artificially optimistic metrics that don't hold up
in production.

Walk-forward validation respects the order of time:

    Fold 1: train [t0 ... t1]  ->  test [t1 ... t1+h]
    Fold 2: train [t0 ... t2]  ->  test [t2 ... t2+h]
    Fold n: train [t0 ... tn]  ->  test [tn ... tn+h]

At each fold, the model is retrained on all available history. The test
set never sees the future. This is the only credible validation for a
forecasting system intended for deployment.

Reference
---------
Hyndman & Athanasopoulos (2021) — Forecasting: Principles and Practice, §5.10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from solarcast.core.logging import get_logger
from solarcast.models.baselines import ClearSkyModel, PersistenceModel
from solarcast.models.metrics import evaluate

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """Result of a single validation fold."""

    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, float]
    y_test: np.ndarray
    y_pred: np.ndarray
    y_persistence: np.ndarray
    test_index: pd.DatetimeIndex


@dataclass
class WalkForwardResult:
    """Aggregated result of the walk-forward validation."""

    model_name: str
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Average of the metrics across all folds."""
        keys = [k for k in self.folds[0].metrics if k != "model"]
        return {
            k: float(np.mean([f.metrics[k] for f in self.folds if not np.isnan(f.metrics.get(k, float("nan")))]))
            for k in keys
        }

    @property
    def all_predictions(self) -> pd.Series:
        """All predictions concatenated in chronological order."""
        parts = [
            pd.Series(f.y_pred, index=f.test_index, name="y_pred")
            for f in self.folds
        ]
        return pd.concat(parts).sort_index()

    @property
    def all_observations(self) -> pd.Series:
        parts = [
            pd.Series(f.y_test, index=f.test_index, name="y_true")
            for f in self.folds
        ]
        return pd.concat(parts).sort_index()

    def summary(self) -> str:
        m = self.mean_metrics
        lines = [
            f"Model         : {self.model_name}",
            f"Folds         : {len(self.folds)}",
            f"RMSE          : {m.get('rmse', float('nan')):.1f} W/m2",
            f"MAE           : {m.get('mae', float('nan')):.1f} W/m2",
            f"nRMSE         : {m.get('nrmse', float('nan')):.3f}",
            f"MBE           : {m.get('mbe', float('nan')):.1f} W/m2",
        ]
        if "skill_score" in m:
            lines.append(f"Skill score   : {m['skill_score']:.3f}")
        return "\n".join(lines)


def walk_forward_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    test_size: int = 24 * 30,
    min_train_size: int = 24 * 60,
    scale: bool = True,
    model_name: str | None = None,
) -> WalkForwardResult:
    """Train and evaluate a model via walk-forward validation.

    Parameters
    ----------
    model:
        scikit-learn estimator (untrained). Cloned at every fold.
    X:
        Feature matrix (sorted DatetimeIndex).
    y:
        Target series (same index as X).
    n_folds:
        Number of test folds.
    test_size:
        Number of time steps per test fold (default: 30 hourly days).
    min_train_size:
        Minimum training set size (default: 60 days).
    scale:
        If True, applies StandardScaler to the features.
    model_name:
        Name shown in the results.

    Returns
    -------
    WalkForwardResult
    """
    name = model_name or type(model).__name__
    result = WalkForwardResult(model_name=name)

    # Baselines read raw physical columns (W/m2) directly as their
    # prediction: standardizing them would break the unit. Scaling is
    # therefore disabled automatically for them, regardless of `scale`.
    is_baseline = isinstance(model, (PersistenceModel, ClearSkyModel))
    effective_scale = scale and not is_baseline

    n = len(X)
    required = min_train_size + n_folds * test_size
    if n < required:
        raise ValueError(
            f"Not enough data: {n} rows available, "
            f"{required} required (min_train={min_train_size}, "
            f"n_folds={n_folds}, test_size={test_size})."
        )

    # Split points: folds are spread evenly across the portion of the
    # series that exceeds min_train_size.
    available = n - min_train_size
    fold_spacing = available // (n_folds + 1)

    for fold_idx in range(n_folds):
        test_end = n - (n_folds - fold_idx - 1) * fold_spacing
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train_size:
            logger.warning(
                "fold skipped: training window too short",
                extra={"context": {"fold": fold_idx + 1, "train_size": train_end}},
            )
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        # Scaling
        if effective_scale:
            scaler = StandardScaler()
            X_train_fit = pd.DataFrame(
                scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
            )
            X_test_fit = pd.DataFrame(
                scaler.transform(X_test), columns=X_test.columns, index=X_test.index
            )
        else:
            X_train_fit, X_test_fit = X_train, X_test

        # Training
        m = clone(model)
        m.fit(X_train_fit, y_train)
        y_pred = m.predict(X_test_fit)

        # Persistence (reference)
        persistence_col = "ghi_lag_1h"
        if persistence_col in X_test.columns:
            y_persistence = X_test[persistence_col].values
        else:
            y_persistence = None

        metrics = evaluate(
            y_test.values,
            y_pred,
            y_persistence=y_persistence,
            label=name,
        )

        fold = FoldResult(
            fold=fold_idx + 1,
            train_size=train_end,
            test_size=len(y_test),
            metrics=metrics,
            y_test=y_test.values,
            y_pred=y_pred,
            y_persistence=y_persistence if y_persistence is not None else np.array([]),
            test_index=X_test.index,
        )
        result.folds.append(fold)

        logger.info(
            "fold finished",
            extra={
                "context": {
                    "fold": fold_idx + 1,
                    "train": train_end,
                    "test": len(y_test),
                    "rmse": round(metrics["rmse"], 1),
                    "skill": round(metrics.get("skill_score", float("nan")), 3),
                }
            },
        )

    return result
