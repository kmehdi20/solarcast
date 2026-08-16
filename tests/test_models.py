"""Tests for the models layer."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from solarcast.models.baselines import ClearSkyModel, PersistenceModel
from solarcast.models.metrics import evaluate, mae, mbe, nrmse, rmse, skill_score
from solarcast.models.registry import get_model, list_models
from solarcast.models.validation import walk_forward_validate

LAT, LON, ALT = 34.261, -6.58, 20


def _make_data(days: int = 120):
    """Synthetic DataFrame and X/y for the tests."""
    from solarcast.features.pipeline import build_features

    index = pd.date_range(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=days * 24, freq="h"
    )
    ghi = np.array([
        max(0.0, 700 * math.sin(math.pi * (h % 24 - 6) / 12))
        if 6 <= (h % 24) <= 18 else 0.0
        for h in range(len(index))
    ])
    temp = 20.0 + 5.0 * np.sin(np.linspace(0, 2 * math.pi, len(index)))
    df = pd.DataFrame({"ghi": ghi, "temp_air": temp}, index=index)
    X, y = build_features(df, LAT, LON, ALT)
    return X, y


# ---------------------------------------------------------------- baselines


def test_persistence_predicts_lag():
    X, y = _make_data()
    model = PersistenceModel(horizon_h=1)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.allclose(pred, X["ghi_lag_1h"].values)


def test_clear_sky_uses_column():
    X, y = _make_data()
    model = ClearSkyModel()
    model.fit(X, y)
    pred = model.predict(X)
    assert np.allclose(pred, X["clear_sky_ghi"].values)


def test_clear_sky_missing_column_raises():
    X, y = _make_data()
    X2 = X.drop(columns=["clear_sky_ghi"])
    model = ClearSkyModel()
    with pytest.raises(ValueError):
        model.fit(X2, y)


# ----------------------------------------------------------------- metrics


def test_rmse_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert rmse(y, y, daytime_only=False) == pytest.approx(0.0)


def test_rmse_daytime_filters_night():
    y_true = np.array([0.0, 0.0, 500.0, 600.0])
    y_pred = np.array([100.0, 100.0, 500.0, 600.0])
    # Night included -> high RMSE; night excluded -> RMSE = 0
    assert rmse(y_true, y_pred, daytime_only=False) > 0
    assert rmse(y_true, y_pred, daytime_only=True) == pytest.approx(0.0)


def test_skill_score_perfect():
    y = np.ones(100) * 500.0
    ref = y + 50.0
    assert skill_score(y, y, ref, daytime_only=False) == pytest.approx(1.0)


def test_skill_score_same_as_ref():
    y = np.ones(100) * 500.0
    pred = y + 50.0
    assert skill_score(y, pred, pred, daytime_only=False) == pytest.approx(0.0)


def test_mbe_sign():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([150.0, 250.0, 350.0])
    assert mbe(y_true, y_pred, daytime_only=False) == pytest.approx(50.0)


def test_evaluate_returns_all_keys():
    y = np.ones(50) * 400.0
    pred = y + 30.0
    ref = y + 60.0
    result = evaluate(y, pred, y_persistence=ref)
    for key in ("rmse", "mae", "nrmse", "mbe", "skill_score"):
        assert key in result


# ---------------------------------------------------------------- registry


def test_list_models_not_empty():
    assert len(list_models()) >= 5


def test_get_model_returns_estimator():
    for name in list_models():
        model = get_model(name)
        assert hasattr(model, "fit") and hasattr(model, "predict")


def test_get_model_unknown_raises():
    with pytest.raises(ValueError):
        get_model("does_not_exist")


# ------------------------------------------------------------- validation


def test_walk_forward_runs():
    X, y = _make_data(days=365)
    from sklearn.linear_model import Ridge
    result = walk_forward_validate(
        Ridge(), X, y,
        n_folds=2,
        test_size=24 * 10,
        min_train_size=24 * 30,
        scale=True,
        model_name="ridge_test",
    )
    assert len(result.folds) == 2
    assert result.mean_metrics["rmse"] >= 0


def test_walk_forward_no_future_leak():
    """The test set must never precede the training set."""
    X, y = _make_data(days=365)
    from sklearn.linear_model import Ridge
    result = walk_forward_validate(
        Ridge(), X, y,
        n_folds=3,
        test_size=24 * 7,
        min_train_size=24 * 30,
    )
    for fold in result.folds:
        train_end = X.index[fold.train_size - 1]
        test_start = fold.test_index[0]
        assert test_start > train_end, f"Fold {fold.fold}: test precedes train"


def test_walk_forward_insufficient_data_raises():
    X, y = _make_data(days=30)
    from sklearn.linear_model import Ridge
    with pytest.raises(ValueError, match="Not enough"):
        walk_forward_validate(
            Ridge(), X, y,
            n_folds=5,
            test_size=24 * 20,
            min_train_size=24 * 120,
        )


def test_gradient_boosting_beats_persistence():
    """GradientBoosting doit avoir un skill score positif."""
    X, y = _make_data(days=365)
    result = walk_forward_validate(
        get_model("gradient_boosting"), X, y,
        n_folds=2,
        test_size=24 * 15,
        min_train_size=24 * 45,
        model_name="gradient_boosting",
    )
    ss = result.mean_metrics.get("skill_score", float("nan"))
    assert ss > 0, f"GradientBoosting skill score is {ss:.3f} — not beating persistence"


def test_persistence_skill_score_is_zero_against_itself():
    """Regression: scaling must never be applied to baselines.

    Historical bug: StandardScaler standardized ghi_lag_1h before
    PersistenceModel read it back as a raw prediction, corrupting the unit
    (W/m2 -> standard deviations) and artificially collapsing its score.
    Persistence compared to itself must always give a skill score of zero.
    """
    X, y = _make_data(days=365)
    result = walk_forward_validate(
        PersistenceModel(horizon_h=1), X, y,
        n_folds=2,
        test_size=24 * 15,
        min_train_size=24 * 45,
        model_name="persistence",
    )
    ss = result.mean_metrics.get("skill_score", float("nan"))
    assert ss == pytest.approx(0.0, abs=1e-6), (
        f"Persistence skill score against itself is {ss:.3f}, expected 0.0 — "
        "scaling is likely leaking into the baseline prediction."
    )
