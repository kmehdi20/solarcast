"""Solar forecast evaluation metrics.

Standard ML metrics (RMSE, MAE) aren't enough here: a model that predicts
0 W/m2 all night gets an excellent MAE simply because night makes up
~60% of hours. We therefore work on daytime hours only, and complement
that with normalized metrics (nRMSE) and the skill score, which measures
relative improvement over persistence.

References
----------
Lauret et al. (2015) — "A benchmarking of machine learning techniques for
solar radiation forecasting in an insular context". Solar Energy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _daytime_mask(y_true: pd.Series | np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """Return a boolean mask, True during daytime hours.

    An hour is considered daytime if the actual GHI value exceeds
    `threshold` W/m2. This is more robust than filtering on the hour of
    day, which varies with season and site.
    """
    arr = np.asarray(y_true)
    return arr > threshold


def rmse(y_true: np.ndarray, y_pred: np.ndarray, daytime_only: bool = True) -> float:
    """Root Mean Square Error (W/m2)."""
    mask = _daytime_mask(y_true) if daytime_only else np.ones(len(y_true), dtype=bool)
    residuals = np.asarray(y_true)[mask] - np.asarray(y_pred)[mask]
    return float(np.sqrt(np.mean(residuals ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray, daytime_only: bool = True) -> float:
    """Mean Absolute Error (W/m2)."""
    mask = _daytime_mask(y_true) if daytime_only else np.ones(len(y_true), dtype=bool)
    residuals = np.abs(np.asarray(y_true)[mask] - np.asarray(y_pred)[mask])
    return float(np.mean(residuals))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, daytime_only: bool = True) -> float:
    """RMSE normalized by the mean of actual daytime values (unitless).

    Allows comparing sites with different average irradiance. An nRMSE of
    0.20 means the root-mean-square error represents 20% of average
    irradiance — a typical value for a good H+1 model.
    """
    mask = _daytime_mask(y_true) if daytime_only else np.ones(len(y_true), dtype=bool)
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]
    mean_obs = np.mean(yt)
    if mean_obs < 1e-6:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)) / mean_obs)


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_ref: np.ndarray,
    daytime_only: bool = True,
) -> float:
    """Skill score relative to a reference model (persistence).

    SS = 1 - RMSE_model / RMSE_ref

    SS = 1   -> perfect forecast
    SS = 0   -> as good as the reference
    SS < 0   -> worse than the reference (useless in production)
    """
    rmse_model = rmse(y_true, y_pred, daytime_only)
    rmse_ref = rmse(y_true, y_ref, daytime_only)
    if rmse_ref < 1e-6:
        return float("nan")
    return float(1.0 - rmse_model / rmse_ref)


def mbe(y_true: np.ndarray, y_pred: np.ndarray, daytime_only: bool = True) -> float:
    """Mean Bias Error (W/m2) — signed average bias.

    Positive = the model overestimates, negative = it underestimates. A
    systematic bias points to a problem in the features or the model.
    """
    mask = _daytime_mask(y_true) if daytime_only else np.ones(len(y_true), dtype=bool)
    return float(np.mean(np.asarray(y_pred)[mask] - np.asarray(y_true)[mask]))


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_persistence: np.ndarray | None = None,
    label: str = "model",
) -> dict[str, float]:
    """Compute every metric and return them in a dictionary.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predictions from the model being evaluated.
    y_persistence:
        Persistence predictions (for the skill score). If None, the skill
        score isn't computed.
    label:
        Model name, included in the result's ``model`` key.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    results: dict[str, float] = {
        "model": label,
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "nrmse": nrmse(yt, yp),
        "mbe": mbe(yt, yp),
    }
    if y_persistence is not None:
        results["skill_score"] = skill_score(yt, yp, np.asarray(y_persistence))

    return results
