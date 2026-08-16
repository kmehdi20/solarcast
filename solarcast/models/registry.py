"""Registry of available models.

Each model is defined with default hyperparameters calibrated for H+1
hourly GHI forecasting. These aren't optimal values — a GridSearchCV or
an Optuna study would refine them — but they give good out-of-the-box
results on North African sites.

GradientBoosting is the primary model: it naturally handles
non-linearities (clear-sky saturation, sunrise/sunset edge effects),
feature interactions, and is robust to moderate outliers.
RandomForest serves as a non-parametric reference.
Ridge serves as a fast linear reference.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from solarcast.models.baselines import ClearSkyModel, PersistenceModel


def get_model(name: str):
    """Return an untrained instance of the requested model.

    Available names
    -----------------
    persistence, clear_sky, ridge, random_forest, gradient_boosting
    """
    models = {
        "persistence": PersistenceModel(horizon_h=1),
        "clear_sky": ClearSkyModel(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        ),
    }
    if name not in models:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(models)}"
        )
    return models[name]


def list_models() -> list[str]:
    return ["persistence", "clear_sky", "ridge", "random_forest", "gradient_boosting"]
