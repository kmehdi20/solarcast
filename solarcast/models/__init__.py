"""Models layer: baselines, metrics, walk-forward validation, registry."""

from solarcast.models.baselines import ClearSkyModel, PersistenceModel
from solarcast.models.metrics import evaluate, mae, nrmse, rmse, skill_score
from solarcast.models.registry import get_model, list_models
from solarcast.models.validation import WalkForwardResult, walk_forward_validate

__all__ = [
    "PersistenceModel",
    "ClearSkyModel",
    "get_model",
    "list_models",
    "evaluate",
    "rmse",
    "mae",
    "nrmse",
    "skill_score",
    "walk_forward_validate",
    "WalkForwardResult",
]
