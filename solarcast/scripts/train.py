"""Model training and evaluation script.

    python -m solarcast.scripts.train --site Kenitra
    python -m solarcast.scripts.train --site Kenitra --model gradient_boosting
    python -m solarcast.scripts.train --site Kenitra --all-models
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from solarcast.core.config import load_settings
from solarcast.core.logging import configure_logging, get_logger
from solarcast.features.pipeline import build_features
from solarcast.models.registry import get_model, list_models
from solarcast.models.validation import walk_forward_validate
from solarcast.storage.repository import LocationRepository, ObservationRepository
from solarcast.storage.session import init_engine, session_scope

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solarcast-train")
    parser.add_argument("--site", required=True)
    parser.add_argument("--model", default="gradient_boosting",
                        choices=list_models())
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--config")
    parser.add_argument("--log-level", default="INFO")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    from solarcast.storage.session import create_schema
    init_engine(settings.database)
    await create_schema()

    loc_cfg = settings.location(args.site)

    async with session_scope() as session:
        location = await LocationRepository(session).get_by_name(args.site)
        if not location:
            print(f"Site '{args.site}' not in database. Run ingestion first.")
            return 1
        frame = await ObservationRepository(session).to_frame(location.id)

    if frame.empty:
        print(f"No data for '{args.site}'. Run ingestion first.")
        return 1

    print(f"\nData loaded : {len(frame)} rows, {list(frame.columns)}")

    X, y = build_features(
        frame,
        latitude=loc_cfg.latitude,
        longitude=loc_cfg.longitude,
        altitude_m=loc_cfg.altitude_m or 0.0,
        horizon_h=args.horizon,
    )
    print(f"Features    : {X.shape[0]} samples × {X.shape[1]} features")

    models_to_run = list_models() if args.all_models else [args.model]

    results = []
    for model_name in models_to_run:
        print(f"\n{'─'*50}")
        print(f"Training : {model_name}")
        model = get_model(model_name)
        try:
            result = walk_forward_validate(
                model, X, y,
                n_folds=args.folds,
                test_size=24 * args.test_days,
                min_train_size=24 * args.min_train_days,
                model_name=model_name,
            )
            print(result.summary())
            results.append(result)
        except ValueError as exc:
            print(f"  Skipped: {exc}")

    if len(results) > 1:
        print(f"\n{'═'*50}")
        print("COMPARISON (mean over folds)")
        print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'nRMSE':>8} {'Skill':>8}")
        print("─" * 58)
        for r in sorted(results, key=lambda r: r.mean_metrics.get("rmse", 9999)):
            m = r.mean_metrics
            print(
                f"{r.model_name:<22} "
                f"{m.get('rmse', float('nan')):>8.1f} "
                f"{m.get('mae', float('nan')):>8.1f} "
                f"{m.get('nrmse', float('nan')):>8.3f} "
                f"{m.get('skill_score', float('nan')):>8.3f}"
            )
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
