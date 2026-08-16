"""Command-line entry point for ingestion.

Examples
--------
Create the schema then load two years of Open-Meteo archive for Kenitra::

    python -m solarcast.scripts.ingest --provider open-meteo \\
        --site Kenitra --start 2023-01-01 --end 2024-12-31

Load every active provider for every declared site::

    python -m solarcast.scripts.ingest --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone

from solarcast.core.config import Settings, load_settings
from solarcast.core.exceptions import SolarCastError
from solarcast.core.logging import configure_logging, get_logger
from solarcast.core.types import RunStatus, Variable
from solarcast.ingestion.service import ingest_all, ingest_historical
from solarcast.storage.session import create_schema, dispose_engine, init_engine

logger = get_logger(__name__)


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date '{value}' — expected format YYYY-MM-DD"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solarcast-ingest",
        description="Acquisition of weather and solar time series.",
    )
    parser.add_argument("--config", help="path to the YAML configuration file")
    parser.add_argument("--start", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="provider to query (repeatable; default: all enabled ones)",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="site to process (repeatable; default: all declared sites)",
    )
    parser.add_argument(
        "--variable",
        action="append",
        dest="variables",
        help="canonical variable to keep (repeatable)",
    )
    parser.add_argument(
        "--log-level", default=None, help="override the configured log level"
    )
    return parser


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    init_engine(settings.database)
    await create_schema()

    variables = (
        [Variable(v) for v in args.variables] if args.variables else None
    )

    try:
        if args.sites:
            locations = [settings.location(name) for name in args.sites]
            providers = args.providers or [
                name for name, cfg in settings.providers.items() if cfg.enabled
            ]
            results = await asyncio.gather(
                *[
                    ingest_historical(
                        provider_name=provider,
                        provider_config=settings.provider(provider),
                        location=location,
                        start=args.start,
                        end=args.end,
                        variables=variables,
                    )
                    for provider in providers
                    for location in locations
                ]
            )
            results = list(results)
        else:
            results = await ingest_all(
                settings,
                start=args.start,
                end=args.end,
                providers=args.providers,
                variables=variables,
            )
    finally:
        await dispose_engine()

    print("\nIngestion results")
    print("-" * 62)
    for result in results:
        flag = "OK " if result.ok else "ERR"
        detail = f"{result.points:>7} points" if result.ok else (result.error or "")
        print(f"[{flag}] {result.provider:<12} {result.location:<14} {detail}")
    print("-" * 62)

    failed = [r for r in results if r.status is RunStatus.FAILED]
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = load_settings(args.config)
    except SolarCastError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    configure_logging(
        level=args.log_level or settings.logging.level,
        json_format=settings.logging.json_format,
    )

    if args.end < args.start:
        print("End date precedes start date.", file=sys.stderr)
        return 2

    try:
        return asyncio.run(_run(args, settings))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
