"""Battery dispatch simulation script.

    python -m solarcast.scripts.simulate --site Kenitra --pv-capacity-kwc 6 --battery-capacity-kwh 10
    python -m solarcast.scripts.simulate --site Kenitra --strategy peak_shaving --daily-kwh 12
"""

from __future__ import annotations

import argparse
import asyncio

from solarcast.core.config import load_settings
from solarcast.core.logging import configure_logging
from solarcast.simulation.battery import BatterySpec
from solarcast.simulation.dispatch import STRATEGIES
from solarcast.simulation.engine import simulate_dispatch, summarize
from solarcast.simulation.load import synthetic_residential_load
from solarcast.simulation.pv_model import ghi_to_pv_power
from solarcast.storage.repository import LocationRepository, ObservationRepository
from solarcast.storage.session import create_schema, init_engine, session_scope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solarcast-simulate")
    parser.add_argument("--site", required=True)
    parser.add_argument("--pv-capacity-kwc", type=float, default=6.0)
    parser.add_argument("--performance-ratio", type=float, default=0.80)
    parser.add_argument("--daily-kwh", type=float, default=10.0,
                        help="Target synthetic daily consumption (kWh/day).")
    parser.add_argument("--battery-capacity-kwh", type=float, default=10.0)
    parser.add_argument("--max-charge-kw", type=float, default=3.0)
    parser.add_argument("--max-discharge-kw", type=float, default=3.0)
    parser.add_argument("--min-soc", type=float, default=0.10)
    parser.add_argument("--max-soc", type=float, default=0.95)
    parser.add_argument("--strategy", default="self_consumption", choices=list(STRATEGIES))
    parser.add_argument("--config")
    parser.add_argument("--log-level", default="WARNING")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    init_engine(settings.database)
    await create_schema()

    async with session_scope() as session:
        location = await LocationRepository(session).get_by_name(args.site)
        if not location:
            print(f"Site '{args.site}' not in database. Run ingestion first.")
            return 1
        frame = await ObservationRepository(session).to_frame(location.id)

    if frame.empty or "ghi" not in frame.columns:
        print(f"No GHI data for '{args.site}'. Run ingestion first.")
        return 1

    pv_kw = ghi_to_pv_power(
        frame, capacity_kwc=args.pv_capacity_kwc, performance_ratio=args.performance_ratio
    )
    load_kw = synthetic_residential_load(frame.index, daily_kwh=args.daily_kwh)

    print(f"\nSite               : {args.site}")
    print(f"Period             : {frame.index.min()} → {frame.index.max()} ({len(frame)} hours)")
    print(f"PV capacity        : {args.pv_capacity_kwc} kWc (PR={args.performance_ratio})")
    print(f"Load               : synthetic, {args.daily_kwh} kWh/day target")
    print(f"Battery            : {args.battery_capacity_kwh} kWh, "
          f"{args.max_charge_kw}/{args.max_discharge_kw} kW charge/discharge")
    print(f"Strategy           : {args.strategy}")

    battery_spec = BatterySpec(
        capacity_kwh=args.battery_capacity_kwh,
        max_charge_kw=args.max_charge_kw,
        max_discharge_kw=args.max_discharge_kw,
        min_soc=args.min_soc,
        max_soc=args.max_soc,
    )

    results = simulate_dispatch(pv_kw, load_kw, battery_spec, strategy=args.strategy)
    stats = summarize(results, battery_spec)

    print(f"\n{'─'*46}")
    print("RESULTS")
    print(f"{'─'*46}")
    print(f"PV production      : {stats['pv_kwh']:>10,.0f} kWh")
    print(f"Load consumption   : {stats['load_kwh']:>10,.0f} kWh")
    print(f"Grid import        : {stats['grid_import_kwh']:>10,.0f} kWh")
    print(f"Grid export        : {stats['grid_export_kwh']:>10,.0f} kWh")
    print(f"Battery charged    : {stats['battery_charge_kwh']:>10,.0f} kWh")
    print(f"Battery discharged : {stats['battery_discharge_kwh']:>10,.0f} kWh")
    print(f"{'─'*46}")
    print(f"Self-consumption   : {stats['self_consumption_pct']:>9.1f} %")
    print(f"Autonomy           : {stats['autonomy_pct']:>9.1f} %")
    print(f"Equivalent cycles  : {stats['equivalent_full_cycles']:>9.1f}")

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
