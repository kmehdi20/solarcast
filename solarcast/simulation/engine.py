"""Battery dispatch simulation engine.

Runs a dispatch strategy step by step over a PV series and a load series,
producing a complete trace (SOC, grid flows) and standard performance
indicators (self-consumption rate, autonomy rate).

Energy balance
---------------
At each timestep:

    net = pv_kw - load_kw
    accepted_kw  = power actually charged (0 if discharging or idle)
    delivered_kw = power actually discharged (0 if charging or idle)
    diff = accepted_kw - delivered_kw - net

    grid_import_kw = max(0, diff)
    grid_export_kw = max(0, -diff)

This formulation is deliberately symmetric and can be checked
algebraically: it guarantees, at every step, the conservation identity

    pv + grid_import + delivered = load + grid_export + accepted

regardless of the sign of `requested_kw` returned by the strategy —
including for a future strategy that would charge from the grid or
discharge to it (arbitrage), a case none of the strategies in
`dispatch.py` use today, but which the engine already handles correctly
without modification. A dedicated test (`test_simulation.py`) checks this
identity step by step on random data.
"""

from __future__ import annotations

import pandas as pd

from solarcast.core.logging import get_logger
from solarcast.simulation.battery import Battery, BatterySpec
from solarcast.simulation.dispatch import DispatchStrategy, get_strategy

logger = get_logger(__name__)


def simulate_dispatch(
    pv_kw: pd.Series,
    load_kw: pd.Series,
    battery_spec: BatterySpec,
    strategy: str | DispatchStrategy = "self_consumption",
    dt_hours: float = 1.0,
    **strategy_kwargs,
) -> pd.DataFrame:
    """Simulate the battery dispatch over the whole period.

    Parameters
    ----------
    pv_kw, load_kw:
        Power series (kW), same index, aligned.
    battery_spec:
        Battery characteristics.
    strategy:
        Name of a registered strategy (`dispatch.STRATEGIES`) or a custom
        function following the `DispatchStrategy` protocol.
    dt_hours:
        Simulation timestep, in hours (1.0 for hourly data).
    strategy_kwargs:
        Extra arguments passed to the strategy (e.g. thresholds for `peak_shaving`).

    Returns
    -------
    pd.DataFrame
        Indexed like `pv_kw`, columns: pv_kw, load_kw, battery_power_kw,
        soc_kwh, soc_pct, grid_import_kw, grid_export_kw.
    """
    if not pv_kw.index.equals(load_kw.index):
        raise ValueError("pv_kw and load_kw must share the same index.")
    if pv_kw.empty:
        raise ValueError("Empty series — nothing to simulate.")

    strat = get_strategy(strategy) if isinstance(strategy, str) else strategy
    battery = Battery(battery_spec)

    n = len(pv_kw)
    battery_power = [0.0] * n
    soc_kwh = [0.0] * n
    grid_import = [0.0] * n
    grid_export = [0.0] * n

    for i, (pv, load) in enumerate(zip(pv_kw.values, load_kw.values)):
        net = float(pv) - float(load)
        requested_kw = strat(float(pv), float(load), battery.soc_fraction, **strategy_kwargs)

        accepted_kw = 0.0
        delivered_kw = 0.0
        if requested_kw > 0:
            accepted_kw = battery.charge(requested_kw, dt_hours)
        elif requested_kw < 0:
            delivered_kw = battery.discharge(-requested_kw, dt_hours)

        diff = accepted_kw - delivered_kw - net
        grid_import[i] = max(0.0, diff)
        grid_export[i] = max(0.0, -diff)
        battery_power[i] = accepted_kw - delivered_kw
        soc_kwh[i] = battery.soc_kwh

    result = pd.DataFrame(
        {
            "pv_kw": pv_kw.values,
            "load_kw": load_kw.values,
            "battery_power_kw": battery_power,
            "soc_kwh": soc_kwh,
            "soc_pct": [s / battery_spec.capacity_kwh * 100.0 for s in soc_kwh],
            "grid_import_kw": grid_import,
            "grid_export_kw": grid_export,
        },
        index=pv_kw.index,
    )

    strategy_name = strategy if isinstance(strategy, str) else getattr(strategy, "__name__", "custom")
    logger.info(
        "dispatch simulation finished",
        extra={"context": {"rows": n, "strategy": strategy_name}},
    )
    return result


def summarize(
    results: pd.DataFrame,
    battery_spec: BatterySpec,
    dt_hours: float = 1.0,
) -> dict[str, float]:
    """Compute the standard performance indicators from the results.

    Indicators
    -----------
    self_consumption_pct:
        Share of PV production consumed on-site, either directly or via
        the battery: (pv_kwh - export_kwh) / pv_kwh.
    autonomy_pct:
        Share of consumption covered without drawing from the grid:
        (load_kwh - import_kwh) / load_kwh.
    equivalent_full_cycles:
        Total energy discharged / capacity — a standard indicator of
        battery stress and aging over the simulated period.
    """
    pv_kwh = float((results["pv_kw"] * dt_hours).sum())
    load_kwh = float((results["load_kw"] * dt_hours).sum())
    import_kwh = float((results["grid_import_kw"] * dt_hours).sum())
    export_kwh = float((results["grid_export_kw"] * dt_hours).sum())

    charge_kwh = float((results["battery_power_kw"].clip(lower=0) * dt_hours).sum())
    discharge_kwh = float((-results["battery_power_kw"].clip(upper=0) * dt_hours).sum())

    self_consumption_pct = (
        100.0 * (pv_kwh - export_kwh) / pv_kwh if pv_kwh > 1e-9 else float("nan")
    )
    autonomy_pct = (
        100.0 * (load_kwh - import_kwh) / load_kwh if load_kwh > 1e-9 else float("nan")
    )
    equivalent_cycles = (
        discharge_kwh / battery_spec.capacity_kwh if battery_spec.capacity_kwh > 0 else 0.0
    )

    return {
        "pv_kwh": round(pv_kwh, 2),
        "load_kwh": round(load_kwh, 2),
        "grid_import_kwh": round(import_kwh, 2),
        "grid_export_kwh": round(export_kwh, 2),
        "battery_charge_kwh": round(charge_kwh, 2),
        "battery_discharge_kwh": round(discharge_kwh, 2),
        "self_consumption_pct": round(self_consumption_pct, 1),
        "autonomy_pct": round(autonomy_pct, 1),
        "equivalent_full_cycles": round(equivalent_cycles, 2),
    }
