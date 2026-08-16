"""Simulation layer: PV model, synthetic load, battery, dispatch, engine."""

from solarcast.simulation.battery import Battery, BatterySpec
from solarcast.simulation.dispatch import get_strategy, peak_shaving, self_consumption
from solarcast.simulation.engine import simulate_dispatch, summarize
from solarcast.simulation.load import synthetic_residential_load
from solarcast.simulation.pv_model import ghi_to_pv_power

__all__ = [
    "Battery",
    "BatterySpec",
    "self_consumption",
    "peak_shaving",
    "get_strategy",
    "simulate_dispatch",
    "summarize",
    "synthetic_residential_load",
    "ghi_to_pv_power",
]
