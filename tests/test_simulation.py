"""Tests de la couche simulation : PV, charge, batterie, dispatch, moteur."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from solarcast.simulation.battery import Battery, BatterySpec
from solarcast.simulation.dispatch import peak_shaving, self_consumption
from solarcast.simulation.engine import simulate_dispatch, summarize
from solarcast.simulation.load import synthetic_residential_load
from solarcast.simulation.pv_model import ghi_to_pv_power


def _hourly_index(days: int = 5) -> pd.DatetimeIndex:
    return pd.date_range(
        datetime(2024, 6, 1, tzinfo=timezone.utc), periods=days * 24, freq="h"
    )


# --------------------------------------------------------------- pv_model


def test_pv_power_zero_at_zero_ghi():
    idx = _hourly_index(1)
    df = pd.DataFrame({"ghi": [0.0] * 24}, index=idx)
    power = ghi_to_pv_power(df, capacity_kwc=5.0)
    assert (power == 0).all()


def test_pv_power_never_negative():
    idx = _hourly_index(2)
    df = pd.DataFrame({"ghi": np.linspace(-10, 900, len(idx))}, index=idx)
    power = ghi_to_pv_power(df, capacity_kwc=5.0)
    assert (power >= 0).all()


def test_pv_power_scales_with_capacity():
    idx = _hourly_index(1)
    df = pd.DataFrame({"ghi": [800.0] * 24}, index=idx)
    p5 = ghi_to_pv_power(df, capacity_kwc=5.0)
    p10 = ghi_to_pv_power(df, capacity_kwc=10.0)
    assert np.allclose(p10.values, 2 * p5.values)


def test_pv_power_without_temp_column_no_derating():
    idx = _hourly_index(1)
    df = pd.DataFrame({"ghi": [1000.0] * 24}, index=idx)
    power = ghi_to_pv_power(df, capacity_kwc=5.0, performance_ratio=1.0)
    # Without a temperature column, derating=1: P = capacity * (GHI/1000) * PR.
    assert np.allclose(power.values, 5.0)


def test_pv_power_hot_day_reduces_output():
    idx = _hourly_index(1)
    cool = pd.DataFrame({"ghi": [800.0] * 24, "temp_air": [15.0] * 24}, index=idx)
    hot = pd.DataFrame({"ghi": [800.0] * 24, "temp_air": [45.0] * 24}, index=idx)
    p_cool = ghi_to_pv_power(cool, capacity_kwc=5.0)
    p_hot = ghi_to_pv_power(hot, capacity_kwc=5.0)
    assert (p_hot < p_cool).all()


def test_pv_power_missing_ghi_raises():
    idx = _hourly_index(1)
    df = pd.DataFrame({"temp_air": [20.0] * 24}, index=idx)
    with pytest.raises(ValueError, match="ghi"):
        ghi_to_pv_power(df, capacity_kwc=5.0)


# ------------------------------------------------------------------- load


def test_load_always_positive():
    idx = _hourly_index(7)
    load = synthetic_residential_load(idx, daily_kwh=10.0)
    assert (load >= 0).all()


def test_load_integrates_to_target():
    idx = _hourly_index(30)
    load = synthetic_residential_load(idx, daily_kwh=10.0)
    total_kwh = load.sum() * 1.0  # dt=1h
    n_days = 30
    assert total_kwh == pytest.approx(10.0 * n_days, rel=0.02)


def test_load_has_morning_and_evening_peaks():
    idx = _hourly_index(1)
    load = synthetic_residential_load(
        idx, daily_kwh=10.0, morning_peak_hour=8.0, evening_peak_hour=20.0, peak_width_h=1.0
    )
    at_8h = load.iloc[8]
    at_3h = load.iloc[3]
    assert at_8h > at_3h


# ---------------------------------------------------------------- battery


def test_battery_charge_respects_max_power():
    spec = BatterySpec(capacity_kwh=100, max_charge_kw=2, max_discharge_kw=2, initial_soc=0.5)
    battery = Battery(spec)
    accepted = battery.charge(requested_kw=10.0, dt_hours=1.0)
    assert accepted <= spec.max_charge_kw + 1e-9


def test_battery_discharge_respects_max_power():
    spec = BatterySpec(capacity_kwh=100, max_charge_kw=2, max_discharge_kw=2, initial_soc=0.5)
    battery = Battery(spec)
    delivered = battery.discharge(requested_kw=10.0, dt_hours=1.0)
    assert delivered <= spec.max_discharge_kw + 1e-9


def test_battery_soc_never_exceeds_max():
    spec = BatterySpec(capacity_kwh=5, max_charge_kw=10, max_discharge_kw=10,
                       initial_soc=0.90, max_soc=0.95)
    battery = Battery(spec)
    for _ in range(20):
        battery.charge(requested_kw=10.0, dt_hours=1.0)
    assert battery.soc_kwh <= spec.max_soc * spec.capacity_kwh + 1e-6


def test_battery_soc_never_below_min():
    spec = BatterySpec(capacity_kwh=5, max_charge_kw=10, max_discharge_kw=10,
                       initial_soc=0.20, min_soc=0.10)
    battery = Battery(spec)
    for _ in range(20):
        battery.discharge(requested_kw=10.0, dt_hours=1.0)
    assert battery.soc_kwh >= spec.min_soc * spec.capacity_kwh - 1e-6


def test_battery_efficiency_loses_energy():
    """With efficiency < 1, energy recovered on discharge is < energy charged.

    The battery starts empty (soc=0) to cleanly isolate the round trip: all
    discharged energy necessarily comes from the charge we just performed,
    with no pre-existing energy to muddy the comparison.
    """
    spec = BatterySpec(
        capacity_kwh=100, max_charge_kw=50, max_discharge_kw=50,
        charge_efficiency=0.9, discharge_efficiency=0.9,
        initial_soc=0.0, min_soc=0.0, max_soc=1.0,
    )
    battery = Battery(spec)
    accepted = battery.charge(requested_kw=10.0, dt_hours=1.0)
    stored = accepted * spec.charge_efficiency  # energy actually stored internally
    assert battery.soc_kwh == pytest.approx(stored)

    delivered = battery.discharge(requested_kw=50.0, dt_hours=1.0)
    # Round-trip efficiency = 0.9 * 0.9 = 0.81: we recover 81% of the
    # energy originally requested for charging, never more.
    assert delivered == pytest.approx(accepted * spec.charge_efficiency * spec.discharge_efficiency)
    assert delivered < stored
    assert battery.soc_kwh == pytest.approx(0.0, abs=1e-9)


def test_battery_zero_request_returns_zero():
    spec = BatterySpec(capacity_kwh=10, max_charge_kw=3, max_discharge_kw=3)
    battery = Battery(spec)
    assert battery.charge(0.0, 1.0) == 0.0
    assert battery.discharge(0.0, 1.0) == 0.0


def test_battery_reset():
    spec = BatterySpec(capacity_kwh=10, max_charge_kw=3, max_discharge_kw=3, initial_soc=0.5)
    battery = Battery(spec)
    battery.charge(3.0, 1.0)
    assert battery.soc_kwh != spec.initial_soc * spec.capacity_kwh
    battery.reset()
    assert battery.soc_kwh == pytest.approx(spec.initial_soc * spec.capacity_kwh)
    assert battery.total_charged_kwh == 0.0


def test_battery_spec_rejects_invalid_soc_bounds():
    with pytest.raises(ValueError):
        BatterySpec(capacity_kwh=10, max_charge_kw=3, max_discharge_kw=3, min_soc=0.9, max_soc=0.5)


def test_battery_spec_rejects_negative_capacity():
    with pytest.raises(ValueError):
        BatterySpec(capacity_kwh=-5, max_charge_kw=3, max_discharge_kw=3)


# --------------------------------------------------------------- dispatch


def test_self_consumption_returns_net():
    assert self_consumption(pv_kw=5.0, load_kw=2.0, soc_fraction=0.5) == pytest.approx(3.0)
    assert self_consumption(pv_kw=2.0, load_kw=5.0, soc_fraction=0.5) == pytest.approx(-3.0)


def test_peak_shaving_inactive_within_thresholds():
    result = peak_shaving(pv_kw=3.0, load_kw=2.0, soc_fraction=0.5,
                          import_threshold_kw=3.0, export_threshold_kw=3.0)
    assert result == pytest.approx(0.0)


def test_peak_shaving_active_beyond_threshold():
    result = peak_shaving(pv_kw=10.0, load_kw=2.0, soc_fraction=0.5,
                          import_threshold_kw=3.0, export_threshold_kw=3.0)
    # net=8, threshold=3 -> charges the surplus beyond the threshold.
    assert result == pytest.approx(5.0)


# ----------------------------------------------------------------- engine


def _random_scenario(days: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = _hourly_index(days)
    pv = pd.Series(rng.uniform(0, 6, len(idx)), index=idx, name="pv_kw")
    load = pd.Series(rng.uniform(0.2, 3, len(idx)), index=idx, name="load_kw")
    spec = BatterySpec(capacity_kwh=8, max_charge_kw=3, max_discharge_kw=3, initial_soc=0.5)
    return pv, load, spec


def test_engine_energy_conservation_identity():
    """The identity pv + import + delivered = load + export + accepted must
    hold exactly (floating-point tolerance) at every timestep, regardless
    of the battery's state at that instant. This is the single most
    important test in this module: any sign error in the engine would
    make it fail.
    """
    pv, load, spec = _random_scenario()
    results = simulate_dispatch(pv, load, spec, strategy="self_consumption")

    accepted = results["battery_power_kw"].clip(lower=0)
    delivered = (-results["battery_power_kw"]).clip(lower=0)
    net = results["pv_kw"] - results["load_kw"]

    lhs = results["pv_kw"] + results["grid_import_kw"] + delivered
    rhs = results["load_kw"] + results["grid_export_kw"] + accepted
    assert np.allclose(lhs.values, rhs.values, atol=1e-8)


def test_engine_soc_stays_within_bounds():
    pv, load, spec = _random_scenario(days=30)
    results = simulate_dispatch(pv, load, spec, strategy="self_consumption")
    min_kwh = spec.min_soc * spec.capacity_kwh
    max_kwh = spec.max_soc * spec.capacity_kwh
    assert (results["soc_kwh"] >= min_kwh - 1e-6).all()
    assert (results["soc_kwh"] <= max_kwh + 1e-6).all()


def test_engine_no_grid_flows_are_negative():
    pv, load, spec = _random_scenario()
    results = simulate_dispatch(pv, load, spec, strategy="self_consumption")
    assert (results["grid_import_kw"] >= 0).all()
    assert (results["grid_export_kw"] >= 0).all()


def test_engine_mismatched_index_raises():
    idx1 = _hourly_index(5)
    idx2 = _hourly_index(5) + pd.Timedelta(hours=1)
    pv = pd.Series(1.0, index=idx1)
    load = pd.Series(1.0, index=idx2)
    spec = BatterySpec(capacity_kwh=10, max_charge_kw=3, max_discharge_kw=3)
    with pytest.raises(ValueError, match="same index"):
        simulate_dispatch(pv, load, spec)


def test_engine_large_battery_absorbs_all_surplus():
    """Oversized battery + constant PV > constant load: never any export."""
    idx = _hourly_index(3)
    pv = pd.Series(5.0, index=idx)
    load = pd.Series(1.0, index=idx)
    spec = BatterySpec(
        capacity_kwh=1000, max_charge_kw=100, max_discharge_kw=100,
        initial_soc=0.0, min_soc=0.0, max_soc=1.0,
    )
    results = simulate_dispatch(pv, load, spec, strategy="self_consumption")
    assert (results["grid_export_kw"] < 1e-6).all()
    assert (results["grid_import_kw"] < 1e-6).all()


def test_engine_no_battery_capacity_passes_through_to_grid():
    """Battery effectively disabled (min_soc=max_soc via tight bounds): everything flows through the grid."""
    idx = _hourly_index(2)
    pv = pd.Series([5.0, 0.0] * 24, index=idx)
    load = pd.Series(1.0, index=idx)
    spec = BatterySpec(
        capacity_kwh=10, max_charge_kw=0.0, max_discharge_kw=0.0, initial_soc=0.5,
    )
    results = simulate_dispatch(pv, load, spec, strategy="self_consumption")
    expected_export = (results["pv_kw"] - results["load_kw"]).clip(lower=0)
    expected_import = (results["load_kw"] - results["pv_kw"]).clip(lower=0)
    assert np.allclose(results["grid_export_kw"].values, expected_export.values, atol=1e-8)
    assert np.allclose(results["grid_import_kw"].values, expected_import.values, atol=1e-8)
    assert (results["battery_power_kw"] == 0).all()


# --------------------------------------------------------------- summarize


def test_summarize_percentages_in_range():
    pv, load, spec = _random_scenario(days=30)
    results = simulate_dispatch(pv, load, spec, strategy="self_consumption")
    stats = summarize(results, spec)
    assert 0 <= stats["self_consumption_pct"] <= 100
    assert 0 <= stats["autonomy_pct"] <= 100
    assert stats["equivalent_full_cycles"] >= 0


def test_summarize_full_battery_improves_self_consumption_vs_no_battery():
    """A bigger battery must never make self-consumption worse."""
    idx = _hourly_index(10)
    rng = np.random.default_rng(1)
    pv = pd.Series(rng.uniform(0, 6, len(idx)), index=idx)
    load = pd.Series(rng.uniform(0.2, 3, len(idx)), index=idx)

    no_battery = BatterySpec(capacity_kwh=0.001, max_charge_kw=0, max_discharge_kw=0,
                             initial_soc=0.5, min_soc=0.0, max_soc=1.0)
    with_battery = BatterySpec(capacity_kwh=10, max_charge_kw=5, max_discharge_kw=5,
                               initial_soc=0.5, min_soc=0.0, max_soc=1.0)

    stats_no_batt = summarize(simulate_dispatch(pv, load, no_battery), no_battery)
    stats_with_batt = summarize(simulate_dispatch(pv, load, with_battery), with_battery)

    assert stats_with_batt["self_consumption_pct"] >= stats_no_batt["self_consumption_pct"] - 1e-6
    assert stats_with_batt["autonomy_pct"] >= stats_no_batt["autonomy_pct"] - 1e-6
