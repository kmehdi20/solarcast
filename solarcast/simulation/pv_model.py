"""Simplified PV model: irradiance -> AC power.

Deliberately simple (not a full single-diode model): the dispatch
simulation needs a realistic hourly PV power profile, not engineering-firm
precision. The model combines:

* a linear power ~ irradiance law, standard for quick sizing;
* thermal derating via the NOCT model, which captures most of the
  efficiency loss on hot days without needing a full I-V curve.

For a fine-grained yield analysis (mismatch, shading, degradation), use a
dedicated tool like PVsyst or a full single-diode model instead.
"""

from __future__ import annotations

import pandas as pd


def ghi_to_pv_power(
    df: pd.DataFrame,
    capacity_kwc: float,
    performance_ratio: float = 0.80,
    noct_c: float = 45.0,
    temp_coefficient: float = -0.004,
    ghi_col: str = "ghi",
    temp_col: str = "temp_air",
) -> pd.Series:
    """Estimate AC PV power (kW) from irradiance and temperature.

    Model
    -----
    T_cell = T_air + (NOCT - 20) / 800 * GHI            (standard NOCT model)
    derating = 1 + temp_coefficient * (T_cell - 25)      (negative coefficient, crystalline silicon)
    P_ac = capacity_kwc * (GHI / 1000) * performance_ratio * derating

    `performance_ratio` absorbs system losses (inverter, wiring, soiling,
    mismatch) that aren't modeled explicitly. 0.80 is a common quick-sizing
    value for a residential installation.

    Parameters
    ----------
    df:
        DataFrame indexed by timestamp, must contain `ghi_col` (W/m2).
        `temp_col` is optional: if absent, thermal derating is skipped.
    capacity_kwc:
        Installed peak power, kWc.
    performance_ratio:
        System performance ratio, dimensionless (0-1).
    noct_c:
        Nominal Operating Cell Temperature, degC (typical datasheet value 44-46degC).
    temp_coefficient:
        Power temperature coefficient, /degC (negative for crystalline
        silicon, typically -0.35 to -0.45 %/degC).

    Returns
    -------
    pd.Series
        AC power in kW, same index as `df`, never negative.
    """
    if ghi_col not in df.columns:
        raise ValueError(f"Column '{ghi_col}' missing from DataFrame.")
    if capacity_kwc <= 0:
        raise ValueError("capacity_kwc must be positive.")

    ghi = df[ghi_col].clip(lower=0.0)

    if temp_col in df.columns:
        t_cell = df[temp_col] + (noct_c - 20.0) / 800.0 * ghi
        derating = 1.0 + temp_coefficient * (t_cell - 25.0)
    else:
        derating = 1.0

    power_kw = capacity_kwc * (ghi / 1000.0) * performance_ratio * derating
    return power_kw.clip(lower=0.0).rename("pv_power_kw")
