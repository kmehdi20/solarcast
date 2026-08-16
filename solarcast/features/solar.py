"""Solar geometry and clear-sky irradiance.

Every function takes a DataFrame whose index is a UTC DatetimeIndex and
returns that same DataFrame enriched with new columns.

pvlib is the reference for this kind of calculation: its models are
validated against measured data and documented in the literature. The
Ineichen/Perez clear-sky model is used here, pvlib's default, which
performs well across North Africa.
"""

from __future__ import annotations

import pandas as pd
import pvlib
from pvlib.location import Location


def add_solar_geometry(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Add solar angles at each timestamp.

    Columns added
    --------------
    solar_zenith : float
        Apparent solar zenith angle (atmospheric refraction included), degrees.
        0deg = sun at zenith, 90deg = sunset/sunrise.
    solar_azimuth : float
        Solar azimuth, degrees (0=North, 90=East, 180=South, 270=West).
    solar_elevation : float
        Solar elevation = 90deg - zenith, degrees.
    """
    if df.empty:
        for col in ("solar_zenith", "solar_azimuth", "solar_elevation"):
            df[col] = pd.Series(dtype=float)
        return df

    location = Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude_m,
        tz=tz,
    )
    # pvlib expects a localized index; we hand it UTC directly.
    solar_pos = location.get_solarposition(df.index)
    df["solar_zenith"] = solar_pos["apparent_zenith"].values
    df["solar_azimuth"] = solar_pos["azimuth"].values
    df["solar_elevation"] = solar_pos["apparent_elevation"].values
    return df


def add_clear_sky(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
    tz: str = "UTC",
    model: str = "ineichen",
) -> pd.DataFrame:
    """Add clear-sky irradiance and the clear-sky index.

    Columns added
    --------------
    clear_sky_ghi : float
        Theoretical clear-sky GHI (W/m2), Ineichen/Perez model by default.
    clear_sky_dni : float
        Theoretical clear-sky DNI (W/m2).
    clear_sky_dhi : float
        Theoretical clear-sky DHI (W/m2).
    clear_sky_index : float
        Ratio of measured GHI to clear-sky GHI. 1 = perfectly clear sky,
        0 = total cloud cover. Capped at 1.5 to filter out abnormal cloud
        reflection spikes.

    Requirements
    ------------
    The ``ghi`` column must be present to compute clear_sky_index.
    """
    if df.empty:
        for col in ("clear_sky_ghi", "clear_sky_dni", "clear_sky_dhi", "clear_sky_index"):
            df[col] = pd.Series(dtype=float)
        return df

    location = Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude_m,
        tz=tz,
    )
    cs = location.get_clearsky(df.index, model=model)
    df["clear_sky_ghi"] = cs["ghi"].values
    df["clear_sky_dni"] = cs["dni"].values
    df["clear_sky_dhi"] = cs["dhi"].values

    if "ghi" in df.columns:
        # Avoid division by zero at night.
        cs_ghi = df["clear_sky_ghi"].replace(0, float("nan"))
        ki = df["ghi"] / cs_ghi
        # At night (cs_ghi=0), the index is undefined — set it to 0
        # (no irradiance, no clouds to characterize).
        df["clear_sky_index"] = ki.clip(upper=1.5).fillna(0.0)
    return df
