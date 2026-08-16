"""Offline demonstration of the full foundation.

Generates a year of synthetic data (simplified clear-sky model), routes
it through the repository, then reads it back as a DataFrame. Serves as
an end-to-end check when the network is unavailable, and as an entry
point for getting familiar with the framework's API.

    python -m solarcast.scripts.demo_offline
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

from solarcast.core.config import DatabaseConfig
from solarcast.core.logging import configure_logging, get_logger
from solarcast.core.types import ObservationPoint, Provider, Variable
from solarcast.storage.repository import LocationRepository, ObservationRepository
from solarcast.storage.session import (
    create_schema,
    dispose_engine,
    init_engine,
    session_scope,
)

logger = get_logger(__name__)

# Reference site: Meknes.
LATITUDE = 33.8935
LONGITUDE = -5.5473


def clear_sky_ghi(moment: datetime, latitude: float, longitude: float) -> float:
    """Clear-sky global horizontal irradiance, simplified model.

    Solar geometry reduced to the essentials: Cooper's declination,
    hour angle corrected for longitude, atmospheric attenuation as a
    power law on air mass. The equation of time is neglected (maximum
    deviation of about a quarter hour).

    Since `moment` is in UTC, the longitude correction is essential:
    without it, solar noon would fall at 12:00 UTC regardless of
    meridian, i.e. about twenty minutes of error in Morocco — exactly
    the kind of silent offset the "everything in UTC" convention is
    meant to make explicit.
    """
    day_of_year = moment.timetuple().tm_yday
    declination = math.radians(23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365)))
    solar_time = moment.hour + moment.minute / 60.0 + longitude / 15.0
    hour_angle = math.radians(15.0 * (solar_time - 12.0))
    lat_rad = math.radians(latitude)

    cos_zenith = math.sin(lat_rad) * math.sin(declination) + math.cos(lat_rad) * math.cos(
        declination
    ) * math.cos(hour_angle)
    if cos_zenith <= 0.0:
        return 0.0

    air_mass = 1.0 / max(cos_zenith, 0.05)
    return 1361.0 * 0.7 ** (air_mass**0.678) * cos_zenith


def synthetic_year(year: int = 2024) -> list[ObservationPoint]:
    """One hourly year of GHI and temperature."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    hours = 366 * 24 if year % 4 == 0 else 365 * 24

    points: list[ObservationPoint] = []
    for offset in range(hours):
        moment = start + timedelta(hours=offset)
        ghi = clear_sky_ghi(moment, LATITUDE, LONGITUDE)

        # Temperature: seasonal cycle + diurnal cycle, plausible for Meknes.
        seasonal = 17.0 - 9.0 * math.cos(2 * math.pi * moment.timetuple().tm_yday / 365)
        diurnal = 6.0 * math.sin(2 * math.pi * (moment.hour - 9) / 24)

        points.append(
            ObservationPoint(Provider.SYNTHETIC, Variable.GHI, moment, round(ghi, 2))
        )
        points.append(
            ObservationPoint(
                Provider.SYNTHETIC,
                Variable.TEMP_AIR,
                moment,
                round(seasonal + diurnal, 2),
            )
        )
    return points


async def main() -> None:
    configure_logging(level="INFO")
    init_engine(DatabaseConfig(url="sqlite+aiosqlite:///data/demo.db"))
    await create_schema()

    try:
        async with session_scope() as session:
            location = await LocationRepository(session).get_or_create(
                "Meknes", LATITUDE, LONGITUDE, altitude_m=550, tz="Africa/Casablanca"
            )
            location_id = location.id

        points = synthetic_year(2024)
        async with session_scope() as session:
            written = await ObservationRepository(session).upsert_many(
                location_id, points
            )

        async with session_scope() as session:
            frame = await ObservationRepository(session).to_frame(location_id)

        print(f"\nPoints written        : {written}")
        print(f"Rows after pivot      : {len(frame)}")
        print(f"Columns               : {list(frame.columns)}")
        print(f"Period                : {frame.index.min()} -> {frame.index.max()}")

        daily = frame["ghi"].resample("D").sum() / 1000.0  # Wh/m2 -> kWh/m2
        print(f"\nAverage irradiation   : {daily.mean():.2f} kWh/m2/day")
        print(f"Sunniest day          : {daily.idxmax().date()} ({daily.max():.2f})")
        print(f"Weakest day           : {daily.idxmin().date()} ({daily.min():.2f})")

        print("\nJune 21 profile (daytime hours):")
        june = frame.loc["2024-06-21 05:00":"2024-06-21 19:00"]
        for timestamp, row in june.iterrows():
            bar = "#" * int(row["ghi"] / 25)
            print(f"  {timestamp:%H:%M}  {row['ghi']:7.1f} W/m2  {bar}")

    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
