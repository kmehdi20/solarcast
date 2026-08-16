"""Foundation tests: configuration, rate limiting, ingestion, persistence.

No real network calls: provider responses are simulated via
`httpx.MockTransport`, which lets parsing and the retry policy be
validated deterministically and offline.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import httpx
import pytest

from solarcast.core.config import ProviderConfig, RetryConfig, load_settings
from solarcast.core.exceptions import ConfigError, ProviderError, ValidationError
from solarcast.core.types import ObservationPoint, Provider, Variable
from solarcast.ingestion.openmeteo import OpenMeteoClient
from solarcast.ingestion.pvgis import PVGISClient
from solarcast.ingestion.ratelimit import TokenBucket
from solarcast.storage.repository import LocationRepository, ObservationRepository
from solarcast.storage.session import (
    create_schema,
    dispose_engine,
    init_engine,
    session_scope,
)
from solarcast.core.config import DatabaseConfig

# `asyncio_mode = auto` (pytest.ini) handles coroutines: no explicit
# marker needed on each test.


# --------------------------------------------------------------------- config


def test_load_settings_reads_yaml() -> None:
    settings = load_settings("config/settings.yaml")
    assert settings.provider("open-meteo").enabled
    assert settings.location("Kenitra").latitude == pytest.approx(34.2610)


def test_unknown_provider_raises() -> None:
    settings = load_settings("config/settings.yaml")
    with pytest.raises(ConfigError):
        settings.provider("meteo-inexistant")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOLARCAST_LOGGING__LEVEL", "DEBUG")
    settings = load_settings("config/settings.yaml")
    assert settings.logging.level == "DEBUG"


# ----------------------------------------------------------------- rate limit


async def test_token_bucket_throttles() -> None:
    """Five acquisitions at 10 req/s with a capacity of 1 take ~0.4s."""
    bucket = TokenBucket(rate=10.0, capacity=1.0)
    start = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.35


# ------------------------------------------------------------------- clients


def _config(handler: httpx.MockTransport, **options) -> ProviderConfig:
    return ProviderConfig(
        base_url="https://example.invalid",
        timeout_s=5.0,
        retry=RetryConfig(max_attempts=3, initial_backoff_s=0.01, jitter=False),
        options=options,
    )


def _mounted(client, transport: httpx.MockTransport):
    """Swap the client's transport for the simulated one."""
    client._client = httpx.AsyncClient(
        base_url=client.config.base_url, transport=transport
    )
    return client


OPEN_METEO_PAYLOAD = {
    "hourly": {
        "time": ["2024-06-01T00:00", "2024-06-01T01:00", "2024-06-01T02:00"],
        "shortwave_radiation": [0.0, 0.0, 12.5],
        "temperature_2m": [18.2, 17.9, 17.5],
    }
}


async def test_openmeteo_parses_archive() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=OPEN_METEO_PAYLOAD)
    )
    client = _mounted(OpenMeteoClient(_config(transport)), transport)

    points = await client.fetch_historical(
        latitude=34.26,
        longitude=-6.58,
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        variables=[Variable.GHI, Variable.TEMP_AIR],
    )
    await client.http.aclose()

    assert len(points) == 6  # 3 timestamps x 2 variables
    ghi = [p for p in points if p.variable is Variable.GHI]
    assert ghi[2].value == pytest.approx(12.5)
    # The timestamp must be localized to UTC, never naive.
    assert ghi[0].timestamp.tzinfo is not None
    assert ghi[0].timestamp.hour == 0
    assert all(not p.is_forecast for p in points)


async def test_openmeteo_rejects_length_mismatch() -> None:
    broken = {
        "hourly": {
            "time": ["2024-06-01T00:00", "2024-06-01T01:00"],
            "shortwave_radiation": [0.0],
        }
    }
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=broken))
    client = _mounted(OpenMeteoClient(_config(transport)), transport)

    with pytest.raises(ValidationError):
        await client.fetch_historical(
            latitude=0.0,
            longitude=0.0,
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            variables=[Variable.GHI],
        )
    await client.http.aclose()


async def test_retry_then_success() -> None:
    """Two 503s then a 200: the client must persist."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json=OPEN_METEO_PAYLOAD)

    transport = httpx.MockTransport(handler)
    client = _mounted(OpenMeteoClient(_config(transport)), transport)

    points = await client.fetch_historical(
        latitude=0.0,
        longitude=0.0,
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        variables=[Variable.GHI],
    )
    await client.http.aclose()

    assert calls["n"] == 3
    assert len(points) == 3


async def test_permanent_error_is_not_retried() -> None:
    """A 400 signals a bad request: retrying won't help."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="invalid parameter")

    transport = httpx.MockTransport(handler)
    client = _mounted(OpenMeteoClient(_config(transport)), transport)

    with pytest.raises(ProviderError):
        await client.fetch_historical(
            latitude=0.0,
            longitude=0.0,
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            variables=[Variable.GHI],
        )
    await client.http.aclose()
    assert calls["n"] == 1


PVGIS_PAYLOAD = {
    "inputs": {"meteo_data": {"radiation_db": "PVGIS-SARAH2"}},
    "outputs": {
        "hourly": [
            {"time": "20200101:0011", "G(i)": 0.0, "Gb(n)": 0.0, "T2m": 11.4},
            {"time": "20200101:1011", "G(i)": 412.3, "Gb(n)": 610.7, "T2m": 16.8},
        ]
    },
}


async def test_pvgis_parses_and_floors_timestamps() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=PVGIS_PAYLOAD))
    client = _mounted(PVGISClient(_config(transport)), transport)

    points = await client.fetch_historical(
        latitude=30.93,
        longitude=-6.94,
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 12, 31, tzinfo=timezone.utc),
    )
    await client.http.aclose()

    # SARAH's ':11' minutes must be floored to the start of the hour.
    assert all(p.timestamp.minute == 0 for p in points)
    # angle=0: G(i) is correctly interpreted as GHI, not POA.
    assert any(p.variable is Variable.GHI for p in points)
    assert all(p.dataset == "PVGIS-SARAH2" for p in points)


async def test_pvgis_tilted_maps_to_poa() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=PVGIS_PAYLOAD))
    client = _mounted(PVGISClient(_config(transport, surface_tilt=30.0)), transport)

    points = await client.fetch_historical(
        latitude=30.93,
        longitude=-6.94,
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 12, 31, tzinfo=timezone.utc),
    )
    await client.http.aclose()

    assert any(p.variable is Variable.POA_GLOBAL for p in points)
    assert not any(p.variable is Variable.GHI for p in points)


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError):
        ObservationPoint(
            provider=Provider.OPEN_METEO,
            variable=Variable.GHI,
            timestamp=datetime(2024, 1, 1),  # no timezone
            value=1.0,
        )


# ------------------------------------------------------------------- storage


@pytest.fixture
async def database():
    init_engine(DatabaseConfig(url="sqlite+aiosqlite:///:memory:"))
    await create_schema()
    yield
    await dispose_engine()


async def test_upsert_is_idempotent(database) -> None:
    """Replaying an ingestion updates values without duplicating rows."""
    ts = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)

    async with session_scope() as session:
        location = await LocationRepository(session).get_or_create(
            "Meknes", 33.8935, -5.5473
        )
        location_id = location.id

    first = [
        ObservationPoint(Provider.OPEN_METEO, Variable.GHI, ts, 800.0),
        ObservationPoint(Provider.OPEN_METEO, Variable.TEMP_AIR, ts, 30.0),
    ]
    async with session_scope() as session:
        await ObservationRepository(session).upsert_many(location_id, first)

    # Same key, revised value: the real-world case of an updated reanalysis.
    revised = [ObservationPoint(Provider.OPEN_METEO, Variable.GHI, ts, 812.5)]
    async with session_scope() as session:
        await ObservationRepository(session).upsert_many(location_id, revised)

    async with session_scope() as session:
        rows = await ObservationRepository(session).fetch(location_id)
        assert len(rows) == 2  # no duplicate
        ghi = next(r for r in rows if r.variable is Variable.GHI)
        assert ghi.value == pytest.approx(812.5)


async def test_to_frame_pivots(database) -> None:
    async with session_scope() as session:
        location = await LocationRepository(session).get_or_create(
            "Kenitra", 34.261, -6.5802
        )
        location_id = location.id

    points = []
    for hour in range(5):
        ts = datetime(2024, 6, 1, hour, tzinfo=timezone.utc)
        points.append(ObservationPoint(Provider.OPEN_METEO, Variable.GHI, ts, hour * 100.0))
        points.append(ObservationPoint(Provider.OPEN_METEO, Variable.TEMP_AIR, ts, 20.0 + hour))

    async with session_scope() as session:
        await ObservationRepository(session).upsert_many(location_id, points)

    async with session_scope() as session:
        frame = await ObservationRepository(session).to_frame(location_id)

    assert list(frame.columns) == ["ghi", "temp_air"]
    assert len(frame) == 5
    assert frame["ghi"].iloc[4] == pytest.approx(400.0)


async def test_forecasts_excluded_by_default(database) -> None:
    """A forecast and a measurement at the same instant coexist without mixing."""
    ts = datetime(2024, 6, 2, 12, tzinfo=timezone.utc)
    run_time = datetime(2024, 6, 1, 0, tzinfo=timezone.utc)

    async with session_scope() as session:
        location = await LocationRepository(session).get_or_create(
            "Ouarzazate", 30.9335, -6.937
        )
        location_id = location.id

    points = [
        ObservationPoint(Provider.OPEN_METEO, Variable.GHI, ts, 900.0),
        ObservationPoint(
            Provider.OPEN_METEO, Variable.GHI, ts, 870.0, reference_time=run_time
        ),
    ]
    async with session_scope() as session:
        await ObservationRepository(session).upsert_many(location_id, points)

    async with session_scope() as session:
        repo = ObservationRepository(session)
        measured = await repo.fetch(location_id)
        everything = await repo.fetch(location_id, forecasts=True)

    assert len(measured) == 1
    assert measured[0].value == pytest.approx(900.0)
    assert len(everything) == 2


async def test_coverage_reports_window(database) -> None:
    async with session_scope() as session:
        location = await LocationRepository(session).get_or_create(
            "Kenitra", 34.261, -6.5802
        )
        location_id = location.id

    points = [
        ObservationPoint(
            Provider.PVGIS,
            Variable.GHI,
            datetime(2020, 1, 1, h, tzinfo=timezone.utc),
            float(h),
        )
        for h in range(24)
    ]
    async with session_scope() as session:
        await ObservationRepository(session).upsert_many(location_id, points)

    async with session_scope() as session:
        first, last = await ObservationRepository(session).coverage(
            location_id, Provider.PVGIS, Variable.GHI
        )

    assert first is not None and last is not None
    assert first.hour == 0
    assert last.hour == 23
