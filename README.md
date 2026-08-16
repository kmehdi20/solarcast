# SolarCast

A modular Python framework for solar energy time-series analysis — from raw
irradiance data to next-hour forecasts to battery sizing decisions.

Live dashboard: **https://solarcast-production.up.railway.app**
Repository: **https://github.com/kmehdi20/solarcast**

---

## What it does

1. **Ingests** hourly solar and weather data from Open-Meteo and PVGIS,
   asynchronously, with retry logic and rate limiting.
2. **Stores** it idempotently in a time-series database — replaying an
   ingestion updates revised values instead of duplicating rows.
3. **Engineers features** from solar geometry (`pvlib`), clear-sky
   modeling, lags, and calendar variables.
4. **Forecasts** next-hour irradiance with GradientBoosting, validated
   with walk-forward testing (never random k-fold, which leaks the future
   into training on time series).
5. **Simulates** battery dispatch with an engine whose energy accounting
   is verified exactly at every timestep, and automates PV sizing by
   sweeping capacity to find the self-consumption/autonomy crossover.
6. Serves all of the above through a **live web dashboard** — no terminal
   required to explore the results.

This is a first step toward a longer-term goal: an open, transparent
alternative to the closed, licensed tools that currently gatekeep serious
PV analysis. It doesn't replace what those tools do — detailed shading
analysis, bankable engineering reports, decades of validation — but it
does something they don't: live, ML-driven forecasting and automated
sizing, in a browser, fully open source.

---

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Offline check of the whole pipeline (no network calls)
python -m solarcast.scripts.demo_offline

# Real ingestion
python -m solarcast.scripts.ingest --provider open-meteo \
    --site Kenitra --start 2023-01-01 --end 2024-12-31

# Train and compare forecasting models
python -m solarcast.scripts.train --site Kenitra --all-models

# Run a battery dispatch simulation
python -m solarcast.scripts.simulate --site Kenitra --pv-capacity-kwc 6 --battery-capacity-kwh 10

# Launch the local dashboard
python -m uvicorn solarcast.scripts.server:app --reload

# Tests
pytest -q
```

The offline demo generates a synthetic hourly year, writes it to the
database, and reads it back as a pivoted DataFrame. It validates the
whole chain without depending on API availability.

---

## Architecture

```
solarcast/
├── core/           config, logging, domain types, exceptions
├── ingestion/      async clients, rate limiting, orchestration
├── storage/        ORM models, sessions, repositories
├── features/       solar geometry, clear-sky index, lags, calendar features
├── models/         baselines, walk-forward validation, GradientBoosting
├── simulation/      PV model, synthetic load, battery, dispatch engine
└── scripts/        CLI entry points + the FastAPI dashboard
```

The flow is one-directional: an ingestion client produces
`ObservationPoint` objects, the service hands them to the repository, the
repository translates them into rows and, later, into pivoted
DataFrames. No client knows about the ORM, no repository knows about
HTTP, no model knows where the data came from.

### Data sources

| Source | Role | Horizon | Forecast |
|---|---|---|---|
| Open-Meteo | Reanalysis + operational forecast | past + D+16 | yes |
| PVGIS (JRC) | Long-term history, TMY, modeled production | past, by year | no |
| NASA POWER | Global history, cross-validation | past, with latency | not implemented yet |

Only Open-Meteo implements `fetch_forecast()`. On the other clients, the
call raises `NotImplementedError` instead of silently returning the past.

---

## Design decisions

**Everything in UTC.** Timestamps are stored exclusively in UTC;
conversion happens on output. This is the single most sensitive point in
the project: a one-hour offset in an irradiance series degrades a
forecasting model without ever raising an error. Open-Meteo is therefore
queried with `timezone=UTC` enforced; a future NASA POWER client will
need to force `time-standard=UTC` — that API returns local solar time by
default.

**Long format.** One row per (site, provider, variable, timestamp). More
verbose than a wide format, but necessary since providers expose neither
the same variables nor the same granularity; a wide schema would force a
migration on every new source.

**Idempotent writes.** `INSERT ... ON CONFLICT DO UPDATE` on the
`uq_observation_point` constraint. Replaying an already-loaded window
updates values instead of duplicating them — necessary because
reanalyses get revised after the fact.

**Measurement vs. forecast.** The `reference_time` column equals
`timestamp` for historical data, and the run's emission time for a
forecast. It's non-nullable, otherwise the uniqueness constraint would be
ineffective: in SQL, two NULLs are never equal. Reads exclude forecasts
by default.

**Rate limiting.** A token bucket for the average rate, a semaphore for
concurrency. Public APIs cap calls — NASA POWER's documentation states
this explicitly — and an unregulated multi-site ingestion gets cut off
within seconds.

**Selective retry.** Only 408, 425, 429, and 5xx are retried, with
exponential backoff and full jitter. A 400 signals a bad request: retrying
would only burn the quota. Jitter prevents multiple tasks from
resynchronizing on the same slot after a shared 429.

**Pinned API version.** PVGIS is called on `v5_2` explicitly. The JRC has
changed default values between versions of the non-interactive service
before.

**No future leakage in validation.** Forecasting models are evaluated
with walk-forward validation — train on `[t0, t]`, test on `[t, t+h]`,
slide forward — never random k-fold, which would let a test fold fall
before a training fold and silently inflate every metric.

**Verified energy conservation.** The battery dispatch engine's core
identity — `pv + grid_import + discharge = load + grid_export + charge`
— is checked exactly, at every simulated hour, on random data. It's the
one test that would catch a sign error immediately.

---

## Source quirks

**PVGIS.** The window is specified in years, not dates: fine-grained
trimming happens client-side. Timestamps arrive as `YYYYMMDD:HHMM` with
non-zero minutes depending on the database — SARAH provides instantaneous
values, ERA5 hourly averages. Timestamps are floored to the start of the
hour and the source database is kept in `dataset`. **Never mix two
radiation databases within the same training set.** Also, `G(i)` means
GHI if `angle=0` and plane-of-array irradiance otherwise; the client
accounts for this.

**NASA POWER (not yet implemented).** `point` endpoint only in hourly
mode, 15 parameters max per request, `RE` community for solar parameters
(the old `SSE` community was removed), timestamps mark the start of the
hour.

---

## Configuration

`config/settings.yaml` holds everything that's safe to version. Any value
can be overridden by an environment variable, with a double underscore
separating levels:

```bash
export SOLARCAST_DATABASE__URL="postgresql+asyncpg://user:pass@host/solarcast"
export SOLARCAST_LOGGING__LEVEL=DEBUG
export SOLARCAST_LOGGING__JSON_FORMAT=true
```

No secrets in the YAML.

---

## Status

**80 automated tests passing**, covering configuration, rate limiting,
both ingestion clients, retry policy, idempotent writes,
measurement/forecast separation, DataFrame export, solar geometry,
clear-sky modeling, lag/rolling features, baseline models, walk-forward
validation, battery physics, and — the one that matters most — the
dispatch engine's energy-conservation identity, checked exactly on random
data.

Ingestion tests run against `httpx.MockTransport` — no real network
calls, so they're reproducible and safe for continuous integration.

On real Kenitra weather data: GradientBoosting reaches a **0.845 skill
score** against persistence for next-hour GHI forecasting, and the
automated PV sizing sweep finds the self-consumption/autonomy crossover
at **~2.75 kWc** for a typical residential load — the point past which,
with no residential net metering in Morocco, oversizing mostly produces
electricity given away for free.

## Next

- NASA POWER ingestion client.
- Alembic migrations for schema changes beyond `create_schema()`.
- Forecast-aware dispatch strategies (e.g. pre-charging ahead of an
  announced cloudy day) — the engine already supports grid-charging
  strategies correctly, none are implemented yet.
- Persisting trained models instead of retraining on every dashboard
  request.
