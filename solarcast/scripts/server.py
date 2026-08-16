"""SolarCast — serveur web local.

Lance avec :
    python -m solarcast.scripts.server

Puis ouvre http://localhost:8000 dans le navigateur.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from solarcast.core.config import load_settings
from solarcast.core.logging import configure_logging, get_logger
from solarcast.core.types import Provider, Variable
from solarcast.features.pipeline import build_features
from solarcast.ingestion.service import ingest_historical
from solarcast.models.registry import get_model, list_models
from solarcast.models.metrics import evaluate
from solarcast.simulation.battery import BatterySpec
from solarcast.simulation.dispatch import STRATEGIES
from solarcast.simulation.engine import simulate_dispatch, summarize
from solarcast.simulation.load import synthetic_residential_load
from solarcast.simulation.pv_model import ghi_to_pv_power
from solarcast.storage.repository import LocationRepository, ObservationRepository
from solarcast.storage.session import create_schema, init_engine, session_scope

logger = get_logger(__name__)

settings = load_settings()
configure_logging(level=settings.logging.level)
init_engine(settings.database)

app = FastAPI(title="SolarCast", version="0.1.0")


@app.on_event("startup")
async def startup():
    await create_schema()


# ------------------------------------------------------------------ REST API


@app.get("/api/sites")
async def list_sites():
    async with session_scope() as session:
        locations = await LocationRepository(session).list_all()
        return [
            {
                "id": loc.id,
                "name": loc.name,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "altitude_m": loc.altitude_m,
                "timezone": loc.timezone,
            }
            for loc in locations
        ]


@app.get("/api/sites/{site_name}/data")
async def get_site_data(
    site_name: str,
    variable: str = "ghi",
    start: str | None = None,
    end: str | None = None,
):
    async with session_scope() as session:
        location = await LocationRepository(session).get_by_name(site_name)
        if not location:
            raise HTTPException(status_code=404, detail=f"Site '{site_name}' not found")

        try:
            var = Variable(variable)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown variable '{variable}'")

        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None

        frame = await ObservationRepository(session).to_frame(
            location.id,
            variables=[var],
            start=start_dt,
            end=end_dt,
        )

    if frame.empty:
        return {"labels": [], "values": [], "variable": variable}

    col = variable
    if col not in frame.columns:
        return {"labels": [], "values": [], "variable": variable}

    series = frame[col].dropna()
    return {
        "labels": [ts.isoformat() for ts in series.index],
        "values": [round(v, 3) for v in series.values],
        "variable": variable,
    }


@app.get("/api/sites/{site_name}/stats")
async def get_site_stats(site_name: str):
    async with session_scope() as session:
        location = await LocationRepository(session).get_by_name(site_name)
        if not location:
            raise HTTPException(404, detail=f"Site '{site_name}' not found")

        frame = await ObservationRepository(session).to_frame(location.id)

    if frame.empty:
        return {"total_points": 0, "period_start": None, "period_end": None, "variables": []}

    return {
        "total_points": int(frame.count().sum()),
        "period_start": frame.index.min().isoformat(),
        "period_end": frame.index.max().isoformat(),
        "variables": list(frame.columns),
    }


@app.get("/api/config")
async def get_config():
    return {
        "providers": list(settings.providers.keys()),
        "sites": [loc.name for loc in settings.locations],
    }


class IngestRequest(BaseModel):
    provider: str
    site: str
    start: str
    end: str


@app.get("/api/ingest/stream")
async def ingest_stream(provider: str, site: str, start: str, end: str):
    """Server-Sent Events stream for live ingestion progress."""

    async def event_generator():
        def send(msg: str, level: str = "info"):
            data = json.dumps({"message": msg, "level": level, "time": datetime.now().strftime("%H:%M:%S")})
            return f"data: {data}\n\n"

        try:
            loc_config = settings.location(site)
            prov_config = settings.provider(provider)
        except Exception as exc:
            yield send(str(exc), "error")
            yield send("__DONE__", "done")
            return

        yield send(f"Starting ingestion: {provider} / {site}")
        yield send(f"Window: {start} → {end}")

        try:
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            yield send(f"Invalid date: {exc}", "error")
            yield send("__DONE__", "done")
            return

        yield send("Connecting to provider...")

        try:
            result = await ingest_historical(
                provider_name=provider,
                provider_config=prov_config,
                location=loc_config,
                start=start_dt,
                end=end_dt,
            )
            if result.ok:
                yield send(f"Done — {result.points:,} points written.", "success")
            else:
                yield send(f"Failed: {result.error}", "error")
        except Exception as exc:
            yield send(f"Error: {traceback.format_exc()}", "error")

        yield send("__DONE__", "done")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sites/{site_name}/forecast")
async def get_forecast(
    site_name: str,
    model: str = "gradient_boosting",
    test_days: int = 14,
):
    """Train a model on the history and evaluate it on the most recent days.

    Simple chronological train/test split (not full walk-forward) to stay
    fast for the web server. Returns the model's predictions, the
    persistence reference, and the associated metrics — enough to draw an
    actual-vs-predicted chart and show a live skill score.
    """
    if model not in list_models():
        raise HTTPException(400, detail=f"Unknown model '{model}'. Choices: {list_models()}")

    async with session_scope() as session:
        location = await LocationRepository(session).get_by_name(site_name)
        if not location:
            raise HTTPException(404, detail=f"Site '{site_name}' not found")
        frame = await ObservationRepository(session).to_frame(location.id)

    if frame.empty or "ghi" not in frame.columns:
        raise HTTPException(
            400, detail="No GHI data for this site. Run an ingestion first."
        )

    try:
        X, y = build_features(
            frame,
            latitude=location.latitude,
            longitude=location.longitude,
            altitude_m=location.altitude_m or 0.0,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))

    test_size = 24 * test_days
    min_train = 24 * 30
    if len(X) < min_train + test_size:
        raise HTTPException(
            400,
            detail=(
                f"Not enough data: {len(X)} rows available, "
                f"{min_train + test_size} needed for a {test_days}-day test window. "
                "Ingest more history or reduce test_days."
            ),
        )

    # Cap the training window to roughly the last 6 months: training on the
    # full history (potentially several years) would push past a minute in
    # the browser for a marginal skill gain. Recent conditions are also
    # more representative for an operational forecast than data that's two
    # years old anyway.
    max_train_rows = 24 * 180
    cap = max_train_rows + test_size
    if len(X) > cap:
        X, y = X.iloc[-cap:], y.iloc[-cap:]

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    estimator = get_model(model)

    from solarcast.models.baselines import ClearSkyModel, PersistenceModel
    from sklearn.preprocessing import StandardScaler

    is_baseline = isinstance(estimator, (PersistenceModel, ClearSkyModel))
    if is_baseline:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
    else:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        estimator.fit(X_train_s, y_train)
        y_pred = estimator.predict(X_test_s)

    y_persistence = (
        X_test["ghi_lag_1h"].values if "ghi_lag_1h" in X_test.columns else None
    )
    metrics = evaluate(y_test.values, y_pred, y_persistence=y_persistence, label=model)
    # NaN/inf aren't JSON-serializable as-is.
    metrics = {
        k: (None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v)
        for k, v in metrics.items()
    }

    return {
        "model": model,
        "site": site_name,
        "test_days": test_days,
        "labels": [ts.isoformat() for ts in X_test.index],
        "actual": [round(v, 2) for v in y_test.values.tolist()],
        "predicted": [round(v, 2) for v in y_pred.tolist()],
        "persistence": (
            [round(v, 2) for v in y_persistence.tolist()] if y_persistence is not None else None
        ),
        "metrics": metrics,
    }


async def _load_site_frame(site_name: str):
    """Load the site and its GHI/temperature DataFrame, or raise an HTTPException."""
    async with session_scope() as session:
        location = await LocationRepository(session).get_by_name(site_name)
        if not location:
            raise HTTPException(404, detail=f"Site '{site_name}' not found")
        frame = await ObservationRepository(session).to_frame(location.id)
    if frame.empty or "ghi" not in frame.columns:
        raise HTTPException(
            400, detail="No GHI data for this site. Run an ingestion first."
        )
    return location, frame


@app.get("/api/sites/{site_name}/simulate")
async def run_simulation(
    site_name: str,
    pv_capacity_kwc: float = 6.0,
    performance_ratio: float = 0.80,
    daily_kwh: float = 10.0,
    battery_capacity_kwh: float = 10.0,
    max_charge_kw: float = 3.0,
    max_discharge_kw: float = 3.0,
    min_soc: float = 0.10,
    max_soc: float = 0.95,
    strategy: str = "self_consumption",
):
    """Simulate the battery dispatch over the site's entire history.

    Returns the aggregated metrics, a compact daily trace (for the chart),
    and a representative week at hourly resolution (to visualize battery
    behavior hour by hour). Returning 17,000+ raw hourly points for two
    years of data would needlessly bloat the response and the chart
    rendering in the browser.
    """
    if strategy not in STRATEGIES:
        raise HTTPException(400, detail=f"Unknown strategy '{strategy}'. Choices: {list(STRATEGIES)}")

    location, frame = await _load_site_frame(site_name)

    try:
        battery_spec = BatterySpec(
            capacity_kwh=battery_capacity_kwh,
            max_charge_kw=max_charge_kw,
            max_discharge_kw=max_discharge_kw,
            min_soc=min_soc,
            max_soc=max_soc,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))

    pv_kw = ghi_to_pv_power(
        frame, capacity_kwc=pv_capacity_kwc, performance_ratio=performance_ratio
    )
    load_kw = synthetic_residential_load(frame.index, daily_kwh=daily_kwh)

    results = simulate_dispatch(pv_kw, load_kw, battery_spec, strategy=strategy)
    stats = summarize(results, battery_spec)

    # Daily trace (aggregates) over the full period — stays lightweight
    # even over two years (~730 points) and feeds the main chart.
    daily = results.resample("D").agg(
        {
            "pv_kw": "sum",
            "load_kw": "sum",
            "grid_import_kw": "sum",
            "grid_export_kw": "sum",
            "soc_pct": "mean",
        }
    )
    daily_trace = {
        "labels": [d.isoformat() for d in daily.index],
        "pv_kwh": [round(v, 2) for v in daily["pv_kw"].values],
        "load_kwh": [round(v, 2) for v in daily["load_kw"].values],
        "grid_import_kwh": [round(v, 2) for v in daily["grid_import_kw"].values],
        "grid_export_kwh": [round(v, 2) for v in daily["grid_export_kw"].values],
        "soc_pct": [round(v, 1) for v in daily["soc_pct"].values],
    }

    # Representative week at hourly resolution — the last 7 days
    # covered, to see the SOC react hour by hour.
    week = results.iloc[-24 * 7 :]
    week_trace = {
        "labels": [t.isoformat() for t in week.index],
        "pv_kw": [round(v, 2) for v in week["pv_kw"].values],
        "load_kw": [round(v, 2) for v in week["load_kw"].values],
        "soc_pct": [round(v, 1) for v in week["soc_pct"].values],
        "grid_import_kw": [round(v, 2) for v in week["grid_import_kw"].values],
        "grid_export_kw": [round(v, 2) for v in week["grid_export_kw"].values],
    }

    return {
        "site": site_name,
        "strategy": strategy,
        "inputs": {
            "pv_capacity_kwc": pv_capacity_kwc,
            "daily_kwh": daily_kwh,
            "battery_capacity_kwh": battery_capacity_kwh,
        },
        "stats": stats,
        "daily": daily_trace,
        "week": week_trace,
    }


@app.get("/api/sites/{site_name}/simulate/sweep")
async def run_sizing_sweep(
    site_name: str,
    daily_kwh: float = 10.0,
    battery_capacity_kwh: float = 5.0,
    max_charge_kw: float = 3.0,
    max_discharge_kw: float = 3.0,
    strategy: str = "self_consumption",
    pv_min_kwc: float = 1.0,
    pv_max_kwc: float = 8.0,
    steps: int = 12,
):
    """Sweep a range of PV capacities and return the indicators for each.

    Automates the usual manual analysis: with load and battery fixed,
    where does the crossover between self-consumption and autonomy rates
    fall? Each simulation is independent and fast (tens of milliseconds
    on two years of hourly data), so the whole sweep stays well within
    the budget of a synchronous HTTP request.
    """
    if strategy not in STRATEGIES:
        raise HTTPException(400, detail=f"Unknown strategy '{strategy}'. Choices: {list(STRATEGIES)}")
    if steps < 2 or steps > 40:
        raise HTTPException(400, detail="steps must be between 2 and 40.")
    if pv_max_kwc <= pv_min_kwc:
        raise HTTPException(400, detail="pv_max_kwc must be greater than pv_min_kwc.")

    location, frame = await _load_site_frame(site_name)

    try:
        battery_spec = BatterySpec(
            capacity_kwh=battery_capacity_kwh,
            max_charge_kw=max_charge_kw,
            max_discharge_kw=max_discharge_kw,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))

    load_kw = synthetic_residential_load(frame.index, daily_kwh=daily_kwh)

    step_size = (pv_max_kwc - pv_min_kwc) / (steps - 1)
    capacities = [round(pv_min_kwc + i * step_size, 3) for i in range(steps)]

    points = []
    crossover_kwc = None
    prev_diff = None
    for cap in capacities:
        pv_kw = ghi_to_pv_power(frame, capacity_kwc=cap)
        results = simulate_dispatch(pv_kw, load_kw, battery_spec, strategy=strategy)
        stats = summarize(results, battery_spec)
        points.append(
            {
                "pv_capacity_kwc": cap,
                "self_consumption_pct": stats["self_consumption_pct"],
                "autonomy_pct": stats["autonomy_pct"],
                "grid_import_kwh": stats["grid_import_kwh"],
                "grid_export_kwh": stats["grid_export_kwh"],
                "pv_kwh": stats["pv_kwh"],
            }
        )
        # Detect the self-consumption/autonomy crossover via sign change.
        diff = stats["self_consumption_pct"] - stats["autonomy_pct"]
        if prev_diff is not None and prev_diff * diff < 0:
            crossover_kwc = cap
        prev_diff = diff

    return {
        "site": site_name,
        "strategy": strategy,
        "battery_capacity_kwh": battery_capacity_kwh,
        "daily_kwh": daily_kwh,
        "points": points,
        "crossover_kwc": crossover_kwc,
    }


# ------------------------------------------------------------------ Frontend


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SolarCast</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    /* Base — deep night-sky blue, not neutral black */
    --bg: #0B1220;
    --surface: #121B2E;
    --surface-2: #16213A;
    --border: #22304A;
    /* Two accents mapped to the tool's two real domains */
    --accent: #F4A93B;     /* solar amber — irradiance, PV, day-side data */
    --accent-dim: rgba(244, 169, 59, 0.14);
    --accent2: #4FC3D9;    /* sky teal — battery, grid, flow data */
    --accent2-dim: rgba(79, 195, 217, 0.14);
    --text: #E7ECF5;
    --muted: #8B96AC;
    --success: #4FD68C;
    --error: #E15554;
    --warn: #F4A93B;
    --radius: 10px;
    --font: 'Inter', system-ui, sans-serif;
    --font-display: 'Space Grotesk', system-ui, sans-serif;
    --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: 24px;
    padding: 14px 28px; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .logo { font-family: var(--font-display); font-size: 21px; font-weight: 700; letter-spacing: -0.3px; }
  .logo span { color: var(--accent); }
  .tagline { color: var(--muted); font-size: 12.5px; margin-left: auto; font-family: var(--font-display); letter-spacing: 0.2px; }

  /* Sun-arc signature widget */
  .sun-widget { display: none; align-items: center; gap: 12px; padding-left: 20px; border-left: 1px solid var(--border); }
  .sun-widget.visible { display: flex; }
  .sun-widget svg { display: block; }
  .sun-meta { display: flex; flex-direction: column; gap: 2px; font-family: var(--font-mono); }
  .sun-site { font-size: 12px; color: var(--text); font-weight: 500; }
  .sun-clock { font-size: 15px; color: var(--accent); font-weight: 600; letter-spacing: 0.3px; }
  .sun-times { font-size: 10.5px; color: var(--muted); }

  .layout { display: grid; grid-template-columns: 300px 1fr; gap: 0; height: calc(100vh - 57px); }

  /* Sidebar */
  .sidebar {
    background: var(--surface); border-right: 1px solid var(--border);
    padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px;
  }
  .section-title { font-family: var(--font-display); font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--muted); margin-bottom: 12px; }

  .field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; }
  .field label { font-size: 12px; color: var(--muted); }
  .field select, .field input {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 8px 10px; font-size: 13px; width: 100%;
    outline: none; transition: border-color .15s; font-family: var(--font);
  }
  .field select:focus, .field input:focus { border-color: var(--accent); }

  .btn {
    width: 100%; padding: 10px; border-radius: 6px; border: none; cursor: pointer;
    font-size: 13px; font-weight: 600; transition: opacity .15s; font-family: var(--font);
  }
  .btn:hover { opacity: 0.85; }
  .btn-primary { background: var(--accent); color: #0B1220; }
  .btn-secondary { background: var(--border); color: var(--text); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Log */
  .log {
    background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 10px; height: 160px; overflow-y: auto; font-family: var(--font-mono); font-size: 11.5px;
  }
  .log-line { padding: 2px 0; display: flex; gap: 8px; }
  .log-time { color: var(--muted); flex-shrink: 0; }
  .log-msg { }
  .log-msg.success { color: var(--success); }
  .log-msg.error { color: var(--error); }
  .log-msg.info { color: var(--text); }

  /* Stats */
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; text-align: center;
  }
  .stat-value { font-family: var(--font-mono); font-size: 19px; font-weight: 600; color: var(--accent); }
  .stat-label { font-size: 10.5px; color: var(--muted); margin-top: 3px; text-transform: uppercase; letter-spacing: 0.4px; }

  /* Main */
  .main { padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }

  .card {
    background: var(--surface); border: 1px solid var(--border); border-top: 2px solid var(--border);
    border-radius: var(--radius); padding: 20px;
  }
  .card--solar { border-top-color: var(--accent); }
  .card--sky { border-top-color: var(--accent2); }
  .eyebrow {
    font-family: var(--font-display); font-size: 10.5px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.4px; color: var(--muted); margin-bottom: 4px; display: block;
  }
  .card--solar .eyebrow { color: var(--accent); }
  .card--sky .eyebrow { color: var(--accent2); }
  .card-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 16px; }
  .card-title { font-family: var(--font-display); font-size: 15px; font-weight: 600; }

  .chart-controls { display: flex; gap: 8px; align-items: center; }
  .chip {
    padding: 4px 10px; border-radius: 20px; font-size: 12px; border: 1px solid var(--border);
    cursor: pointer; background: transparent; color: var(--muted); transition: all .15s; font-family: var(--font);
  }
  .chip.active { background: var(--accent); color: #0B1220; border-color: var(--accent); font-weight: 600; }

  .chart-wrap { position: relative; height: 260px; }
  .empty { display: flex; align-items: center; justify-content: center; height: 260px;
    color: var(--muted); font-size: 13px; border: 1px dashed var(--border); border-radius: 8px; }

  .sites-list { display: flex; flex-direction: column; gap: 8px; }
  .site-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; cursor: pointer; transition: border-color .15s;
  }
  .site-row:hover, .site-row.active { border-color: var(--accent); }
  .site-name { font-weight: 600; font-size: 13px; }
  .site-coords { font-size: 11px; color: var(--muted); font-family: var(--font-mono); }
  .site-badge {
    font-family: var(--font-mono); font-size: 11px; padding: 2px 8px; border-radius: 20px;
    background: var(--accent); color: #0B1220; font-weight: 600;
  }
  .site-badge.empty { background: var(--border); color: var(--muted); }

  .resample-row { display: flex; gap: 6px; flex-wrap: wrap; }
</style>
</head>
<body>

<header>
  <div class="logo">Solar<span>Cast</span></div>

  <div class="sun-widget" id="sun-widget">
    <svg viewBox="0 0 100 56" width="100" height="56">
      <line x1="6" y1="46" x2="94" y2="46" stroke="var(--border)" stroke-width="1"/>
      <path d="M6,46 A44,44 0 0,1 94,46" fill="none" stroke="var(--border)" stroke-width="1.5" stroke-dasharray="2,3"/>
      <circle id="sun-dot" cx="6" cy="46" r="4" fill="var(--accent)" style="display:none"/>
    </svg>
    <div class="sun-meta">
      <div class="sun-site" id="sun-site-name">—</div>
      <div class="sun-clock" id="sun-clock">--:--</div>
      <div class="sun-times" id="sun-times">sunrise — · sunset —</div>
    </div>
  </div>

  <div class="tagline">Solar &amp; battery forecasting for Morocco</div>
</header>

<div class="layout">
  <!-- Sidebar -->
  <div class="sidebar">
    <div>
      <div class="section-title">Ingest Data</div>
      <div class="field">
        <label>Provider</label>
        <select id="sel-provider">
          <option value="">Loading...</option>
        </select>
      </div>
      <div class="field">
        <label>Site</label>
        <select id="sel-site">
          <option value="">Loading...</option>
        </select>
      </div>
      <div class="field">
        <label>Start date</label>
        <input type="date" id="inp-start" value="2024-01-01">
      </div>
      <div class="field">
        <label>End date</label>
        <input type="date" id="inp-end" value="2024-12-31">
      </div>
      <button class="btn btn-primary" id="btn-ingest">Run Ingestion</button>
    </div>

    <div>
      <div class="section-title">Live Log</div>
      <div class="log" id="log-box">
        <div class="log-line"><span class="log-msg info">Ready — configure and run an ingestion above.</span></div>
      </div>
    </div>

    <div>
      <div class="section-title">Site Stats</div>
      <div class="stats-grid" id="stats-grid">
        <div class="stat-card"><div class="stat-value">—</div><div class="stat-label">Total points</div></div>
        <div class="stat-card"><div class="stat-value">—</div><div class="stat-label">Variables</div></div>
        <div class="stat-card"><div class="stat-value">—</div><div class="stat-label">Period start</div></div>
        <div class="stat-card"><div class="stat-value">—</div><div class="stat-label">Period end</div></div>
      </div>
    </div>
  </div>

  <!-- Main -->
  <div class="main">
    <!-- Sites -->
    <div class="card card--solar">
      <div class="card-header">
        <div>
          <span class="eyebrow">Registered locations</span>
          <div class="card-title">Sites in Database</div>
        </div>
        <button class="btn btn-secondary" style="width:auto;padding:6px 14px" onclick="loadSites()">Refresh</button>
      </div>
      <div class="sites-list" id="sites-list">
        <div style="color:var(--muted);font-size:13px">Loading...</div>
      </div>
    </div>

    <!-- Chart -->
    <div class="card card--solar">
      <div class="card-header">
        <div>
          <span class="eyebrow">Measured history</span>
          <div class="card-title" id="chart-title">Select a site to view data</div>
        </div>
        <div class="chart-controls">
          <div class="resample-row" id="resample-row">
            <button class="chip active" onclick="setResample('raw',this)">Raw</button>
            <button class="chip" onclick="setResample('daily',this)">Daily avg</button>
            <button class="chip" onclick="setResample('monthly',this)">Monthly</button>
          </div>
        </div>
      </div>
      <div class="chart-controls" style="margin-bottom:12px;gap:6px" id="var-chips"></div>
      <div id="chart-area"><div class="empty">Select a site from the list above</div></div>
    </div>

    <!-- Forecast -->
    <div class="card card--solar">
      <div class="card-header">
        <div>
          <span class="eyebrow">Machine learning</span>
          <div class="card-title">Forecast — predicted vs actual (GHI)</div>
        </div>
        <div class="chart-controls" style="gap:8px">
          <select id="sel-forecast-model" style="background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:6px 10px;font-size:12px">
            <option value="gradient_boosting">Gradient Boosting</option>
            <option value="random_forest">Random Forest</option>
            <option value="ridge">Ridge</option>
            <option value="clear_sky">Clear Sky</option>
            <option value="persistence">Persistence</option>
          </select>
          <button class="btn btn-primary" style="width:auto;padding:6px 16px" id="btn-forecast">Train &amp; Forecast</button>
        </div>
      </div>
      <div id="forecast-skill" style="display:none;gap:12px;margin-bottom:14px" class="stats-grid"></div>
      <div id="forecast-area"><div class="empty">Pick a site above, then click "Train &amp; Forecast"</div></div>
    </div>

    <!-- Battery Simulation -->
    <div class="card card--sky">
      <div class="card-header">
        <div>
          <span class="eyebrow">Physical simulation</span>
          <div class="card-title">Battery dispatch simulation</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px">
        <div class="field">
          <label>PV capacity (kWc)</label>
          <input type="number" id="sim-pv" value="6" min="0.5" step="0.5">
        </div>
        <div class="field">
          <label>Load (kWh/day)</label>
          <input type="number" id="sim-load" value="10" min="1" step="0.5">
        </div>
        <div class="field">
          <label>Battery (kWh)</label>
          <input type="number" id="sim-battery" value="10" min="1" step="0.5">
        </div>
        <div class="field">
          <label>Charge/discharge (kW)</label>
          <input type="number" id="sim-power" value="3" min="0.5" step="0.5">
        </div>
        <div class="field">
          <label>Strategy</label>
          <select id="sim-strategy">
            <option value="self_consumption">Self-consumption</option>
            <option value="peak_shaving">Peak shaving</option>
          </select>
        </div>
      </div>
      <button class="btn btn-primary" style="width:auto;padding:8px 20px" id="btn-simulate">Run Simulation</button>

      <div id="sim-stats" style="display:none;gap:10px;margin:16px 0" class="stats-grid"></div>
      <div id="sim-area"></div>
    </div>

    <!-- Sizing Sweep -->
    <div class="card card--sky">
      <div class="card-header">
        <div>
          <span class="eyebrow">System sizing</span>
          <div class="card-title">PV sizing sweep — find the self-consumption / autonomy crossover</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px">
        <div class="field">
          <label>PV range (kWc)</label>
          <div style="display:flex;gap:6px">
            <input type="number" id="sweep-min" value="1" min="0.5" step="0.5" style="width:50%">
            <input type="number" id="sweep-max" value="8" min="1" step="0.5" style="width:50%">
          </div>
        </div>
        <div class="field">
          <label>Load (kWh/day)</label>
          <input type="number" id="sweep-load" value="10" min="1" step="0.5">
        </div>
        <div class="field">
          <label>Battery (kWh)</label>
          <input type="number" id="sweep-battery" value="5" min="1" step="0.5">
        </div>
        <div class="field">
          <label>&nbsp;</label>
          <button class="btn btn-primary" id="btn-sweep">Run Sweep</button>
        </div>
      </div>
      <div id="sweep-crossover" style="display:none;margin-bottom:12px;padding:10px 14px;background:var(--bg);border:1px solid var(--accent);border-radius:8px;font-size:13px"></div>
      <div id="sweep-area"><div class="empty">Pick a site above, then click "Run Sweep"</div></div>
    </div>
  </div>
</div>

<script>
let forecastChart = null;
let chart = null;
let currentSite = null;
let currentVar = 'ghi';
let currentResample = 'raw';
let allData = { labels: [], values: [] };
let siteCoords = {};      // name -> {latitude, longitude, timezone}
let sunTickHandle = null;

// ---------------------------------------------------------------- init
async function init() {
  const cfg = await fetch('/api/config').then(r => r.json());

  const pSel = document.getElementById('sel-provider');
  pSel.innerHTML = cfg.providers.map(p => `<option value="${p}">${p}</option>`).join('');

  const sSel = document.getElementById('sel-site');
  sSel.innerHTML = cfg.sites.map(s => `<option value="${s}">${s}</option>`).join('');

  await loadSites();
}

// ---------------------------------------------------------------- sun-arc
// Simplified solar geometry (Cooper's declination, hour angle corrected
// for longitude) — same family of formulas as the server-side synthetic
// model. Used only for the header widget; real forecasts and features
// go through pvlib on the Python side.
function sunPosition(lat, lon, utcDate) {
  const rad = Math.PI / 180;
  const dayOfYear = Math.floor((utcDate - new Date(Date.UTC(utcDate.getUTCFullYear(), 0, 0))) / 86400000);
  const decl = 23.45 * Math.sin(rad * 360 * (284 + dayOfYear) / 365);
  const solarTime = utcDate.getUTCHours() + utcDate.getUTCMinutes() / 60 + lon / 15;
  const hourAngle = 15 * (solarTime - 12);

  const latR = lat * rad, declR = decl * rad, haR = hourAngle * rad;
  const elevation = Math.asin(
    Math.sin(latR) * Math.sin(declR) + Math.cos(latR) * Math.cos(declR) * Math.cos(haR)
  ) / rad;

  const cosH0 = -Math.tan(latR) * Math.tan(declR);
  let sunriseSolar = 6, sunsetSolar = 18;
  if (cosH0 >= -1 && cosH0 <= 1) {
    const h0 = Math.acos(cosH0) / rad;
    sunriseSolar = 12 - h0 / 15;
    sunsetSolar = 12 + h0 / 15;
  }
  const fraction = (solarTime - sunriseSolar) / (sunsetSolar - sunriseSolar);

  return { elevation, fraction, sunriseSolar, sunsetSolar, solarTime, lon };
}

function fmtSolarHour(solarHour, lon) {
  // Reconvertit l'heure solaire approximative en heure UTC pour affichage.
  let utcHour = solarHour - lon / 15;
  utcHour = ((utcHour % 24) + 24) % 24;
  const h = Math.floor(utcHour), m = Math.round((utcHour - h) * 60);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;
}

function updateSunWidget() {
  const widget = document.getElementById('sun-widget');
  if (!currentSite || !siteCoords[currentSite]) { widget.classList.remove('visible'); return; }
  widget.classList.add('visible');

  const { latitude, longitude, timezone } = siteCoords[currentSite];
  const now = new Date();
  const pos = sunPosition(latitude, longitude, now);

  document.getElementById('sun-site-name').textContent = currentSite;

  try {
    const localTime = new Intl.DateTimeFormat('en-GB', {
      timeZone: timezone, hour: '2-digit', minute: '2-digit', hour12: false
    }).format(now);
    document.getElementById('sun-clock').textContent = localTime;
  } catch (e) {
    document.getElementById('sun-clock').textContent = now.toISOString().slice(11,16) + ' UTC';
  }

  document.getElementById('sun-times').textContent =
    `sunrise ~${fmtSolarHour(pos.sunriseSolar, longitude)} · sunset ~${fmtSolarHour(pos.sunsetSolar, longitude)}`;

  const dot = document.getElementById('sun-dot');
  if (pos.fraction >= 0 && pos.fraction <= 1) {
    const phi = (180 - pos.fraction * 180) * Math.PI / 180;
    const cx = 50, cy = 46, R = 44;
    const x = cx + R * Math.cos(phi);
    const y = cy - R * Math.sin(phi);
    dot.setAttribute('cx', x.toFixed(1));
    dot.setAttribute('cy', y.toFixed(1));
    dot.style.display = 'block';
  } else {
    dot.style.display = 'none';
  }
}

function startSunClock() {
  if (sunTickHandle) clearInterval(sunTickHandle);
  updateSunWidget();
  sunTickHandle = setInterval(updateSunWidget, 60000);
}

// ---------------------------------------------------------------- sites
async function loadSites() {
  const res = await fetch('/api/sites').then(r => r.json());
  const el = document.getElementById('sites-list');

  if (!res.length) {
    el.innerHTML = '<div style="color:var(--muted);font-size:13px">No sites in database yet. Run an ingestion first.</div>';
    return;
  }

  el.innerHTML = '';
  for (const s of res) {
    siteCoords[s.name] = { latitude: s.latitude, longitude: s.longitude, timezone: s.timezone || 'UTC' };
    const stats = await fetch(`/api/sites/${s.name}/stats`).then(r => r.json());
    const hasData = stats.total_points > 0;
    const div = document.createElement('div');
    div.className = 'site-row' + (currentSite === s.name ? ' active' : '');
    div.innerHTML = `
      <div>
        <div class="site-name">${s.name}</div>
        <div class="site-coords">${s.latitude.toFixed(4)}°N, ${s.longitude.toFixed(4)}°E · ${s.altitude_m ?? '?'} m</div>
      </div>
      <span class="site-badge ${hasData ? '' : 'empty'}">${hasData ? stats.total_points.toLocaleString() + ' pts' : 'no data'}</span>
    `;
    div.onclick = () => selectSite(s.name);
    el.appendChild(div);
  }
}

// ---------------------------------------------------------------- select site
async function selectSite(name) {
  currentSite = name;
  document.querySelectorAll('.site-row').forEach(r => r.classList.remove('active'));
  event.currentTarget.classList.add('active');

  startSunClock();

  const stats = await fetch(`/api/sites/${name}/stats`).then(r => r.json());
  updateStats(stats);
  buildVarChips(stats.variables || []);

  if (stats.variables && stats.variables.length) {
    currentVar = stats.variables.includes('ghi') ? 'ghi' : stats.variables[0];
    updateVarChips();
    await loadChart();
  }
}

function updateStats(stats) {
  const grid = document.getElementById('stats-grid');
  const fmt = v => v ? v.slice(0, 10) : '—';
  grid.innerHTML = `
    <div class="stat-card"><div class="stat-value">${stats.total_points ? stats.total_points.toLocaleString() : '—'}</div><div class="stat-label">Total points</div></div>
    <div class="stat-card"><div class="stat-value">${stats.variables ? stats.variables.length : '—'}</div><div class="stat-label">Variables</div></div>
    <div class="stat-card"><div class="stat-value">${fmt(stats.period_start)}</div><div class="stat-label">Period start</div></div>
    <div class="stat-card"><div class="stat-value">${fmt(stats.period_end)}</div><div class="stat-label">Period end</div></div>
  `;
}

function buildVarChips(variables) {
  const el = document.getElementById('var-chips');
  el.innerHTML = variables.map(v =>
    `<button class="chip ${v === currentVar ? 'active' : ''}" onclick="selectVar('${v}',this)">${v}</button>`
  ).join('');
}

function updateVarChips() {
  document.querySelectorAll('#var-chips .chip').forEach(c => {
    c.classList.toggle('active', c.textContent === currentVar);
  });
}

async function selectVar(v, el) {
  currentVar = v;
  document.querySelectorAll('#var-chips .chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  await loadChart();
}

function setResample(mode, el) {
  currentResample = mode;
  document.querySelectorAll('#resample-row .chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  if (allData.labels.length) renderChart(allData.labels, allData.values);
}

// ---------------------------------------------------------------- chart
async function loadChart() {
  if (!currentSite) return;
  document.getElementById('chart-title').textContent = `${currentSite} — ${currentVar}`;
  document.getElementById('chart-area').innerHTML = '<div class="empty">Loading...</div>';

  const data = await fetch(`/api/sites/${currentSite}/data?variable=${currentVar}`).then(r => r.json());
  allData = data;

  if (!data.labels.length) {
    document.getElementById('chart-area').innerHTML = '<div class="empty">No data for this variable</div>';
    return;
  }
  renderChart(data.labels, data.values);
}

function resampleData(labels, values) {
  if (currentResample === 'raw') return { labels, values };

  const buckets = {};
  labels.forEach((l, i) => {
    const d = new Date(l);
    const key = currentResample === 'daily'
      ? d.toISOString().slice(0, 10)
      : d.toISOString().slice(0, 7);
    if (!buckets[key]) buckets[key] = [];
    if (values[i] !== null) buckets[key].push(values[i]);
  });

  const outLabels = Object.keys(buckets).sort();
  const outValues = outLabels.map(k => {
    const arr = buckets[k];
    return arr.length ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : null;
  });
  return { labels: outLabels, values: outValues };
}

function renderChart(labels, values) {
  const { labels: L, values: V } = resampleData(labels, values);

  const area = document.getElementById('chart-area');
  area.innerHTML = '<div class="chart-wrap"><canvas id="myChart"></canvas></div>';
  const ctx = document.getElementById('myChart').getContext('2d');

  if (chart) chart.destroy();

  const units = { ghi: 'W/m²', dni: 'W/m²', dhi: 'W/m²', temp_air: '°C',
    wind_speed: 'm/s', relative_humidity: '%', cloud_cover: '%', precipitation: 'mm' };

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: L,
      datasets: [{
        label: `${currentVar} (${units[currentVar] || ''})`,
        data: V,
        borderColor: '#F4A93B',
        backgroundColor: 'rgba(244,169,59,0.08)',
        borderWidth: currentResample === 'raw' ? 1 : 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#8B96AC', font: { size: 12 } } },
        tooltip: { backgroundColor: '#121B2E', borderColor: '#22304A', borderWidth: 1,
          titleColor: '#E7ECF5', bodyColor: '#8B96AC' }
      },
      scales: {
        x: { ticks: { color: '#8B96AC', maxTicksLimit: 10, font: { size: 11 } },
          grid: { color: '#1A2540' } },
        y: { ticks: { color: '#8B96AC', font: { size: 11 } },
          grid: { color: '#1A2540' } }
      }
    }
  });
}

// ---------------------------------------------------------------- ingestion
document.getElementById('btn-ingest').onclick = async () => {
  const provider = document.getElementById('sel-provider').value;
  const site = document.getElementById('sel-site').value;
  const start = document.getElementById('inp-start').value;
  const end = document.getElementById('inp-end').value;

  if (!provider || !site || !start || !end) return;

  const btn = document.getElementById('btn-ingest');
  btn.disabled = true;
  btn.textContent = 'Running...';
  const log = document.getElementById('log-box');
  log.innerHTML = '';

  const url = `/api/ingest/stream?provider=${provider}&site=${encodeURIComponent(site)}&start=${start}&end=${end}`;
  const es = new EventSource(url);

  es.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.message === '__DONE__') {
      es.close();
      btn.disabled = false;
      btn.textContent = 'Run Ingestion';
      loadSites();
      if (currentSite === site) loadChart();
      return;
    }
    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = `<span class="log-time">${d.time}</span><span class="log-msg ${d.level}">${d.message}</span>`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  };

  es.onerror = () => {
    es.close();
    btn.disabled = false;
    btn.textContent = 'Run Ingestion';
  };
};

// ---------------------------------------------------------------- forecast
document.getElementById('btn-forecast').onclick = async () => {
  if (!currentSite) {
    alert('Pick a site from the list above first.');
    return;
  }
  const model = document.getElementById('sel-forecast-model').value;
  const btn = document.getElementById('btn-forecast');
  btn.disabled = true;
  btn.textContent = 'Training...';
  document.getElementById('forecast-area').innerHTML = '<div class="empty">Training model on recent history — usually 10-30 seconds...</div>';

  try {
    const res = await fetch(`/api/sites/${currentSite}/forecast?model=${model}&test_days=14`);
    if (!res.ok) {
      const err = await res.json();
      document.getElementById('forecast-area').innerHTML = `<div class="empty">${err.detail || 'Error running forecast'}</div>`;
      return;
    }
    const data = await res.json();
    renderForecast(data);
  } catch (e) {
    document.getElementById('forecast-area').innerHTML = '<div class="empty">Request failed. Check the server logs.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Train & Forecast';
  }
};

function renderForecast(data) {
  const skillBox = document.getElementById('forecast-skill');
  const m = data.metrics;
  const fmt = (v, d=1) => (v === null || v === undefined) ? '—' : v.toFixed(d);
  skillBox.style.display = 'grid';
  skillBox.style.gridTemplateColumns = 'repeat(4, 1fr)';
  skillBox.innerHTML = `
    <div class="stat-card"><div class="stat-value">${fmt(m.rmse)}</div><div class="stat-label">RMSE (W/m²)</div></div>
    <div class="stat-card"><div class="stat-value">${fmt(m.mae)}</div><div class="stat-label">MAE (W/m²)</div></div>
    <div class="stat-card"><div class="stat-value">${fmt(m.nrmse, 3)}</div><div class="stat-label">nRMSE</div></div>
    <div class="stat-card"><div class="stat-value" style="color:${(m.skill_score||0) > 0 ? 'var(--success)' : 'var(--error)'}">${fmt(m.skill_score, 3)}</div><div class="stat-label">Skill vs persistence</div></div>
  `;

  const area = document.getElementById('forecast-area');
  area.innerHTML = '<div class="chart-wrap"><canvas id="forecastCanvas"></canvas></div>';
  const ctx = document.getElementById('forecastCanvas').getContext('2d');

  if (forecastChart) forecastChart.destroy();

  const datasets = [
    {
      label: 'Actual',
      data: data.actual,
      borderColor: '#4FC3D9',
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2,
    },
    {
      label: `Predicted (${data.model})`,
      data: data.predicted,
      borderColor: '#F4A93B',
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      borderDash: [4, 3],
      pointRadius: 0,
      tension: 0.2,
    },
  ];

  forecastChart = new Chart(ctx, {
    type: 'line',
    data: { labels: data.labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#8B96AC', font: { size: 12 } } },
        tooltip: { backgroundColor: '#121B2E', borderColor: '#22304A', borderWidth: 1,
          titleColor: '#E7ECF5', bodyColor: '#8B96AC' }
      },
      scales: {
        x: { ticks: { color: '#8B96AC', maxTicksLimit: 10, font: { size: 11 } }, grid: { color: '#1A2540' } },
        y: { ticks: { color: '#8B96AC', font: { size: 11 } }, grid: { color: '#1A2540' }, title: { display: true, text: 'W/m²', color: '#8B96AC' } }
      }
    }
  });
}

// ---------------------------------------------------------------- simulation
let simDailyChart = null;
let simWeekChart = null;

document.getElementById('btn-simulate').onclick = async () => {
  if (!currentSite) { alert('Pick a site from the list above first.'); return; }

  const params = new URLSearchParams({
    pv_capacity_kwc: document.getElementById('sim-pv').value,
    daily_kwh: document.getElementById('sim-load').value,
    battery_capacity_kwh: document.getElementById('sim-battery').value,
    max_charge_kw: document.getElementById('sim-power').value,
    max_discharge_kw: document.getElementById('sim-power').value,
    strategy: document.getElementById('sim-strategy').value,
  });

  const btn = document.getElementById('btn-simulate');
  btn.disabled = true;
  btn.textContent = 'Running...';
  document.getElementById('sim-area').innerHTML = '<div class="empty">Simulating...</div>';

  try {
    const res = await fetch(`/api/sites/${currentSite}/simulate?${params}`);
    if (!res.ok) {
      const err = await res.json();
      document.getElementById('sim-area').innerHTML = `<div class="empty">${err.detail || 'Error'}</div>`;
      return;
    }
    const data = await res.json();
    renderSimulation(data);
  } catch (e) {
    document.getElementById('sim-area').innerHTML = '<div class="empty">Request failed.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Simulation';
  }
};

function renderSimulation(data) {
  const s = data.stats;
  const statsBox = document.getElementById('sim-stats');
  statsBox.style.display = 'grid';
  statsBox.style.gridTemplateColumns = 'repeat(4, 1fr)';
  statsBox.innerHTML = `
    <div class="stat-card"><div class="stat-value">${s.self_consumption_pct}%</div><div class="stat-label">Self-consumption</div></div>
    <div class="stat-card"><div class="stat-value">${s.autonomy_pct}%</div><div class="stat-label">Autonomy</div></div>
    <div class="stat-card"><div class="stat-value">${s.grid_import_kwh.toLocaleString()}</div><div class="stat-label">Grid import (kWh)</div></div>
    <div class="stat-card"><div class="stat-value">${s.grid_export_kwh.toLocaleString()}</div><div class="stat-label">Grid export (kWh)</div></div>
  `;

  const area = document.getElementById('sim-area');
  area.innerHTML = `
    <div style="margin-bottom:8px;font-size:12px;color:var(--muted)">Daily energy flow — full period</div>
    <div class="chart-wrap" style="height:220px"><canvas id="simDailyCanvas"></canvas></div>
    <div style="margin:16px 0 8px;font-size:12px;color:var(--muted)">Battery SOC & power — last 7 days</div>
    <div class="chart-wrap" style="height:220px"><canvas id="simWeekCanvas"></canvas></div>
  `;

  if (simDailyChart) simDailyChart.destroy();
  if (simWeekChart) simWeekChart.destroy();

  const dailyCtx = document.getElementById('simDailyCanvas').getContext('2d');
  simDailyChart = new Chart(dailyCtx, {
    type: 'line',
    data: {
      labels: data.daily.labels,
      datasets: [
        { label: 'PV (kWh)', data: data.daily.pv_kwh, borderColor: '#F4A93B', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.2 },
        { label: 'Grid import (kWh)', data: data.daily.grid_import_kwh, borderColor: '#E15554', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.2 },
        { label: 'Grid export (kWh)', data: data.daily.grid_export_kwh, borderColor: '#4FC3D9', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.2 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8B96AC', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#8B96AC', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#1A2540' } },
        y: { ticks: { color: '#8B96AC', font: { size: 10 } }, grid: { color: '#1A2540' } }
      }
    }
  });

  const weekCtx = document.getElementById('simWeekCanvas').getContext('2d');
  simWeekChart = new Chart(weekCtx, {
    data: {
      labels: data.week.labels,
      datasets: [
        { type: 'line', label: 'SOC (%)', data: data.week.soc_pct, borderColor: '#4FD68C', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.2, yAxisID: 'y1' },
        { type: 'line', label: 'PV (kW)', data: data.week.pv_kw, borderColor: '#F4A93B', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.2, yAxisID: 'y' },
        { type: 'line', label: 'Load (kW)', data: data.week.load_kw, borderColor: '#8B96AC', backgroundColor: 'transparent', borderWidth: 1, borderDash: [3,3], pointRadius: 0, tension: 0.2, yAxisID: 'y' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8B96AC', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#8B96AC', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#1A2540' } },
        y: { position: 'left', ticks: { color: '#8B96AC', font: { size: 10 } }, grid: { color: '#1A2540' }, title: { display: true, text: 'kW', color: '#8B96AC' } },
        y1: { position: 'right', min: 0, max: 100, ticks: { color: '#4FD68C', font: { size: 10 } }, grid: { display: false }, title: { display: true, text: 'SOC %', color: '#4FD68C' } }
      }
    }
  });
}

// ---------------------------------------------------------------- sweep
let sweepChart = null;

document.getElementById('btn-sweep').onclick = async () => {
  if (!currentSite) { alert('Pick a site from the list above first.'); return; }

  const params = new URLSearchParams({
    pv_min_kwc: document.getElementById('sweep-min').value,
    pv_max_kwc: document.getElementById('sweep-max').value,
    daily_kwh: document.getElementById('sweep-load').value,
    battery_capacity_kwh: document.getElementById('sweep-battery').value,
    steps: 12,
  });

  const btn = document.getElementById('btn-sweep');
  btn.disabled = true;
  btn.textContent = 'Sweeping...';
  document.getElementById('sweep-area').innerHTML = '<div class="empty">Running sweep...</div>';

  try {
    const res = await fetch(`/api/sites/${currentSite}/simulate/sweep?${params}`);
    if (!res.ok) {
      const err = await res.json();
      document.getElementById('sweep-area').innerHTML = `<div class="empty">${err.detail || 'Error'}</div>`;
      return;
    }
    const data = await res.json();
    renderSweep(data);
  } catch (e) {
    document.getElementById('sweep-area').innerHTML = '<div class="empty">Request failed.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Sweep';
  }
};

function renderSweep(data) {
  const crossoverBox = document.getElementById('sweep-crossover');
  if (data.crossover_kwc !== null) {
    crossoverBox.style.display = 'block';
    crossoverBox.innerHTML = `Self-consumption and autonomy curves cross near <strong style="color:var(--accent)">${data.crossover_kwc} kWc</strong> — the sizing sweet spot for this load and battery, given no grid injection compensation.`;
  } else {
    crossoverBox.style.display = 'block';
    crossoverBox.innerHTML = `No crossover found in this PV range — try widening it.`;
  }

  const area = document.getElementById('sweep-area');
  area.innerHTML = '<div class="chart-wrap"><canvas id="sweepCanvas"></canvas></div>';
  const ctx = document.getElementById('sweepCanvas').getContext('2d');

  if (sweepChart) sweepChart.destroy();

  sweepChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.points.map(p => p.pv_capacity_kwc),
      datasets: [
        { label: 'Self-consumption (%)', data: data.points.map(p => p.self_consumption_pct), borderColor: '#F4A93B', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, tension: 0.3 },
        { label: 'Autonomy (%)', data: data.points.map(p => p.autonomy_pct), borderColor: '#4FC3D9', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, tension: 0.3 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8B96AC', font: { size: 12 } } } },
      scales: {
        x: { title: { display: true, text: 'PV capacity (kWc)', color: '#8B96AC' }, ticks: { color: '#8B96AC', font: { size: 11 } }, grid: { color: '#1A2540' } },
        y: { min: 0, max: 100, ticks: { color: '#8B96AC', font: { size: 11 } }, grid: { color: '#1A2540' }, title: { display: true, text: '%', color: '#8B96AC' } }
      }
    }
  });
}

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTML
