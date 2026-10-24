"""GET /api/ready - hosting readiness for the web runtime (deployment audit).

Reports, without any secret value:
  * alive + mode + version,
  * the runtime assets the web request path reads (traced with an audit hook over health / catalogue / WALK / COMPARE /
    TRACK / REPLAY in engine mode): present or missing, with sizes,
  * engines loadable (engine mode only: loading them here is the safe warm-up; the first WALK / COMPARE / TRACK are then
    served from warm caches),
  * realtime configuration: whether the TfNSW key variable is set in the API process's environment (true / false, never
    the value, never the variable name in the response - the app's secret guard refuses it), the replay sources available, the planned-shape cache coverage for the services' trips.
Status: "ready" when every required asset is present and (engine mode) the engines loaded; otherwise "not_ready" with
the missing items named.  The endpoint changes nothing on disk.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

from . import catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_KEY_VARIABLE = "TFNSW_API_KEY"

# (path, required, purpose) - the files the web request path opens at runtime (scripts/audit_web_runtime_bundle.py traces them)
RUNTIME_ASSETS: List[tuple] = [
    ("data/cache/glebe_walk.graphml", True, "walking graph"),
    ("data/cache/terrain_cost/edge_terrain_cost.csv", True, "terrain edge costs (Tobler / Minetti inputs)"),
    ("data/cache/stair_model/stair_costs.csv", True, "stair model"),
    ("data/cache/stair_model/metadata.json", True, "stair model metadata"),
    ("data/cache/mapped_steps.json", True, "mapped stairs"),
    ("data/cache/glebe_stops_snapped.csv", True, "candidate stops"),
    ("data/boundaries/glebe_suburb.geojson", True, "study area"),
    ("data/gtfs/trips.txt", True, "static GTFS trips (route / direction / shape_id)"),
    ("data/gtfs/routes.txt", True, "static GTFS routes"),
    ("data/gtfs/feed_provenance.json", True, "static snapshot identity"),
    ("results/service_intent/candidate_stop_trips.csv", True, "exact trips per candidate stop"),
    ("results/service_intent/candidate_stop_services.csv", True, "route + direction choices"),
    ("results/service_intent/service_calendar.csv", True, "service calendar"),
    ("results/service_intent/service_calendar_dates.csv", True, "service calendar exceptions"),
    ("results/service_intent/provenance.json", True, "service intent provenance"),
    ("results/service_intent/inspection.json", True, "service intent inspection"),
    ("data/cache/basemap/inner_west_basemap.json", False, "street basemap (display)"),
    ("data/cache/terrain_display/terrain_metadata.json", False, "terrain display surface (display)"),
    ("data/cache/terrain_display/elevation_tint.png", False, "terrain display surface (display)"),
    ("data/cache/terrain_display/hillshade.png", False, "terrain display surface (display)"),
    ("data/cache/terrain_display/major_contours.geojson", False, "terrain display surface (display)"),
    ("data/cache/gtfs_shapes", False, "planned-shape cache (TRACK; scripts/build_shape_cache.py)"),
]


def asset_status() -> Dict[str, Any]:
    present, missing = [], []
    for rel, required, purpose in RUNTIME_ASSETS:
        path = PROJECT_ROOT / rel
        ok = path.exists()
        entry = {"path": rel, "required": required, "purpose": purpose,
                 "bytes": (sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size) if ok else None}
        (present if ok else missing).append(entry)
    return {"present": present, "missing": missing,
            "required_missing": [m["path"] for m in missing if m["required"]]}


def shape_cache_status(app) -> Dict[str, Any]:
    """How many of the services' shape_ids the cache already holds (the rest would stream shapes.txt from the zip, if present)."""
    try:
        import pandas as pd
        store = app.host.shape_store
        trips = pd.read_csv(PROJECT_ROOT / "results" / "service_intent" / "candidate_stop_trips.csv", usecols=["trip_id"], dtype=str)
        mapping = store.load_trip_shape_ids()
        needed = {mapping[t] for t in set(trips["trip_id"]) if t in mapping}
        cached = {s for s in needed if store.cache_path(s).exists()}
        source, kind = store.source()
        return {"shape_ids_needed": len(needed), "cached": len(cached), "missing": sorted(needed - cached),
                "shapes_source_available": source is not None, "shapes_source_kind": kind,
                "note": "a missing shape with no shapes.txt source makes TRACK report the planned route as unavailable (no fake line)"}
    except Exception as error:   # noqa: BLE001 - readiness must not fail on a diagnostic
        return {"error": type(error).__name__}


def report(app) -> Dict[str, Any]:
    started = time.perf_counter()
    assets = asset_status()
    engines: Dict[str, Any] = {"mode": app.mode}
    if app.mode == "engine":
        try:
            t = time.perf_counter()
            facts = app.host.graph_facts()                   # loads the engines on the first call: the warm-up
            engines.update({"loaded": True, "load_seconds_this_call": round(time.perf_counter() - t, 2), "graph": facts})
        except Exception as error:   # noqa: BLE001
            engines.update({"loaded": False, "error": type(error).__name__})
    replays = [r for r in catalog.replays()["replays"] if r["available"]]
    terrain = catalog.terrain()
    realtime = {"api_key_configured": bool((os.environ.get(API_KEY_VARIABLE) or "").strip()),
                "live_note": "LIVE fetches TfNSW feeds from the API process only when a request asks for run_mode LIVE; the key is "
                             "read from the process environment / .env (the TfNSW variable) and never returned",
                "replays_available": [r["name"] for r in replays]}
    shapes = shape_cache_status(app) if app.mode == "engine" and engines.get("loaded") else {"skipped": "engine mode only"}
    ready = not assets["required_missing"] and (app.mode != "engine" or engines.get("loaded", False))
    return {"status": "ready" if ready else "not_ready", "mode": app.mode,
            "assets": assets, "engines": engines, "realtime": realtime,
            "terrain_display": {"available": terrain.get("available", False)},
            "shape_cache": shapes, "replay_fallback": bool(replays),
            "report_seconds": round(time.perf_counter() - started, 2)}
