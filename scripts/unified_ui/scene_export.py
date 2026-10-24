"""Scene export from the unified window - the existing schema and exporters only.

WALK: the same ``analysis_level = "walking_only"`` scene the Week 9 primary
window writes (origin, all candidate stops, the selected stop's three routes,
``route_identity``, ``selected_stop``), written with
``phase2_decision.scene_to_geojson`` / ``write_scene`` (schema 1.1).

COMPARE: the Phase 2 scene the engine returned (selected trip, trip
candidates, per-stop catchability, highlights, Pareto), with
``selected_stop`` set to the stop being inspected, written with the same two
functions. No second schema exists.

TRACK: the same Phase 2 scene plus the schema-1.1 ``transit`` block the Week 9
primary window writes while a trip is tracked (``planned_route`` = static
GTFS shape as [lon, lat], ``reported_positions`` = raw VehiclePosition
reports, ``tracking_state``), which ``scene_to_geojson`` already turns into
the ``planned_transit_route`` / ``reported_vehicle_position`` features.
WGS84 only; no projector coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .walk_engine import WalkEngine, phase2

DEFAULT_OUT_DIR = phase2.RESULTS_DIR          # results/ - what the Sandbox reads today


def walk_scene(state, engine: WalkEngine, mode_label: str) -> Optional[Dict[str, Any]]:
    """The stop-first (walking-only) scene, field-compatible with the primary window's."""
    origin = state.origin
    if origin is None:
        return None
    result = state.walking_result
    selected = state.selected_stop if result else None
    stops_out = []
    for stop in engine.stops.itertuples(index=False):
        stop_id = str(stop.stop_id)
        entry = {
            "stop_id": stop_id, "stop_name": stop.stop_name, "mode": getattr(stop, "mode", ""),
            "lat": float(stop.stop_lat), "lon": float(stop.stop_lon),
            "eligibility": None, "status_reason": "", "serves_selected_service": None,
            "effort_rank": None, "near_tie_group": None, "effort_j_per_kg": None, "distance_m": None,
            "time_route_walking_time_s": None, "time_route_walking_time_min": None,
            "routes": {"distance": None, "effort": None, "time": None},
            "route_identity": None, "catchability": None,
            "pareto_status": phase2.PARETO_NOT_COMPARED, "dominated_by": [], "highlights": [],
        }
        if result and stop_id == selected:
            routes = result.get("routes") or {}
            entry["eligibility"] = "compared" if result.get("available") else "no_trusted_cost_path"
            entry["status_reason"] = result.get("reason") or ""
            for obj in ("distance", "effort", "time"):
                entry["routes"][obj] = phase2.route_summary(routes.get(obj))
            e, t = routes.get("effort"), routes.get("time")
            entry["effort_j_per_kg"] = round(e["effort"], 4) if e else None
            entry["distance_m"] = round(e["distance"], 3) if e else None
            if t is not None:
                entry["time_route_walking_time_s"] = round(t["walking_time_s"], 3)
                entry["time_route_walking_time_min"] = round(t["walking_time_s"] / 60.0, 3)
            entry["route_identity"] = result.get("route_identity")
            entry["effort_saving_vs_shortest_pct"] = result.get("effort_saving_vs_shortest_pct")
            entry["extra_distance_vs_shortest_pct"] = result.get("extra_distance_vs_shortest_pct")
        stops_out.append(entry)
    return {
        "schema_version": phase2.SCENE_SCHEMA_VERSION,
        "generated_at": int(time.time()),
        "analysis_level": "walking_only",
        "ui_mode": "WALK",
        "mode": state.run_mode, "mode_label": mode_label,
        "realtime": {"mode": state.run_mode, "never_label_as_live": state.run_mode != phase2.MODE_LIVE},
        "origin": {"label": "", "lat": float(origin["input_latitude"]), "lon": float(origin["input_longitude"]),
                   "nearest_node": int(origin["nearest_node"]),
                   "snap_distance_m": round(float(origin["snap_distance_m"]), 3), "component_id": None},
        "selected_stop": selected,
        "intent": {"key": None, "route_short_name": None, "direction_id": None, "route_direction": None,
                   "headsigns": [], "label": "none (stop-first workflow)", "stop_ids": []},
        "parameters": {"near_tie_pct": None, "margin_s": None},
        "model_statements": phase2.MODEL_STATEMENTS,
        "no_combined_score": True,
        "summary": {"analysis_level": "walking_only", "selected_stop": selected,
                    "compared": 1 if (result and result.get("available")) else 0,
                    "candidates_total": len(stops_out)},
        "stops": stops_out,
        "transit": None,
    }


def transit_block(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The schema-1.1 transit block from a TrackEngine result (field names as the primary window's)."""
    if not result:
        return None
    return {
        "trip_id": result["trip_id"], "route_short_name": result.get("route_short_name"),
        "destination": result.get("destination"), "tracking_state": result.get("tracking_state"),
        "planned_route": [[float(lon), float(lat)] for lon, lat in (result.get("planned_route") or [])],
        "planned_route_status": result.get("shape_status"),
        "reported_positions": [{"lat": float(p["lat"]), "lon": float(p["lon"]), "vehicle_id": p.get("vehicle_id"),
                                "reported_at": p.get("reported_at"), "observed_at": p.get("observed_at")}
                               for p in (result.get("positions") or [])],
        "source_stop_id": result.get("source_stop_id"),
        "boarding_stop": {"stop_id": result.get("source_stop_id"),
                          "expected_arrival_epoch": result.get("expected_arrival_epoch"),
                          "expected_time_source": result.get("expected_time_source"),
                          "realtime_confirmed": bool(result.get("realtime_confirmed")),
                          "pedestrian_arrival_epoch": result.get("pedestrian_arrival_epoch"),
                          "slack_s": result.get("slack_s"), "timing_status": result.get("timing_status")},
        "vehicle_feed": result.get("vehicle_snapshot"),
        "last_refresh_status": result.get("last_refresh_status"),
        "provenance": ("planned_route: static GTFS shapes.txt; reported_positions: raw GTFS-Realtime "
                       "VehiclePosition, not snapped, not interpolated, no trajectory"),
    }


def export(state, engine: WalkEngine, mode_label: str,
           out_dir: Optional[Path] = None) -> Optional[Tuple[Path, Path]]:
    """Write current_scene.json / .geojson for the current UI state; None before an origin."""
    if state.ui_mode == "TRACK" and state.comparison_scene is not None and state.tracking_result is not None:
        scene = dict(state.comparison_scene)          # the cached Compare scene is not mutated
        scene["ui_mode"] = "TRACK"
        scene["selected_stop"] = state.tracked_stop_id
        scene["transit"] = transit_block(state.tracking_result)
    elif state.ui_mode == "COMPARE" and state.comparison_scene is not None:
        scene = state.comparison_scene
        scene["ui_mode"] = "COMPARE"
        scene["selected_stop"] = state.selected_compare_stop
        scene["transit"] = None
    else:
        scene = walk_scene(state, engine, mode_label)
    if scene is None:
        return None
    geojson = phase2.scene_to_geojson(scene, engine.graph)
    return phase2.write_scene(scene, geojson, Path(out_dir) if out_dir else DEFAULT_OUT_DIR)
