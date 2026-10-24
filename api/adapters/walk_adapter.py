"""POST /api/walk — the WALK contract (docs/WEB_MIGRATION_CONTRACT.md §3) over the validated engines.

Exactly the desktop calls, in the desktop order (``shell.UnifiedApp.set_origin`` / ``select_stop``):
``engine.snap_origin`` (Week 5 snapping) -> ``engine.search_from`` (the three Week 9 searches) ->
``engine.solve`` (trusted graph; the Step-free profile) and, for the Standard profile,
``StandardContext.search`` / ``solve`` on the validated hybrid graph.  The triplet of every route is the one the
engine returned (``phase2.route_summary``); the geometry is the stored OSM geometry of the route's own edges
(``phase2.route_linestring``), WGS84 lon/lat.  No distance, time or effort is computed in this file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import schemas
from .engines import EngineHost

MODEL_STATEMENTS = {
    "walking_time": "Tobler hiking function on the validated terrain model: modelled, not measured.",
    "effort": "Minetti metabolic cost per kg (J/kg) on the validated terrain model.",
    "standard": "Standard: eligible mapped stair segments (classes A / P / B) may be used; class C excluded.",
    "step_free": "Step-free: mapped stair segments excluded. Step-free does not mean wheelchair accessible.",
    "triplet": "One route -> one internally consistent distance + time + effort triplet (never mixed across routes).",
}


def route_block(host: EngineHost, route: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One objective's route as the API shape; the numbers are the engine's own summary."""
    if route is None:
        return None
    from unified_ui.walk_engine import phase2                               # the validated Week 9 helpers
    summary = phase2.route_summary(route)
    coords = phase2.route_linestring(host.engine.graph, route["edges"])
    block = {
        "distance_m": summary["distance_m"],
        "walking_time_s": summary["walking_time_s"],
        "walking_time_min": summary["walking_time_min"],
        "effort_j_per_kg": summary["effort_j_per_kg"],
        "edges": summary["edges"],
        "edge_keys": summary["edge_keys"],
        "geometry": schemas.line_string(coords),
    }
    if "stair_edge_count" in route:                                       # Standard hybrid routes carry stair facts
        block["stair_edge_count"] = int(route["stair_edge_count"])
        block["stair_edges"] = [[int(u), int(v), int(k)] for u, v, k in route.get("stair_edges", [])]
        block["stairs"] = stairs_on_route(host, route)
    return block


def stairs_on_route(host: EngineHost, route: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The mapped stairs this route walks over, in travel order, each with the direction the walker climbs them.

    Direction and vertical change come from the Standard stair cache row of the DIRECTED edge (u -> v is the travel
    direction of the route): ``direction`` ('up' / 'down') and ``signed_vertical_change_m`` were computed offline by
    scripts/build_stair_cost_cache.py from the DEM profile.  Nothing is derived here; a stair edge without a cache
    row reports ``elevation unknown`` rather than a guess.
    """
    standard = host.standard
    if standard is None:
        return []
    out = []
    for u, v, k in route.get("stair_edges", []):
        edge = (int(u), int(v), int(k))
        row = standard.stair_rows.get(edge) or {}
        record = (standard.mapped_steps.by_edge.get(edge) if standard.mapped_steps is not None else None) or \
                 (standard.mapped_steps.by_edge.get((int(v), int(u), int(k))) if standard.mapped_steps is not None else None) or {}
        rise = row.get("signed_vertical_change_m")
        rise = None if rise is None or rise != rise else round(float(rise), 2)
        direction = str(row.get("direction")) if row.get("direction") not in (None, "") else None
        parts = record.get("stair_parts_lonlat") or []
        out.append({
            "u": edge[0], "v": edge[1], "key": edge[2],
            "name": _text(row.get("name")) or _text(record.get("name")) or "",
            "direction": direction,                                            # 'up' = ascent, 'down' = descent
            "vertical_change_m": rise,                                         # signed, along the travel direction
            "stair_length_m": _float(row.get("stair_length_m")) if row else _float(record.get("stair_length_m")),
            "step_count": _float(row.get("step_count")) if row else _float(record.get("step_count")),
            "confidence_class": row.get("confidence_class"),
            "elevation_source": "stair model cache (directed edge)" if rise is not None else "elevation unknown",
            "geometry": {"type": "MultiLineString",
                         "coordinates": [[[float(lon), float(lat)] for lon, lat in part] for part in parts]},
        })
    return out


def _text(value: Any) -> str:
    """A string field that may arrive as float NaN from a CSV-backed row (an unnamed stair): never NaN in a response."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value)
    return "" if text in ("", "nan", "None") else text


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None or value != value else float(value)
    except (TypeError, ValueError):
        return None


def mapped_stairs(host: EngineHost, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    standard = host.standard
    if standard is None or not result.get("uses_stairs"):
        return []
    out = []
    for record in standard.stair_segments(result):                      # the mapped-steps cache records, as stored
        item = dict(record)
        parts = record.get("stair_parts_lonlat") or []
        item["geometry"] = {"type": "MultiLineString",
                            "coordinates": [[[float(lon), float(lat)] for lon, lat in part] for part in parts]}
        if record.get("marker_lon") is not None:
            item["marker"] = schemas.point(float(record["marker_lon"]), float(record["marker_lat"]))
        out.append(item)
    return out


def walk(host: EngineHost, request: Dict[str, Any]) -> Dict[str, Any]:
    """Validated request -> WALK response (engine source)."""
    engine = host.engine
    lat, lon = request["origin"]["lat"], request["origin"]["lon"]
    if not engine.inside_study_area(lat, lon):
        raise schemas.ValidationError("origin is outside the Glebe study area")
    if engine.stop_row(request["stop_id"]) is None:
        raise schemas.ValidationError(f"unknown stop_id {request['stop_id']}")
    from .attach import attach_origin, no_route_block, stop_access
    profile = request["profile"]
    # bounded nearest-reachable attachment (api/adapters/attach.py): the validated nearest node when it reaches the
    # stop (then identical to the desktop), else the nearest connected candidate within the radius, else a reason
    origin = attach_origin(host, lat, lon, request["stop_id"], profile if host.standard is not None else "step_free")
    node = int(origin["nearest_node"])
    search = engine.search_from(node)
    trusted = engine.solve(search, request["stop_id"], engine.origin_component(node))
    standard = host.standard
    if profile == "standard" and standard is not None:
        result = standard.solve(standard.search(node), request["stop_id"])
        basis = "standard_hybrid"
    else:
        result = trusted
        basis = "trusted"
        if profile == "standard":
            profile = "step_free"                                            # the flag is off: the trusted basis only
    routes = {name: route_block(host, (result.get("routes") or {}).get(key))
              for name, key in schemas.OBJECTIVE_KEYS.items()}
    return {
        "profile": profile,
        "routing_basis": basis,
        "origin": {"lat": float(origin.get("tap_latitude", origin["input_latitude"])), "lon": float(origin.get("tap_longitude", origin["input_longitude"])),
                   "nearest_node": node, "snap_distance_m": round(float(origin["snap_distance_m"]), 3),
                   "node_lat": float(origin["node_lat"]), "node_lon": float(origin["node_lon"]),
                   "geometry": schemas.point(float(origin.get("tap_longitude", origin["input_longitude"])), float(origin.get("tap_latitude", origin["input_latitude"]))),
                   "attachment": origin.get("attachment")},
        "selected_stop": {"stop_id": result["stop_id"], "name": result["stop_name"], "mode": result.get("mode", ""),
                          "stop_node": result.get("stop_node"), "stop_access": stop_access(host, result["stop_id"])},
        "available": bool(result.get("available")),
        "status": result.get("status"),
        "reason": result.get("reason") or "",
        "no_route": None if result.get("available") else (no_route_block(origin, result.get("reason") or result.get("status") or "")
                                                          or {"code": "no_reachable_snap_candidate", "message": "No connected walking path from this location.",
                                                              "detail": "The attached origin node does not reach the stop under this profile.",
                                                              "internal_reason": result.get("reason") or result.get("status") or ""}),
        "routes": routes,
        "route_identity": result.get("route_identity"),
        "comparisons": {"effort_saving_vs_shortest_pct": result.get("effort_saving_vs_shortest_pct"),
                        "extra_distance_vs_shortest_pct": result.get("extra_distance_vs_shortest_pct"),
                        "time_saving_vs_shortest_s": result.get("time_saving_vs_shortest_s")},
        "uses_stairs": bool(result.get("uses_stairs")),
        "stair_usage": result.get("stair_usage") or {},
        "mapped_stairs": mapped_stairs(host, result),
        "step_free_available": bool(trusted.get("available")),
        "model_provenance": {**MODEL_STATEMENTS, "graph": host.graph_facts()},
        "provenance": schemas.provenance(schemas.SOURCE_ENGINE,
                                         "computed by the validated desktop engines (WalkEngine / StandardContext)"),
    }
