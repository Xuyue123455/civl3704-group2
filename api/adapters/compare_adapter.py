"""POST /api/compare — the COMPARE contract (docs/WEB_MIGRATION_CONTRACT.md §4) over the validated scene.

The scene is the engine's own ``comparison_scene`` (schema 1.1, the one ``scene_export`` writes): exact trips,
catchability, slack, ranks, highlights, objective winners.  This adapter only selects and renames fields and adds
WGS84 geometry for the selected stop's routes and the winner routes.  No catchability, slack, rank or winner is
computed here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import schemas
from .engines import EngineHost
from .attach import stop_access
from .session import ApiSession

WINNER_NAMES = {"effort": "gentlest", "time": "fastest", "distance": "shortest"}
SUMMARY_WINNER_KEYS = {"gentlest": "lowest_terrain_effort_stop", "fastest": "lowest_modelled_walking_time_stop",
                       "shortest": "shortest_walk_stop"}


def geometry_for(host: EngineHost, edge_keys: Optional[List[List[int]]]) -> Optional[Dict[str, Any]]:
    if not edge_keys:
        return None
    from unified_ui.walk_engine import phase2
    return schemas.line_string(phase2.route_linestring(host.engine.graph, [tuple(e) for e in edge_keys]))


def route_summary_block(host: EngineHost, summary: Optional[Dict[str, Any]], with_geometry: bool) -> Optional[Dict[str, Any]]:
    if summary is None:
        return None
    block = {"distance_m": summary.get("distance_m"), "walking_time_s": summary.get("walking_time_s"),
             "walking_time_min": summary.get("walking_time_min"), "effort_j_per_kg": summary.get("effort_j_per_kg"),
             "edges": summary.get("edges"), "edge_keys": summary.get("edge_keys")}
    if with_geometry:
        block["geometry"] = geometry_for(host, summary.get("edge_keys"))
        block.update(stair_block(host, summary.get("edge_keys")))
    return block


def stair_block(host: EngineHost, edge_keys: Optional[List[List[int]]]) -> Dict[str, Any]:
    """The mapped stairs a COMPARE path walks over (the same directed-edge membership the Standard routes carry;
    an empty list on the trusted basis).  Presentation passthrough of the stair cache, nothing computed."""
    from .walk_adapter import stairs_on_route
    standard = host.standard
    if standard is None or not edge_keys:
        return {"stair_edge_count": 0, "stair_edges": [], "stairs": []}
    stair_edges = [[int(u), int(v), int(k)] for u, v, k in edge_keys if (int(u), int(v), int(k)) in standard.stair_keys]
    return {"stair_edge_count": len(stair_edges), "stair_edges": stair_edges,
            "stairs": stairs_on_route(host, {"stair_edges": [tuple(e) for e in stair_edges]})}


def candidate_record(host: EngineHost, entry: Dict[str, Any], scene: Dict[str, Any],
                     with_geometry: bool) -> Dict[str, Any]:
    """One candidate stop: exact identity, route-specific metrics, catchability, slack, ranks."""
    catch = entry.get("catchability") or {}
    feasible = scene["summary"].get("feasible_stops") or []
    return {
        "stop_id": entry["stop_id"], "stop_name": entry["stop_name"], "mode": entry.get("mode"),
        "geometry": schemas.point(float(entry["lon"]), float(entry["lat"])),
        "eligibility": entry.get("eligibility"), "status_reason": entry.get("status_reason"),
        "serves_selected_service": entry.get("serves_selected_service"),
        "routes": {name: route_summary_block(host, (entry.get("routes") or {}).get(key), with_geometry)
                   for name, key in schemas.OBJECTIVE_KEYS.items()},
        "route_identity": entry.get("route_identity"),
        "catchability": {k: v for k, v in catch.items() if k != "trips"},
        "trips_at_stop": catch.get("trips"),
        "slack_s": catch.get("slack_s"),
        "effort_rank": entry.get("effort_rank"), "near_tie_group": entry.get("near_tie_group"),
        "feasible_rank": (feasible.index(entry["stop_id"]) + 1) if entry["stop_id"] in feasible else None,
        "feasible_count": len(feasible),
        "highlights": entry.get("highlights") or [], "pareto_status": entry.get("pareto_status"),
        "stop_access": stop_access(host, entry["stop_id"]),           # display-only connector node -> GTFS marker
    }


def compare(host: EngineHost, request: Dict[str, Any], run_mode: str = "ANALYTICAL", replay: Optional[str] = None,
            observed_at: Optional[int] = None, geometry_for_candidates: bool = False) -> Dict[str, Any]:
    from unified_ui.profile_phase2 import objective_rows
    session = ApiSession(host, run_mode, replay, observed_at, profile=request["walking_profile"])
    try:
        comp = session.compare
        anchor_stop = request["selected_stop_id"]
        session.walk_to(request["origin"]["lat"], request["origin"]["lon"],
                        anchor_stop or comp.choice(f"{request['route_short_name']}/{request['direction_id']}").stop_ids[0])
        scene = session.compare_service(request["route_short_name"], request["direction_id"], request["walking_profile"],
                                        request["selected_trip_id"], anchor_stop)
        state = session.state
        selected_stop = state.selected_compare_stop
        entry = comp.stop_entry(scene, selected_stop) if selected_stop else None
        winners = []
        for block in objective_rows(scene):
            winners.append({"objectives": [WINNER_NAMES[o] for o in block["objectives"]],
                            "stop_id": block["stop_id"], "stop_name": block["stop_name"],
                            "distance_m": block["distance_m"], "walking_time_s": block["walking_time_s"],
                            "effort_j_per_kg": block["effort_j_per_kg"], "slack_s": block.get("slack_s"),
                            "status": block.get("status"), "highlights": block.get("highlights"),
                            "same_stop_other_path": bool(block.get("same_stop_other_path")),
                            "edge_keys": [[int(x) for x in e] for e in (block.get("edge_keys") or [])],
                            "geometry": geometry_for(host, block.get("edge_keys")),
                            "stop_access": stop_access(host, block["stop_id"]),
                            **stair_block(host, block.get("edge_keys"))})
        candidates = [candidate_record(host, s, scene, geometry_for_candidates) for s in scene["stops"]
                      if s.get("serves_selected_service")]
        return {
            "run_mode": state.run_mode,
            "route": {"route_short_name": scene["intent"]["route_short_name"], "route_id": scene["intent"].get("route_id"),
                      "mode": scene["intent"].get("mode"), "label": scene["intent"].get("label"),
                      "stop_ids": scene["intent"].get("stop_ids")},
            "direction": {"direction_id": int(scene["intent"]["direction_id"]),
                          "route_direction": scene["intent"].get("route_direction"),
                          "headsigns": scene["intent"].get("headsigns")},
            "walking_profile": state.compare_profile or state.walking_profile,
            "routing_basis": state.compare_basis,
            "origin": {"lat": request["origin"]["lat"], "lon": request["origin"]["lon"],
                       "nearest_node": (state.origin or {}).get("nearest_node"), "attachment": session.origin_attachment},
            "observed_at": scene["realtime"].get("observed_at"),
            "observed_at_sydney": scene["realtime"].get("observed_at_sydney"),
            "upcoming_trips": comp.trip_rows(scene, scene["trip_selection"]["horizon"]),
            "selected_trip": scene.get("selected_trip"),
            "trip_selection_mode": state.trip_selection_mode,
            "next_feasible_trip": {"trip_id": scene["summary"].get("next_feasible_trip_id"),
                                   "policy": scene["trip_selection"].get("next_feasible_policy")},
            "candidate_stops": candidates,
            "objective_winners": {name: scene["summary"].get(key) for name, key in SUMMARY_WINNER_KEYS.items()},
            "winner_blocks": winners,
            "selected_stop": selected_stop,
            "selected_stop_status": (None if entry is None else
                                     {"record": candidate_record(host, entry, scene, True),
                                      "rank_lines": comp.rank_lines(scene, entry)}),
            "summary": scene["summary"],
            "parameters": scene["parameters"],
            "realtime_provenance": scene["realtime"],
            "model_statements": scene.get("model_statements"),
            "no_combined_score": bool(scene.get("no_combined_score")),
            "provenance": schemas.provenance(schemas.SOURCE_ENGINE,
                                             "computed by the validated desktop COMPARE engine (UI-free journey session)",
                                             scene_schema_version=scene.get("schema_version"), replay=replay),
        }
    finally:
        session.close()
