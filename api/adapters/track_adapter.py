"""POST /api/track — the TRACK contract (docs/WEB_MIGRATION_CONTRACT.md §5) over the validated tracking result.

The result is ``state.tracking_result`` of the Week 7 tracking chain (exact trip pinned by trip_id, boarding-stop
timing, Tobler pedestrian arrival on the Fastest route, slack, the vehicle state, the planned static GTFS route,
the raw reported positions).  The walking block is the Fastest route of the tracked stop under the inherited
profile - the route the arrival and the slack are based on.  Nothing is snapped, interpolated or predicted here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import schemas
from .engines import EngineHost
from .attach import stop_access
from .compare_adapter import stair_block
from .session import ApiSession

STATE_HEADINGS = {"LIVE": "POSITION REPORTED", "LAST SEEN": "LAST SEEN", "POSITION UNAVAILABLE": "POSITION UNAVAILABLE",
                  "AMBIGUOUS": "POSITION AMBIGUOUS"}


def service_of_trip(host: EngineHost, trip_id: str):
    rows = host.tables.trips
    match = rows[rows["trip_id"].astype(str) == str(trip_id)]
    if match.empty:
        raise schemas.ValidationError(f"trip_id {trip_id} is not in the study-area timetable")
    first = match.iloc[0]
    return str(first["route_short_name"]), int(first["direction_id"])


def track(host: EngineHost, request: Dict[str, Any], run_mode: str = "ANALYTICAL", replay: Optional[str] = None,
          observed_at: Optional[int] = None) -> Dict[str, Any]:
    from unified_ui.scene_export import transit_block
    from unified_ui.walk_engine import phase2
    route_short_name, direction_id = service_of_trip(host, request["trip_id"])
    session = ApiSession(host, run_mode, replay, observed_at, profile=request["walking_profile"])
    try:
        comp = session.compare
        session.walk_to(request["origin"]["lat"], request["origin"]["lon"], request["boarding_stop_id"])
        session.compare_service(route_short_name, direction_id, request["walking_profile"], None, request["boarding_stop_id"])
        result = session.track_trip(request["trip_id"], request["boarding_stop_id"])
        state = session.state
        scene = state.comparison_scene
        entry = comp.stop_entry(scene, request["boarding_stop_id"])
        fastest = (entry.get("routes") or {}).get("time") if entry else None
        block = transit_block(result)                                   # the validated schema-1.1 transit block
        positions = result.get("positions") or []
        return {
            "run_mode": state.run_mode,
            "trip": {"trip_id": result["trip_id"], "route_short_name": result.get("route_short_name"),
                     "destination": result.get("destination"), "direction_id": direction_id,
                     "in_upcoming": result.get("in_upcoming"), "trip_update_status": result.get("trip_update_status")},
            "boarding_stop": {"stop_id": result["source_stop_id"], "name": result.get("source_stop_name"),
                              "geometry": (schemas.point(float(entry["lon"]), float(entry["lat"])) if entry else None),
                              "stop_access": stop_access(host, result["source_stop_id"])},
            "origin": {"lat": request["origin"]["lat"], "lon": request["origin"]["lon"],
                       "nearest_node": (state.origin or {}).get("nearest_node"), "attachment": session.origin_attachment},
            "walking_profile": state.compare_profile or state.walking_profile,
            "walking": {"objective": "fastest",
                        "distance_m": fastest.get("distance_m") if fastest else None,
                        "walking_time_s": result.get("walking_time_s"),
                        "effort_j_per_kg": fastest.get("effort_j_per_kg") if fastest else None,
                        "margin_s": result.get("margin_s"),
                        "model": "Tobler walking time on the validated terrain model: modelled, not measured",
                        "geometry": (schemas.line_string(phase2.route_linestring(host.engine.graph,
                                                                                [tuple(e) for e in fastest["edge_keys"]]))
                                     if fastest and fastest.get("edge_keys") else None),
                        "edge_keys": fastest.get("edge_keys") if fastest else None,
                        **stair_block(host, fastest.get("edge_keys") if fastest else None)},
            "timing": {"scheduled": result.get("scheduled_epoch"), "expected": result.get("expected_arrival_epoch"),
                       "expected_time_source": result.get("expected_time_source"),
                       "realtime_confirmed": bool(result.get("realtime_confirmed")),
                       "delay_s": result.get("delay_s"), "timing_status": result.get("timing_status"),
                       "pedestrian_arrival": result.get("pedestrian_arrival_epoch"),
                       "slack_s": result.get("slack_s"), "observed_at": result.get("observed_at"),
                       "pinned_expected": result.get("pinned_expected_epoch"),
                       "pinned_expected_basis": result.get("pinned_expected_basis")},
            "vehicle": {"state": STATE_HEADINGS.get(result.get("tracking_state"), result.get("tracking_state")),
                        "week7_state": result.get("tracking_state"),
                        "reported_position": (schemas.point(float(positions[0]["lon"]), float(positions[0]["lat"]))
                                              if len(positions) == 1 else None),
                        "reported_positions": block["reported_positions"],
                        "feed_status": result.get("vehicle_feed_status"), "feed_error": result.get("vehicle_feed_error"),
                        "invalid_coordinates": result.get("invalid_coordinates"),
                        "snapshot": result.get("vehicle_snapshot"),
                        "meaning": "a missing position is not a cancellation; the point is the raw GTFS-Realtime "
                                   "report, never snapped, interpolated or extrapolated"},
            "planned_route": {"source": "static_gtfs", "shape_id": result.get("shape_id"),
                              "status": result.get("shape_status"), "points": result.get("shape_points"),
                              "geometry": (schemas.line_string(block["planned_route"]) if block["planned_route"] else None),
                              "meaning": "planned line (trip_id -> trips.txt -> shape_id -> shapes.txt), not an observed trajectory"},
            "refresh": {"last_status": result.get("last_refresh_status"), "run_mode": state.run_mode,
                        "auto_refresh": False, "note": "the API answers one request; there is no server-side timer"},
            "provenance": schemas.provenance(schemas.SOURCE_ENGINE,
                                             "computed by the validated desktop TRACK chain (UI-free journey session)",
                                             replay=replay, transit_provenance=block["provenance"]),
        }
    finally:
        session.close()
