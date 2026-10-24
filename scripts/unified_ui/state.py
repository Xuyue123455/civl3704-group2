"""The explicit application state of the unified window (Phase A: WALK).

Three WALK phases, in order:

    NO_ORIGIN        nothing chosen; the map waits for a click
    ORIGIN_SELECTED  origin snapped and the three origin searches done
    STOP_SELECTED    an exact stop_id chosen; the three routes solved

Phase B adds a second UI state, COMPARE (``ui_mode``), which keeps the WALK
fields intact and adds its own: the service intent (route + direction), the
listed exact trips, the selected trip, the Phase 2 scene the engine
returned, and the stop being inspected in that scene. ``ui_mode`` (WALK /
COMPARE / TRACK) is a different axis from ``run_mode`` (ANALYTICAL / LIVE /
REPLAY).

Phase C adds TRACK: one pinned exact ``trip_id`` at one boarding stop, the
Week 7 tracking state of its vehicle, the latest raw reported position, the
planned static geometry and the last refresh outcome. The COMPARE fields are
left untouched while tracking, so "Back to Compare" restores them.

Everything the window shows is derived from this object; callbacks only
transition it. ``timings`` records what the click handlers measured, so the
performance figures in the design notes come from the same code path the
viewer uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import MODE_WALK

PHASE_NO_ORIGIN = "no_origin"
PHASE_ORIGIN_SELECTED = "origin_selected"
PHASE_STOP_SELECTED = "stop_selected"

UI_WALK = "WALK"
UI_COMPARE = "COMPARE"
UI_TRACK = "TRACK"
ROUTES_ALL = "all"                 # walking_route_visibility: all three routes
ROUTES_FASTEST = "fastest"         # the time-optimal route only
TRIP_AUTOMATIC = "automatic"
TRIP_MANUAL = "manual"


@dataclass
class AppState:
    mode: str = MODE_WALK
    run_mode: str = "ANALYTICAL"                     # ANALYTICAL | LIVE (accepted, not exposed)
    origin: Optional[Dict[str, Any]] = None          # snap_origin_coordinate() dict
    origin_node: Optional[int] = None
    search: Optional[Dict[str, Any]] = None          # time_optimal_route.search_all() result
    selected_stop: Optional[str] = None              # exact GTFS stop_id, never collapsed
    walking_result: Optional[Dict[str, Any]] = None  # WalkEngine.solve() result
    technical_details_visible: bool = False
    message: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    searches_run: int = 0
    solves_run: int = 0
    # ---- Phase B: COMPARE (additive; WALK fields above are untouched) ----
    ui_mode: str = UI_WALK
    selected_route: Optional[str] = None             # route_short_name, e.g. "433"
    selected_direction: Optional[str] = None         # direction_id, "0" / "1"
    trip_candidates: list = field(default_factory=list)   # scene["trip_candidates"] (exact trips)
    selected_trip_id: Optional[str] = None
    trip_selection_mode: str = TRIP_AUTOMATIC        # automatic (next feasible) | manual
    comparison_scene: Optional[Dict[str, Any]] = None     # phase2_decision.run_phase2() scene
    selected_compare_stop: Optional[str] = None      # exact stop_id inspected in COMPARE
    compare_status: str = ""                         # one human line about the comparison
    realtime_snapshot_metadata: Optional[Dict[str, Any]] = None   # RealtimeContext.as_dict()
    compare_runs: int = 0
    # ---- Phase C: TRACK (additive; COMPARE fields above are the cached Compare state) ----
    tracked_trip_id: Optional[str] = None            # the pinned exact trip (never a route)
    tracked_stop_id: Optional[str] = None            # the boarding stop it was pinned at
    tracking_result: Optional[Dict[str, Any]] = None # TrackEngine.result(): plain, JSON-able
    vehicle_status: Optional[str] = None             # LIVE | LAST SEEN | POSITION UNAVAILABLE | AMBIGUOUS
    vehicle_position: Optional[Dict[str, Any]] = None  # the latest raw report drawn (LIVE / LAST SEEN), else None
    vehicle_last_seen_at: Optional[int] = None
    planned_trip_geometry: Optional[list] = None     # [[lon, lat], ...] static GTFS shape, or None
    track_snapshot_metadata: Optional[Dict[str, Any]] = None  # vehicle snapshot facts (fetched_at, feed_timestamp...)
    track_last_refresh_status: str = "none"          # none | ok | failed | not requested
    track_runs: int = 0
    track_feed_refreshes: int = 0                    # explicit refreshes made while in TRACK
    walking_route_visibility: str = ROUTES_ALL
    # ---- AUTO-REFRESH Phase A (additive; a mirror of the controller, never written to the scene export) ----
    auto_refresh_enabled: bool = False               # LIVE only, not with --no-auto-refresh
    auto_refresh_interval_s: float = 30.0
    auto_refresh_paused: bool = False                # the viewer's preference, kept for the session
    auto_refresh_running: bool = False               # the one timer is scheduled (COMPARE / TRACK only)
    auto_refresh_in_progress: bool = False
    auto_refresh_last_started_at: Optional[float] = None
    auto_refresh_last_completed_at: Optional[float] = None
    auto_refresh_last_succeeded_at: Optional[float] = None
    auto_refresh_last_status: str = "none"           # none | ok | failed
    auto_refresh_last_reason: str = ""               # why the last automatic refresh failed (Technical details)
    auto_refresh_ticks: int = 0                      # automatic refreshes that ran
    # ---- STEP-FREE Phase A (additive; a WALK presentation preference, never a ui_mode) ----
    stair_shortcuts_enabled: bool = False            # --stair-shortcuts and the mapped-steps cache present
    avoid_steps: bool = False                        # the viewer's preference: show the trusted (step-free) Shortest
    raw_shortest_result: Optional[Dict[str, Any]] = None   # raw_shortest.RawShortestLayer.route(): distance only
    raw_shortest_has_steps: bool = False
    raw_shortest_step_segments: list = field(default_factory=list)   # mapped-steps cache records on the raw route
    step_free_status: str = "not_applicable"         # not_applicable | mapped_steps_found | step_free | no_step_free_route
    # ---- STANDARD HYBRID Phase A (additive; --hybrid-stairs; a walking PROFILE, never a ui_mode) ----
    hybrid_stairs_enabled: bool = False              # the flag is on and the stair cost cache was found
    walking_profile: str = "standard"                # standard | step_free (meaningful only when hybrid_stairs_enabled)
    standard_search: Optional[Dict[str, Any]] = None # StandardContext.search(): the three searches from the origin
    standard_result: Optional[Dict[str, Any]] = None # StandardContext.solve(): the three Standard routes, all three metrics
    profile_switches: int = 0
    # ---- Phase B-1: the routing basis the cached Compare scene was computed on (follows the profile) ----
    compare_basis: Optional[str] = None             # standard_hybrid | trusted_step_free | None (no scene / flag off)
    compare_profile: Optional[str] = None           # the walking profile at the time the scene was computed
    last_profile_switch: Optional[Dict[str, Any]] = None   # Phase B-2: what the last Compare switch preserved (facts)

    @property
    def service_key(self) -> Optional[str]:
        if self.selected_route is None or self.selected_direction is None:
            return None
        return f"{self.selected_route}/{self.selected_direction}"

    def invalidate_comparison(self) -> None:
        """A new origin or intent: the scene no longer describes the state."""
        self.trip_candidates = []
        self.selected_trip_id = None
        self.comparison_scene = None
        self.selected_compare_stop = None
        self.compare_status = ""
        self.compare_basis = None
        self.compare_profile = None
        # the pinned trip is NOT dropped here: a new origin changes the walking side only;
        # a new intent or a different trip calls invalidate_tracking() explicitly

    def invalidate_tracking(self) -> None:
        """A different trip (or no trip): nothing of the previous vehicle state may survive."""
        self.tracked_trip_id = None
        self.tracked_stop_id = None
        self.tracking_result = None
        self.vehicle_status = None
        self.vehicle_position = None
        self.vehicle_last_seen_at = None
        self.planned_trip_geometry = None
        self.track_snapshot_metadata = None
        self.track_last_refresh_status = "none"

    def set_tracking(self, result: Dict[str, Any]) -> None:
        self.tracking_result = result
        self.tracked_trip_id = result["trip_id"]
        self.tracked_stop_id = result["source_stop_id"]
        self.vehicle_status = result["tracking_state"]
        positions = result.get("positions") or []
        self.vehicle_position = positions[0] if len(positions) == 1 else None
        self.vehicle_last_seen_at = result.get("last_seen_at")
        self.planned_trip_geometry = result.get("planned_route") or None
        self.track_snapshot_metadata = result.get("vehicle_snapshot")
        self.track_last_refresh_status = result.get("last_refresh_status") or "none"
        self.track_runs += 1

    def set_scene(self, scene: Dict[str, Any]) -> None:
        self.comparison_scene = scene
        self.trip_candidates = list(scene.get("trip_candidates") or [])
        chosen = scene.get("selected_trip")
        self.selected_trip_id = chosen["trip_id"] if chosen else None
        self.compare_runs += 1

    @property
    def phase(self) -> str:
        if self.origin is None:
            return PHASE_NO_ORIGIN
        if self.selected_stop is None or self.walking_result is None:
            return PHASE_ORIGIN_SELECTED
        return PHASE_STOP_SELECTED

    def clear_origin(self) -> None:
        self.origin = None
        self.origin_node = None
        self.search = None
        self.selected_stop = None
        self.walking_result = None
        self.standard_search = None
        self.standard_result = None
        self.invalidate_comparison()
        self.invalidate_tracking()
        self.reset_step_free()

    def set_origin(self, origin: Dict[str, Any], search: Dict[str, Any]) -> None:
        self.origin = origin
        self.origin_node = int(origin["nearest_node"])
        self.search = search
        self.searches_run += 1
        # a new origin invalidates the previous routes; the stop choice is kept
        self.walking_result = None
        self.standard_search = None
        self.standard_result = None
        self.invalidate_comparison()
        self.reset_step_free()                       # STEP-FREE: a new origin resets avoid_steps

    def set_result(self, stop_id: str, result: Optional[Dict[str, Any]]) -> None:
        self.selected_stop = str(stop_id)
        self.walking_result = result
        self.standard_result = None
        self.solves_run += 1
        self.reset_step_free()                       # STEP-FREE: a new destination stop resets avoid_steps

    # ---- STEP-FREE Phase A ----
    def reset_step_free(self) -> None:
        """A new origin or stop: the preference and the raw route start again (pan / zoom / details never call this).

        With the hybrid profile the preference is the walking PROFILE, which persists for the session; avoid_steps
        then mirrors it so the two never disagree."""
        self.avoid_steps = (self.walking_profile == "step_free") if self.hybrid_stairs_enabled else False
        self.raw_shortest_result = None
        self.raw_shortest_has_steps = False
        self.raw_shortest_step_segments = []
        self.step_free_status = "not_applicable"

    def set_walking_profile(self, profile: str) -> None:
        if profile not in ("standard", "step_free"):
            raise ValueError(f"walking profile must be standard or step_free, not {profile!r}")
        if profile != self.walking_profile:
            self.profile_switches += 1
            if self.hybrid_stairs_enabled and self.comparison_scene is not None:
                self.invalidate_comparison()        # Phase B-1: the scene was computed on the other profile's basis
                self.invalidate_tracking()
        self.walking_profile = profile
        self.avoid_steps = profile == "step_free" if self.hybrid_stairs_enabled else self.avoid_steps

    def set_raw_shortest(self, raw: Optional[Dict[str, Any]]) -> None:
        self.raw_shortest_result = raw
        self.raw_shortest_has_steps = bool(raw and raw.get("has_mapped_steps"))
        self.raw_shortest_step_segments = list((raw or {}).get("step_edges") or [])

    def as_dict(self) -> Dict[str, Any]:
        """A plain summary for tests and the Technical details box."""
        return {
            "mode": self.mode, "run_mode": self.run_mode, "phase": self.phase,
            "origin": self.origin, "origin_node": self.origin_node,
            "selected_stop": self.selected_stop,
            "has_result": self.walking_result is not None,
            "technical_details_visible": self.technical_details_visible,
            "timings": dict(self.timings),
            "searches_run": self.searches_run, "solves_run": self.solves_run,
            "ui_mode": self.ui_mode, "service_key": self.service_key,
            "selected_route": self.selected_route, "selected_direction": self.selected_direction,
            "selected_trip_id": self.selected_trip_id, "trip_selection_mode": self.trip_selection_mode,
            "trips_listed": len(self.trip_candidates), "has_comparison": self.comparison_scene is not None,
            "selected_compare_stop": self.selected_compare_stop, "compare_status": self.compare_status,
            "realtime_snapshot_metadata": self.realtime_snapshot_metadata, "compare_runs": self.compare_runs,
            "tracked_trip_id": self.tracked_trip_id, "tracked_stop_id": self.tracked_stop_id,
            "vehicle_status": self.vehicle_status, "vehicle_last_seen_at": self.vehicle_last_seen_at,
            "has_planned_geometry": bool(self.planned_trip_geometry),
            "track_last_refresh_status": self.track_last_refresh_status, "track_runs": self.track_runs,
            "walking_route_visibility": self.walking_route_visibility,
            "auto_refresh": {"enabled": self.auto_refresh_enabled, "interval_s": self.auto_refresh_interval_s,
                             "paused": self.auto_refresh_paused, "running": self.auto_refresh_running,
                             "in_progress": self.auto_refresh_in_progress,
                             "last_status": self.auto_refresh_last_status, "ticks": self.auto_refresh_ticks},
            "walking_profile": {"hybrid_stairs_enabled": self.hybrid_stairs_enabled, "profile": self.walking_profile,
                                "has_standard_result": self.standard_result is not None, "switches": self.profile_switches,
                                "compare_basis": self.compare_basis, "compare_profile": self.compare_profile},
            "step_free": {"enabled": self.stair_shortcuts_enabled, "avoid_steps": self.avoid_steps,
                          "status": self.step_free_status, "raw_shortest_has_steps": self.raw_shortest_has_steps,
                          "raw_shortest_distance_m": ((self.raw_shortest_result or {}).get("distance_m")
                                                      if self.raw_shortest_result else None)},
        }
