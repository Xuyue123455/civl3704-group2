"""The API's session over the validated engines: the desktop steps by name, without a window.

Mobile Web Phase 2: ``ApiSession`` drives ``scripts/unified_ui/journey.JourneySession`` - the UI-free copy of
the ``UnifiedApp`` orchestration (same engine calls, same ``AppState`` writes, same order) - so a COMPARE or
TRACK request no longer builds a Matplotlib figure, a map and three panels to obtain a scene.  The heavy,
read-only objects are shared across requests through ``EngineHost`` (engines, service tables, the static GTFS
snapshot, the planned-shape store); everything a request mutates (the ``CompareEngine`` snapshot, the
``TrackEngine`` states, the ``AppState``) is built per request, as the window builds it per window.

``HeadlessSession`` (Phase 1: a real ``UnifiedApp`` on an Agg figure) is kept as the reference the equivalence
test drives against (``tests/test_mobile_web_phase2.py``); the adapters never use it.

Replay sources are server-side named files; the browser never sends a path.  LIVE runs only when the API
process holds the TfNSW credential (environment / .env on the server) - the browser never does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .. import schemas
from .engines import PROJECT_ROOT, EngineHost

# named replay sources: recorded, real feed captures kept in the repository (never fabricated)
REPLAY_SOURCES: Dict[str, Dict[str, Any]] = {
    "l1_2026-10-17_0917": {
        "trip_updates": PROJECT_ROOT / "tests" / "fixtures" / "week11_l1" / "l1_trip_updates.pb",
        "vehicle_positions": PROJECT_ROOT / "tests" / "fixtures" / "week11_l1" / "l1_vehicle_positions.pb",
        "description": "TfNSW GTFS-Realtime light-rail capture of 2026-10-17 09:17:01 AEDT (Trip Updates + Vehicle "
                       "Positions), the Week 11 L1 integration fixture",
    },
}


class _SharedShapesTrackEngine:
    """A ``TrackEngine`` whose planned-shape store is the process-wide one (``EngineHost.shape_store``).

    ``TrackEngine.ensure_states`` builds a fresh ``TripShapeStore`` per engine (per window on the desktop); the API
    builds one engine per request, so without sharing every TRACK request re-reads trips.txt for the trip ->
    shape map.  The store is a read-once cache over the same static snapshot (keyed by its sha256), so sharing
    it changes no output.  The frozen ``track_engine.py`` is not edited: the swap happens after the validated
    ``ensure_states`` ran, on the two references the tracker reads (``engine.shapes`` / ``tracker.shapes``).
    """

    @staticmethod
    def make(track_engine_module, comp, run_mode, vehicles, want_shape, shared):
        base = track_engine_module.TrackEngine

        class SharedShapesTrackEngine(base):
            def ensure_states(self) -> None:
                first = self.tracker is None
                base.ensure_states(self)
                if first and self.want_shape and shared is not None:
                    self.shapes = shared
                    self.tracker.shapes = shared

        return SharedShapesTrackEngine(comp, run_mode, vehicles, want_shape=want_shape)


def _engines_for(host: EngineHost, run_mode: str, replay: Optional[str], observed_at: Optional[int],
                 want_shape: bool, shared_shapes: bool):
    """The per-request CompareEngine + TrackEngine, built the way the validated suites build them."""
    from unified_ui import compare_engine, track_engine, walk_engine

    run_mode = str(run_mode).upper()
    if run_mode not in schemas.RUN_MODES:
        raise schemas.ValidationError(f"run_mode must be one of {schemas.RUN_MODES}")
    # LIVE: the engines fetch the TfNSW feeds themselves through the validated Week 7 chain, reading the
    # credential from the API process's own environment / .env; nothing about it enters a response.  Without a
    # credential or network the engine records realtime status "unavailable" and COMPARE falls back to the
    # timetable - the same behaviour as the desktop window.
    source = None
    if run_mode == "REPLAY":
        if replay not in REPLAY_SOURCES:
            raise schemas.ValidationError(f"replay must name a server-side source: {sorted(REPLAY_SOURCES)}")
        source = REPLAY_SOURCES[replay]
        if not Path(source["trip_updates"]).exists():
            raise schemas.ValidationError(f"replay source {replay} is not available on this server")
    comp = compare_engine.CompareEngine(host.engine, host.tables, run_mode,
                                        Path(source["trip_updates"]) if source else None)
    comp.loaded_static = host.real_static
    if run_mode == "ANALYTICAL" and observed_at is not None:
        comp.realtime = walk_engine.phase2.RealtimeContext(mode="ANALYTICAL", status="none",
                                                           observed_at=int(observed_at), anchor="api_request")
    vehicles = Path(source["vehicle_positions"]) if source and source.get("vehicle_positions") else None
    if shared_shapes and want_shape:
        track = _SharedShapesTrackEngine.make(track_engine, comp, run_mode, vehicles, want_shape, host.shape_store)
    else:
        track = track_engine.TrackEngine(comp, run_mode, vehicles, want_shape=want_shape)
    return run_mode, comp, track


class _Steps:
    """The desktop steps by name, over anything exposing ``state`` / ``standard`` and the transition methods."""

    host: EngineHost
    compare: Any
    core: Any                      # the object with set_origin / select_stop / enter_compare / ... (app or journey)
    intent_before_compare = False  # True: set route / direction in WALK, then enter Compare once (ApiSession)

    @property
    def state(self):
        return self.core.state

    @property
    def standard(self):
        return self.core.standard

    attach_origins = False         # True: bounded nearest-reachable attachment for the profile (ApiSession)

    def walk_to(self, lat: float, lon: float, stop_id: str) -> None:
        core = self.core
        if not core.engine.inside_study_area(lat, lon):
            raise schemas.ValidationError("origin is outside the Glebe study area")
        if core.engine.stop_row(stop_id) is None:
            raise schemas.ValidationError(f"unknown stop_id {stop_id}")
        if self.attach_origins:
            from .attach import attach_origin
            origin = attach_origin(self.host, lat, lon, stop_id, core.state.walking_profile)
            self.origin_attachment = origin.get("attachment")
            core.set_origin_at(origin)
        else:
            core.set_origin(lat, lon)
        core.select_stop(stop_id)

    def compare_service(self, route_short_name: str, direction_id: int, profile: str,
                        selected_trip_id: Optional[str], selected_stop_id: Optional[str]) -> Dict[str, Any]:
        core = self.core
        if route_short_name not in self.compare.routes_available():
            raise schemas.ValidationError(f"route_short_name {route_short_name} is not a study-area service")
        choices = self.compare.directions_for(route_short_name)             # ServiceChoice objects
        if str(direction_id) not in [str(getattr(c, "direction_id", c)) for c in choices]:
            raise schemas.ValidationError(f"direction_id {direction_id} is not available for {route_short_name}")
        wanted = f"{route_short_name}/{direction_id}"
        if self.intent_before_compare and core.state.service_key != wanted:
            # Phase 2: the request names its service, so the intent is set while still in WALK (where set_route /
            # set_direction only record it) and Compare is entered once - one engine run instead of three.  The
            # scene is the one the desktop reaches by entering Compare on the stop's default service and then
            # changing route and direction (the equivalence test proves it against HeadlessSession).
            core.set_route(route_short_name)
            core.set_direction(str(direction_id))
        core.enter_compare()
        if core.state.service_key != wanted:
            core.set_route(route_short_name)
            core.set_direction(str(direction_id))
        if core.standard is not None and core.state.compare_profile != profile:
            core.set_walking_profile(profile)                        # Phase B-2: the scene is recomputed
        scene = core.state.comparison_scene
        if selected_trip_id is not None:
            listed = [c["trip_id"] for c in scene["trip_candidates"]]
            if selected_trip_id not in listed:
                raise schemas.ValidationError(f"selected_trip_id {selected_trip_id} is not an upcoming exact trip")
            core.select_trip(selected_trip_id)
            scene = core.state.comparison_scene
        if selected_stop_id is not None:
            if self.compare.stop_entry(scene, selected_stop_id) is None:
                raise schemas.ValidationError(f"selected_stop_id {selected_stop_id} is not a candidate stop")
            core.select_compare_stop(selected_stop_id)
        return core.state.comparison_scene

    def track_trip(self, trip_id: str, boarding_stop_id: str) -> Dict[str, Any]:
        core = self.core
        scene = core.state.comparison_scene
        listed = [c["trip_id"] for c in scene["trip_candidates"]]
        if trip_id not in listed:
            raise schemas.ValidationError(f"trip_id {trip_id} is not an upcoming exact trip of this service")
        if trip_id != core.state.selected_trip_id:
            core.select_trip(trip_id)
        core.select_compare_stop(boarding_stop_id)
        if not core.can_track():
            raise schemas.ValidationError("this exact trip is not timed at the boarding stop")
        result = core.enter_track()
        if result is None:
            raise schemas.ValidationError("tracking could not start")
        return result


class ApiSession(_Steps):
    """Phase 2: the UI-free journey (``unified_ui.journey.JourneySession``) - what the adapters use."""

    intent_before_compare = True
    attach_origins = True
    origin_attachment: Optional[Dict[str, Any]] = None

    def __init__(self, host: EngineHost, run_mode: str = "ANALYTICAL", replay: Optional[str] = None,
                 observed_at: Optional[int] = None, profile: str = "standard", want_shape: bool = True) -> None:
        from unified_ui.journey import JourneySession

        run_mode, comp, track = _engines_for(host, run_mode, replay, observed_at, want_shape, shared_shapes=True)
        self.host, self.run_mode, self.replay, self.compare = host, run_mode, replay, comp
        self.core = JourneySession(host.engine, run_mode, compare=comp, track=track, standard=host.standard,
                                   initial_profile=profile)
        self.journey = self.core

    def close(self) -> None:
        return None


class HeadlessSession(_Steps):
    """Phase 1 reference: one real ``UnifiedApp`` on an Agg figure, built the way the validated suites build it.
    Kept for the equivalence test only (slow: it draws the window after every step)."""

    def __init__(self, host: EngineHost, run_mode: str = "ANALYTICAL", replay: Optional[str] = None,
                 observed_at: Optional[int] = None, profile: str = "standard", want_shape: bool = True) -> None:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from unified_ui import shell

        run_mode, comp, track = _engines_for(host, run_mode, replay, observed_at, want_shape, shared_shapes=False)
        self.host, self.run_mode, self.replay, self.compare = host, run_mode, replay, comp
        self.app = shell.UnifiedApp(host.engine, run_mode, figure=plt.figure(figsize=(16.0, 9.0)), compare=comp,
                                    track=track, hybrid_stairs=host.hybrid_stairs,
                                    stair_cache_dir=host.stair_cache_dir, initial_profile=profile,
                                    auto_refresh=False)
        self.core = self.app

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close(self.app.figure)
