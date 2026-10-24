"""The journey without the window: WALK -> COMPARE -> TRACK orchestration with no Matplotlib dependency.

Reader's summary (Mobile Web Phase 2)
-------------------------------------
Responsibility  the Compare / Track *service* layer: the exact call sequences ``shell.UnifiedApp`` makes on
                the engines for every transition (set origin, select stop, switch profile, enter Compare,
                change route / direction, select trip, inspect stop, enter Track, refresh), writing the same
                ``AppState`` fields in the same order, with no figure, no panel, no map and no timer.
Inputs          a ``WalkEngine``; optionally a ``CompareEngine``, a ``TrackEngine``, a ``StandardContext``
                (``--hybrid-stairs``) and a ``RawShortestLayer`` (``--stair-shortcuts``); the run mode and the
                initial walking profile - exactly what the window receives.
Outputs         ``AppState`` transitions (``state.comparison_scene``, ``state.tracking_result``,
                ``state.timings`` ...), the same keys the desktop writes.
Depends on      ``state.py`` and the engines only.  It never imports ``shell``, ``map_view`` or a panel.
Do not infer    any analytical rule from this file: every quantity comes from the engines it calls.  This
                module exists so that a web request does not have to build a desktop window to get a scene;
                its outputs must stay identical to the window's (``tests/test_mobile_web_phase2.py`` holds
                the equivalence check against ``api.adapters.session.HeadlessSession``).

Why a copy of the shell's sequences rather than the shell itself
-----------------------------------------------------------------
``UnifiedApp`` interleaves presentation (acknowledge the click, draw the origin, re-render, status line) with
the engine calls.  The desktop is the validated product (Desktop Gold, tag ``desktop-v42-demo-ready-2026-10-23``)
and is left untouched here; this module reproduces the engine-facing half of each method, line for line, and the
window keeps its own copy until the desktop is re-based on it in a later, separately validated pass.  Any
behavioural difference between the two is a defect of this file, never of the window.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from . import MODE_WALK
from .profile_phase2 import PROFILE_LABEL, PROFILE_WORDS, run_scene as run_profile_scene, view_for_profile
from .standard_hybrid import PROFILE_STANDARD, PROFILE_STEP_FREE, PROFILES
from .state import TRIP_AUTOMATIC, TRIP_MANUAL, UI_COMPARE, UI_TRACK, UI_WALK, AppState

RUN_MODES = ("ANALYTICAL", "LIVE", "REPLAY")


class JourneySession:
    """One viewer's journey over the engines, without a window (the Compare / Track service layer).

    The methods carry the names of the ``UnifiedApp`` transitions they reproduce; a method that the window
    answers with a status-strip message answers here with ``state.message`` set and ``None`` returned - the
    same contract the window's tests rely on.
    """

    def __init__(self, engine: Any, run_mode: str = "ANALYTICAL", compare: Optional[Any] = None,
                 track: Optional[Any] = None, standard: Optional[Any] = None, raw_layer: Optional[Any] = None,
                 initial_profile: str = PROFILE_STANDARD, state: Optional[AppState] = None) -> None:
        run_mode = str(run_mode).upper()
        if run_mode not in RUN_MODES:
            raise ValueError(f"run mode must be ANALYTICAL, LIVE or REPLAY, not {run_mode!r}")
        if initial_profile not in PROFILES:
            raise ValueError(f"initial profile must be one of {PROFILES}, not {initial_profile!r}")
        self.engine = engine
        self.compare = compare
        self.track = track if compare is not None else None
        self.standard = standard
        self.raw_layer = raw_layer
        self.state = state if state is not None else AppState(mode=MODE_WALK, run_mode=run_mode)
        self.state.stair_shortcuts_enabled = bool(raw_layer is not None and getattr(raw_layer, "available", False))
        self.state.hybrid_stairs_enabled = standard is not None
        self.state.walking_profile = initial_profile
        self.state.avoid_steps = self.state.hybrid_stairs_enabled and initial_profile == PROFILE_STEP_FREE
        self.last_good_realtime: Dict[str, Any] = {}
        self.last_refresh_failures: list = []

    # ----- WALK ------------------------------------------------------------------------ #
    def set_origin(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        if not self.engine.inside_study_area(latitude, longitude):
            self.state.message = "Please click inside the Glebe study area."
            return None
        origin = self.engine.snap_origin(latitude, longitude)          # Week 5 snapping
        return self.set_origin_at(origin)

    def set_origin_at(self, origin: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The window's ``set_origin`` from an already snapped origin record (``WalkEngine.snap_origin`` shape).

        The API's bounded nearest-reachable attachment (``api/adapters/attach.py``) produces such a record when the
        single nearest node cannot reach the chosen stop; with the nearest node it is exactly ``snap_origin``'s."""
        started = time.perf_counter()
        kept_stop = self.state.selected_stop
        self.state.origin = origin
        self.state.origin_node = int(origin["nearest_node"])
        self.state.walking_result = None
        self.state.timings["origin_feedback_s"] = time.perf_counter() - started
        search = self.engine.search_from(origin["nearest_node"])        # Week 9 search_all
        self.state.set_origin(origin, search)                           # also invalidates any comparison
        self.state.timings["origin_search_ready_s"] = time.perf_counter() - started
        if self.standard is not None:                                   # STANDARD HYBRID: the three searches
            hybrid_started = time.perf_counter()
            self.state.standard_search = self.standard.search(int(origin["nearest_node"]))
            self.state.timings["standard_search_s"] = time.perf_counter() - hybrid_started
        self.state.message = ""
        if kept_stop is not None:
            self.select_stop(kept_stop)                                 # one click keeps the stop
        if self.state.ui_mode in (UI_COMPARE, UI_TRACK):
            self.run_comparison()                                       # recompute for the new origin
        return origin

    def select_stop(self, stop_id: str) -> Optional[Dict[str, Any]]:
        if self.state.search is None:
            self.state.message = "Set an origin first."
            return None
        started = time.perf_counter()
        self.state.selected_stop = str(stop_id)
        result = self.engine.solve(self.state.search, str(stop_id),
                                   self.engine.origin_component(self.state.origin_node))
        self.state.set_result(str(stop_id), result)
        self.state.timings["stop_result_ready_s"] = time.perf_counter() - started
        if self.state.stair_shortcuts_enabled and self.standard is None:   # STEP-FREE: the raw layer (not with hybrid)
            raw_started = time.perf_counter()
            self.state.set_raw_shortest(self.raw_layer.route(int(self.state.origin_node), int(result["stop_node"])))
            self.state.timings["raw_shortest_s"] = time.perf_counter() - raw_started
        if self.standard is not None and self.state.standard_search is not None:   # STANDARD HYBRID: the three routes
            hybrid_started = time.perf_counter()
            self.state.standard_result = self.standard.solve(self.state.standard_search, str(stop_id))
            self.state.timings["standard_solve_s"] = time.perf_counter() - hybrid_started
        return result

    def displayed_walking_result(self) -> Optional[Dict[str, Any]]:
        """The WALK result shown: Standard (hybrid) or Step-free (trusted, unchanged)."""
        if self.standard is not None and self.state.walking_profile == PROFILE_STANDARD:
            return self.state.standard_result
        return self.state.walking_result

    # ----- the walking profile --------------------------------------------------------- #
    def set_walking_profile(self, profile: str) -> Optional[Dict[str, Any]]:
        """WALK: the displayed routes switch, nothing is re-solved.  COMPARE: the scene is recomputed on the
        profile's basis (intent, manual trip and stop preserved where valid).  TRACK: read-only."""
        if self.standard is None:
            self.state.message = "The Standard profile is not available (start with --hybrid-stairs)."
            return None
        if self.state.ui_mode == UI_TRACK:
            self.state.message = "The walking profile is read-only in Track; go back to Compare to change it."
            return None
        if self.state.ui_mode == UI_COMPARE:
            return self.switch_compare_profile(profile)
        self.state.set_walking_profile(profile)
        self.state.message = ""
        return self.displayed_walking_result()

    def switch_compare_profile(self, profile: str) -> Optional[Dict[str, Any]]:
        state = self.state
        if self.compare is None or state.comparison_scene is None or state.service_key is None:
            state.set_walking_profile(profile)
            return None
        if profile == state.walking_profile:
            return state.comparison_scene
        started = time.perf_counter()
        previous_profile = state.walking_profile
        previous_stop = state.selected_compare_stop
        previous_entry = self.compare.stop_entry(state.comparison_scene, previous_stop)
        previous_name = str((previous_entry or {}).get("stop_name") or previous_stop or "")
        manual_trip = state.selected_trip_id if state.trip_selection_mode == TRIP_MANUAL else None
        state.set_walking_profile(profile)                 # drops the other basis's scene and pinned trip
        if manual_trip is not None:
            state.selected_trip_id = manual_trip             # keep the viewer's exact trip while listed
            state.trip_selection_mode = TRIP_MANUAL
        scene = self.run_comparison(preferred_stop=previous_stop)
        label = PROFILE_WORDS.get(profile, profile)
        notes = []
        if manual_trip is not None and state.trip_selection_mode != TRIP_MANUAL:
            notes.append(f"Trip {manual_trip} is no longer listed under the {label} profile; the next feasible trip is shown.")

        def available(e):
            return bool(e) and bool(e.get("serves_selected_service")) and e.get("eligibility") == "compared"
        entry = self.compare.stop_entry(scene, previous_stop) if scene is not None else None
        if previous_stop is not None and not available(entry):
            state.selected_compare_stop = self.compare.default_compare_stop(scene, None)   # the normal default rule
            if available(previous_entry):
                notes.append(f"Previous stop ({previous_name}) is not available under the {label} profile.")
            if not available(self.compare.stop_entry(scene, state.selected_compare_stop)):
                notes.append(f"No stop of this route and direction has a {label} walking route from this origin.")
        state.message = " ".join(notes)
        state.last_profile_switch = {"from": previous_profile, "to": profile, "manual_trip": manual_trip,
                                     "manual_trip_kept": bool(manual_trip) and state.trip_selection_mode == TRIP_MANUAL,
                                     "previous_stop": previous_stop, "stop_kept": state.selected_compare_stop == previous_stop,
                                     "notes": list(notes)}
        state.timings["compare_profile_switch_s"] = time.perf_counter() - started
        return scene

    # ----- COMPARE ---------------------------------------------------------------------- #
    def enter_compare(self) -> Optional[Dict[str, Any]]:
        if self.compare is None or self.state.phase != "stop_selected":
            self.state.message = "Choose a stop first."
            return None
        started = time.perf_counter()
        self.state.ui_mode = UI_COMPARE
        if self.state.service_key is None:
            choices = self.compare.choices_for_stop(self.state.selected_stop) or self.compare.tables.choices()
            if choices:
                self.state.selected_route = choices[0].route_short_name
                self.state.selected_direction = choices[0].direction_id
        self.state.trip_selection_mode = TRIP_AUTOMATIC
        scene = self.run_comparison(preferred_stop=self.state.selected_stop)
        self.state.timings["walk_to_compare_s"] = time.perf_counter() - started
        return scene

    def back_to_walk(self) -> None:
        started = time.perf_counter()
        self.state.ui_mode = UI_WALK
        self.state.invalidate_tracking()
        self.state.timings["compare_to_walk_s"] = time.perf_counter() - started

    def set_route(self, route_short_name: str) -> None:
        if self.compare is None:
            return
        started = time.perf_counter()
        route = str(route_short_name)
        if route == self.state.selected_route:
            return
        self.state.selected_route = route
        directions = self.compare.directions_for(route)
        ids = [d.direction_id for d in directions]
        if self.state.selected_direction not in ids:
            self.state.selected_direction = ids[0] if ids else None
        self.state.trip_selection_mode = TRIP_AUTOMATIC
        self.state.invalidate_comparison()
        self.state.invalidate_tracking()
        self.state.timings["route_to_directions_ready_s"] = time.perf_counter() - started
        if self.state.ui_mode == UI_COMPARE:
            self.run_comparison(preferred_stop=self.state.selected_compare_stop)

    def set_direction(self, direction_id: str) -> None:
        if self.compare is None:
            return
        if str(direction_id) == self.state.selected_direction:
            return
        self.state.selected_direction = str(direction_id)
        self.state.trip_selection_mode = TRIP_AUTOMATIC
        self.state.invalidate_comparison()
        self.state.invalidate_tracking()
        if self.state.ui_mode == UI_COMPARE:
            self.run_comparison(preferred_stop=self.state.selected_compare_stop)

    def phase2_view(self):
        return view_for_profile(self.standard, self.state.walking_profile)

    def run_scene(self, trip_id: Optional[str]) -> Dict[str, Any]:
        started = time.perf_counter()
        view = self.phase2_view()
        if view is None:
            scene = self.compare.run(self.state.origin, self.state.service_key, trip_id)
            self.state.compare_basis = "trusted_step_free" if self.standard is not None else None
        else:
            scene = run_profile_scene(self.compare, view, self.state.origin, self.state.service_key, trip_id)
            self.state.compare_basis = view.basis
        self.state.compare_profile = self.state.walking_profile if self.standard is not None else None
        self.state.timings["phase2_run_s"] = time.perf_counter() - started
        return scene

    def profile_label(self) -> str:
        if self.standard is None:
            return ""
        return PROFILE_LABEL.get(self.state.compare_profile or self.state.walking_profile, "")

    def run_comparison(self, preferred_stop: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.compare is None or self.state.origin is None or self.state.service_key is None:
            return None
        started = time.perf_counter()
        trip_id = self.state.selected_trip_id if self.state.trip_selection_mode == TRIP_MANUAL else None
        try:
            scene = self.run_scene(trip_id)
        except Exception as error:                     # e.g. the manual trip no longer listed
            if trip_id is not None:
                self.state.trip_selection_mode = TRIP_AUTOMATIC
                scene = self.run_scene(None)
            else:
                raise error
        self.state.set_scene(scene)
        self.state.realtime_snapshot_metadata = scene.get("realtime")
        self.remember_good_realtime()
        preferred = preferred_stop or self.state.selected_compare_stop or self.state.selected_stop
        self.state.selected_compare_stop = self.compare.default_compare_stop(scene, preferred)
        self.state.compare_status = (scene.get("summary") or {}).get("feasibility_note") or ""
        self.state.timings["direction_to_trips_s"] = time.perf_counter() - started
        if self.state.ui_mode == UI_TRACK:
            self.sync_track(refresh_status=self.state.track_last_refresh_status)
        return scene

    def select_trip(self, trip_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if self.compare is None or self.state.comparison_scene is None:
            return None
        started = time.perf_counter()
        self.state.trip_selection_mode = TRIP_MANUAL if trip_id is not None else TRIP_AUTOMATIC
        scene = self.compare.select_trip(self.state.comparison_scene, trip_id)   # no re-routing, no feed access
        self.state.set_scene(scene)
        if self.state.tracked_trip_id is not None and self.state.tracked_trip_id != self.state.selected_trip_id:
            self.state.invalidate_tracking()
        self.state.selected_compare_stop = self.compare.default_compare_stop(scene, self.state.selected_compare_stop)
        self.state.compare_status = (scene.get("summary") or {}).get("feasibility_note") or ""
        self.state.timings["trip_to_feasibility_s"] = time.perf_counter() - started
        return scene

    def select_compare_stop(self, stop_id: str) -> Optional[Dict[str, Any]]:
        if self.state.comparison_scene is None:
            return None
        started = time.perf_counter()
        entry = self.compare.stop_entry(self.state.comparison_scene, stop_id)
        if entry is None:
            return None
        self.state.selected_compare_stop = str(stop_id)
        self.state.timings["compare_stop_to_routes_ready_s"] = time.perf_counter() - started
        return entry

    def refresh_feed(self) -> Optional[Dict[str, Any]]:
        """LIVE only: one explicit fetch, then the same intent with the automatic trip policy."""
        if self.compare is None or self.state.run_mode != "LIVE":
            return None
        self.compare.set_feed_mode(self.feed_mode())
        self.compare.snapshot(refresh=True)
        self.remember_good_realtime()
        self.state.trip_selection_mode = TRIP_AUTOMATIC
        return self.run_comparison(preferred_stop=self.state.selected_compare_stop)

    def remember_good_realtime(self) -> None:
        context = self.compare.realtime if self.compare is not None else None
        if context is not None and getattr(context, "status", None) == "ok":
            self.last_good_realtime[self.feed_mode()] = context

    def feed_mode(self) -> str:
        from .feed_adapter import MODE_BUS
        if self.compare is None:
            return MODE_BUS
        if self.state.ui_mode == UI_TRACK and self.state.tracked_trip_id is not None and self.track is not None:
            return self.track.trip_mode(self.state.tracked_trip_id)
        return self.compare.mode_of_service(self.state.service_key)

    # ----- TRACK ------------------------------------------------------------------------ #
    def can_track(self) -> bool:
        state = self.state
        if self.track is None or state.comparison_scene is None or state.selected_trip_id is None:
            return False
        entry = self.compare.stop_entry(state.comparison_scene, state.selected_compare_stop)
        if entry is None or not entry.get("serves_selected_service") or entry.get("eligibility") != "compared":
            return False
        return (entry.get("catchability") or {}).get("trip_id") == state.selected_trip_id

    def enter_track(self) -> Optional[Dict[str, Any]]:
        if not self.can_track():
            self.state.message = "Choose an exact trip and a boarding stop first."
            return None
        started = time.perf_counter()
        state = self.state
        if state.tracked_trip_id not in (None, state.selected_trip_id) or state.tracked_stop_id not in (
                None, state.selected_compare_stop):
            state.invalidate_tracking()
        state.ui_mode = UI_TRACK
        self.track.ensure_states()
        self.track.sync_trip_updates(self.compare.realtime)            # the snapshot COMPARE already holds
        vehicle_status = self.track.refresh_vehicles(                  # one Vehicle Position request (LIVE)
            mode=self.compare.mode_of_service(state.service_key))
        refresh_status = ("ok" if vehicle_status == "ok" else
                          "not requested" if state.run_mode == "ANALYTICAL" or
                          (state.run_mode == "REPLAY" and self.track.vehicle_replay_path is None)
                          else "unavailable")
        self.sync_track(refresh_status=refresh_status, trip_id=state.selected_trip_id,
                        stop_id=state.selected_compare_stop)
        state.timings["compare_to_track_s"] = time.perf_counter() - started
        return state.tracking_result

    def sync_track(self, refresh_status: str, trip_id: Optional[str] = None,
                   stop_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        state = self.state
        trip_id = trip_id or state.tracked_trip_id
        stop_id = stop_id or state.tracked_stop_id
        scene = state.comparison_scene
        if scene is None or trip_id is None or stop_id is None:
            return None
        row = self.engine.stop_row(stop_id)
        stop_name = str(row["stop_name"]) if row is not None else str(stop_id)
        stop_mode = str(row.get("mode", "bus")) if row is not None else "bus"
        self.track.pin(scene, trip_id, stop_id, stop_name)
        self.track.evaluate()
        timing = self.track.expected_at_stop(scene, trip_id, stop_id, stop_mode)
        result = self.track.result(timing, refresh_status)
        state.set_tracking(result)
        state.message = "" if result.get("planned_route") else "Planned route geometry unavailable for this trip."
        return result

    def back_to_compare(self) -> None:
        started = time.perf_counter()
        self.state.ui_mode = UI_COMPARE
        self.state.timings["track_to_compare_s"] = time.perf_counter() - started

    def refresh_track(self) -> Optional[Dict[str, Any]]:
        """Manual refresh: Trip Updates (through COMPARE's snapshot) and Vehicle Positions, independently."""
        if self.track is None or self.state.ui_mode != UI_TRACK or self.state.tracking_result is None:
            return None
        started = time.perf_counter()
        state = self.state
        failed: list = []
        if state.run_mode == "ANALYTICAL":
            refresh_status = "not requested"
        else:
            self.compare.set_feed_mode(self.feed_mode())
            try:
                context = self.compare.snapshot(refresh=True)
                if context.status != "ok":
                    failed.append(f"trip updates: {context.error or 'unavailable'}")
                elif state.run_mode == "LIVE":
                    state.track_feed_refreshes += 1
                    self.remember_good_realtime()
            except Exception as error:
                context = self.compare.realtime
                failed.append(f"trip updates: {error}")
            self.track.sync_trip_updates(context)
            vehicle_status = self.track.refresh_vehicles(mode=self.feed_mode())
            if vehicle_status != "ok":
                failed.append(f"vehicle positions: {self.track.vehicle_state.error or vehicle_status}")
            refresh_status = "failed" if failed else "ok"
        if refresh_status == "ok" and state.run_mode == "LIVE":
            # the Trip Update snapshot changed: the comparison scene is re-derived for the same intent and trip,
            # so the boarding-stop timing and the Compare cache describe the SAME snapshot
            trip_id = state.selected_trip_id if state.trip_selection_mode == TRIP_MANUAL else None
            try:
                scene = self.run_scene(trip_id)
            except Exception:
                scene = self.run_scene(None)
            state.set_scene(scene)
            state.realtime_snapshot_metadata = scene.get("realtime")
            state.selected_compare_stop = self.compare.default_compare_stop(scene, state.tracked_stop_id)
            state.compare_status = (scene.get("summary") or {}).get("feasibility_note") or ""
        state.message = "; ".join(failed) if failed else ""
        self.last_refresh_failures = list(failed)
        self.sync_track(refresh_status=refresh_status)
        state.timings["track_refresh_s"] = time.perf_counter() - started
        return state.tracking_result
