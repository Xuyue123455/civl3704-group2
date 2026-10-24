"""The adapter between the unified TRACK state and the validated Week 7 tracking backend.

Reader's summary (cleanup pass 2)
---------------------------------
Responsibility  TRACK: pin ONE exact trip at ONE boarding stop, refresh the raw vehicle
                snapshot, and return one result dict the panel and the scene export read.
Inputs          the COMPARE scene (trip_id, boarding stop, the Trip Update snapshot that
                COMPARE already holds), the run mode (ANALYTICAL / REPLAY / LIVE), a
                Vehicle Position feed or a saved ``.pb`` for REPLAY, static GTFS
                (``trips.txt`` + ``shapes.txt`` through ``TripShapeStore``).
Outputs         ``result()`` -> trip identity, expected / scheduled time with source and
                delay, modelled pedestrian arrival and slack (Fastest route), the vehicle
                state (POSITION REPORTED / LAST SEEN / POSITION UNAVAILABLE / AMBIGUOUS),
                the raw reported position(s), the planned static route, refresh provenance.
Depends on      ``scripts/demo_week7_realtime.py`` (V3 Trip Updates), ``demo_week7_live_
                vehicles.py`` (V4B Vehicle Positions), ``demo_week7_trip_tracking.py``
                (V4D pinning, states, shapes), ``scripts/phase2_decision.py``
                (catchability_for_stop), ``feed_adapter.py`` (which TfNSW feed per mode).
Do not infer    a trajectory, a snapped position or a prediction from the vehicle point
                (it is drawn as reported); a cancellation from a missing position; a
                route-number match (the only identity is the exact trip_id); a live claim
                from a REPLAY snapshot. TRACK is Fastest-only by product rule.

Nothing here decides a tracking state, matches a vehicle to a map, reads a
shape file, or predicts anything. Every TRACK quantity comes from:

* ``demo_week7_realtime.RealtimeState``        the Trip Update side (V3): the decoded
                                               ArrivalEvents by stop, the observation clock
* ``demo_week7_live_vehicles.VehicleState``    the Vehicle Position side (V4B): one raw
                                               snapshot, indexed by trip_id, replaced whole
* ``demo_week7_trip_tracking``                 ``pin_event`` (what is pinned), ``TripTracker.
                                               evaluate`` (LIVE / LAST SEEN / POSITION
                                               UNAVAILABLE / AMBIGUOUS), ``resolve_shape`` and
                                               ``TripShapeStore`` (trip_id -> trips.txt
                                               shape_id -> shapes.txt, static geometry only),
                                               ``positions`` (what may be drawn), ``age_phrase``
                                               and ``running_phrase`` (wording of age / delay)
* ``phase2_decision.catchability_for_stop``    the pinned trip's expected arrival at the
                                               boarding stop and the modelled pedestrian
                                               arrival / slack, re-read on every refresh

The Week 7 ``TripTracker`` is constructed HEADLESS: its ``__init__`` only
records the lower-layer objects it evaluates against (``panel.state`` = the
V3 state, ``overlay.state`` = the V4B state), and ``build()`` - the legacy
tracking window - is never called. Trip Updates and Vehicle Positions stay
independent feeds: the Trip Update snapshot is the SAME one COMPARE already
holds (no second request for it), the vehicle snapshot is fetched here, and
a missing VehiclePosition never implies anything about the service.

The transit line drawn from this engine is planned static GTFS geometry.
The vehicle marker is the latest raw GTFS-Realtime reported position. No map
matching, interpolation, trajectory reconstruction, or GPS snapping is
performed.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from . import feed_adapter
from .compare_engine import MODE_ANALYTICAL, MODE_LIVE, MODE_REPLAY, CompareEngine
from .walk_engine import load_script, phase2

v4d = load_script("demo_week7_trip_tracking")       # the validated tracking backend (no window built)
v4b = v4d.v4b                                       # VehicleState, fetchers
v3 = v4d.v3                                         # RealtimeState, LiveSnapshot, clock
v4a = v4d.v4a                                       # clock for vehicle timestamps

STATE_LIVE = v4d.STATE_LIVE
STATE_LAST_SEEN = v4d.STATE_LAST_SEEN
STATE_UNAVAILABLE = v4d.STATE_UNAVAILABLE
STATE_AMBIGUOUS = v4d.STATE_AMBIGUOUS
TRACKING_STATES = (STATE_LIVE, STATE_LAST_SEEN, STATE_UNAVAILABLE, STATE_AMBIGUOUS)

SHAPE_UNAVAILABLE = "Planned route geometry unavailable"
PROVENANCE_NOTE = ("The transit line is planned static GTFS geometry. The vehicle marker is the latest raw "
                   "GTFS-Realtime reported position. No map matching, interpolation, trajectory "
                   "reconstruction, or GPS snapping is performed.")
REFRESH_OK = "ok"
REFRESH_FAILED = "failed"
REFRESH_NONE = "none"
REFRESH_NOT_REQUESTED = "not requested"

TIME_NOT_LISTED = "selected_trip_time_unavailable_at_stop"


def _no_vehicle_fetcher(reason: str) -> Callable[[], bytes]:
    """A fetcher that explains why no vehicle feed is requested (ANALYTICAL; REPLAY without a file)."""
    def read() -> bytes:
        raise v4a.RealtimeError(reason)
    return read


class AnchoredVehicleState(v4b.VehicleState):
    """V4B's state with the observation clock pinned to a recorded capture time (REPLAY only)."""

    anchor_epoch: Optional[int] = None

    def refresh(self, fetcher=None) -> str:
        status = super().refresh(fetcher)
        if self.snapshot is not None and self.anchor_epoch is not None:
            self.snapshot.observed_at = int(self.anchor_epoch)
        return status


class TrackEngine:
    """One instance per window; vehicle positions are fetched explicitly, never on a timer."""

    def __init__(self, compare: CompareEngine, run_mode: str, vehicle_replay_path: Optional[Path] = None,
                 want_shape: bool = True, limit: int = v3.NEXT_SERVICES) -> None:
        self.compare = compare
        self.run_mode = str(run_mode).upper()
        self.vehicle_replay_path = Path(vehicle_replay_path) if vehicle_replay_path else None
        self.want_shape = bool(want_shape)
        self.limit = int(limit)
        self.realtime_state: Optional[Any] = None       # v3.RealtimeState (Trip Updates, synced from COMPARE)
        self.vehicle_state: Optional[Any] = None        # v4b.VehicleState (Vehicle Positions)
        self.shapes: Optional[Any] = None               # v4d.TripShapeStore
        self.tracker: Optional[Any] = None              # v4d.TripTracker, headless
        self.vehicle_fetcher: Optional[Callable[[], bytes]] = None   # explicit override (tests / REPLAY files)
        self.vehicle_fetches = 0
        # Week 11: in LIVE the vehicle feed follows the pinned trip's mode (bus / light_rail); a snapshot of
        # one mode is never searched for a trip of the other (the state is reset on a mode change)
        self.vehicle_mode: Optional[str] = None
        self.vehicle_fetches_by_mode: Dict[str, int] = {}
        self._overlay: Optional[Any] = None
        self.static_load_s: Optional[float] = None
        self.last_vehicle_status: str = v4b.STATUS_NEVER

    @classmethod
    def load(cls, compare: CompareEngine, run_mode: str, vehicle_replay_path: Optional[Path] = None,
             want_shape: bool = True) -> "TrackEngine":
        return cls(compare, run_mode, vehicle_replay_path, want_shape)

    # ----- the lower-layer objects (built once, lazily) ------------------------------ #
    def ensure_states(self) -> None:
        """The V3 and V4B state objects over the static snapshot COMPARE already loaded."""
        if self.tracker is not None:
            return
        if self.compare.loaded_static is None:
            started = time.perf_counter()
            self.compare.loaded_static = v3.load_static_once(want_scheduled=self.run_mode != MODE_ANALYTICAL)
            self.static_load_s = time.perf_counter() - started
        loaded = self.compare.loaded_static
        self.realtime_state = v3.RealtimeState(stops=loaded["stops"], static=loaded["static"],
                                               scheduled=loaded["scheduled"])
        anchor = None
        if self.run_mode == MODE_REPLAY:
            self.vehicle_state = AnchoredVehicleState(static=loaded["static"])
            if self.vehicle_replay_path is not None:
                sidecar = phase2.read_replay_sidecar(self.vehicle_replay_path)
                if sidecar.get("captured_at") is None and self.compare.replay_path is not None:
                    sidecar = phase2.read_replay_sidecar(self.compare.replay_path)
                anchor = int(sidecar["captured_at"]) if sidecar.get("captured_at") is not None else None
                self.vehicle_fetcher = v4b.file_vehicle_fetcher(self.vehicle_replay_path)
            else:
                self.vehicle_fetcher = _no_vehicle_fetcher(
                    "REPLAY without --replay-vehicles: no recorded vehicle snapshot to read.")
            self.vehicle_state.anchor_epoch = anchor
        else:
            self.vehicle_state = v4b.VehicleState(static=loaded["static"])
            if self.run_mode == MODE_ANALYTICAL:
                self.vehicle_fetcher = _no_vehicle_fetcher("ANALYTICAL mode - realtime not requested.")
            else:
                self.vehicle_fetcher = None            # V4B default: one bounded request, key from the environment
        self.shapes = v4d.shape_store_for(self.realtime_state) if self.want_shape else None
        self._overlay = SimpleNamespace(state=self.vehicle_state)
        window = {"controller": None,
                  "panel": SimpleNamespace(state=self.realtime_state, limit=self.limit),
                  "overlay": self._overlay,
                  "explorer": None, "figure": None}
        self.tracker = v4d.TripTracker(window, shapes=self.shapes, want_shape=self.want_shape)

    # ----- the two realtime feeds, independent ------------------------------------------ #
    def sync_trip_updates(self, context: Any) -> str:
        """Mirror COMPARE's Trip Update snapshot into the V3 state (no second request)."""
        self.ensure_states()
        state = self.realtime_state
        state.refreshes += 1
        if context is None or context.mode == MODE_ANALYTICAL:
            state.status = v3.STATUS_NEVER
            return state.status
        if context.status != "ok":
            state.error = context.error
            state.status = v3.STATUS_STALE if state.has_snapshot else v3.STATUS_UNAVAILABLE
            return state.status
        state.snapshot = v3.LiveSnapshot(
            feed_timestamp=context.feed_timestamp, observed_at=int(context.observed_at),
            fetched_at=int(context.fetched_at or context.observed_at),
            entity_count=int(context.entity_count), trip_update_count=int(context.trip_update_count),
            events_by_stop=context.events_by_stop)
        state.error = None
        state.status = v3.STATUS_OK
        state.successes += 1
        return state.status

    def trip_mode(self, trip_id: Optional[str]) -> str:
        """The mode of an exact trip through the service-intent tables (never the route name)."""
        return feed_adapter.mode_for_trip(self.compare.tables, trip_id)

    def select_vehicle_mode(self, mode: Optional[str]) -> str:
        """LIVE: the vehicle feed of this mode is used from now on. A change of mode drops the other mode's
        snapshot (a bus snapshot can never describe a light-rail trip, and the reverse)."""
        wanted = feed_adapter.normalise_mode(mode)
        self.ensure_states()
        if self.vehicle_mode is not None and wanted != self.vehicle_mode and self.run_mode == MODE_LIVE:
            self.vehicle_state = v4b.VehicleState(static=self.compare.loaded_static["static"])
            if self._overlay is not None:
                setattr(self._overlay, "state", self.vehicle_state)   # the tracker evaluates the fresh state
            self.last_vehicle_status = v4b.STATUS_NEVER
        self.vehicle_mode = wanted
        return wanted

    def refresh_vehicles(self, mode: Optional[str] = None) -> str:
        """ONE vehicle-position request (LIVE, the feed of ``mode``), one file read (REPLAY), none (ANALYTICAL).
        An explicit ``vehicle_fetcher`` (REPLAY file, tests) is used whatever the mode."""
        self.ensure_states()
        if mode is not None:
            self.select_vehicle_mode(mode)
        fetcher = self.vehicle_fetcher
        if fetcher is None and self.run_mode == MODE_LIVE:
            fetcher = feed_adapter.vehicle_position_fetcher(self.vehicle_mode or feed_adapter.MODE_BUS)
        self.vehicle_fetches += 1
        key = self.vehicle_mode or feed_adapter.MODE_BUS
        self.vehicle_fetches_by_mode[key] = self.vehicle_fetches_by_mode.get(key, 0) + 1
        self.last_vehicle_status = self.vehicle_state.refresh(fetcher)
        return self.last_vehicle_status

    # ----- pinning ------------------------------------------------------------------------ #
    def event_for(self, trip_id: str, stop_id: str) -> Optional[Any]:
        """The V2 ArrivalEvent of this exact trip at this stop in the current Trip Update snapshot."""
        state = self.realtime_state
        if state is None or not state.has_snapshot:
            return None
        for event in state.snapshot.events_for(str(stop_id)):
            if str(event.trip_id) == str(trip_id):
                return event
        return None

    def pin(self, scene: Dict[str, Any], trip_id: str, stop_id: str, stop_name: str) -> Any:
        """Pin ONE exact trip at ONE boarding stop; realtime through pin_event, else from the timetable."""
        self.ensure_states()
        tracked = self.tracker.tracked
        if tracked is not None and tracked.trip_id == str(trip_id) and tracked.source_stop_id == str(stop_id):
            return tracked                                   # same pin: keep its observation history
        event = self.event_for(trip_id, stop_id)
        if event is not None:
            tracked = v4d.pin_event(event, str(stop_id), str(stop_name))
        else:
            candidate = next((c for c in scene.get("trip_candidates") or [] if c["trip_id"] == str(trip_id)), None)
            call = self.scene_call(scene, trip_id, stop_id)
            scheduled = (call or {}).get("scheduled_epoch")
            tracked = v4d.TrackedTrip(
                trip_id=str(trip_id),
                route=str((candidate or {}).get("route_short_name") or "route unresolved"),
                route_short_name=(candidate or {}).get("route_short_name"),
                destination=(candidate or {}).get("trip_headsign"),
                source_stop_id=str(stop_id), source_stop_name=str(stop_name),
                scheduled_text=v3.clock(scheduled) if scheduled is not None else None,
                scheduled_epoch=int(scheduled) if scheduled is not None else None,
                expected_epoch=int(scheduled) if scheduled is not None else None,
                expected_basis=phase2.SOURCE_SCHEDULE if scheduled is not None else None,
                delay_seconds=None, pinned_at=int(time.time()))
        self.tracker.tracked = tracked               # a different trip: fresh object, no stale vehicle state
        self.tracker.selected_stop = str(stop_id)
        if self.want_shape:
            self.tracker.resolve_shape()
        return tracked

    @staticmethod
    def scene_call(scene: Dict[str, Any], trip_id: str, stop_id: str) -> Optional[Dict[str, Any]]:
        entry = CompareEngine.stop_entry(scene, stop_id)
        for call in ((entry or {}).get("catchability") or {}).get("trips") or []:
            if call["trip_id"] == str(trip_id):
                return call
        return None

    # ----- evaluation ---------------------------------------------------------------------- #
    def evaluate(self) -> Optional[str]:
        """The Week 7 state rule, unchanged: usable records -> LIVE / AMBIGUOUS; else LAST SEEN / UNAVAILABLE."""
        if self.tracker is None or self.tracker.tracked is None:
            return None
        return self.tracker.evaluate()

    def positions(self) -> List[Any]:
        return self.tracker.positions() if self.tracker is not None else []

    def expected_at_stop(self, scene: Dict[str, Any], trip_id: str, stop_id: str,
                         stop_mode: str) -> Dict[str, Any]:
        """The pinned trip's timing at the boarding stop NOW (Phase 2 inventory, re-read each refresh)."""
        entry = CompareEngine.stop_entry(scene, stop_id) or {}
        walking_time_s = entry.get("time_route_walking_time_s")
        realtime = self.compare.realtime
        service_key = (scene.get("intent") or {}).get("key")
        choice = self.compare.choice(service_key) if service_key else None
        if realtime is None or choice is None or walking_time_s is None:
            return {"status": TIME_NOT_LISTED, "reason": "no timing available", "call": None,
                    "pedestrian_arrival_epoch": None, "walking_time_s": walking_time_s}
        inventory = phase2.catchability_for_stop(str(stop_id), stop_mode, float(walking_time_s), realtime, choice,
                                                 self.compare.tables, margin_s=self.compare.margin_s,
                                                 schedule_fallback=self.compare.schedule_fallback)
        call = next((c for c in inventory.get("trips") or [] if c["trip_id"] == str(trip_id)), None)
        return {"status": call["status"] if call else TIME_NOT_LISTED,
                "reason": "" if call else (inventory.get("inventory_reason") or "trip not listed at this stop now"),
                "call": call,
                "pedestrian_arrival_epoch": inventory.get("pedestrian_arrival_epoch"),
                "walking_time_s": walking_time_s, "margin_s": inventory.get("margin_s"),
                "observed_at": inventory.get("observed_at")}

    # ----- the plain result the panel / export read ----------------------------------------- #
    def result(self, timing: Dict[str, Any], refresh_status: str) -> Dict[str, Any]:
        tracked = self.tracker.tracked
        vehicle = self.vehicle_state
        snapshot = vehicle.snapshot if vehicle is not None and vehicle.has_snapshot else None
        positions = []
        for obs in self.positions():
            positions.append({"lat": float(obs.latitude), "lon": float(obs.longitude),
                              "vehicle_id": obs.vehicle_id, "reported_at": obs.vehicle_timestamp,
                              "observed_at": obs.observed_at, "feed_timestamp": obs.feed_timestamp,
                              "age_text": v4d.age_phrase(obs, snapshot) if snapshot is not None else "age not available",
                              "reported_clock": v4a.clock(obs.vehicle_timestamp)})
        call = timing.get("call") or {}
        return {
            "trip_id": tracked.trip_id, "route_short_name": tracked.route_short_name, "route": tracked.route,
            "destination": tracked.destination,
            "source_stop_id": tracked.source_stop_id, "source_stop_name": tracked.source_stop_name,
            "pinned_scheduled_epoch": tracked.scheduled_epoch, "pinned_expected_epoch": tracked.expected_epoch,
            "pinned_expected_basis": tracked.expected_basis, "pinned_delay_s": tracked.delay_seconds,
            "tracking_state": tracked.state,
            "vehicle_feed_status": vehicle.status if vehicle is not None else v4b.STATUS_NEVER,
            "vehicle_feed_error": vehicle.error if vehicle is not None else None,
            "vehicle_snapshot": None if snapshot is None else {
                "fetched_at": snapshot.fetched_at, "feed_timestamp": snapshot.feed_timestamp,
                "observed_at": snapshot.observed_at, "record_count": snapshot.record_count,
                "valid_coordinate_count": snapshot.valid_coordinate_count},
            "trip_update_status": self.realtime_state.status if self.realtime_state is not None else v3.STATUS_NEVER,
            "positions": positions, "n_current": len(tracked.current),
            "last_seen_at": tracked.last_seen_at, "invalid_coordinates": tracked.invalid_coordinates,
            "refresh_failed": tracked.refresh_failed,
            "in_upcoming": tracked.in_upcoming, "in_raw_events": tracked.in_raw_events,
            "shape_id": tracked.shape_id, "shape_status": tracked.shape_status,
            "shape_points": len(tracked.shape_points or []),
            "planned_route": [[float(lon), float(lat)] for lat, lon in (tracked.shape_points or [])],
            # the boarding-stop timing NOW (Phase 2 inventory)
            "timing_status": timing.get("status"), "timing_reason": timing.get("reason"),
            "expected_arrival_epoch": call.get("expected_arrival_epoch"),
            "expected_time_source": call.get("expected_time_source"),
            "realtime_confirmed": bool(call.get("realtime_confirmed")),
            "scheduled_epoch": call.get("scheduled_epoch"), "delay_s": call.get("delay_s"),
            "slack_s": call.get("slack_s"),
            "pedestrian_arrival_epoch": timing.get("pedestrian_arrival_epoch"),
            "walking_time_s": timing.get("walking_time_s"), "margin_s": timing.get("margin_s"),
            "observed_at": timing.get("observed_at"),
            "last_refresh_status": refresh_status,
        }


# --------------------------------------------------------------------------- #
# wording (presentation only; every number comes from the dicts above)
# --------------------------------------------------------------------------- #
def clock(epoch: Optional[int]) -> str:
    return v3.clock(int(epoch)) if epoch is not None else "-"


def running_phrase(delay_s: Optional[int]) -> str:
    return v4d.running_phrase(delay_s)


def spare_phrase(result: Dict[str, Any]) -> str:
    """'About 5 min to spare' / 'About 5 min by timetable' / 'Not confirmed reachable in time'."""
    slack = result.get("slack_s")
    if slack is None:
        return "No timing for this trip at your stop in the current snapshot"
    if slack < 0:
        return "Not confirmed reachable in time" + ("" if result.get("realtime_confirmed") else " (timetable)")
    minutes = int(round(slack / 60.0))
    size = "Less than a minute" if slack < 30 else f"About {max(minutes, 1)} min"
    return f"{size} to spare" if result.get("realtime_confirmed") else f"{size} by timetable"


def expected_lines(result: Dict[str, Any]) -> List[str]:
    """EXPECTED AT YOUR STOP: realtime expected + scheduled + delay, or timetable + 'confirmation unavailable'."""
    if result.get("timing_status") == TIME_NOT_LISTED or result.get("expected_arrival_epoch") is None:
        lines = ["Not listed at your stop in the current snapshot"]
        if result.get("pinned_expected_epoch") is not None:
            lines.append(f"Expected when selected: {clock(result['pinned_expected_epoch'])}"
                         + ("" if result.get("pinned_expected_basis") != phase2.SOURCE_SCHEDULE else " (timetable)"))
        return lines
    if result.get("realtime_confirmed"):
        lines = [f"Expected {clock(result['expected_arrival_epoch'])}"]
        if result.get("scheduled_epoch") is not None:
            lines.append(f"Scheduled {clock(result['scheduled_epoch'])}  ·  {running_phrase(result.get('delay_s'))}")
        else:
            lines.append(f"Running: {running_phrase(result.get('delay_s'))}")
        return lines
    return [f"Scheduled {clock(result['expected_arrival_epoch'])}", "Realtime confirmation unavailable"]


def vehicle_lines(result: Dict[str, Any], run_mode: str) -> List[str]:
    """VEHICLE: the Week 7 state, the report age, and what an absence does NOT mean."""
    state = result.get("tracking_state")
    lines = [state or STATE_UNAVAILABLE]
    positions = result.get("positions") or []
    if state == STATE_AMBIGUOUS:
        lines.append(f"{len(positions)} vehicles report this same trip; none is chosen.")
    for p in positions:
        heading = "Reported " if state in (STATE_LIVE, STATE_AMBIGUOUS) else "Last reported "
        lines.append(f"{heading}{p['reported_clock']} ({p['age_text']})")
    if state == STATE_LAST_SEEN:
        if result.get("refresh_failed"):
            lines.append("Refresh failed; the position above is the last successful report.")
        else:
            lines.append("The current vehicle snapshot does not carry this trip. That is not a cancellation.")
    if state == STATE_UNAVAILABLE:
        if run_mode == MODE_ANALYTICAL:
            lines.append("No realtime requested in ANALYTICAL mode.")
        elif result.get("vehicle_feed_status") in (v4b.STATUS_UNAVAILABLE, v4b.STATUS_NEVER):
            lines.append("Vehicle feed unavailable. That says nothing about the service.")
        else:
            lines.append("No vehicle position reported for this trip yet. That says nothing about the service.")
    if result.get("invalid_coordinates"):
        lines.append(f"{result['invalid_coordinates']} report(s) with invalid coordinates ignored.")
    return lines


def technical_lines(result: Dict[str, Any]) -> List[str]:
    """The identifiers and raw values, for the Technical details box only."""
    lines = [
        "",
        "TRACK (demo_week7_trip_tracking backend, headless)",
        f"trip_id {result['trip_id']}  route_short_name {result.get('route_short_name')}  "
        f"destination {result.get('destination')}",
        f"boarding stop_id {result['source_stop_id']}  tracking_state {result['tracking_state']}",
        f"vehicle feed status {result.get('vehicle_feed_status')}  error {result.get('vehicle_feed_error') or '-'}  "
        f"trip update status {result.get('trip_update_status')}  last refresh {result.get('last_refresh_status')}",
    ]
    snap = result.get("vehicle_snapshot")
    if snap:
        lines.append(f"vehicle snapshot fetched_at {snap['fetched_at']}  feed_timestamp {snap['feed_timestamp']}  "
                     f"observed_at {snap['observed_at']}  records {snap['record_count']}  "
                     f"valid coordinates {snap['valid_coordinate_count']}")
    for p in result.get("positions") or []:
        lines.append(f"  raw position lat {p['lat']:.6f} lon {p['lon']:.6f}  vehicle_id {p['vehicle_id']}  "
                     f"vehicle timestamp {p['reported_at']}  age {p['age_text']}")
    lines += [
        f"in source stop's upcoming list {result.get('in_upcoming')}  in raw events {result.get('in_raw_events')}  "
        f"refresh_failed {result.get('refresh_failed')}  last_seen_at {result.get('last_seen_at')}",
        f"shape_id {result.get('shape_id') or '-'}  shape status '{result.get('shape_status')}'  "
        f"points {result.get('shape_points')}",
        f"expected arrival {result.get('expected_arrival_epoch')} = {clock(result.get('expected_arrival_epoch'))}  "
        f"source {result.get('expected_time_source') or '-'}  scheduled {result.get('scheduled_epoch')}  "
        f"delay_s {result.get('delay_s')}",
        f"pedestrian arrival {result.get('pedestrian_arrival_epoch')} = {clock(result.get('pedestrian_arrival_epoch'))}  "
        f"(observed_at {result.get('observed_at')} + walking_time_s {result.get('walking_time_s')} + "
        f"margin {result.get('margin_s')})  slack_s {result.get('slack_s')}  timing status {result.get('timing_status')}",
        PROVENANCE_NOTE[:PROVENANCE_NOTE.index(" No map matching")],
        PROVENANCE_NOTE[PROVENANCE_NOTE.index(" No map matching") + 1:],
    ]
    return lines
