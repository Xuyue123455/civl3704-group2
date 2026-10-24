"""Week 9 Phase 2 - decision support for ONE origin and ONE service intention.

THE QUESTION
------------
Given an origin inside the Glebe study area, an intended service (route +
direction, see ``service_intent.py``) and, where available, the current
transit snapshot: which of the boarding stops that service actually calls at
can the pedestrian reach in time for ONE exact trip, and - among those
stops only - which is attractive in terms of

    * terrain-related walking effort      (Minetti, effort-optimal route)
    * modelled walking time               (Tobler, time-optimal route)
    * walking distance                    (distance-optimal route)

shown SIDE BY SIDE. There is no combined score: no alpha * effort +
beta * time, no universal "best stop".

CATCHABILITY IS A CONSTRAINT, NOT AN OBJECTIVE
----------------------------------------------
Earlier versions of this engine ranked stops by "earliest catchable
boarding". That was wrong: on the SAME exact trip, boarding at an upstream
stop is earlier only because the bus has not yet reached the downstream
stop - the pedestrian arrives at the destination at the same time either
way, having walked further or climbed more. Boarding time is therefore not
optimised anywhere here. The logical order is now

    1. route + direction                       (service intent)
    2. the upcoming EXACT trips of that service (exact trip_id, never parsed)
    3. ONE trip: chosen manually, or the NEXT FEASIBLE TRIP automatically
    4. for that same trip, which stops are catchable  (the constraint)
    5. compare ONLY those stops by effort, walking time and distance

CATCHABILITY SEMANTICS (one stop i, one exact trip)
---------------------------------------------------
    pedestrian_arrival_i = observed_at + time_route_walking_time_i + margin_s
    bus_arrival_i        = that trip's expected arrival at stop i
    slack_i              = bus_arrival_i - pedestrian_arrival_i

``margin_s`` defaults to 0 (there is no documented policy for a margin).
``slack_i >= 0`` is ``catchable_under_current_model`` when the expected time
is realtime (an absolute feed time, or scheduled + realtime delay), and the
weaker ``scheduled_service_after_modelled_arrival`` when it comes from the
static timetable only (calendar-checked; nothing has confirmed the trip is
running). ``slack_i < 0`` is ``not_proven_catchable_under_current_model`` -
never "missed": a faster walker, or a late bus, is outside the model. The
pedestrian side is the MINIMUM modelled Tobler time on the trusted graph.

NEXT FEASIBLE TRIP
------------------
The upcoming exact trips of the chosen service are ordered by their
trip-level reference time: the earliest expected arrival, at or after the
observation instant, at any candidate stop (the trip's first upcoming call
in the study area). The next feasible trip is the first trip in that order
with at least one catchable stop (realtime-confirmed trips first; if none,
the first trip feasible on the timetable only). "Next" refers to the TRIP,
not to an earliest boarding stop.

What this never claims: that Tobler time is a measured pedestrian time; that
a realtime expected time is our prediction (it is feed information); that
one snapshot says anything about the next one.

MODES
-----
    LIVE        one request to the TfNSW Trip Update feed of the selected
                service's mode now (bus, or Inner West light rail v2 since
                Week 11; the caller passes the fetcher and ``feed_mode``)
    REPLAY      a previously saved snapshot (.pb + .json sidecar); the clock
                is the capture time and the scene says so. Never "live".
    ANALYTICAL  terrain + ranking + walking time; no realtime at all

Light rail (Week 11): the snapshot carries ``feed_mode``; a stop whose mode the
loaded snapshot does not cover gets the explicit status
``realtime_not_loaded_for_this_mode`` (formerly the unconditional
``light_rail_realtime_not_integrated``). Everything else - catchability, slack,
ranking, selection - is identical for every mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tor = load_script("time_optimal_route")        # three routes; imports compare_terrain_routes
rank = load_script("rank_candidate_stops")     # Phase 1 ranking + near ties
intent_mod = load_script("service_intent")     # route + direction choices

MODE_LIVE = "LIVE"
MODE_REPLAY = "REPLAY"
MODE_ANALYTICAL = "ANALYTICAL"
MODES = (MODE_LIVE, MODE_REPLAY, MODE_ANALYTICAL)

# catchability labels (one stop, one exact trip)
CATCHABLE = "catchable_under_current_model"
NOT_PROVEN = "not_proven_catchable_under_current_model"
SCHEDULED_AFTER = "scheduled_service_after_modelled_arrival"
NO_SERVICE_FOUND = "no_service_found"
TRIP_TIME_UNAVAILABLE = "selected_trip_time_unavailable_at_stop"
FEASIBLE_STATUSES = (CATCHABLE, SCHEDULED_AFTER)      # slack >= 0, either provenance

# trip-level feasibility (one exact trip across the candidate stops)
TRIP_FEASIBLE = "feasible"
TRIP_NOT_PROVEN = "not_proven_feasible"
BASIS_REALTIME = "realtime"
BASIS_SCHEDULE = "schedule_only"
BASIS_MIXED = "mixed"
SELECTION_AUTO = "automatic_next_feasible"
SELECTION_MANUAL = "manual"
DEFAULT_TRIP_HORIZON = 12                              # upcoming exact trips kept in the scene
TRIP_REFERENCE_DEFINITION = ("earliest expected arrival at or after observed_at at any "
                             "candidate stop the service calls at (its first upcoming call "
                             "in the study area); trips with no upcoming call are excluded")
NEXT_FEASIBLE_POLICY = ("trips in chronological order of their reference time; the first "
                        "with >= 1 realtime-confirmed catchable stop, else the first with "
                        ">= 1 stop feasible on the static timetable only; 'next' is the "
                        "trip, not an earliest boarding stop")

# what the stop comparison was made over
COMPARISON_FEASIBLE = "feasible_stops_of_selected_trip"
COMPARISON_NO_FEASIBLE_TRIP = "all_served_stops_no_feasible_trip"
COMPARISON_EMPTY = "none_selected_trip_not_feasible"

# why no service was found / what the realtime side could say
REALTIME_OK = "realtime_snapshot"
REALTIME_NONE_ANALYTICAL = "no_realtime_analytical_mode"
REALTIME_UNAVAILABLE = "realtime_unavailable"
REALTIME_MODE_NOT_LOADED = "realtime_not_loaded_for_this_mode"   # the loaded snapshot is another mode's feed
REALTIME_LIGHT_RAIL = REALTIME_MODE_NOT_LOADED                     # Week 9 name, kept for readers of the old status
REALTIME_NO_MATCH = "no_matching_service_in_snapshot"
SCHEDULE_NO_MATCH = "no_matching_service_in_static_timetable"
SCHEDULE_UNAVAILABLE = "schedule_fallback_unavailable"
SCHEDULE_DATE_NOT_COVERED = "static_snapshot_does_not_cover_date"
SCHEDULE_DISABLED = "schedule_fallback_disabled"
STOP_NOT_SERVED = "stop_not_served_by_selected_service"
STOP_NOT_ROUTABLE = "no_time_route_to_stop"

# where an expected time came from
SOURCE_REALTIME = "realtime"                       # absolute time in the feed
SOURCE_DERIVED = "derived_scheduled_plus_delay"    # V2: scheduled + feed delay
SOURCE_SCHEDULE = intent_mod.SCHEDULE_BASIS        # static timetable, calendar-checked

PARETO_NON_DOMINATED = "non_dominated"
PARETO_DOMINATED = "dominated"
PARETO_NOT_COMPARED = "not_compared"

DEFAULT_MARGIN_S = 0.0
RESULTS_DIR = CONFIG_ROOT / "results"
SCENE_JSON = "current_scene.json"
SCENE_GEOJSON = "current_scene.geojson"
SCENE_SCHEMA_VERSION = "1.1"

MODEL_STATEMENTS = [
    "Walking time is the Tobler (1993) hiking function integrated over 5 m DEM "
    "gradient cells along the time-optimal route on the trusted subgraph: a "
    "modelled minimum, not a measured pedestrian time.",
    "Terrain effort is the Minetti et al. (2002) cost of walking integrated the "
    "same way along the effort-optimal route (J/kg). Effort, time and distance "
    "are separate objectives and are never combined into a score.",
    "Catchability is a constraint, not an objective: one exact trip is chosen, "
    "the stops where slack = trip expected arrival - modelled pedestrian arrival "
    "is >= 0 are feasible, and only those stops are compared. Boarding time is "
    "not optimised: boarding the same trip upstream is not an earlier arrival.",
    "Expected service times are GTFS-Realtime feed information (an absolute "
    "time, or scheduled + delay) matched on the exact trip_id; they are not a "
    "prediction made here. One snapshot says nothing about the next one.",
    "A stop whose slack is negative is 'not proven catchable under the current "
    "model', never 'missed'.",
    "Scheduled-only times come from the static timetable (calendar-checked) "
    "and have not been confirmed by realtime.",
]


class Phase2Error(RuntimeError):
    """An input is missing or inconsistent; nothing is guessed."""


# --------------------------------------------------------------------------- #
# realtime: one snapshot, one of three modes
# --------------------------------------------------------------------------- #
@dataclass
class RealtimeContext:
    """The realtime side of one run: mode, snapshot, clock anchor, status."""

    mode: str
    status: str                          # ok | unavailable | none
    observed_at: Optional[int] = None    # the clock the scene is anchored to
    anchor: str = ""                     # where observed_at came from
    feed_timestamp: Optional[int] = None
    fetched_at: Optional[int] = None
    source_label: str = ""               # LIVE feed / REPLAY file / none
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    entity_count: int = 0
    trip_update_count: int = 0
    error: Optional[str] = None
    events_by_stop: Dict[str, List[Any]] = field(default_factory=dict)
    static_loaded: bool = False
    is_synthetic: bool = False
    feed_mode: Optional[str] = None     # which mode's feed this snapshot is (bus / light_rail); None = unspecified

    @property
    def has_snapshot(self) -> bool:
        return self.status == "ok" and self.observed_at is not None

    def covers_mode(self, mode: Any) -> bool:
        """True unless this snapshot is known to be another mode's feed."""
        return self.feed_mode is None or mode is None or str(self.feed_mode) == str(mode)

    def events_for(self, stop_id: Any) -> List[Any]:
        return list(self.events_by_stop.get(str(stop_id), []))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "status": self.status,
            "observed_at": self.observed_at,
            "observed_at_sydney": sydney_text(self.observed_at),
            "clock_anchor": self.anchor,
            "feed_timestamp": self.feed_timestamp,
            "feed_timestamp_sydney": sydney_text(self.feed_timestamp),
            "fetched_at": self.fetched_at,
            "source_label": self.source_label, "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "entity_count": self.entity_count,
            "trip_update_count": self.trip_update_count,
            "error": self.error, "is_synthetic": self.is_synthetic,
            "is_live": self.mode == MODE_LIVE and self.status == "ok",
            "never_label_as_live": self.mode != MODE_LIVE,
        }


def sydney_text(epoch: Optional[int]) -> Optional[str]:
    if epoch is None:
        return None
    from zoneinfo import ZoneInfo
    return datetime.fromtimestamp(int(epoch), ZoneInfo(intent_mod.SYDNEY_TZ)
                                  ).strftime("%Y-%m-%d %H:%M:%S %Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_replay_sidecar(payload_path: Path) -> Dict[str, Any]:
    """The .json beside a saved .pb: capture time and provenance, never a key."""
    sidecar = payload_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def load_realtime(mode: str, replay_path: Optional[Path] = None,
                  loaded_static: Optional[Dict[str, Any]] = None,
                  fetcher: Optional[Callable[[], bytes]] = None,
                  want_scheduled: bool = True,
                  feed_mode: Optional[str] = None, feed_label: Optional[str] = None) -> RealtimeContext:
    """Build the realtime side for one run.

    LIVE fetches once through V3 (API key from the environment, used inside
    V1 and discarded). REPLAY reads a saved payload through V3's file
    fetcher and anchors the clock on the sidecar's ``captured_at`` (or, if
    absent, the feed header timestamp). ANALYTICAL loads nothing.
    """
    mode = str(mode).upper()
    if mode not in MODES:
        raise Phase2Error(f"mode must be one of {MODES}, not {mode!r}")
    if mode == MODE_ANALYTICAL:
        return RealtimeContext(mode=mode, status="none", anchor="wall_clock",
                               observed_at=int(time.time()),
                               source_label="no realtime (analytical mode)")

    v3 = load_script("demo_week7_realtime")
    v2 = v3.v2
    if loaded_static is None:
        loaded_static = v3.load_static_once(want_scheduled=want_scheduled)
    state = v3.RealtimeState(stops=loaded_static["stops"],
                             static=loaded_static["static"],
                             scheduled=loaded_static["scheduled"])
    context = RealtimeContext(mode=mode, status="unavailable",
                              static_loaded=loaded_static["static"] is not None)

    if mode == MODE_REPLAY:
        if replay_path is None and fetcher is None:
            raise Phase2Error("REPLAY mode needs a saved snapshot path.")
        sidecar = read_replay_sidecar(Path(replay_path)) if replay_path else {}
        source = fetcher if fetcher is not None else v3.file_fetcher(Path(replay_path))
        context.source_label = f"REPLAY of saved snapshot {Path(replay_path).name if replay_path else '<bytes>'}"
        context.source_path = str(replay_path) if replay_path else None
        if replay_path and Path(replay_path).exists():
            context.source_sha256 = file_sha256(Path(replay_path))
        context.is_synthetic = bool(sidecar.get("synthetic", False))
    else:
        source = fetcher
        context.source_label = f"LIVE TfNSW {feed_label or 'bus'} Trip Update feed"
        sidecar = {}
    context.feed_mode = feed_mode

    status = state.refresh(source)
    if status != v3.STATUS_OK or state.snapshot is None:
        context.status = "unavailable"
        context.error = state.error
        context.observed_at = int(time.time())
        context.anchor = "wall_clock"
        return context

    snapshot = state.snapshot
    context.status = "ok"
    context.feed_timestamp = snapshot.feed_timestamp
    context.fetched_at = snapshot.fetched_at
    context.entity_count = snapshot.entity_count
    context.trip_update_count = snapshot.trip_update_count
    context.events_by_stop = snapshot.events_by_stop
    if mode == MODE_REPLAY:
        if sidecar.get("captured_at") is not None:
            context.observed_at = int(sidecar["captured_at"])
            context.anchor = "replay_sidecar_captured_at"
        elif snapshot.feed_timestamp is not None:
            context.observed_at = int(snapshot.feed_timestamp)
            context.anchor = "replay_feed_header_timestamp"
        else:
            raise Phase2Error(
                "The replay snapshot has neither a sidecar captured_at nor a "
                "feed header timestamp, so the scene cannot be anchored in time.")
    else:
        context.observed_at = int(snapshot.observed_at)
        context.anchor = "live_decode_time"
    return context


# --------------------------------------------------------------------------- #
# per-stop catchability
# --------------------------------------------------------------------------- #
def event_matches_intent(event: Any, choice: Any, tables: Any) -> bool:
    """Exact trip_id -> static trip -> (route_short_name, direction_id)."""
    if event.trip_id is None:
        return False
    return bool(tables.trip_matches(event.trip_id, choice))


def expected_time_source(event: Any) -> Optional[str]:
    if event.sort_basis == "realtime":
        return SOURCE_REALTIME
    if event.sort_basis == "derived":
        return SOURCE_DERIVED
    return None


def catchability_for_stop(
    stop_id: str, mode: str, walking_time_s: Optional[float], realtime: RealtimeContext,
    choice: Any, tables: Any, margin_s: float = DEFAULT_MARGIN_S,
    schedule_fallback: bool = True,
) -> Dict[str, Any]:
    """The exact upcoming trips of the chosen service at ONE stop, with slack.

    This is an INVENTORY, not a decision: every matching trip at this stop
    (realtime first; the static timetable only when the snapshot carries
    nothing for this stop) is listed with its expected arrival, its
    provenance and its slack against THIS stop's modelled pedestrian
    arrival. No trip is chosen here and no "first catchable" is picked -
    the trip is chosen across stops by ``select_next_feasible_trip`` /
    ``apply_trip_selection``, which then fill the selected-trip fields
    (``status``, ``slack_s``, ``trip_expected_arrival_epoch`` ...).
    """
    out: Dict[str, Any] = {
        "stop_id": str(stop_id),
        "observed_at": realtime.observed_at,
        "observed_at_sydney": sydney_text(realtime.observed_at),
        "time_route_walking_time_s": walking_time_s,
        "margin_s": float(margin_s),
        "pedestrian_arrival_epoch": None,
        "pedestrian_arrival_sydney": None,
        "pedestrian_modelled_arrival_epoch": None,       # same value; kept for schema 1.0 readers
        "pedestrian_modelled_arrival_sydney": None,
        "realtime_status": "",
        "inventory_reason": "",
        "services_matching_intent_in_snapshot": 0,
        "services_already_past_at_observation": 0,
        "trips": [],
        # filled per selected trip by apply_trip_selection
        "status": NO_SERVICE_FOUND, "reason": "",
        "trip_id": None,
        "trip_expected_arrival_epoch": None, "trip_expected_arrival_sydney": None,
        "slack_s": None,
        "expected_time_source": None,
        "realtime_confirmed": False,
    }
    if walking_time_s is None or not np.isfinite(walking_time_s) or realtime.observed_at is None:
        out["inventory_reason"] = out["reason"] = STOP_NOT_ROUTABLE
        out["realtime_status"] = REALTIME_NONE_ANALYTICAL if realtime.mode == MODE_ANALYTICAL else realtime.status
        return out
    pedestrian = int(round(realtime.observed_at + float(walking_time_s) + float(margin_s)))
    for key in ("pedestrian_arrival_epoch", "pedestrian_modelled_arrival_epoch"):
        out[key] = pedestrian
    for key in ("pedestrian_arrival_sydney", "pedestrian_modelled_arrival_sydney"):
        out[key] = sydney_text(pedestrian)

    # ---- realtime side -------------------------------------------------- #
    candidates: List[Dict[str, Any]] = []
    if realtime.mode != MODE_ANALYTICAL and not realtime.covers_mode(mode):
        out["realtime_status"] = REALTIME_MODE_NOT_LOADED        # e.g. a bus snapshot while a light-rail stop is asked
    elif realtime.mode == MODE_ANALYTICAL:
        out["realtime_status"] = REALTIME_NONE_ANALYTICAL
    elif not realtime.has_snapshot:
        out["realtime_status"] = REALTIME_UNAVAILABLE
    else:
        matching = [e for e in realtime.events_for(stop_id)
                    if event_matches_intent(e, choice, tables) and e.sort_epoch is not None]
        out["services_matching_intent_in_snapshot"] = len(matching)
        out["realtime_status"] = REALTIME_OK if matching else REALTIME_NO_MATCH
        for event in matching:
            candidates.append({
                "trip_id": str(event.trip_id),
                "route_short_name": event.route_short_name or choice.route_short_name,
                "trip_headsign": event.trip_headsign or (choice.headsigns[0] if choice.headsigns else None),
                "direction_id": choice.direction_id,
                "expected_arrival_epoch": int(event.sort_epoch),
                "expected_arrival_sydney": sydney_text(int(event.sort_epoch)),
                "expected_time_source": expected_time_source(event),
                "scheduled_epoch": event.scheduled_epoch,
                "delay_s": (event.arrival.get("delay") if event.arrival.get("delay") is not None
                            else event.departure.get("delay")),
                "schedule_relationship": event.schedule_relationship,
                "realtime_confirmed": True,
            })

    # ---- schedule fallback (only when realtime gave nothing here) -------- #
    if not candidates:
        if not schedule_fallback:
            out["inventory_reason"] = SCHEDULE_DISABLED
        elif not tables.has_calendar:
            out["inventory_reason"] = SCHEDULE_UNAVAILABLE
        else:
            service_date = intent_mod.sydney_service_date(realtime.observed_at)
            start, end = tables_calendar_range(tables)
            if start and end and not (start <= service_date <= end):
                out["inventory_reason"] = SCHEDULE_DATE_NOT_COVERED
            else:
                visits = tables.scheduled_visits(stop_id, choice, service_date)
                for row in visits.itertuples():
                    candidates.append({
                        "trip_id": str(row.trip_id),
                        "route_short_name": row.route_short_name,
                        "trip_headsign": row.trip_headsign,
                        "direction_id": row.direction_id,
                        "expected_arrival_epoch": int(row.scheduled_epoch),
                        "expected_arrival_sydney": sydney_text(int(row.scheduled_epoch)),
                        "expected_time_source": SOURCE_SCHEDULE,
                        "scheduled_epoch": int(row.scheduled_epoch),
                        "delay_s": None, "schedule_relationship": None,
                        "realtime_confirmed": False,
                        "service_date": service_date,
                    })
                if not candidates:
                    out["inventory_reason"] = SCHEDULE_NO_MATCH
    if not candidates:
        if not out["inventory_reason"]:
            out["inventory_reason"] = out["realtime_status"] or REALTIME_NO_MATCH
        out["reason"] = out["inventory_reason"]
        return out

    # ---- slack of every trip against THIS stop's pedestrian arrival ------ #
    # Calls already past at the observation instant are kept in the list
    # (their slack is negative by construction) but counted separately and
    # never used as a trip's reference time. Nothing is fabricated.
    candidates.sort(key=lambda c: (c["expected_arrival_epoch"], c["trip_id"]))
    for c in candidates:
        c["already_past_at_observation"] = bool(c["expected_arrival_epoch"] < realtime.observed_at)
        c["slack_s"] = int(c["expected_arrival_epoch"] - pedestrian)
        c["status"] = slack_status(c["slack_s"], c["realtime_confirmed"])
    out["services_already_past_at_observation"] = sum(1 for c in candidates if c["already_past_at_observation"])
    out["trips"] = candidates
    out["reason"] = ""
    return out


def slack_status(slack_s: Optional[int], realtime_confirmed: bool) -> str:
    """slack >= 0 -> catchable (realtime) / scheduled-after (timetable); < 0 -> not proven."""
    if slack_s is None:
        return NO_SERVICE_FOUND
    if slack_s < 0:
        return NOT_PROVEN
    return CATCHABLE if realtime_confirmed else SCHEDULED_AFTER


def tables_calendar_range(tables: Any) -> Tuple[Optional[str], Optional[str]]:
    if tables.calendar is None or tables.calendar.empty:
        return None, None
    return str(tables.calendar.start_date.min()), str(tables.calendar.end_date.max())


# --------------------------------------------------------------------------- #
# exact trips across stops: grouping, next feasible trip, selection
# --------------------------------------------------------------------------- #
def build_trip_candidates(stops_out: List[Dict[str, Any]], observed_at: Optional[int],
                          horizon: int = DEFAULT_TRIP_HORIZON) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group the per-stop inventories by EXACT trip_id and order the trips.

    One entry per exact trip (never collapsed by route, headsign or time).
    The chronological reference of a trip is its earliest expected arrival
    at or after ``observed_at`` at any candidate stop; a trip whose every
    call is already past has no reference and is left out (counted). The
    first ``horizon`` trips are kept; the rest are counted, not listed.
    """
    visits: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for stop in stops_out:
        catch = stop.get("catchability") or {}
        for trip in catch.get("trips") or []:
            visits.setdefault(trip["trip_id"], []).append((stop["stop_id"], trip))
    candidates: List[Dict[str, Any]] = []
    past_trips = 0
    for trip_id, calls in visits.items():
        upcoming = [(s, t) for s, t in calls if not t["already_past_at_observation"]]
        if not upcoming:
            past_trips += 1
            continue
        ref_stop, ref = min(upcoming, key=lambda st: (st[1]["expected_arrival_epoch"], st[0]))
        catchable = sorted(s for s, t in calls if t["status"] == CATCHABLE)
        schedule_only = sorted(s for s, t in calls if t["status"] == SCHEDULED_AFTER)
        not_proven = sorted(s for s, t in calls if t["status"] == NOT_PROVEN)
        sources = {t["expected_time_source"] for _s, t in calls}
        if sources <= {SOURCE_REALTIME, SOURCE_DERIVED}:
            basis = BASIS_REALTIME
        elif sources == {SOURCE_SCHEDULE}:
            basis = BASIS_SCHEDULE
        else:
            basis = BASIS_MIXED
        feasible = catchable + schedule_only
        candidates.append({
            "trip_id": trip_id,
            "route_short_name": ref["route_short_name"],
            "trip_headsign": ref["trip_headsign"],
            "direction_id": ref["direction_id"],
            "reference_epoch": int(ref["expected_arrival_epoch"]),
            "reference_sydney": ref["expected_arrival_sydney"],
            "reference_stop_id": ref_stop,
            "n_stops_with_times": len(calls),
            "n_upcoming_calls": len(upcoming),
            "n_catchable_stops": len(catchable),
            "n_schedule_only_stops": len(schedule_only),
            "n_not_proven_stops": len(not_proven),
            "n_feasible_stops": len(feasible),
            "catchable_stop_ids": catchable,
            "schedule_only_stop_ids": schedule_only,
            "not_proven_stop_ids": not_proven,
            "feasible_stop_ids": sorted(feasible),
            "feasibility": TRIP_FEASIBLE if feasible else TRIP_NOT_PROVEN,
            "basis": basis,
            "is_next_feasible": False,
            "is_selected": False,
        })
    candidates.sort(key=lambda c: (c["reference_epoch"], c["trip_id"]))
    kept, beyond = candidates[:horizon], candidates[horizon:]
    next_id = select_next_feasible_trip(kept)
    for c in kept:
        c["is_next_feasible"] = c["trip_id"] == next_id
    info = {
        "reference_definition": TRIP_REFERENCE_DEFINITION,
        "next_feasible_policy": NEXT_FEASIBLE_POLICY,
        "horizon": int(horizon),
        "trips_listed": len(kept),
        "trips_beyond_horizon": len(beyond),
        "trips_already_past_at_observation": past_trips,
        "next_feasible_trip_id": next_id,
    }
    return kept, info


def select_next_feasible_trip(candidates: Sequence[Dict[str, Any]]) -> Optional[str]:
    """NEXT_FEASIBLE_POLICY: chronological; realtime-confirmed feasibility first."""
    for c in candidates:
        if c["n_catchable_stops"] >= 1:
            return c["trip_id"]
    for c in candidates:
        if c["n_schedule_only_stops"] >= 1:
            return c["trip_id"]
    return None


def trip_source_summary(calls: Sequence[Dict[str, Any]]) -> Optional[str]:
    sources = {t.get("expected_time_source") for t in calls if t.get("expected_time_source")}
    if not sources:
        return None
    if sources <= {SOURCE_REALTIME, SOURCE_DERIVED}:
        return BASIS_REALTIME
    if sources == {SOURCE_SCHEDULE}:
        return BASIS_SCHEDULE
    return BASIS_MIXED


# --------------------------------------------------------------------------- #
# Pareto over the WALKING objectives (effort, modelled walking time) - among
# the feasible stops of one exact trip only. Catchability never enters it.
# --------------------------------------------------------------------------- #
def pareto_status(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Non-dominated set over (effort_j_per_kg asc, walking_time_s asc).

    Only rows with both numbers take part; a row is dominated when another
    row is no worse on both and strictly better on at least one. Equal pairs
    are both kept. Nothing is scored or ranked here; dominance is reported
    with the dominating stop named so the reader can check it.
    """
    comparable = [r for r in rows if r.get("effort_j_per_kg") is not None
                  and r.get("walking_time_s") is not None]
    out: Dict[str, Dict[str, Any]] = {
        r["stop_id"]: {"pareto_status": PARETO_NOT_COMPARED, "dominated_by": []}
        for r in rows}
    for a in comparable:
        dominators = []
        for b in comparable:
            if a is b:
                continue
            no_worse = (b["effort_j_per_kg"] <= a["effort_j_per_kg"]
                        and b["walking_time_s"] <= a["walking_time_s"])
            strictly = (b["effort_j_per_kg"] < a["effort_j_per_kg"]
                        or b["walking_time_s"] < a["walking_time_s"])
            if no_worse and strictly:
                dominators.append(b["stop_id"])
        out[a["stop_id"]] = {
            "pareto_status": PARETO_DOMINATED if dominators else PARETO_NON_DOMINATED,
            "dominated_by": sorted(dominators),
        }
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def phase1_table(table: pd.DataFrame, routes_by_stop: Dict[str, Dict[str, Any]],
                 cost_table: pd.DataFrame, near_tie_pct: float) -> pd.DataFrame:
    """The Phase 1 effort ranking of ALL candidates, by the Week 9 code."""
    edge_rows = []
    for stop_id, routes in routes_by_stop.items():
        for obj in (tor.OBJECTIVE_DISTANCE, tor.OBJECTIVE_EFFORT):
            route = routes.get(obj)
            if route is None:
                continue
            for hop in route["hops"]:
                edge_rows.append({"stop_id": stop_id, "metric": obj, **hop})
    route_edges = pd.DataFrame(edge_rows, columns=["stop_id", "metric", "u", "v", "key"]) \
        if not edge_rows else pd.DataFrame(edge_rows)
    times = rank.route_walking_times(route_edges, cost_table)
    return rank.build_experiment_table(table, times, near_tie_pct)


def _clean(value: Any) -> Any:
    """JSON-safe scalars: NaN -> None, numpy -> python."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def route_summary(route: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if route is None:
        return None
    seconds = route.get("walking_time_s")
    return {
        "distance_m": round(float(route["distance"]), 3),
        "effort_j_per_kg": round(float(route["effort"]), 4),
        "walking_time_s": round(float(seconds), 3) if seconds is not None else None,
        "walking_time_min": round(float(seconds) / 60.0, 3) if seconds is not None else None,
        "edges": len(route["edges"]),
        "nodes": [int(n) for n in route["nodes"]],
        "edge_keys": [[int(u), int(v), int(k)] for u, v, k in route["edges"]],
    }


def route_identity(routes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Which of the three routes share one path - so no consumer draws a line twice.

    Purely a comparison of the edge sequences the searches returned; nothing
    is recomputed. ``groups`` lists the objectives per distinct path, in the
    order distance, effort, time.
    """
    present = [(obj, tuple(tuple(e) for e in routes[obj]["edges"]))
               for obj in (tor.OBJECTIVE_DISTANCE, tor.OBJECTIVE_EFFORT, tor.OBJECTIVE_TIME)
               if routes.get(obj) is not None]
    if not present:
        return None
    groups: List[List[str]] = []
    paths: List[Any] = []
    for obj, path in present:
        if path in paths:
            groups[paths.index(path)].append(obj)
        else:
            paths.append(path)
            groups.append([obj])
    return {
        "distinct_paths": len(groups),
        "groups": groups,
        "distance_equals_effort": any({"distance", "effort"} <= set(g) for g in groups),
        "time_equals_effort": any({"effort", "time"} <= set(g) for g in groups),
        "time_equals_distance": any({"distance", "time"} <= set(g) for g in groups),
        "all_same": len(groups) == 1 and len(present) == 3,
    }


def run_phase2(
    context: Dict[str, Any], tables: Any, latitude: float, longitude: float,
    service_key: str, realtime: RealtimeContext, cost_table: pd.DataFrame,
    margin_s: float = DEFAULT_MARGIN_S, near_tie_pct: float = rank.DEFAULT_NEAR_TIE_PCT,
    schedule_fallback: bool = True, stops: Optional[pd.DataFrame] = None,
    origin_label: str = "", trip_id: Optional[str] = None,
    trip_horizon: int = DEFAULT_TRIP_HORIZON,
) -> Dict[str, Any]:
    """One origin, one intention, one snapshot -> the scene (a plain dict).

    ``trip_id`` chooses an exact trip manually; ``None`` lets
    ``select_next_feasible_trip`` choose the next feasible trip. Either way
    the per-stop catchability, the highlights and the Pareto shortlist are
    derived by ``apply_trip_selection`` from the same inventory, so a later
    manual choice needs no re-routing.
    """
    choice = tables.choice(service_key)
    table, routes_by_stop, provenance = tor.solve_origin(context, latitude, longitude, stops)
    phase1 = phase1_table(table, routes_by_stop, cost_table, near_tie_pct)
    phase1 = phase1.set_index(phase1.stop_id.astype(str))
    table = table.set_index(table.stop_id.astype(str), drop=False)

    stops_out: List[Dict[str, Any]] = []
    for stop_id, row in table.iterrows():
        routes = routes_by_stop.get(stop_id, {})
        p1 = phase1.loc[stop_id] if stop_id in phase1.index else None
        serves = choice.serves(stop_id)
        time_route = routes.get(tor.OBJECTIVE_TIME)
        walking_time_s = (float(time_route["walking_time_s"]) if time_route else None)
        entry: Dict[str, Any] = {
            "stop_id": stop_id, "stop_name": row["stop_name"], "mode": row["mode"],
            "lat": float(row["stop_lat"]), "lon": float(row["stop_lon"]),
            "eligibility": row["comparison_status"],
            "status_reason": row["status_reason"],
            "serves_selected_service": bool(serves),
            "effort_rank": (int(p1["effort_rank"]) if p1 is not None and pd.notna(p1["effort_rank"]) else None),
            "near_tie_group": _clean(p1["near_tie_group"]) if p1 is not None else None,
            "effort_gap_to_previous_pct": _clean(p1["effort_gap_to_previous_pct"]) if p1 is not None else None,
            "effort_j_per_kg": _clean(row.get("effort_route_effort_j_per_kg")),
            "distance_m": _clean(row.get("effort_route_distance_m")),
            "shortest_distance_m": _clean(row.get("distance_route_distance_m")),
            "time_route_walking_time_s": _clean(row.get("time_route_walking_time_s")),
            "time_route_walking_time_min": _clean(row.get("time_route_walking_time_min")),
            "time_route_effort_j_per_kg": _clean(row.get("time_route_effort_j_per_kg")),
            "time_route_distance_m": _clean(row.get("time_route_distance_m")),
            "time_route_same_as_effort_route": _clean(row.get("time_route_same_as_effort_route")),
            "effort_saving_vs_shortest_pct": _clean(row.get("effort_saving_pct")),
            "extra_distance_vs_shortest_pct": _clean(row.get("distance_detour_pct")),
            "route_identity": route_identity(routes),
            "routes": {
                "distance": route_summary(routes.get(tor.OBJECTIVE_DISTANCE)),
                "effort": route_summary(routes.get(tor.OBJECTIVE_EFFORT)),
                "time": route_summary(time_route),
            },
            "catchability": None,
            "pareto_status": PARETO_NOT_COMPARED, "dominated_by": [],
            "highlights": [],
        }
        if serves:
            entry["catchability"] = catchability_for_stop(
                stop_id, row["mode"], walking_time_s, realtime, choice, tables,
                margin_s, schedule_fallback)
        else:
            entry["catchability"] = {"status": NO_SERVICE_FOUND, "reason": STOP_NOT_SERVED,
                                     "stop_id": stop_id, "trips": []}
        stops_out.append(entry)

    # ---- exact trips across the stops the service calls at --------------- #
    trip_candidates, trip_info = build_trip_candidates(stops_out, realtime.observed_at, trip_horizon)
    listed = {c["trip_id"] for c in trip_candidates}
    for s in stops_out:
        catch = s["catchability"]
        if catch and catch.get("trips"):
            catch["trips"] = [t for t in catch["trips"] if t["trip_id"] in listed]

    scene = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "generated_at": int(time.time()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": realtime.mode,
        "mode_label": mode_label(realtime),
        "realtime": realtime.as_dict(),
        "origin": {
            "label": origin_label, "lat": float(latitude), "lon": float(longitude),
            "nearest_node": provenance["origin"]["nearest_node"],
            "snap_distance_m": provenance["origin"]["snap_distance_m"],
            "component_id": provenance["origin"]["component_id"],
        },
        "intent": choice.as_dict(),
        "parameters": {
            "margin_s": float(margin_s), "margin_policy": "none documented; default 0",
            "near_tie_pct": float(near_tie_pct),
            "schedule_fallback": bool(schedule_fallback),
            "service_date": intent_mod.sydney_service_date(realtime.observed_at)
            if realtime.observed_at else None,
            "static_snapshot_calendar_range": list(tables_calendar_range(tables)),
            "trip_horizon": int(trip_horizon),
        },
        "model_statements": MODEL_STATEMENTS,
        "no_combined_score": True,
        "catchability_is_a_constraint_not_an_objective": True,
        "trip_selection": trip_info,
        "trip_candidates": trip_candidates,
        "selected_trip": None,
        "summary": {},
        "stops": stops_out,
        "provenance": {
            "routing": {k: v for k, v in provenance.items() if k != "origin"},
            "cost_table_columns": int(len(cost_table.columns)),
            "service_intent": tables.provenance,
        },
    }
    return apply_trip_selection(scene, trip_id)


def apply_trip_selection(scene: Dict[str, Any], trip_id: Optional[str] = None) -> Dict[str, Any]:
    """Choose ONE exact trip (manual ``trip_id`` or the next feasible trip) and
    derive, from the scene's existing inventory, the per-stop catchability for
    that trip, the feasible-stop comparison and the summary. No routing, no
    feed access: this is cheap and the UI calls it when the viewer changes
    trip. Returns the same scene object, updated in place.
    """
    candidates = scene["trip_candidates"]
    by_id = {c["trip_id"]: c for c in candidates}
    selection = SELECTION_MANUAL
    if trip_id is None:
        trip_id = scene["trip_selection"].get("next_feasible_trip_id")
        selection = SELECTION_AUTO
    elif str(trip_id) not in by_id:
        raise Phase2Error(f"trip {trip_id!r} is not among the listed upcoming trips of this service")
    trip_id = str(trip_id) if trip_id is not None else None
    for c in candidates:
        c["is_selected"] = c["trip_id"] == trip_id
    chosen = by_id.get(trip_id) if trip_id else None

    # ---- per-stop catchability for the selected trip --------------------- #
    in_scope = [s for s in scene["stops"] if s["serves_selected_service"]
                and s["eligibility"] == tor.STATUS_COMPARED]
    selected_calls: List[Dict[str, Any]] = []
    for s in in_scope:
        catch = s["catchability"]
        trips = catch.get("trips") or []
        call = next((t for t in trips if t["trip_id"] == trip_id), None) if trip_id else None
        catch["trip_id"] = trip_id
        if call is not None:
            catch["status"] = call["status"]
            catch["reason"] = ""
            catch["trip_expected_arrival_epoch"] = call["expected_arrival_epoch"]
            catch["trip_expected_arrival_sydney"] = call["expected_arrival_sydney"]
            catch["slack_s"] = call["slack_s"]
            catch["expected_time_source"] = call["expected_time_source"]
            catch["realtime_confirmed"] = bool(call["realtime_confirmed"])
            selected_calls.append(call)
            continue
        catch["trip_expected_arrival_epoch"] = catch["trip_expected_arrival_sydney"] = None
        catch["slack_s"] = None
        catch["expected_time_source"] = None
        catch["realtime_confirmed"] = False
        if trip_id is not None:
            catch["status"] = TRIP_TIME_UNAVAILABLE
            catch["reason"] = f"no expected time for trip {trip_id} at this stop"
        elif trips:
            catch["status"] = NOT_PROVEN
            catch["reason"] = "no listed upcoming trip has a non-negative slack at this stop"
        else:
            catch["status"] = NO_SERVICE_FOUND
            catch["reason"] = catch.get("inventory_reason") or REALTIME_NO_MATCH

    # ---- the comparison: feasible stops of the selected trip only --------- #
    feasible = [s for s in in_scope if s["catchability"]["status"] in FEASIBLE_STATUSES]
    if chosen is not None:
        pool = feasible
        basis = COMPARISON_FEASIBLE if feasible else COMPARISON_EMPTY
    else:
        pool = in_scope
        basis = COMPARISON_NO_FEASIBLE_TRIP
    pool_ids = {s["stop_id"] for s in pool}
    rows_for_pareto = [{"stop_id": s["stop_id"], "effort_j_per_kg": s["effort_j_per_kg"],
                        "walking_time_s": s["time_route_walking_time_s"]} for s in pool]
    pareto = pareto_status(rows_for_pareto)
    for s in scene["stops"]:
        s["highlights"] = []
        s.update(pareto.get(s["stop_id"], {"pareto_status": PARETO_NOT_COMPARED, "dominated_by": []}))

    def pick(key: Callable[[Dict[str, Any]], Any]) -> Optional[str]:
        valid = [s for s in pool if key(s) is not None]
        if not valid:
            return None
        return min(valid, key=lambda s: (key(s), s["stop_id"]))["stop_id"]

    lowest_effort = pick(lambda s: s["effort_j_per_kg"])
    lowest_time = pick(lambda s: s["time_route_walking_time_s"])
    shortest = pick(lambda s: s["shortest_distance_m"])
    shortlist = sorted(s["stop_id"] for s in pool if s["pareto_status"] == PARETO_NON_DOMINATED)
    for s in scene["stops"]:
        if s["stop_id"] == lowest_effort:
            s["highlights"].append("lowest_terrain_effort")
        if s["stop_id"] == lowest_time:
            s["highlights"].append("lowest_modelled_walking_time")
        if s["stop_id"] == shortest:
            s["highlights"].append("shortest_walk")
        if s["stop_id"] in shortlist:
            s["highlights"].append("pareto_non_dominated")

    near_tie_pct = scene["parameters"]["near_tie_pct"]
    near_tie_note = None
    if lowest_effort is not None:
        anchor = next(s for s in pool if s["stop_id"] == lowest_effort)
        if anchor["near_tie_group"] is not None:
            peers = [s["stop_id"] for s in pool
                     if s["near_tie_group"] == anchor["near_tie_group"] and s["stop_id"] != lowest_effort]
            if peers:
                near_tie_note = (f"{lowest_effort} is provisionally near-tied (within "
                                 f"{near_tie_pct:g} %) with {', '.join(peers)} in the Phase 1 "
                                 "ranking; it is not shown to be lower-effort than them.")

    realtime_catchable = sorted(s["stop_id"] for s in in_scope if s["catchability"]["status"] == CATCHABLE)
    scheduled_only = sorted(s["stop_id"] for s in in_scope if s["catchability"]["status"] == SCHEDULED_AFTER)
    not_proven = sorted(s["stop_id"] for s in in_scope if s["catchability"]["status"] == NOT_PROVEN)
    if chosen is not None:
        if feasible:
            feasibility_note = (f"This exact trip ({chosen['route_short_name']} -> {chosen['trip_headsign']}, "
                                f"reference {chosen['reference_sydney']}) is feasible via "
                                f"{', '.join(sorted(pool_ids))}; the stops are compared on walking "
                                "effort, walking time and distance only.")
        else:
            feasibility_note = (f"No stop is proven catchable for the selected trip {trip_id} under the "
                                "current model; no stop comparison is made for it.")
    elif candidates:
        feasibility_note = ("No listed upcoming trip is proven catchable from this origin under the "
                            "current model; the walking comparison below is among all served stops "
                            "and says nothing about catchability.")
    else:
        feasibility_note = ("No upcoming trip of this service could be timed (no realtime match and "
                            "no usable timetable); the walking comparison below is among all served "
                            "stops and says nothing about catchability.")

    scene["selected_trip"] = None if chosen is None else {
        "trip_id": chosen["trip_id"],
        "route_short_name": chosen["route_short_name"],
        "trip_headsign": chosen["trip_headsign"],
        "direction_id": chosen["direction_id"],
        "route_direction": scene["intent"].get("route_direction"),
        "reference_epoch": chosen["reference_epoch"],
        "reference_sydney": chosen["reference_sydney"],
        "reference_stop_id": chosen["reference_stop_id"],
        "reference_definition": TRIP_REFERENCE_DEFINITION,
        "expected_time_source": trip_source_summary(selected_calls),
        "selection": selection,
        "is_next_feasible": bool(chosen["is_next_feasible"]),
        "n_feasible_stops": len(feasible),
        "feasible_stop_ids": sorted(pool_ids) if feasible else [],
        "catchable_stop_ids": realtime_catchable,
        "schedule_only_stop_ids": scheduled_only,
        "not_proven_stop_ids": not_proven,
    }
    scene["summary"] = {
        "candidates_total": len(scene["stops"]),
        "candidates_serving_selected_service": sum(
            1 for s in scene["stops"] if s["serves_selected_service"]),
        "compared": len(in_scope),
        "selected_trip_id": trip_id,
        "trip_selection": selection if chosen is not None else None,
        "next_feasible_trip_id": scene["trip_selection"].get("next_feasible_trip_id"),
        "trips_listed": len(candidates),
        "comparison_basis": basis,
        "feasible_stops": sorted(pool_ids) if chosen is not None else [],
        "stops_realtime_catchable": realtime_catchable,
        "stops_scheduled_only": scheduled_only,
        "stops_not_proven": not_proven,
        "lowest_terrain_effort_stop": lowest_effort,
        "lowest_modelled_walking_time_stop": lowest_time,
        "shortest_walk_stop": shortest,
        "pareto_shortlist": shortlist,
        "pareto_objectives": ["effort_j_per_kg (effort-optimal route)",
                              "time_route_walking_time_s (time-optimal route)"],
        "near_tie_note": near_tie_note,
        "feasibility_note": feasibility_note,
    }
    return scene


def mode_label(realtime: RealtimeContext) -> str:
    if realtime.mode == MODE_LIVE:
        feed = {"light_rail": "light rail"}.get(realtime.feed_mode or "bus", "bus")
        return (f"LIVE - TfNSW {feed} Trip Updates fetched " + (sydney_text(realtime.fetched_at) or "")
                if realtime.status == "ok" else "LIVE requested - feed UNAVAILABLE; analytical results only")
    if realtime.mode == MODE_REPLAY:
        tag = "SYNTHETIC " if realtime.is_synthetic else ""
        return (f"REPLAY - {tag}saved snapshot as of {sydney_text(realtime.observed_at)} (not live)"
                if realtime.status == "ok" else "REPLAY requested - snapshot could not be read")
    return "ANALYTICAL - terrain, ranking and walking time only; no realtime"


# --------------------------------------------------------------------------- #
# GeoJSON
# --------------------------------------------------------------------------- #
def edge_coordinates(graph: Any, u: int, v: int, key: int) -> List[List[float]]:
    """WGS84 [lon, lat] vertices of one directed edge, oriented u -> v.

    Stored OSM geometry when present, otherwise the two node positions - a
    drawing fallback that changes no distance (the length attribute is what
    the routing used).
    """
    data = graph.edges[u, v, key]
    geometry = data.get("geometry")
    if geometry is not None:
        coords = [[float(x), float(y)] for x, y in geometry.coords]
        ux, uy = float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])
        first = (coords[0][0] - ux) ** 2 + (coords[0][1] - uy) ** 2
        last = (coords[-1][0] - ux) ** 2 + (coords[-1][1] - uy) ** 2
        if last < first:
            coords.reverse()
        return coords
    return [[float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])],
            [float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])]]


def route_linestring(graph: Any, edge_keys: Sequence[Sequence[int]]) -> List[List[float]]:
    coords: List[List[float]] = []
    for u, v, k in edge_keys:
        piece = edge_coordinates(graph, int(u), int(v), int(k))
        if coords and coords[-1] == piece[0]:
            piece = piece[1:]
        coords.extend(piece)
    return coords


def scene_to_geojson(scene: Dict[str, Any], graph: Any,
                     include_distance_routes: bool = True) -> Dict[str, Any]:
    """Geographic features for the UI and, later, the Sandbox projector.

    Coordinates are WGS84 lon/lat only. No projector or sandbox coordinates
    appear here; Adi's projection branch transforms these.
    """
    features: List[Dict[str, Any]] = []
    origin = scene["origin"]
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [origin["lon"], origin["lat"]]},
        "properties": {"feature_type": "origin", "label": origin.get("label") or "origin",
                       "nearest_node": origin["nearest_node"],
                       "snap_distance_m": origin["snap_distance_m"],
                       "mode": scene["mode"], "mode_label": scene["mode_label"]},
    })
    selected_trip = scene.get("selected_trip") or {}
    for stop in scene["stops"]:
        catch = stop["catchability"] or {}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [stop["lon"], stop["lat"]]},
            "properties": {
                "feature_type": "candidate_stop",
                "stop_id": stop["stop_id"], "stop_name": stop["stop_name"],
                "mode": stop["mode"], "eligibility": stop["eligibility"],
                "status_reason": stop["status_reason"],
                "serves_selected_service": stop["serves_selected_service"],
                "route_short_name": scene["intent"]["route_short_name"],
                "direction_id": scene["intent"]["direction_id"],
                "route_direction": scene["intent"]["route_direction"],
                "effort_rank": stop["effort_rank"],
                "near_tie_group": stop["near_tie_group"],
                "effort_j_per_kg": stop["effort_j_per_kg"],
                "distance_m": stop["distance_m"],
                "walking_time_min": stop["time_route_walking_time_min"],
                "walking_time_s": stop["time_route_walking_time_s"],
                # catchability of THIS stop for the ONE selected exact trip (a constraint)
                "catchability_status": catch.get("status"),
                "catchability_reason": catch.get("reason"),
                "trip_id": catch.get("trip_id") or selected_trip.get("trip_id"),
                "trip_headsign": selected_trip.get("trip_headsign"),
                "trip_expected_arrival_epoch": catch.get("trip_expected_arrival_epoch"),
                "trip_expected_arrival_sydney": catch.get("trip_expected_arrival_sydney"),
                "expected_arrival_epoch": catch.get("trip_expected_arrival_epoch"),   # schema 1.0 name
                "expected_arrival_sydney": catch.get("trip_expected_arrival_sydney"),
                "expected_time_source": catch.get("expected_time_source"),
                "pedestrian_arrival_epoch": catch.get("pedestrian_arrival_epoch"),
                "pedestrian_modelled_arrival_epoch": catch.get("pedestrian_modelled_arrival_epoch"),
                "slack_s": catch.get("slack_s"),
                "feasible_for_selected_trip": catch.get("status") in FEASIBLE_STATUSES,
                "pareto_status": stop["pareto_status"],
                "dominated_by": stop["dominated_by"],
                "highlights": stop["highlights"],
                "selected": bool(scene.get("selected_stop") == stop["stop_id"]),
                "highlighted": bool(stop["highlights"]),
            },
        })
        for objective in ("effort", "time", "distance"):
            if objective == "distance" and not include_distance_routes:
                continue
            route = stop["routes"].get(objective)
            if route is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": route_linestring(graph, route["edge_keys"])},
                "properties": {
                    "feature_type": f"{objective}_optimal_route",
                    "objective": objective, "stop_id": stop["stop_id"],
                    "stop_name": stop["stop_name"],
                    "serves_selected_service": stop["serves_selected_service"],
                    "distance_m": route["distance_m"],
                    "effort_j_per_kg": route["effort_j_per_kg"],
                    "walking_time_s": route["walking_time_s"],
                    "walking_time_min": route["walking_time_min"],
                    "edges": route["edges"],
                    "highlights": stop["highlights"],
                    "selected": bool(scene.get("selected_stop") == stop["stop_id"]),
                },
            })
    # Transit layer (exact-trip tracking), kept SEPARATE from the walking
    # layer: a planned route from static GTFS and raw reported positions
    # from GTFS-Realtime. Present only when a trip is being tracked.
    transit = scene.get("transit") or {}
    if transit.get("planned_route"):
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[float(lon), float(lat)] for lon, lat in transit["planned_route"]]},
            "properties": {"feature_type": "planned_transit_route", "layer": "transit",
                           "trip_id": transit.get("trip_id"), "route_short_name": transit.get("route_short_name"),
                           "destination": transit.get("destination"), "source": "static GTFS shapes.txt",
                           "tracking_state": transit.get("tracking_state")},
        })
    for position in transit.get("reported_positions") or []:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(position["lon"]), float(position["lat"])]},
            "properties": {"feature_type": "reported_vehicle_position", "layer": "transit",
                           "trip_id": transit.get("trip_id"), "vehicle_id": position.get("vehicle_id"),
                           "reported_at": position.get("reported_at"), "observed_at": position.get("observed_at"),
                           "tracking_state": transit.get("tracking_state"),
                           "source": "GTFS-Realtime VehiclePosition (raw, not snapped)"},
        })
    return {
        "type": "FeatureCollection",
        "name": "civl3704_phase2_scene",
        "crs_note": "WGS84 longitude/latitude (EPSG:4326); no projector coordinates",
        "schema_version": scene["schema_version"],
        "mode": scene["mode"], "mode_label": scene["mode_label"],
        "generated_at": scene["generated_at"],
        "intent": scene["intent"], "summary": scene["summary"],
        "selected_trip": scene.get("selected_trip"),
        "trip_candidates": scene.get("trip_candidates") or [],
        "features": features,
    }


def scene_json_ready(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-clean a scene (NaN -> null, numpy -> python) for json.dumps."""
    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(v) for v in value]
        return _clean(value)
    return walk(scene)


def write_scene(scene: Dict[str, Any], geojson: Dict[str, Any],
                out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / SCENE_JSON
    geo_path = out_dir / SCENE_GEOJSON
    json_path.write_text(json.dumps(scene_json_ready(scene), indent=2), encoding="utf-8")
    geo_path.write_text(json.dumps(scene_json_ready(geojson)), encoding="utf-8")
    return json_path, geo_path


# --------------------------------------------------------------------------- #
# text report
# --------------------------------------------------------------------------- #
def format_report(scene: Dict[str, Any]) -> str:
    s = scene["summary"]
    chosen = scene.get("selected_trip")
    lines = [
        f"MODE   {scene['mode_label']}",
        f"ORIGIN {scene['origin']['lat']:.6f}, {scene['origin']['lon']:.6f} "
        f"(node {scene['origin']['nearest_node']}, snap {scene['origin']['snap_distance_m']} m)",
        f"INTENT {scene['intent']['label']}  [{scene['intent']['key']}]",
        f"       boards at {len(scene['intent']['stop_ids'])} candidate stops; "
        f"{s['compared']} compared",
        "",
        "UPCOMING EXACT TRIPS (chronological by first upcoming call in the study area)",
    ]
    for c in scene.get("trip_candidates") or []:
        mark = ">" if c["is_selected"] else ("n" if c["is_next_feasible"] else " ")
        ref = (c["reference_sydney"] or "")[11:16]
        lines.append(f"  {mark} {ref:5s}  {c['route_short_name']} -> {str(c['trip_headsign'])[:28]:28s} "
                     f"{c['n_feasible_stops']:2d} feasible stop(s)  [{c['basis']}]  {c['trip_id']}")
    info = scene.get("trip_selection") or {}
    if info.get("trips_beyond_horizon"):
        lines.append(f"    (+{info['trips_beyond_horizon']} later trips not listed; horizon {info['horizon']})")
    lines.append("  > selected   n next feasible (automatic)")
    lines.append("")
    if chosen is None:
        lines.append("SELECTED TRIP   none")
    else:
        lines.append(f"SELECTED TRIP   {chosen['trip_id']}  ({chosen['selection']}; "
                     f"times {chosen['expected_time_source'] or '-'})")
    lines.append(f"  {s['feasibility_note']}")
    lines += [
        "",
        f"Among the compared stops [{s['comparison_basis']}] - catchability is a constraint, not an objective:",
        f"  lowest terrain effort          {s['lowest_terrain_effort_stop']}",
        f"  lowest modelled walking time   {s['lowest_modelled_walking_time_stop']}",
        f"  shortest walk                  {s['shortest_walk_stop']}",
        f"  Pareto shortlist (effort, walking time)   {', '.join(s['pareto_shortlist']) or '-'}",
    ]
    if s.get("near_tie_note"):
        lines.append(f"  near tie: {s['near_tie_note']}")
    lines.append("")
    header = (f"{'stop':8s} {'name':38s} {'rank':>4s} {'effort':>8s} {'dist':>7s} "
              f"{'t_min':>6s} {'ped_at':>6s} {'bus_at':>6s} {'slack':>7s} {'status':40s} src")
    lines.append(header)
    for stop in scene["stops"]:
        if not stop["serves_selected_service"]:
            continue
        c = stop["catchability"] or {}
        bus = (c.get("trip_expected_arrival_sydney") or "")[11:16] or "-"
        ped = (c.get("pedestrian_arrival_sydney") or "")[11:16] or "-"
        slack = c.get("slack_s")
        lines.append(
            f"{stop['stop_id']:8s} {str(stop['stop_name'])[:38]:38s} "
            f"{str(stop['effort_rank'] or '-'):>4s} "
            f"{(stop['effort_j_per_kg'] if stop['effort_j_per_kg'] is not None else float('nan')):8.1f} "
            f"{(stop['distance_m'] if stop['distance_m'] is not None else float('nan')):7.1f} "
            f"{(stop['time_route_walking_time_min'] if stop['time_route_walking_time_min'] is not None else float('nan')):6.2f} "
            f"{ped:>6s} {bus:>6s} {('%+d' % slack) if slack is not None else '-':>7s} "
            f"{(c.get('status') or '')[:40]:40s} {c.get('expected_time_source') or c.get('reason') or ''}"
            + (f"   <- {', '.join(stop['highlights'])}" if stop["highlights"] else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--origin", nargs=2, type=float, metavar=("LAT", "LON"))
    parser.add_argument("--origin-label", default="")
    parser.add_argument("--service", default=None, help="route/direction key, e.g. 433/0")
    parser.add_argument("--mode", default=MODE_ANALYTICAL,
                        choices=[m.lower() for m in MODES] + list(MODES))
    parser.add_argument("--replay", default=None, help="saved snapshot .pb (REPLAY mode)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_S,
                        help="seconds added to the modelled walking time (default 0)")
    parser.add_argument("--near-tie-pct", type=float, default=rank.DEFAULT_NEAR_TIE_PCT)
    parser.add_argument("--no-schedule-fallback", action="store_true")
    parser.add_argument("--trip", default=None,
                        help="exact trip_id to evaluate (default: the next feasible trip)")
    parser.add_argument("--trip-horizon", type=int, default=DEFAULT_TRIP_HORIZON,
                        help="how many upcoming exact trips to list (default 12)")
    parser.add_argument("--cost-dir", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--out", default=None, help="scene output dir (default results/)")
    parser.add_argument("--list-services", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tables = intent_mod.ServiceIntentTables.load()
    if args.list_services:
        print(intent_mod.describe_choices(tables))
        return 0
    if args.origin is None or args.service is None:
        print("--origin LAT LON and --service KEY are required (or --list-services)",
              file=sys.stderr)
        return 2
    cost_dir = Path(args.cost_dir) if args.cost_dir else tor.cmp.cost_dir()
    cost_table = tor.cmp.load_cost_table(cost_dir)
    context = tor.build_context(cost_table, Path(args.graph) if args.graph else None)
    realtime = load_realtime(args.mode.upper(), Path(args.replay) if args.replay else None)
    scene = run_phase2(context, tables, args.origin[0], args.origin[1], args.service,
                       realtime, cost_table, margin_s=args.margin,
                       near_tie_pct=args.near_tie_pct,
                       schedule_fallback=not args.no_schedule_fallback,
                       origin_label=args.origin_label, trip_id=args.trip,
                       trip_horizon=args.trip_horizon)
    geojson = scene_to_geojson(scene, context["graph"])
    json_path, geo_path = write_scene(scene, geojson, Path(args.out) if args.out else None)
    print(format_report(scene))
    print("")
    print(f"written: {json_path}")
    print(f"written: {geo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
