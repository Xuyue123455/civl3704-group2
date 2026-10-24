"""Week 7 V2 - live TfNSW bus Trip Updates, narrowed to Glebe candidate stops.

WHAT THIS IS
------------
V1 (``scripts/inspect_gtfs_realtime.py``) proved that a live TfNSW bus Trip
Update feed can be fetched and decoded. V2 takes that same feed - the whole
Sydney bus network - and answers one further question:

    which of it, if any, concerns THIS project's existing Glebe candidate
    bus stops, and what does the existing static GTFS snapshot say about it?

The GTFS-Realtime fetching and decoding are NOT reimplemented. This script
imports V1 and calls its tested ``fetch_feed`` / ``parse_feed`` /
``load_api_key`` / ``redact`` / ``to_sydney`` directly. V1 is unchanged.

BUS ONLY
--------
The verified feed is the **bus** Trip Update feed. The Inner West Light Rail
v2 endpoint is still unverified (D-24), so Glebe's four light-rail candidate
stops get no realtime information here and the report says so. Nothing printed
below describes all Glebe transit modes.

WHAT V2 DOES NOT ESTABLISH
--------------------------
No walking route, no terrain effort, no stop ranking, no catchability, no
journey planning, no accessibility statement. The Week 5 GUI is untouched -
that is V3.

And absence is not a finding. A Glebe stop that does not appear in one feed
snapshot has NOT been shown to be unserved, unreachable or inaccessible. It
means only that this snapshot carried no matching live update.

TIME FIELDS
-----------
Three quantities are kept apart and labelled apart:

    scheduled   the static stop_times.txt time for this trip at this stop
    realtime    an ABSOLUTE time supplied by GTFS-R itself
    derived     scheduled + realtime delay, and ONLY when every condition in
                ``derive_time`` holds

A ``delay`` of 0 is a real zero. A missing ``delay`` stays missing and is
never printed as "on time". Nothing is derived unless the trip, the stop
occurrence and the service date are all matched exactly.

TWO ROUTE NAMESPACES
--------------------
In the observed TfNSW bus feed the realtime ``route_id`` (e.g. ``2459_433``)
and the static ``route_id`` (e.g. ``11-431-sj2-1``) had **zero** direct
overlap. A route number is therefore obtained ONLY through the chain

    realtime trip_id -> trips.txt -> that trip's static route_id
                     -> routes.txt -> route_short_name / route_long_name

and never by parsing or splitting the realtime route_id string. Both
identifiers are kept, separately, and neither overwrites the other.

Usage
-----
    python scripts/show_glebe_live_arrivals.py
    python scripts/show_glebe_live_arrivals.py --audit
    python scripts/show_glebe_live_arrivals.py --stop-id 203732 --limit 5
    python scripts/show_glebe_live_arrivals.py --parse-file captured.pb
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The tested V1 ingestion layer, imported and used as-is. There is no second
# GTFS-Realtime decoder in this project.
v1 = load_script("inspect_gtfs_realtime")

RealtimeError = v1.RealtimeError
MissingApiKey = v1.MissingApiKey
to_sydney = v1.to_sydney
field_or_none = v1.field_or_none
redact = v1.redact

SYDNEY_TZ = v1.SYDNEY_TZ
BUS_FEED = "buses"
BUS_MODE = "bus"
BUS_ROUTE_TYPE = "700"          # GTFS route_type 700 = bus, as Stage 2 uses
LIGHT_RAIL_MODE = "light_rail"  # GTFS route_type 900, the project's normalised mode (config mode_policy)
# the candidate modes a realtime Trip Update feed may carry events for (Week 11: bus + Inner West light rail);
# which feed is fetched is decided by the caller (unified_ui.feed_adapter), never here
REALTIME_MODES = (BUS_MODE, LIGHT_RAIL_MODE)

CANDIDATE_STOPS_NAME = "glebe_stops_snapped.csv"
GTFS_DIRNAME = "gtfs"
PROVENANCE_NAME = "feed_provenance.json"
EXTRACT_SUBDIR = "gtfs_realtime"

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
STOP_TIMES_CHUNK = 500_000

ABSENCE_NOTE = (
    "A candidate stop missing from this snapshot has NOT been shown to be "
    "unserved, unreachable or inaccessible. It means only that this feed "
    "snapshot carried no matching live update.")

SCOPE_NOTE = [
    "Bus Trip Update feed only - the verified TfNSW feed (D-24).",
    "No light rail realtime: the Inner West Light Rail v2 endpoint is still",
    "unverified, so Glebe's light-rail candidate stops are excluded here.",
    "No walking route, no terrain effort, no ranking, no accessibility claim.",
    "Route numbers come from the matched trip_id via trips.txt and routes.txt,",
    "never from parsing the realtime route_id: the two are different",
    "identifier namespaces and had zero direct overlap in the observed feed.",
]


class GlebeRealtimeError(RealtimeError):
    """Raised with an actionable message when V2 cannot continue."""


# --------------------------------------------------------------------------- #
# existing project inputs
# --------------------------------------------------------------------------- #
def candidate_stops_path(root: Optional[Path] = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache"
            / CANDIDATE_STOPS_NAME)


def load_candidate_stops(path: Optional[Path] = None) -> pd.DataFrame:
    """The Stage 4 candidate stops. Read-only; nothing is re-derived.

    ``stop_id`` is normalised to a string because GTFS-Realtime carries it as
    a string and the CSV would otherwise read it as an integer.
    """
    target = Path(path) if path else candidate_stops_path()
    if not target.exists():
        raise GlebeRealtimeError(
            f"{target} is missing. Run the Week 4 foundation first:\n"
            "    python scripts/run_week4_foundation.py")
    frame = pd.read_csv(target)
    for column in ("stop_id", "stop_name", "mode"):
        if column not in frame.columns:
            raise GlebeRealtimeError(f"{target} has no {column!r} column.")
    frame = frame.copy()
    frame["stop_id"] = frame["stop_id"].astype(str).str.strip()
    return frame


def bus_candidates(stops: pd.DataFrame) -> pd.DataFrame:
    """Only the bus candidate stops (the bus feed audit and the bus CLI)."""
    return stops[stops["mode"].astype(str) == BUS_MODE].reset_index(drop=True)


def realtime_candidates(stops: pd.DataFrame,
                        modes: Sequence[str] = REALTIME_MODES) -> pd.DataFrame:
    """The candidate stops whose mode has a realtime feed (bus and, since Week 11, light rail).

    A feed only ever carries the stop_ids of its own mode, so keeping both
    modes here never mixes them: bus events land on bus stops, light-rail
    events on the light-rail platforms, exactly as the stop_time_update says.
    """
    wanted = {str(m) for m in modes}
    return stops[stops["mode"].astype(str).isin(wanted)].reset_index(drop=True)


@dataclass
class StaticGtfs:
    """The parts of the existing static snapshot V2 actually joins to.

    ``stop_times.txt`` is ~400 MB, so it is NEVER loaded whole and never
    loaded at all unless a scheduled time is asked for.
    """

    directory: Path
    routes: Optional[pd.DataFrame] = None
    trips: Optional[pd.DataFrame] = None
    scheduled: Optional[pd.DataFrame] = None
    missing: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_routes(self) -> bool:
        return self.routes is not None and not self.routes.empty

    @property
    def has_trips(self) -> bool:
        return self.trips is not None and not self.trips.empty


def gtfs_dir(root: Optional[Path] = None) -> Path:
    return Path(root or CONFIG_ROOT) / "data" / GTFS_DIRNAME


def read_gtfs_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """One GTFS text file, as strings, with the required columns checked.

    TfNSW ships these files with a UTF-8 BOM, hence ``utf-8-sig``.
    """
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    present = [name for name in columns if name in header.columns]
    missing = [name for name in columns if name not in header.columns]
    if missing:
        raise GlebeRealtimeError(
            f"{path.name} is missing the column(s) {missing}. Its columns "
            f"are {list(header.columns)}. The static snapshot may not be the "
            "one this code was written against.")
    return pd.read_csv(path, usecols=present, dtype=str,
                       encoding="utf-8-sig").fillna("")


def load_static_gtfs(directory: Optional[Path] = None,
                     want_routes: bool = True,
                     want_trips: bool = True) -> StaticGtfs:
    """Load routes.txt and trips.txt if they are present.

    A missing file is RECORDED, not fatal: V2's realtime matching still works
    without the static join, it just reports less.
    """
    target = Path(directory) if directory else gtfs_dir()
    static = StaticGtfs(directory=target)

    provenance = target / PROVENANCE_NAME
    if provenance.exists():
        with open(provenance, "r", encoding="utf-8") as handle:
            static.provenance = json.load(handle)

    if want_routes:
        path = target / "routes.txt"
        if path.exists():
            static.routes = read_gtfs_table(
                path, ["route_id", "route_short_name", "route_long_name",
                       "route_type"])
        else:
            static.missing.append("routes.txt")
    if want_trips:
        path = target / "trips.txt"
        if path.exists():
            static.trips = read_gtfs_table(
                path, ["trip_id", "route_id", "service_id", "trip_headsign",
                       "direction_id"])
        else:
            static.missing.append("trips.txt")
    return static


# --------------------------------------------------------------------------- #
# scheduled stop times - a bounded, cached extract, built only on request
# --------------------------------------------------------------------------- #
def extract_dir(root: Optional[Path] = None) -> Path:
    return Path(root or CONFIG_ROOT) / "data" / "cache" / EXTRACT_SUBDIR


def extract_key(stop_ids: Set[str], provenance: Dict[str, Any]) -> str:
    """A cache key over BOTH the stop set and the static snapshot identity.

    Keying on the snapshot's own sha256 means a re-downloaded GTFS bundle
    cannot be silently served from a stale extract.
    """
    digest = hashlib.sha256()
    digest.update(",".join(sorted(stop_ids)).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(provenance.get("zip_sha256", "unknown")).encode("utf-8"))
    return digest.hexdigest()[:16]


def build_scheduled_extract(stop_times_path: Path, stop_ids: Set[str],
                            chunk_size: int = STOP_TIMES_CHUNK
                            ) -> pd.DataFrame:
    """Stream stop_times.txt once, keeping only the wanted stops.

    The full file is ~400 MB, so it is read in chunks and never held whole.
    """
    kept: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
            stop_times_path,
            usecols=["trip_id", "arrival_time", "departure_time", "stop_id",
                     "stop_sequence"],
            dtype=str, encoding="utf-8-sig", chunksize=chunk_size):
        chunk["stop_id"] = chunk["stop_id"].astype(str).str.strip()
        kept.append(chunk[chunk["stop_id"].isin(stop_ids)])
    if not kept:
        return pd.DataFrame(columns=["trip_id", "arrival_time",
                                     "departure_time", "stop_id",
                                     "stop_sequence"])
    return pd.concat(kept, ignore_index=True).fillna("")


def load_scheduled(static: StaticGtfs, stop_ids: Set[str],
                   cache_dir: Optional[Path] = None,
                   rebuild: bool = False) -> Optional[pd.DataFrame]:
    """The cached Glebe-only slice of stop_times.txt, building it if needed."""
    stop_times = static.directory / "stop_times.txt"
    if not stop_times.exists():
        static.missing.append("stop_times.txt")
        return None
    directory = Path(cache_dir) if cache_dir else extract_dir()
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"glebe_stop_times_{extract_key(stop_ids, static.provenance)}.csv"
    if cache.exists() and not rebuild:
        return pd.read_csv(cache, dtype=str).fillna("")
    frame = build_scheduled_extract(stop_times, stop_ids)
    frame.to_csv(cache, index=False)
    return frame


# --------------------------------------------------------------------------- #
# GTFS time arithmetic
# --------------------------------------------------------------------------- #
def parse_gtfs_time(text: Any) -> Optional[int]:
    """``HH:MM:SS`` as seconds after midnight. HH may legitimately exceed 23.

    ``25:10:00`` is 1:10 am on the NEXT calendar day of the same SERVICE day
    and must not be rejected or wrapped.
    """
    if text is None:
        return None
    parts = str(text).strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return None
    if hours < 0 or not (0 <= minutes < 60) or not (0 <= seconds < 60):
        return None
    return hours * 3600 + minutes * 60 + seconds


def service_epoch(start_date: str, seconds_after_midnight: int
                  ) -> Optional[int]:
    """A GTFS service date plus seconds-after-midnight, as a POSIX second.

    GTFS defines the day's origin as "noon minus twelve hours" on the service
    date, which is what makes times correct across a daylight-saving
    transition. That is what is computed here, in Australia/Sydney.
    """
    try:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(SYDNEY_TZ)
    except Exception:
        return None
    try:
        day = datetime.strptime(str(start_date).strip(), "%Y%m%d")
    except (TypeError, ValueError):
        return None
    noon = datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=zone)
    origin = noon - timedelta(hours=12)
    return int((origin + timedelta(seconds=int(seconds_after_midnight)))
               .timestamp())


# --------------------------------------------------------------------------- #
# walking the feed
# --------------------------------------------------------------------------- #
def iter_stop_time_updates(message: Any) -> Iterator[Tuple[Any, Any, Any]]:
    """Every (entity, trip_update, stop_time_update) in the feed."""
    for entity in message.entity:
        if not entity.HasField("trip_update"):
            continue
        update = entity.trip_update
        for stop_time_update in update.stop_time_update:
            yield entity, update, stop_time_update


def event_fields(stop_time_update: Any, name: str) -> Dict[str, Any]:
    """One StopTimeEvent, by presence. Absent stays absent; 0 stays 0."""
    event = field_or_none(stop_time_update, name)
    if event is None:
        return {"present": False, "time": None, "delay": None,
                "uncertainty": None}
    return {"present": True,
            "time": field_or_none(event, "time"),
            "delay": field_or_none(event, "delay"),
            "uncertainty": field_or_none(event, "uncertainty")}


# --------------------------------------------------------------------------- #
# the matching audit
# --------------------------------------------------------------------------- #
@dataclass
class MatchAudit:
    """Measured overlap between one feed snapshot and the project's inputs."""

    candidate_stops_total: int = 0
    candidate_bus_stops: int = 0
    candidate_light_rail_stops: int = 0
    live_stop_ids: int = 0
    live_stop_time_updates: int = 0
    live_trip_updates: int = 0
    matched_stop_ids: List[str] = field(default_factory=list)
    unmatched_candidate_stop_ids: List[str] = field(default_factory=list)
    matched_stop_time_updates: int = 0
    matched_trip_updates: int = 0
    live_trip_ids: int = 0
    matched_trip_ids: int = 0
    unmatched_trip_id_examples: List[str] = field(default_factory=list)
    live_route_ids: int = 0
    matched_route_ids: int = 0
    unmatched_route_id_examples: List[str] = field(default_factory=list)
    static_available: bool = False
    static_missing: List[str] = field(default_factory=list)

    # ----- Glebe-local resolution, not the whole Sydney feed ------------- #
    glebe_trip_ids: int = 0
    glebe_trip_ids_in_static: int = 0
    glebe_trips_with_static_route: int = 0
    glebe_trips_with_route_short_name: int = 0
    glebe_realtime_route_ids: int = 0
    glebe_realtime_route_ids_in_static: int = 0
    glebe_unresolved_trip_examples: List[str] = field(default_factory=list)

    @property
    def matched_stop_count(self) -> int:
        return len(self.matched_stop_ids)

    @property
    def matched_stop_pct(self) -> Optional[float]:
        if not self.candidate_bus_stops:
            return None
        return 100.0 * self.matched_stop_count / self.candidate_bus_stops

    def update_glebe_resolution(self, glebe_trip_ids: Set[str],
                                static: "StaticGtfs") -> None:
        """Walk the adopted route-resolution chain for the Glebe trips only.

        Each step is counted separately, so a failure can be located: does the
        trip exist statically, does that trip name a static route, and does
        that route carry a human-readable short name?
        """
        if not static.has_trips:
            return
        trips = {row.trip_id: row for row in static.trips.itertuples(
            index=False)}
        routes = ({} if not static.has_routes else
                  {row.route_id: row for row in static.routes.itertuples(
                      index=False)})
        unresolved: List[str] = []
        for trip_id in sorted(glebe_trip_ids):
            trip_row = trips.get(trip_id)
            if trip_row is None:
                unresolved.append(trip_id)
                continue
            self.glebe_trip_ids_in_static += 1
            static_route_id = (trip_row.route_id or "") or None
            if static_route_id is None:
                continue
            route_row = routes.get(static_route_id)
            if route_row is None:
                continue
            self.glebe_trips_with_static_route += 1
            if (route_row.route_short_name or "").strip():
                self.glebe_trips_with_route_short_name += 1
        self.glebe_unresolved_trip_examples = unresolved[:5]


def audit_match(message: Any, stops: pd.DataFrame,
                static: Optional[StaticGtfs] = None) -> MatchAudit:
    """Measure, do not assume, how the live feed relates to this project.

    Every identifier that does not match is COUNTED, and a bounded set of
    examples is kept, rather than being silently discarded.
    """
    buses = bus_candidates(stops)
    wanted = set(buses["stop_id"])
    audit = MatchAudit(
        candidate_stops_total=int(len(stops)),
        candidate_bus_stops=int(len(buses)),
        candidate_light_rail_stops=int(
            (stops["mode"].astype(str) == "light_rail").sum()),
    )

    live_stops: Set[str] = set()
    live_trips: Set[str] = set()
    live_routes: Set[str] = set()
    matched_stops: Set[str] = set()
    matched_entities: Set[str] = set()
    glebe_trips: Set[str] = set()
    glebe_routes: Set[str] = set()

    for entity, update, stop_time_update in iter_stop_time_updates(message):
        audit.live_stop_time_updates += 1
        stop_id = field_or_none(stop_time_update, "stop_id")
        if stop_id is None or str(stop_id) not in wanted:
            if stop_id is not None:
                live_stops.add(str(stop_id))
            continue
        live_stops.add(str(stop_id))
        audit.matched_stop_time_updates += 1
        matched_stops.add(str(stop_id))
        matched_entities.add(entity.id)
        trip = field_or_none(update, "trip")
        if trip is None:
            continue
        trip_id = field_or_none(trip, "trip_id")
        route_id = field_or_none(trip, "route_id")
        if trip_id is not None:
            glebe_trips.add(str(trip_id))
        if route_id is not None:
            glebe_routes.add(str(route_id))

    for entity in message.entity:
        if not entity.HasField("trip_update"):
            continue
        audit.live_trip_updates += 1
        trip = field_or_none(entity.trip_update, "trip")
        if trip is None:
            continue
        trip_id = field_or_none(trip, "trip_id")
        route_id = field_or_none(trip, "route_id")
        if trip_id is not None:
            live_trips.add(str(trip_id))
        if route_id is not None:
            live_routes.add(str(route_id))

    audit.live_stop_ids = len(live_stops)
    audit.live_trip_ids = len(live_trips)
    audit.live_route_ids = len(live_routes)
    audit.matched_stop_ids = sorted(matched_stops)
    audit.unmatched_candidate_stop_ids = sorted(wanted - matched_stops)
    audit.matched_trip_updates = len(matched_entities)
    audit.glebe_trip_ids = len(glebe_trips)
    audit.glebe_realtime_route_ids = len(glebe_routes)

    if static is not None:
        audit.static_missing = list(static.missing)
        audit.static_available = static.has_trips or static.has_routes
        if static.has_trips:
            known = set(static.trips["trip_id"])
            matched = live_trips & known
            audit.matched_trip_ids = len(matched)
            audit.unmatched_trip_id_examples = sorted(live_trips - known)[:5]
        if static.has_routes:
            known = set(static.routes["route_id"])
            matched = live_routes & known
            audit.matched_route_ids = len(matched)
            audit.unmatched_route_id_examples = sorted(live_routes - known)[:5]
            audit.glebe_realtime_route_ids_in_static = len(glebe_routes & known)
        audit.update_glebe_resolution(glebe_trips, static)
    return audit


# --------------------------------------------------------------------------- #
# upcoming events
# --------------------------------------------------------------------------- #
TIME_REALTIME = "realtime"
TIME_DERIVED = "derived"
TIME_NONE = None


def derive_time(scheduled_epoch: Optional[int], delay: Optional[int]
                ) -> Optional[int]:
    """scheduled + delay, and only when BOTH are actually available.

    A missing delay does not become zero, so a scheduled time is never
    quietly presented as a realtime prediction.
    """
    if scheduled_epoch is None or delay is None:
        return None
    return int(scheduled_epoch) + int(delay)


def scheduled_lookup(scheduled: Optional[pd.DataFrame]
                     ) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    """(trip_id, stop_id) -> its stop_times rows, for exact-occurrence checks."""
    index: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    if scheduled is None or scheduled.empty:
        return index
    for row in scheduled.itertuples(index=False):
        key = (str(row.trip_id), str(row.stop_id))
        index.setdefault(key, []).append({
            "arrival_time": str(getattr(row, "arrival_time", "")),
            "departure_time": str(getattr(row, "departure_time", "")),
            "stop_sequence": str(getattr(row, "stop_sequence", "")),
        })
    return index


def resolve_scheduled(index: Dict[Tuple[str, str], List[Dict[str, str]]],
                      trip_id: Optional[str], stop_id: Optional[str],
                      stop_sequence: Optional[int]) -> Optional[Dict[str, str]]:
    """The ONE static stop_times row this realtime record refers to.

    Returns None unless the occurrence is unambiguous: either the realtime
    record names a ``stop_sequence`` that matches exactly one row, or the
    (trip_id, stop_id) pair occurs exactly once in the static data. A loop
    service that visits the same stop twice therefore yields nothing rather
    than a coin toss.
    """
    if trip_id is None or stop_id is None:
        return None
    rows = index.get((str(trip_id), str(stop_id)))
    if not rows:
        return None
    if stop_sequence is not None:
        exact = [row for row in rows
                 if row["stop_sequence"] == str(int(stop_sequence))]
        return exact[0] if len(exact) == 1 else None
    return rows[0] if len(rows) == 1 else None


@dataclass
class ArrivalEvent:
    """One realtime stop event at one Glebe candidate bus stop.

    ``realtime_route_id`` and ``static_route_id`` are two DIFFERENT
    identifiers in two different namespaces. In the observed TfNSW bus feed
    they had zero direct overlap, so neither is allowed to stand in for the
    other and neither is overwritten by the other.
    """

    stop_id: str
    stop_name: str
    entity_id: str
    trip_id: Optional[str] = None
    realtime_route_id: Optional[str] = None
    static_route_id: Optional[str] = None
    route_short_name: Optional[str] = None
    route_long_name: Optional[str] = None
    route_type: Optional[str] = None
    trip_headsign: Optional[str] = None
    start_date: Optional[str] = None
    start_time: Optional[str] = None
    schedule_relationship: Optional[str] = None
    stop_schedule_relationship: Optional[str] = None
    stop_sequence: Optional[int] = None
    arrival: Dict[str, Any] = field(default_factory=dict)
    departure: Dict[str, Any] = field(default_factory=dict)
    scheduled_arrival_text: Optional[str] = None
    scheduled_epoch: Optional[int] = None
    derived_epoch: Optional[int] = None
    trip_in_static: Optional[bool] = None
    sort_epoch: Optional[int] = None
    sort_basis: Optional[str] = None


def build_events(message: Any, stops: pd.DataFrame,
                 static: Optional[StaticGtfs] = None,
                 scheduled: Optional[pd.DataFrame] = None,
                 stop_id_filter: Optional[str] = None,
                 modes: Optional[Sequence[str]] = None) -> List[ArrivalEvent]:
    """Every realtime stop event that lands on a Glebe candidate stop of a realtime mode
    (bus and light rail by default; ``modes`` narrows it)."""
    from google.transit import gtfs_realtime_pb2

    candidates = realtime_candidates(stops, modes if modes is not None else REALTIME_MODES)
    names = dict(zip(candidates["stop_id"], candidates["stop_name"]))
    wanted = set(names)
    if stop_id_filter is not None:
        wanted = {str(stop_id_filter)} & wanted

    routes = ({} if static is None or not static.has_routes else
              {row.route_id: row for row in static.routes.itertuples(
                  index=False)})
    trips = ({} if static is None or not static.has_trips else
             {row.trip_id: row for row in static.trips.itertuples(
                 index=False)})
    index = scheduled_lookup(scheduled)

    events: List[ArrivalEvent] = []
    for entity, update, stop_time_update in iter_stop_time_updates(message):
        stop_id = field_or_none(stop_time_update, "stop_id")
        if stop_id is None or str(stop_id) not in wanted:
            continue
        stop_id = str(stop_id)
        trip = field_or_none(update, "trip")
        trip_id = field_or_none(trip, "trip_id") if trip else None
        route_id = field_or_none(trip, "route_id") if trip else None

        event = ArrivalEvent(
            stop_id=stop_id, stop_name=str(names.get(stop_id, "")),
            entity_id=entity.id,
            trip_id=str(trip_id) if trip_id is not None else None,
            realtime_route_id=(str(route_id) if route_id is not None
                               else None),
            start_date=field_or_none(trip, "start_date") if trip else None,
            start_time=field_or_none(trip, "start_time") if trip else None,
            schedule_relationship=v1.enum_name(
                gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship
                .DESCRIPTOR,
                field_or_none(trip, "schedule_relationship")) if trip else None,
            stop_schedule_relationship=v1.enum_name(
                gtfs_realtime_pb2.TripUpdate.StopTimeUpdate
                .ScheduleRelationship.DESCRIPTOR,
                field_or_none(stop_time_update, "schedule_relationship")),
            stop_sequence=field_or_none(stop_time_update, "stop_sequence"),
            arrival=event_fields(stop_time_update, "arrival"),
            departure=event_fields(stop_time_update, "departure"),
        )

        # THE ADOPTED ROUTE-RESOLUTION CHAIN.
        #   realtime trip_id -> trips.txt -> that trip's static route_id
        #                    -> routes.txt -> short/long name, route_type
        # The realtime route_id is NEVER used to look up routes.txt, and is
        # never parsed or split to guess a route number: in the observed feed
        # the two identifier namespaces had zero overlap, so a realtime
        # route_id such as "2459_433" is an opaque token here, not evidence
        # that the service is route 433.
        trip_row = trips.get(event.trip_id) if event.trip_id else None
        if trips:
            event.trip_in_static = trip_row is not None
        if trip_row is not None:
            event.trip_headsign = trip_row.trip_headsign or None
            event.static_route_id = (trip_row.route_id or "") or None
        route_row = (routes.get(event.static_route_id)
                     if event.static_route_id else None)
        if route_row is not None:
            event.route_short_name = (route_row.route_short_name or "") or None
            event.route_long_name = (route_row.route_long_name or "") or None
            event.route_type = (route_row.route_type or "") or None

        # A scheduled time, and a derived one, ONLY when everything matches.
        if index and event.trip_in_static and event.start_date:
            row = resolve_scheduled(index, event.trip_id, stop_id,
                                    event.stop_sequence)
            if row is not None:
                text = row["arrival_time"] or row["departure_time"]
                seconds = parse_gtfs_time(text)
                if seconds is not None:
                    event.scheduled_arrival_text = text
                    event.scheduled_epoch = service_epoch(event.start_date,
                                                          seconds)
        delay = (event.arrival.get("delay")
                 if event.arrival.get("delay") is not None
                 else event.departure.get("delay"))
        event.derived_epoch = derive_time(event.scheduled_epoch, delay)

        absolute = (event.arrival.get("time")
                    if event.arrival.get("time") is not None
                    else event.departure.get("time"))
        if absolute is not None:
            event.sort_epoch = int(absolute)
            event.sort_basis = TIME_REALTIME
        elif event.derived_epoch is not None:
            event.sort_epoch = int(event.derived_epoch)
            event.sort_basis = TIME_DERIVED
        events.append(event)
    return events


def upcoming(events: Sequence[ArrivalEvent], now_epoch: int,
             limit: int = DEFAULT_LIMIT) -> Tuple[List[ArrivalEvent], int]:
    """The next N events that carry an orderable time, plus how many do not.

    An event with neither an absolute realtime time nor a safely derived one
    cannot be ordered, so it is EXCLUDED from this list and counted instead.
    Excluding it is not a judgement about the service; it is an admission
    that no time is available for it.
    """
    orderable = [event for event in events if event.sort_epoch is not None]
    future = [event for event in orderable if event.sort_epoch >= now_epoch]
    future.sort(key=lambda event: (event.sort_epoch, event.stop_id,
                                   event.entity_id))
    bounded = max(0, min(int(limit), MAX_LIMIT))
    return future[:bounded], len(events) - len(orderable)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
RULE = "-" * 72


def show(value: Any) -> str:
    return "-" if value is None else str(value)


def format_event_time(epoch: Optional[int]) -> str:
    if epoch is None:
        return "-"
    return f"{epoch}  ({to_sydney(epoch)})"


def format_matching(audit: MatchAudit) -> List[str]:
    lines = [
        "Matching",
        f"  Glebe candidate stops (all modes): "
        f"{audit.candidate_stops_total}",
        f"  Glebe candidate BUS stops:         {audit.candidate_bus_stops}",
        f"  Glebe candidate light-rail stops:  "
        f"{audit.candidate_light_rail_stops}  (served by the light-rail feed, not audited by this bus tool)",
        f"  unique stop_ids in this snapshot:  {audit.live_stop_ids}",
        f"  live StopTimeUpdates:              "
        f"{audit.live_stop_time_updates}",
        f"  live TripUpdates:                  {audit.live_trip_updates}",
        "",
        f"  Glebe bus stops represented:       {audit.matched_stop_count}"
        + (f" of {audit.candidate_bus_stops}"
           f" ({audit.matched_stop_pct:.1f} %)"
           if audit.matched_stop_pct is not None else ""),
        f"  matching StopTimeUpdates:          "
        f"{audit.matched_stop_time_updates}",
        f"  matching TripUpdates:              {audit.matched_trip_updates}",
    ]
    if audit.static_available:
        lines += [
            "",
            f"  live trip_ids:                     {audit.live_trip_ids}",
            f"  trip_ids found in static trips:    {audit.matched_trip_ids}",
            f"  live route_ids:                    {audit.live_route_ids}",
            f"  route_ids found in static routes:  {audit.matched_route_ids}",
        ]
        if audit.unmatched_trip_id_examples:
            lines.append("  unmatched trip_id examples:        "
                         + ", ".join(audit.unmatched_trip_id_examples))
        if audit.unmatched_route_id_examples:
            lines.append("  unmatched route_id examples:       "
                         + ", ".join(audit.unmatched_route_id_examples))
        if audit.live_route_ids and not audit.matched_route_ids:
            lines.append(
                "  NOTE: realtime route_id and static route_id are different "
                "identifier namespaces")
            lines.append(
                "        in this feed, so routes are resolved through the "
                "matched trip_id instead.")
        lines += [
            "",
            "  Route resolution for the GLEBE-matched TripUpdates only",
            "  (realtime trip_id -> trips.txt -> static route_id -> "
            "routes.txt):",
            f"    unique Glebe realtime trip_ids:  {audit.glebe_trip_ids}",
            f"    of those, found in trips.txt:    "
            f"{audit.glebe_trip_ids_in_static}",
            f"    of those, resolving a route:     "
            f"{audit.glebe_trips_with_static_route}",
            f"    of those, with a short name:     "
            f"{audit.glebe_trips_with_route_short_name}",
            f"    Glebe realtime route_ids:        "
            f"{audit.glebe_realtime_route_ids}"
            f"  (direct static matches: "
            f"{audit.glebe_realtime_route_ids_in_static})",
        ]
        if audit.glebe_unresolved_trip_examples:
            lines.append("    Glebe trip_ids not in trips.txt: "
                         + ", ".join(audit.glebe_unresolved_trip_examples))
    else:
        lines.append("  static GTFS join unavailable: "
                     + (", ".join(audit.static_missing) or "no files found"))
    if audit.unmatched_candidate_stop_ids:
        lines += [
            "",
            f"  Glebe bus stops NOT in this snapshot "
            f"({len(audit.unmatched_candidate_stop_ids)}):",
            "    " + ", ".join(audit.unmatched_candidate_stop_ids),
            f"  {ABSENCE_NOTE}",
        ]
    return lines


def route_label(event: ArrivalEvent) -> str:
    """The human-readable route line.

    A route NUMBER is printed only when it came through the adopted chain -
    matched trip_id, that trip's static route_id, then routes.txt. When the
    chain did not complete, the raw realtime route_id is shown and explicitly
    labelled as unresolved. It is never parsed, split or reinterpreted: a
    realtime route_id such as "2459_433" is an opaque token here.
    """
    if event.route_short_name:
        return f"Route {event.route_short_name}"
    if event.realtime_route_id:
        return (f"Route unresolved  (raw realtime route_id "
                f"{event.realtime_route_id}, not a route number)")
    return "Route -"


def format_event(event: ArrivalEvent, index: int) -> List[str]:
    lines = [
        f"  [{index}] Stop {event.stop_id} - {event.stop_name or '-'}",
        f"       {route_label(event)}"
        + (f"  ({event.route_long_name})" if event.route_long_name else "")
        + (f"  route_type {event.route_type}" if event.route_type else ""),
        f"       realtime route_id {show(event.realtime_route_id)}"
        f"   static route_id {show(event.static_route_id)}",
        f"       trip_id {show(event.trip_id)}"
        + (f"  headsign {event.trip_headsign}" if event.trip_headsign else ""),
        f"       trip in static GTFS: {show(event.trip_in_static)}"
        f"   start_date {show(event.start_date)}"
        f"   start_time {show(event.start_time)}",
        f"       schedule_relationship {show(event.schedule_relationship)}"
        f"   stop-level {show(event.stop_schedule_relationship)}"
        f"   stop_sequence {show(event.stop_sequence)}",
        f"       scheduled  {show(event.scheduled_arrival_text)}"
        + (f"  -> {format_event_time(event.scheduled_epoch)}"
           if event.scheduled_epoch is not None else ""),
    ]
    for name in ("arrival", "departure"):
        payload = getattr(event, name)
        if not payload.get("present"):
            lines.append(f"       {name:<10} (field absent)")
            continue
        lines.append(
            f"       {name:<10} realtime {format_event_time(payload['time'])}"
            if payload["time"] is not None else
            f"       {name:<10} realtime -")
        lines.append(
            f"                  delay {show(payload['delay'])}"
            f"   uncertainty {show(payload['uncertainty'])}")
    lines.append(f"       derived    {format_event_time(event.derived_epoch)}"
                 + ("  (scheduled + delay)"
                    if event.derived_epoch is not None else ""))
    lines.append(f"       ordered by {show(event.sort_basis)} time")
    return lines


def format_report(summary: Any, audit: MatchAudit,
                  events: Sequence[ArrivalEvent], unorderable: int,
                  limit: int, stop_id: Optional[str],
                  static: Optional[StaticGtfs],
                  fetch: Optional[Any] = None) -> str:
    lines = [
        RULE,
        "Week 7 V2 - Glebe live bus arrivals",
        RULE,
        f"Feed updated: {show(summary.header_timestamp)}  "
        f"({show(summary.header_timestamp_local)})",
    ]
    if summary.age_seconds is None:
        lines.append("Feed age:     the header carries no timestamp.")
    else:
        lines.append(f"Feed age:     {summary.age_seconds:.0f} s "
                     "(observation minus feed header; measured, not judged)")
    if fetch is not None:
        lines.append(f"Transport:    HTTP {fetch.http_status}, "
                     f"{fetch.byte_count} bytes, {fetch.elapsed_s:.3f} s")
    if static is not None and static.provenance:
        lines.append(
            f"Static GTFS:  downloaded "
            f"{static.provenance.get('download_date', '?')}, sha256 "
            f"{str(static.provenance.get('zip_sha256', '?'))[:16]}...")
    lines.append("")
    lines += format_matching(audit)
    lines.append("")

    heading = ("Upcoming realtime events"
               + (f" at stop {stop_id}" if stop_id else
                  " across the Glebe candidate bus stops"))
    lines.append(f"{heading} (next {limit}, bounded)")
    if not events:
        lines.append("  none: no matching event in this snapshot carries a "
                     "future orderable time.")
    for index, event in enumerate(events, start=1):
        lines += format_event(event, index)
        lines.append("")
    if unorderable:
        lines.append(
            f"  {unorderable} matching event(s) carry neither an absolute "
            "realtime time nor a safely")
        lines.append(
            "  derivable one, so they cannot be ordered and are not listed. "
            "That is a")
        lines.append(
            "  statement about the available fields, not about the service.")
        lines.append("")

    lines.append("Scope")
    for note in SCOPE_NOTE:
        lines.append(f"  {note}")
    lines += [
        "  scheduled / realtime / derived are three different quantities and",
        "  are labelled separately. A delay of 0 is a real zero; a missing",
        "  delay stays missing and is never printed as 'on time'.",
        "  Realtime matching is by stop_id only; V3 is the GUI integration.",
        "  realtime route_id and static route_id are kept as two separate",
        "  identifiers and neither stands in for the other.",
        RULE,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 V2: live TfNSW bus Trip Updates narrowed to this "
                    "project's Glebe candidate bus stops. Bus only.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"upcoming events to show "
                             f"(default {DEFAULT_LIMIT}, max {MAX_LIMIT})")
    parser.add_argument("--stop-id", default=None,
                        help="restrict to one Glebe candidate bus stop")
    parser.add_argument("--audit", action="store_true",
                        help="print the matching audit only")
    parser.add_argument("--scheduled", action="store_true",
                        help="also join static scheduled times (streams "
                             "stop_times.txt once, then caches a Glebe-only "
                             "extract under data/cache/)")
    parser.add_argument("--rebuild-extract", action="store_true",
                        help="rebuild the cached scheduled-time extract")
    parser.add_argument("--parse-file", default=None,
                        help="decode a saved .pb payload instead of fetching")
    parser.add_argument("--no-static", action="store_true",
                        help="skip the static GTFS join entirely")
    options = parser.parse_args(argv)

    api_key: Optional[str] = None
    try:
        stops = load_candidate_stops()
        buses = bus_candidates(stops)
        if options.stop_id is not None:
            known = set(buses["stop_id"])
            if str(options.stop_id) not in known:
                raise GlebeRealtimeError(
                    f"{options.stop_id} is not a Glebe candidate BUS stop. "
                    f"The {len(known)} bus stop_ids are: "
                    f"{', '.join(sorted(known))}.")

        static = None if options.no_static else load_static_gtfs()
        scheduled = None
        if options.scheduled and static is not None:
            print("Building the Glebe scheduled-time extract "
                  "(streams stop_times.txt once; cached afterwards)...")
            scheduled = load_scheduled(static, set(buses["stop_id"]),
                                       rebuild=options.rebuild_extract)

        if options.parse_file:
            source = Path(options.parse_file)
            if not source.exists():
                raise GlebeRealtimeError(f"{source} does not exist.")
            payload = source.read_bytes()
            fetch = None
        else:
            definition = v1.resolve_feed(BUS_FEED)
            api_key = v1.load_api_key()
            print(f"Requesting {definition.url}")
            print("Authorization header: apikey <redacted>")
            fetch = v1.fetch_feed(definition.url, api_key)
            payload = fetch.payload

        message = v1.parse_feed(payload)
        summary = v1.summarise_feed(message, sample=0)
        audit = audit_match(message, stops, static)
        events = ([] if options.audit else
                  build_events(message, stops, static, scheduled,
                               options.stop_id))
        listed, unorderable = upcoming(events, int(summary.observed_at or
                                                   time.time()),
                                       options.limit)
    except MissingApiKey as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except RealtimeError as error:
        print(f"ERROR: {redact(error, api_key)}", file=sys.stderr)
        return 1

    print(redact(format_report(summary, audit, listed, unorderable,
                               options.limit, options.stop_id, static, fetch),
                 api_key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
