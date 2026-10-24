"""Week 7 V4D - select ONE service, and keep tracking it after it passes.

WHY THIS EXISTS
---------------
V4B and V4C show vehicles for the services the selected stop is CURRENTLY
displaying. That list comes from V2's ``upcoming(...)``, which keeps only
events whose effective time is at or after the snapshot's observation time. So
the moment a bus passes the stop, its service leaves the list, its vehicle
stops being matched, and the marker disappears. V4C's omission bridge cannot
help: it is deliberately bounded by that same service's own expected arrival,
which has just passed.

V4D.1 therefore adds a SECOND, INDEPENDENT question:

    "Where is THIS service - the one I picked - right now?"

The user picks one exact service from the stop's current list. That pins its
**exact trip_id**, and from then on the vehicle feed is queried directly by
that trip_id:

    vehicle_state.snapshot.records_for(tracked.trip_id)

which does not care whether the service is still in ``upcoming(...)`` for any
stop. The tracked trip survives the stop panel moving on to the next service.

WHAT IS AND IS NOT MATCHED
--------------------------
The link is still, and only, exact equality:

    ArrivalEvent.trip_id  ==  VehicleRecord.trip_id

"Route 433" is NOT an identity: two Route 433 trips are two different
services, and the pinned identity is the trip_id, never the route number. No
route parsing, no nearest-vehicle guess, no bearing or time heuristic, and no
silent choice when one trip_id carries two simultaneous positions.

WHAT IT REUSES, AND THEREFORE DOES NOT REDO
-------------------------------------------
``scripts/demo_week7_live_motion.py``    V4C: the whole composed window and
                                         its coordinated ``Refresh all``
``scripts/demo_week7_live_vehicles.py``  V4B: ``VehicleState``, the extent
                                         test and the projection helper
``scripts/demo_week7_realtime.py``       V3: the LIVE TRANSIT panel and its
                                         time/delay wording helpers
``scripts/show_glebe_live_arrivals.py``  V2: ``upcoming`` and ``route_label``

None of those files is modified. ``python scripts/demo_week7_live_motion.py``,
``python scripts/demo_week7_live_vehicles.py`` and
``python scripts/demo_week7_realtime.py`` all still run exactly as before.

THE WHOLE-TRIP ROUTE MAP (V4D.2)
--------------------------------
The tracking window also draws the selected trip's PLANNED route. It is read
from the SAME configured static snapshot, by the join the timetable itself
defines:

    trip_id -> trips.txt(shape_id) -> shapes.txt, ordered by shape_pt_sequence

and the latest raw GTFS-Realtime ``VehiclePosition`` is drawn ON TOP of it, at
the coordinate TfNSW supplied. The two are different kinds of statement and
are labelled as such on the panel: the line is planned geometry from static
GTFS, the dot is one reported position from GTFS-Realtime. The dot is never
moved onto the line, no part of the line is presented as travelled, and the
arrows along it come only from the static shape's own point order - they are
the PLANNED direction of the trip, not observed motion.

A shape is fetched once per ``shape_id`` and cached in memory and on disk, so
a refresh never re-reads the ~950 MB ``shapes.txt`` again. When a trip has no
usable shape the panel says so and tracking continues unaffected.

WHAT IT DOES NOT DO
-------------------
No polling, no timer, no animation, no interpolation, no prediction, no road
snapping, no map matching, no trajectory line, no ETA of its own, and no
change to any Week 5 walking number. In particular, nothing here detects that
a bus has PASSED a stop: a tracked trip leaving the stop's arrivals list is a
fact about that list, not a spatial observation.

Usage
-----
    set TFNSW_API_KEY=<key>
    python scripts/demo_week7_trip_tracking.py
    python scripts/demo_week7_trip_tracking.py --debug-live
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ONE import chain: V4C already carries V4B, V3, V4A, V2, V1 and the Week 5
# explorer, so nothing is loaded twice and there is only one of each class.
v4c = load_script("demo_week7_live_motion")
v4b = v4c.v4b
v3 = v4c.v3
v4a = v4c.v4a
v2 = v4c.v2
week5 = v4c.week5

STATUS_OK = v4c.STATUS_OK

# ----- tracking states ------------------------------------------------------ #
STATE_LIVE = "LIVE"
STATE_LAST_SEEN = "LAST SEEN"
STATE_UNAVAILABLE = "POSITION UNAVAILABLE"
STATE_AMBIGUOUS = "AMBIGUOUS"

# ----- the tracking window -------------------------------------------------- #
# A LANDSCAPE window: the picker across the top, the whole-trip route map and
# the selected-trip summary side by side beneath it, and the controls along the
# bottom. The earlier portrait layout put the figure title, the panel titles
# and the summary text into each other, which is not a font-size problem and
# is not fixed by shrinking text.
TRACKING_FIGURE_SIZE = (11.0, 7.5)
TRACKING_WINDOW_TITLE = "Week 7 trip tracking"
PICKER_RECT = (0.045, 0.660, 0.915, 0.275)
TRIP_MAP_RECT = (0.045, 0.205, 0.550, 0.400)
DETAIL_RECT = (0.635, 0.205, 0.325, 0.400)
REFRESH_RECT = (0.635, 0.055, 0.150, 0.046)
UNTRACK_RECT = (0.805, 0.055, 0.155, 0.046)
STATUS_XY = (0.045, 0.105)        # left of the buttons, never under them
FOOTNOTE_XY = (0.045, 0.090)

PICKER_TITLE = "STOP: UPCOMING SERVICES - click one to track it"
TRIP_MAP_TITLE = "WHOLE TRIP - planned route (static GTFS)"
DETAIL_TITLE = "SELECTED TRIP"
REFRESH_LABEL = "Refresh all"
UNTRACK_LABEL = "Stop tracking"

PICKER_TOP = 0.86                 # axes fraction of the heading
PICKER_FIRST_ROW = 0.16           # heading -> first row, in axes fraction
PICKER_ROW = 0.125                # axes fraction per row
PICKER_FONTSIZE = 9.0
PICKER_HIT_PAD_PX = 5.0           # a whole row is clickable, not just glyphs
HINT_XY = (0.58, 0.86)            # beside the rows, never over them
DETAIL_FONTSIZE = 8.0
PANEL_WRAP = 46                   # the picker rows, which are wide
DETAIL_WRAP = 38                  # the narrower right-hand summary

PICK_HINT = ("The pinned trip is kept by its exact trip_id, even after\n"
             "that service leaves this stop's upcoming list.")

# The captured ArrivalEvent time is NOT a live ETA, and nothing recomputes it.
# Once the service has left the stop's list a bare "Expected" read as one.
EXPECTED_LABEL = "Expected when selected:"

# Two different Route 433 trips can legitimately be on this screen at once -
# one in the list above, one pinned below. Saying so is the whole point of
# V4D, and it is said in words rather than by putting trip_ids on screen.
SEPARATE_LIST_NOTE = "Upcoming services above are separate from the selected trip."
SAME_ROUTE_NOTE = "The Route {route} shown above is a different trip."

# ----- the tracked-vehicle marker ------------------------------------------- #
# A plain CIRCLE, deliberately: the bearing triangle read as a direction claim.
# A teal that is not already carrying meaning on this map (the routes are blue
# and orange, the origin green, the selected stop purple, V4B's vehicles
# magenta).
TRACK_COLOUR = "#00688b"
TRACK_MARKER_SIZE = 170
TRACK_LAST_SEEN_ALPHA = 0.55
TRACK_ZORDER = v4b.VEHICLE_ZORDER + 2
TRACK_LEGEND_LABEL = "tracked trip (current position)"

# V4B's panel names the vehicles of the CURRENT bounded upcoming services. In
# the V4D screen a second, separately pinned trip is also on the map, so the
# title is made specific HERE - by re-titling the axes after V4C built the
# window. demo_week7_live_vehicles.py is not touched and keeps its own title
# when it is run on its own.
UPCOMING_VEHICLES_TITLE = "UPCOMING SERVICE VEHICLES"

# The two windows answer two DIFFERENT questions, and a real refresh makes
# that plain: the stop's next service can legitimately be a 431 while the
# separately pinned 370 is still live. So the V4D composition names the
# question each panel answers. V3's own module keeps its own title when it is
# run on its own; only the composed screen is retitled.
STOP_ARRIVALS_TITLE = "STOP LIVE ARRIVALS"

# ----- focused tracking mode (V4D.2) ---------------------------------------- #
# While ONE trip is pinned, the V4D map shows only that trip's vehicle. The
# lower layers keep every bit of their state and their panels keep reporting
# it; their MAP artists are merely hidden here, and restored when tracking
# stops. Two different Route 433 trips can legitimately be on the map at once,
# and the demonstration has to make clear which one was selected.
FOCUS_LEGEND_LABEL = "selected tracked vehicle"
FOCUS_SHAPE_LEGEND_LABEL = "selected bus route (static GTFS)"

# ----- the whole-trip map (V4D.2) ------------------------------------------- #
SHAPE_COLOUR = "#4f6d7a"
SHAPE_LINEWIDTH = 2.2
SHAPE_ZORDER = 2
# On the GLEBE map the bus geometry is CONTEXT, not the finding: Week 5's two
# walking routes are drawn at zorder 3 and 4, so this sits behind them, thin,
# dashed and semi-transparent, and cannot be mistaken for a Week 5 result.
SHAPE_MAIN_MAP_LINEWIDTH = 1.6
SHAPE_MAIN_MAP_ALPHA = 0.35
SHAPE_MAIN_MAP_ZORDER = 1.5
SHAPE_MAIN_MAP_DASHES = (0, (6, 3))
TRIP_MAP_MARGIN = 0.15            # share of the shape's own span
NO_POSITION_WINDOW_M = 600.0      # a box around a lone point, when no shape

# Trip start, destination and the stop the user actually clicked. All three are
# drawn at their own STATIC coordinates; none of them is moved onto anything.
TRIP_START_COLOUR = "#2e7d32"
TRIP_END_COLOUR = "#c2185b"
SOURCE_STOP_COLOUR = "#7b3fa0"    # the same purple family the map already uses
TRIP_START_LABEL = "Trip start"
TRIP_END_PREFIX = "Destination: "
SOURCE_STOP_PREFIX = "Selected stop: "
TRIP_MAP_DIRECTION_LABEL = "Planned direction (static shape order)"
DIRECTION_ARROWS = 3
DIRECTION_AT = (0.2, 0.5, 0.8)    # fractions along the ORDERED shape
ENDPOINT_LABEL_WRAP = 26
TRIP_MAP_LEGEND_DROP = -0.035      # axes fractions BELOW the whole-trip axes

SHAPE_STATUS_NONE = "not requested"
SHAPE_STATUS_OK = "loaded"
SHAPE_STATUS_NO_COLUMN = "the static trips table has no shape_id column"
SHAPE_STATUS_NO_TRIP = "trip_id not in the static trips table"
SHAPE_STATUS_NO_SHAPE_ID = "no shape_id for this trip"
SHAPE_STATUS_NO_SOURCE = "no shapes.txt in the static snapshot"
SHAPE_STATUS_NOT_FOUND = "shape_id not found in shapes.txt"
SHAPE_UNAVAILABLE_TEXT = "Static trip shape unavailable for this trip"

SHAPE_CACHE_SUBDIR = "gtfs_shapes"
SHAPE_COLUMNS = ("shape_id", "shape_pt_lat", "shape_pt_lon",
                 "shape_pt_sequence")

# The two data sources on the whole-trip map are NAMED, because they are not
# the same kind of statement: one is the planned geometry of the timetable,
# the other is a position a vehicle reported. Neither is derived from the
# other, and the reported point is never moved onto the line.
TRIP_MAP_SHAPE_LABEL = "Planned route: static GTFS"
TRIP_MAP_LIVE_LABEL = "Vehicle position: GTFS-Realtime"
TRIP_MAP_LAST_SEEN_LABEL = "Last reported position: GTFS-Realtime (not current)"
TRIP_MAP_IDLE_TEXT = ("The selected trip's whole planned route\n"
                      "(static GTFS) will be drawn here.")
TRIP_MAP_NO_POSITION_TEXT = "no VehiclePosition for this trip in this snapshot"
TRIP_MAP_ENDPOINT_SIZE = 90
TRIP_MAP_STOP_SIZE = 210
TRIP_MAP_POSITION_SIZE = 150
TRIP_MAP_LABEL_FONTSIZE = 7.5

# The neither-of-these-is-a-trajectory sentence, in the demonstration's own
# words, on the panel that shows the two together.
TRIP_MAP_FOOTNOTE = ("Line = planned geometry (static GTFS).  Dot = one "
                     "reported position (GTFS-Realtime).\nNeither is an "
                     "observed path: no interpolation, no map matching, no "
                     "snapping.")



# --------------------------------------------------------------------------- #
# static trip geometry - the PLANNED route, never an observed one
# --------------------------------------------------------------------------- #
def shape_cache_dir(root: Optional[Path] = None) -> Path:
    return Path(root or week5.CONFIG_ROOT) / "data" / "cache" / SHAPE_CACHE_SUBDIR


def short(text: str, width: int) -> str:
    """Shorten a label for a map annotation. Nothing else uses this."""
    value = " ".join(str(text).split())
    if len(value) <= width:
        return value
    return value[:max(1, width - 1)].rstrip() + "…"


def running_phrase(delay_seconds: Optional[int]) -> str:
    """A signed delay, in words a reader does not have to decode.

    This is PRESENTATION of the number V3 already produced. The sign
    convention is GTFS-Realtime's own - negative is early - and no arithmetic
    beyond splitting the seconds into minutes happens here. An ABSENT delay
    is still absent: it is never shown as zero and never as "on time".
    """
    if delay_seconds is None:
        return "not supplied"
    value = int(delay_seconds)
    if value == 0:
        return "on time"
    minutes, seconds = divmod(abs(value), 60)
    size = f"{minutes} min {seconds} s" if minutes else f"{seconds} s"
    return f"{size} {'late' if value > 0 else 'early'}"


def safe_name(value: str) -> str:
    """A file-name-safe form of a shape_id. Identity is unchanged."""
    return "".join(character if character.isalnum() or character in "-._"
                   else "_" for character in str(value))


@dataclass
class TripShapeStore:
    """trip_id -> shape_id -> the ordered points of the PLANNED route.

    This is STATIC GTFS geometry. It is not a trajectory, it is not derived
    from any VehiclePosition, and the live marker is never moved onto it.

    Cost, and why it is bounded
    ---------------------------
    In the real TfNSW snapshot ``shapes.txt`` is about 950 MB and 12.1 million
    rows, and it is not one of the files Stage 2 extracts - only the source
    .zip in the same folder holds it. So a shape is fetched ONCE per shape_id,
    by a single streaming pass that keeps only the rows of the wanted shape,
    and the result (a few hundred points) is cached in memory and on disk.
    A refresh never touches any of this.
    """

    static_dir: Path
    cache_dir: Optional[Path] = None
    snapshot_key: str = "unknown"
    points: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    trip_shape_ids: Optional[Dict[str, str]] = None
    trip_ids: Optional[set] = None
    has_shape_id_column: Optional[bool] = None
    scans = 0

    def cache_path(self, shape_id: str) -> Path:
        directory = Path(self.cache_dir) if self.cache_dir else \
            shape_cache_dir()
        return directory / f"{self.snapshot_key}_{safe_name(shape_id)}.csv"

    # ----- trip_id -> shape_id -------------------------------------------- #
    def load_trip_shape_ids(self) -> Dict[str, str]:
        """The trip -> shape map, read once from the SAME static snapshot.

        V2 loads trips.txt without the ``shape_id`` column, and V2 is not
        being redesigned for this, so this narrow helper reads the two columns
        it needs from the same configured file.
        """
        if self.trip_shape_ids is not None:
            return self.trip_shape_ids
        path = Path(self.static_dir) / "trips.txt"
        if not path.exists():
            self.has_shape_id_column = None
            self.trip_ids = set()
            self.trip_shape_ids = {}
            return self.trip_shape_ids
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        self.has_shape_id_column = "shape_id" in header.columns
        if not self.has_shape_id_column:
            self.trip_ids = set()
            self.trip_shape_ids = {}
            return self.trip_shape_ids
        frame = pd.read_csv(path, usecols=["trip_id", "shape_id"], dtype=str,
                            encoding="utf-8-sig").fillna("")
        self.trip_ids = {str(value) for value in frame["trip_id"]}
        self.trip_shape_ids = {str(row.trip_id): str(row.shape_id)
                               for row in frame.itertuples(index=False)
                               if str(row.shape_id).strip()}
        return self.trip_shape_ids

    def why_no_shape_id(self, trip_id: str) -> str:
        """WHICH of the three different absences this is.

        A missing column, a trip the static snapshot does not contain, and a
        trip whose shape_id is blank are three different facts about the
        data, and they are not reported as one.
        """
        self.load_trip_shape_ids()
        if not self.has_shape_id_column:
            return SHAPE_STATUS_NO_COLUMN
        if self.trip_ids is not None and str(trip_id) not in self.trip_ids:
            return SHAPE_STATUS_NO_TRIP
        return SHAPE_STATUS_NO_SHAPE_ID

    def shape_id_for(self, trip_id: str) -> Optional[str]:
        return self.load_trip_shape_ids().get(str(trip_id))

    # ----- shape_id -> ordered points -------------------------------------- #
    def source(self) -> Tuple[Optional[Path], Optional[str]]:
        """Where shapes.txt can be read from: the file, else the source zip."""
        directory = Path(self.static_dir)
        extracted = directory / "shapes.txt"
        if extracted.exists():
            return (extracted, "file")
        for archive in sorted(directory.glob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as bundle:
                    if any(Path(name).name == "shapes.txt"
                           for name in bundle.namelist()):
                        return (archive, "zip")
            except (zipfile.BadZipFile, OSError):
                continue
        return (None, None)

    def points_for(self, shape_id: Optional[str]
                   ) -> Tuple[Optional[List[Tuple[float, float]]], str]:
        """The ordered points of one shape, and why there are none."""
        if not shape_id:
            return (None, SHAPE_STATUS_NO_SHAPE_ID)
        key = str(shape_id)
        if key in self.points:
            return (self.points[key], SHAPE_STATUS_OK)
        cached = self.read_cache(key)
        if cached is not None:
            self.points[key] = cached
            return (cached, SHAPE_STATUS_OK)
        path, kind = self.source()
        if path is None:
            return (None, SHAPE_STATUS_NO_SOURCE)
        rows = self.scan(path, kind, key)
        if not rows:
            return (None, SHAPE_STATUS_NOT_FOUND)
        self.points[key] = rows
        self.write_cache(key, rows)
        return (rows, SHAPE_STATUS_OK)

    def scan(self, path: Path, kind: Optional[str],
             shape_id: str) -> List[Tuple[float, float]]:
        """ONE streaming pass, keeping only the wanted shape's rows.

        The whole file is never held in memory, and the rows are ordered by
        ``shape_pt_sequence`` as GTFS defines - never by the order they
        happened to appear in.
        """
        TripShapeStore.scans += 1
        collected: List[Tuple[int, float, float]] = []
        for row in self.iter_rows(path, kind):
            if row.get("shape_id") != shape_id:
                continue
            try:
                collected.append((int(row["shape_pt_sequence"]),
                                  float(row["shape_pt_lat"]),
                                  float(row["shape_pt_lon"])))
            except (TypeError, ValueError):
                continue
        collected.sort(key=lambda item: item[0])
        return [(latitude, longitude)
                for _sequence, latitude, longitude in collected]

    def iter_rows(self, path: Path, kind: Optional[str]):
        if kind == "zip":
            with zipfile.ZipFile(path) as bundle:
                name = next(member for member in bundle.namelist()
                            if Path(member).name == "shapes.txt")
                with bundle.open(name) as raw:
                    stream = io.TextIOWrapper(raw, encoding="utf-8-sig",
                                              newline="")
                    for row in csv.DictReader(stream):
                        yield row
            return
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield row

    # ----- the small per-shape cache --------------------------------------- #
    def read_cache(self, shape_id: str) -> Optional[List[Tuple[float, float]]]:
        path = self.cache_path(shape_id)
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError):
            return None
        return [(float(row.shape_pt_lat), float(row.shape_pt_lon))
                for row in frame.itertuples(index=False)]

    def write_cache(self, shape_id: str,
                    rows: Sequence[Tuple[float, float]]) -> Optional[Path]:
        path = self.cache_path(shape_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=["shape_pt_lat",
                                        "shape_pt_lon"]).to_csv(path,
                                                                index=False)
        except OSError:
            return None
        return path


def shape_store_for(realtime_state: Any,
                    cache_dir: Optional[Path] = None
                    ) -> Optional[TripShapeStore]:
    """A shape store over the SAME static snapshot V2 already configured.

    The directory and the snapshot identity are taken from the loaded
    ``StaticGtfs`` object, so no path is written into this file and a
    re-downloaded bundle cannot be served from a stale cache: the cache key
    carries the snapshot's own sha256, exactly as V2's extract cache does.
    """
    static = getattr(realtime_state, "static", None)
    directory = getattr(static, "directory", None)
    if directory is None:
        return None
    provenance = getattr(static, "provenance", None) or {}
    key = str(provenance.get("zip_sha256", "unknown"))[:16] or "unknown"
    cache = (Path(cache_dir) if cache_dir is not None
             else Path(directory).parent / "cache" / SHAPE_CACHE_SUBDIR)
    return TripShapeStore(static_dir=Path(directory), cache_dir=cache,
                          snapshot_key=key)


# --------------------------------------------------------------------------- #
# what was observed, and what was pinned
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservedVehicle:
    """One VehiclePosition report for the tracked trip. A report, not a path."""

    latitude: float
    longitude: float
    vehicle_id: Optional[str]
    vehicle_timestamp: Optional[int]
    observed_at: int
    feed_timestamp: Optional[int]

    @classmethod
    def from_record(cls, record: Any, observed_at: int,
                    feed_timestamp: Optional[int]) -> "ObservedVehicle":
        return cls(latitude=float(record.latitude),
                   longitude=float(record.longitude),
                   vehicle_id=record.vehicle_id,
                   vehicle_timestamp=record.timestamp,
                   observed_at=int(observed_at),
                   feed_timestamp=feed_timestamp)

    def age(self, reference: Optional[int]) -> Optional[int]:
        """Seconds between a reference timestamp and this report.

        Negative means the vehicle timestamp is AHEAD of the reference. That
        is returned as it is; it is the caller's job to say so honestly
        rather than print a negative "ago".
        """
        if self.vehicle_timestamp is None or reference is None:
            return None
        return int(reference) - int(self.vehicle_timestamp)


AGE_BASIS_FEED = "feed timestamp"
AGE_BASIS_CLOCK = "local fetch clock"


def age_reference(snapshot: Any) -> Tuple[Optional[int], Optional[str]]:
    """What a VehiclePosition timestamp should be compared against.

    The feed's OWN header timestamp, because that and the vehicle timestamp
    come from the same realtime domain. The local fetch clock is only a
    fallback for a header that carried no timestamp, and it is named as such
    wherever it is used: the two clocks are not the same thing.
    """
    if snapshot is None:
        return (None, None)
    if snapshot.feed_timestamp is not None:
        return (int(snapshot.feed_timestamp), AGE_BASIS_FEED)
    return (int(snapshot.observed_at), AGE_BASIS_CLOCK)


def age_phrase(observation: "ObservedVehicle", snapshot: Any) -> str:
    """How old the report is - and never a negative "ago"."""
    reference, basis = age_reference(snapshot)
    seconds = observation.age(reference)
    if seconds is None:
        return "age not available"
    if seconds < 0:
        return f"{-seconds} s AHEAD of the {basis}"
    if basis == AGE_BASIS_FEED:
        return f"{seconds} s ago"
    return f"{seconds} s ago, vs {basis}"


@dataclass
class TrackedTrip:
    """ONE pinned service. Its identity is the trip_id, never the route.

    Everything except the observed position is captured from the ArrivalEvent
    the user picked, using V2's own already-resolved fields: nothing is
    re-derived, and the route number only ever comes from the adopted static
    chain V2 already applied.
    """

    trip_id: str
    route: str
    route_short_name: Optional[str]
    destination: Optional[str]
    source_stop_id: str
    source_stop_name: str
    scheduled_text: Optional[str]
    scheduled_epoch: Optional[int]
    expected_epoch: Optional[int]
    expected_basis: Optional[str]
    delay_seconds: Optional[int]
    realtime_route_id: Optional[str] = None
    static_route_id: Optional[str] = None
    pinned_at: int = 0

    state: str = STATE_UNAVAILABLE
    current: List[ObservedVehicle] = field(default_factory=list)
    last_position: Optional[ObservedVehicle] = None
    last_seen_at: Optional[int] = None
    invalid_coordinates: int = 0
    in_raw_events: Optional[bool] = None
    in_upcoming: Optional[bool] = None
    refresh_failed: bool = False

    # the PLANNED route from static GTFS - never an observed trajectory
    shape_id: Optional[str] = None
    shape_points: Optional[List[Tuple[float, float]]] = None
    shape_status: str = SHAPE_STATUS_NONE

    @property
    def label(self) -> str:
        """The map label: a resolved number when there is one, never a raw id."""
        if self.route_short_name:
            return self.route_short_name
        return "route unresolved"


def pin_event(event: Any, stop_id: str, stop_name: str) -> TrackedTrip:
    """Capture one displayed service, exactly as V2 already described it."""
    return TrackedTrip(
        trip_id=str(event.trip_id),
        route=v2.route_label(event),
        route_short_name=event.route_short_name,
        destination=(event.trip_headsign or event.route_long_name or None),
        source_stop_id=str(stop_id), source_stop_name=str(stop_name),
        scheduled_text=event.scheduled_arrival_text,
        scheduled_epoch=event.scheduled_epoch,
        expected_epoch=event.sort_epoch,
        expected_basis=event.sort_basis,
        delay_seconds=v3.event_delay(event),
        realtime_route_id=event.realtime_route_id,
        static_route_id=event.static_route_id,
        pinned_at=int(time.time()))


def source_presence_text(tracked: "TrackedTrip") -> str:
    """The precise source-stop diagnostic, kept for ``--debug-live``.

    The graphical panel now says the same thing in a sentence a reader can
    act on. The exact wording that describes WHICH of the two absences this
    is stays available here, unchanged.
    """
    if tracked.in_upcoming is False and tracked.in_raw_events is False:
        return "not in the source stop's snapshot now"
    if tracked.in_upcoming is False:
        return "not in the source stop's next few services"
    return "still in the source stop's next few services"


def stop_services(realtime_state: Any, stop_id: Optional[str],
                  limit: int) -> List[Any]:
    """The services V3 is displaying for this stop - V2's own bounded list."""
    return v4c.displayed_services(realtime_state, stop_id, limit)


def service_row(index: int, event: Any) -> str:
    """One selectable line. The route NUMBER never stands in for identity."""
    number = (event.route_short_name or "route unresolved")
    when = v3.clock(event.sort_epoch, seconds=False)
    destination = (event.trip_headsign or event.route_long_name or "")
    row = f"{index + 1})  {number:<6}{when:<7}{destination}"
    return row[:PANEL_WRAP + 12]


# --------------------------------------------------------------------------- #
# the tracker
# --------------------------------------------------------------------------- #
class TripTracker:
    """The pinned-service layer: one picker, one detail panel, one marker.

    It owns only its own figure, its own artists and its own state. It never
    reroutes, never moves a vehicle, never widens the map, and never edits
    what Week 5, V3, V4B or V4C drew.
    """

    def __init__(self, window: Dict[str, Any], debug: bool = False,
                 shapes: Optional[TripShapeStore] = None,
                 want_shape: bool = True):
        self.window = window
        self.controller = window["controller"]        # V4C
        self.panel = window["panel"]                  # V3
        self.overlay = window["overlay"]              # V4B
        self.explorer = window["explorer"]            # Week 5
        self.map_figure = window["figure"]

        self.figure = None
        self.picker_axis = None
        self.picker_text = None
        self.hint_text = None
        self.trip_map_axis = None
        self.trip_map_text = None
        self.trip_map_legend = None
        self.detail_axis = None
        self.detail_text = None
        self.status_text = None
        self.footnote_text = None
        self.refresh_button = None
        self.untrack_button = None
        self.legend = None

        self.rows: List[Any] = []                     # the ArrivalEvents shown
        self.row_artists: List[Tuple[Any, Any]] = []  # (Text, ArrivalEvent)
        self.tracked: Optional[TrackedTrip] = None
        self.artists: List[Any] = []
        self.shape_artists: List[Any] = []            # on the MAIN map
        self.trip_map_artists: List[Any] = []
        self.hidden: List[Any] = []                   # lower-layer map artists
        self.focused = False
        self.shapes: Optional[TripShapeStore] = None
        self.want_shape = True
        self.selected_stop: Optional[str] = None
        self.refresh_cycles = 0
        self.debug = bool(debug)
        self.debug_output: List[str] = []
        self.shapes = shapes
        self.want_shape = bool(want_shape)

    # ----- construction --------------------------------------------------- #
    def build(self, figure: Any = None):
        from matplotlib.lines import Line2D
        from matplotlib.widgets import Button

        import matplotlib.pyplot as plt

        # V4C's own Refresh all is hidden HERE so the screen has exactly one
        # coordinated control; its Button object and handler are untouched and
        # demo_week7_live_motion.py still shows it when run on its own.
        v4c.hide_button(self.controller.button)
        if self.overlay.axis is not None:
            self.overlay.axis.set_title(UPCOMING_VEHICLES_TITLE, fontsize=11,
                                        color=week5.plots.COLOUR_INK, pad=6.0)
        if self.panel.axis is not None:
            self.panel.axis.set_title(STOP_ARRIVALS_TITLE, fontsize=12,
                                      color=week5.plots.COLOUR_INK, pad=8.0)

        self.figure = figure if figure is not None else plt.figure(
            figsize=TRACKING_FIGURE_SIZE)
        # NO figure suptitle: it sat on top of the picker's own panel title.
        # The window's own title bar carries the name instead.
        self.name_window()

        self.picker_axis = self._panel_axes(PICKER_RECT, PICKER_TITLE)
        # The heading and the hint are their own artists; every SELECTABLE
        # service gets its own Text artist in ``row_artists``, so a click is
        # hit-tested against what was actually rendered.
        self.picker_text = self.picker_axis.text(
            0.02, PICKER_TOP, "", transform=self.picker_axis.transAxes,
            ha="left", va="top", fontsize=PICKER_FONTSIZE, family="monospace",
            color=week5.plots.COLOUR_INK)
        self.hint_text = self.picker_axis.text(
            HINT_XY[0], HINT_XY[1], "", transform=self.picker_axis.transAxes,
            ha="left", va="top", fontsize=7.5,
            color=week5.plots.COLOUR_MUTED)

        self.trip_map_axis = self._panel_axes(TRIP_MAP_RECT, TRIP_MAP_TITLE)
        self.trip_map_axis.set_aspect("equal", adjustable="box")
        self.trip_map_text = self.trip_map_axis.text(
            0.5, 0.5, "", transform=self.trip_map_axis.transAxes,
            ha="center", va="center", fontsize=9.0,
            color=week5.plots.COLOUR_MUTED)

        self.detail_axis = self._panel_axes(DETAIL_RECT, DETAIL_TITLE)
        self.detail_text = self.detail_axis.text(
            0.05, 0.95, "", transform=self.detail_axis.transAxes, ha="left",
            va="top", fontsize=DETAIL_FONTSIZE, family="monospace",
            color=week5.plots.COLOUR_INK)

        self.refresh_button = Button(self.figure.add_axes(REFRESH_RECT),
                                     REFRESH_LABEL)
        self.refresh_button.label.set_fontsize(9)
        self.refresh_button.on_clicked(self.on_refresh_all)
        self.untrack_button = Button(self.figure.add_axes(UNTRACK_RECT),
                                     UNTRACK_LABEL)
        self.untrack_button.label.set_fontsize(9)
        self.untrack_button.on_clicked(self.on_untrack)

        self.status_text = self.figure.text(
            STATUS_XY[0], STATUS_XY[1], "", ha="left", va="bottom",
            fontsize=7.5, color=week5.plots.COLOUR_MUTED)
        self.footnote_text = self.figure.text(
            FOOTNOTE_XY[0], FOOTNOTE_XY[1], TRIP_MAP_FOOTNOTE,
            ha="left", va="top", fontsize=6.6,
            color=week5.plots.COLOUR_MUTED)

        # One more legend entry on the MAP, so the teal circle is explained.
        if self.controller.legend is not None:
            handles = list(getattr(self.controller.legend, "legend_handles",
                                   None)
                           or getattr(self.controller.legend,
                                      "legendHandles", []))
            labels = [text.get_text()
                      for text in self.controller.legend.get_texts()]
            handles.append(Line2D([0], [0], marker="o", color="none",
                                  markersize=8, markerfacecolor=TRACK_COLOUR,
                                  markeredgecolor="#ffffff",
                                  label=TRACK_LEGEND_LABEL))
            labels.append(TRACK_LEGEND_LABEL)
            self.replace_map_legend(handles, labels)

        self.refresh_picker()
        self.draw_trip_map()
        self.render()
        return self

    def name_window(self) -> bool:
        """Put the name in the WINDOW's title bar, not on the canvas.

        Some backends (the Agg one the tests use) have no window manager, so
        this is allowed to do nothing rather than fail.
        """
        manager = getattr(self.figure.canvas, "manager", None)
        setter = getattr(manager, "set_window_title", None)
        if setter is None:
            return False
        try:
            setter(TRACKING_WINDOW_TITLE)
        except Exception:                         # a backend without a window
            return False
        return True

    def _panel_axes(self, rect, title):
        axis = self.figure.add_axes(rect)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#bbbbbb")
        axis.set_title(title, fontsize=10, color=week5.plots.COLOUR_INK,
                       pad=6.0)
        return axis

    def replace_map_legend(self, handles, labels) -> None:
        """Re-draw V4C's vehicle legend with one extra entry, in this window."""
        from matplotlib.legend import Legend

        self.controller.legend.remove()
        legend = Legend(self.explorer.map_axis, handles, labels,
                        loc="upper left", bbox_to_anchor=(0.012, 0.845),
                        bbox_transform=self.explorer.map_axis.transAxes,
                        fontsize=6.4, framealpha=0.92, handlelength=1.6,
                        handletextpad=0.5, borderpad=0.4, borderaxespad=0.0)
        self.explorer.map_axis.add_artist(legend)
        self.controller.legend = legend
        self.legend = legend

    def connect(self):
        """Two subscriptions: stop clicks on the map, service clicks here.

        The stop under the pointer is hit-tested with the Week 5 explorer's
        own primitive, and the picker is rebuilt from V3's snapshot, so this
        does not depend on the order in which Matplotlib dispatches the map
        handlers.
        """
        self.map_figure.canvas.mpl_connect("button_press_event",
                                           self.on_map_click)
        self.figure.canvas.mpl_connect("button_press_event",
                                       self.on_picker_click)
        return self

    # ----- events ---------------------------------------------------------- #
    def on_map_click(self, event) -> Optional[str]:
        """A stop click rebuilds the picker. It never drops the tracked trip."""
        if getattr(event, "inaxes", None) is not self.explorer.map_axis:
            return None
        if getattr(event, "button", 1) != 1:
            return None
        stop_id = self.explorer.hit_test_stop(event)
        if stop_id is None:
            return None
        self.selected_stop = str(stop_id)
        self.refresh_picker()
        self.render()
        return self.selected_stop

    def on_picker_click(self, event) -> Optional[str]:
        if getattr(event, "inaxes", None) is not self.picker_axis:
            return None
        if getattr(event, "button", 1) != 1:
            return None
        index = self.row_at(event)
        if index is None:
            return None
        return self.select_service(index)

    def row_at(self, event) -> Optional[int]:
        """Which RENDERED service row the click landed on, or None.

        The test is the row artist's own drawn extent, widened to the full
        panel width so the whole line is clickable. Nothing here reproduces
        the text layout arithmetically: the arithmetic version did not match
        what was on screen, and clicking row 1 could select row 2.
        """
        if not self.row_artists or self.picker_axis is None:
            return None
        if getattr(event, "x", None) is None or event.y is None:
            return None
        for index, (artist, _event) in enumerate(self.row_artists):
            box = self.row_extent(artist)
            if box is not None and box.contains(float(event.x),
                                                float(event.y)):
                return index
        return None

    def row_extent(self, artist: Any):
        """One row's drawn box, widened a little so the whole line is clickable.

        The box stops short of the hint column on the right, so nothing in
        the hint can ever be counted as a click on a service.
        """
        from matplotlib.transforms import Bbox

        try:
            renderer = self.figure.canvas.get_renderer()
            box = artist.get_window_extent(renderer=renderer)
        except (AttributeError, RuntimeError, ValueError):
            try:
                box = artist.get_window_extent()
            except Exception:                      # no renderer yet
                return None
        panel = self.picker_axis.get_window_extent()
        right = min(panel.x1, panel.x0 + HINT_XY[0] * panel.width)
        return Bbox.from_extents(panel.x0, box.y0 - PICKER_HIT_PAD_PX,
                                 max(right, box.x1 + PICKER_HIT_PAD_PX),
                                 box.y1 + PICKER_HIT_PAD_PX)

    def clear_row_artists(self) -> None:
        while self.row_artists:
            artist, _event = self.row_artists.pop()
            artist.remove()

    def draw_rows(self) -> List[Tuple[Any, Any]]:
        """One Text artist per selectable service, tied to its own event."""
        self.clear_row_artists()
        if self.picker_axis is None:
            return self.row_artists
        for index, event in enumerate(self.rows):
            artist = self.picker_axis.text(
                0.02, PICKER_TOP - PICKER_FIRST_ROW - index * PICKER_ROW,
                service_row(index, event),
                transform=self.picker_axis.transAxes, ha="left", va="top",
                fontsize=PICKER_FONTSIZE, family="monospace",
                color=week5.plots.COLOUR_INK)
            self.row_artists.append((artist, event))
        return self.row_artists

    def on_refresh_all(self, _event=None) -> Tuple[str, str]:
        """ONE coordinated cycle, then re-ask the feed about the pinned trip.

        The cycle itself is V4C's, unchanged: one Trip Update request and one
        Vehicle Position request, no polling and no timer.
        """
        self.refresh_cycles += 1
        statuses = self.controller.on_refresh_all()
        self.after_refresh()
        return statuses

    def on_untrack(self, _event=None) -> None:
        """Forget the pinned service, and give the map back to the layers.

        The lower layers never lost their state: their map artists were only
        hidden while one trip was focused, and they become visible again here.
        """
        self.tracked = None
        self.clear_artists()
        self.clear_shape_artists()
        self.draw_trip_map()
        self.apply_focus()
        self.render()
        self.redraw()

    # ----- tracking -------------------------------------------------------- #
    def select_service(self, index: int) -> Optional[str]:
        """Pin the EXACT service at this ROW - its trip_id, not its route.

        The event comes from the row artist that was actually clicked, so
        what the user sees and what gets pinned cannot drift apart.
        """
        if index < 0 or index >= len(self.row_artists):
            return None
        _artist, event = self.row_artists[index]
        if event.trip_id is None:
            return None
        stop_id = self.selected_stop or str(event.stop_id)
        self.tracked = pin_event(event, stop_id, event.stop_name)
        self.resolve_shape()
        self.evaluate()
        self.draw_main_shape()
        self.draw_tracked()
        self.draw_trip_map()
        self.apply_focus()
        self.render()
        self.redraw()
        return self.tracked.trip_id

    # ----- the planned route ---------------------------------------------- #
    def resolve_shape(self) -> str:
        """trip_id -> trips.txt -> shape_id -> shapes.txt, once per shape.

        STATIC geometry only. It is never inferred from a VehiclePosition, a
        realtime route_id or anything else the vehicle feed said.
        """
        tracked = self.tracked
        if tracked is None:
            return SHAPE_STATUS_NONE
        if not self.want_shape or self.shapes is None:
            tracked.shape_status = SHAPE_STATUS_NONE
            return tracked.shape_status
        shape_id = self.shapes.shape_id_for(tracked.trip_id)
        tracked.shape_id = shape_id
        if shape_id is None:
            tracked.shape_points = None
            tracked.shape_status = self.shapes.why_no_shape_id(tracked.trip_id)
            return tracked.shape_status
        if self.trip_map_text is not None:
            self.trip_map_text.set_text("Loading the planned route from the\n"
                                        "static GTFS snapshot (once, then\n"
                                        "cached) ...")
            self.trip_map_text.set_visible(True)
            self.redraw()
        points, status = self.shapes.points_for(shape_id)
        tracked.shape_points = points
        tracked.shape_status = status
        return status

    def after_refresh(self) -> Optional[str]:
        """Re-evaluate the pinned trip against the new snapshots.

        The planned route is NOT re-read: it is static geometry keyed by
        shape_id, so a refresh never touches ``shapes.txt`` again. Only the
        reported position, and the layers' own newly drawn map artists,
        change here.
        """
        self.refresh_picker()
        state = self.evaluate()
        self.draw_main_shape()
        self.draw_tracked()
        self.draw_trip_map()
        self.apply_focus()
        self.render()
        self.redraw()
        if self.debug:
            self.debug_output = self.debug_lines()
            print("\n".join(self.debug_output))
        return state

    def refresh_picker(self) -> List[Any]:
        """The rows are V3's own displayed services for the selected stop."""
        self.rows = stop_services(self.panel.state, self.selected_stop,
                                  self.panel.limit)
        self.draw_rows()
        return self.rows

    def evaluate(self) -> Optional[str]:
        """Ask the vehicle feed about the pinned trip_id, and nothing else.

        ``records_for(trip_id)`` is queried directly, so the answer does not
        depend on whether the service is still in any stop's upcoming list.
        The pinned trip is never dropped because a snapshot lacks it.
        """
        tracked = self.tracked
        if tracked is None:
            return None
        state = self.overlay.state
        snapshot = state.snapshot if state.has_snapshot else None
        records = (snapshot.records_for(tracked.trip_id) if snapshot is not None
                   else [])
        usable = [record for record in records if record.coordinate_valid]
        tracked.invalid_coordinates = len(records) - len(usable)
        self.update_source_presence(tracked)

        # A FAILED vehicle refresh leaves V4B holding its previous successful
        # snapshot. That snapshot is still good evidence of where the vehicle
        # WAS, but it is not confirmation from the refresh that just failed,
        # so nothing here may call it current.
        tracked.refresh_failed = state.status != STATUS_OK
        if tracked.refresh_failed:
            usable = []

        if usable and snapshot is not None:
            observations = [ObservedVehicle.from_record(
                record, snapshot.observed_at, snapshot.feed_timestamp)
                for record in usable]
            tracked.current = observations
            tracked.last_seen_at = snapshot.observed_at
            if len(observations) > 1:
                # Two simultaneous positions for one trip_id is an AMBIGUITY.
                # Both are kept and shown; none is chosen, and none of them is
                # promoted to "the" last known position.
                tracked.state = STATE_AMBIGUOUS
            else:
                tracked.state = STATE_LIVE
                tracked.last_position = observations[0]
        else:
            tracked.current = []
            if tracked.last_position is not None:
                # The current snapshot does not carry it. The last position
                # that WAS observed is retained and labelled as such - it is
                # never presented as the current position.
                tracked.state = STATE_LAST_SEEN
            else:
                tracked.state = STATE_UNAVAILABLE
        return tracked.state

    def update_source_presence(self, tracked: TrackedTrip) -> None:
        """Where the pinned trip stands at ITS OWN source stop.

        Not at whatever stop the user has clicked since: the picker follows
        the map, but the tracked service stays pinned to the stop it was
        selected at, and these two diagnostics describe THAT stop.
        """
        source = tracked.source_stop_id
        tracked.in_raw_events = tracked.trip_id in v4c.all_stop_trip_ids(
            self.panel.state, source)
        tracked.in_upcoming = any(
            str(event.trip_id) == tracked.trip_id
            for event in v4c.displayed_services(self.panel.state, source,
                                                self.panel.limit))

    # ----- drawing --------------------------------------------------------- #
    def clear_artists(self) -> None:
        """Remove ONLY this layer's marker and label."""
        while self.artists:
            self.artists.pop().remove()

    def positions(self) -> List[ObservedVehicle]:
        """What to draw for the tracked trip, if anything."""
        tracked = self.tracked
        if tracked is None:
            return []
        if tracked.current:
            return list(tracked.current)
        if tracked.state == STATE_LAST_SEEN and tracked.last_position:
            return [tracked.last_position]
        return []

    def draw_tracked(self) -> None:
        """A plain CIRCLE for the tracked vehicle - no bearing triangle.

        A point outside the current displayed extent is not drawn and the
        extent is never changed to bring it in. The coordinate is exactly the
        one TfNSW supplied: nothing is snapped, moved or interpolated.
        """
        self.clear_artists()
        axis = self.explorer.map_axis
        tracked = self.tracked
        if axis is None or tracked is None:
            return
        extent = self.overlay.extent()
        live = tracked.state in (STATE_LIVE, STATE_AMBIGUOUS)
        alpha = 1.0 if live else TRACK_LAST_SEEN_ALPHA
        for observation in self.positions():
            point = week5.plots.project([observation.longitude],
                                        [observation.latitude])[0]
            position = (float(point[0]), float(point[1]))
            if not v4b.inside_extent(position, extent):
                continue
            self.artists.append(axis.scatter(
                [position[0]], [position[1]], marker="o",
                s=TRACK_MARKER_SIZE, facecolor=TRACK_COLOUR,
                edgecolor="#ffffff", linewidth=1.8, alpha=alpha,
                zorder=TRACK_ZORDER))
            offset, ha, va = self.label_offset(position)
            self.artists.append(axis.annotate(
                self.marker_label(), position, textcoords="offset points",
                xytext=offset, ha=ha, va=va, fontsize=8.5,
                color=TRACK_COLOUR, alpha=alpha, zorder=TRACK_ZORDER + 1,
                bbox={"facecolor": "#ffffff", "edgecolor": TRACK_COLOUR,
                      "boxstyle": "round,pad=0.2", "alpha": 0.92}))

    def marker_label(self) -> str:
        """The route number and the tracking state. Nothing else.

        In focused mode this is the ONLY realtime vehicle emphasised on the
        map, so the source stop and the expected time no longer have to be
        carried in the annotation - they are named in the SELECTED TRIP panel
        and on the whole-trip route map, where there is room for them.
        """
        tracked = self.tracked
        if tracked is None:
            return ""
        state = ("AMBIGUOUS ?" if tracked.state == STATE_AMBIGUOUS
                 else tracked.state)
        return f"{tracked.label} \u00b7 {state}"

    def label_offset(self, position: Tuple[float, float]):
        """Keep the label off Week 5's top-left instruction box.

        The marker itself never moves: only the side the text sits on
        changes, and only when the marker is under that corner.
        """
        axis = self.explorer.map_axis
        if axis is None:
            return ((0, 13), "center", "bottom")
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        if right == left or top == bottom:
            return ((0, 13), "center", "bottom")
        fx = (position[0] - left) / (right - left)
        fy = (position[1] - bottom) / (top - bottom)
        if fx < 0.42 and fy > 0.62:              # under the instruction box
            return ((13, -6), "left", "top")
        return ((0, 13), "center", "bottom")

    # ----- the planned route, drawn ---------------------------------------- #
    def clear_shape_artists(self) -> None:
        while self.shape_artists:
            self.shape_artists.pop().remove()

    def project_shape(self, points: Sequence[Tuple[float, float]]
                      ) -> List[Tuple[float, float]]:
        """Shape points into Week 5's metric frame. Nothing is resampled."""
        if not points:
            return []
        projected = week5.plots.project([longitude for _lat, longitude in points],
                                        [latitude for latitude, _lon in points])
        return [(float(point[0]), float(point[1])) for point in projected]

    def draw_main_shape(self) -> int:
        """Only the PORTION of the planned route already inside the map.

        The Glebe extent is read, never changed: a route that runs to Balmain
        is not a reason to widen the study-area map. Points outside it are
        dropped, and only contiguous runs of inside points are joined, so the
        drawn line never bridges across geometry that is off the screen.
        """
        self.clear_shape_artists()
        axis = self.explorer.map_axis
        tracked = self.tracked
        if axis is None or tracked is None or not tracked.shape_points:
            return 0
        extent = self.overlay.extent()
        if extent is None:
            return 0
        runs: List[List[Tuple[float, float]]] = []
        run: List[Tuple[float, float]] = []
        for position in self.project_shape(tracked.shape_points):
            if v4b.inside_extent(position, extent):
                run.append(position)
            else:
                if len(run) > 1:
                    runs.append(run)
                run = []
        if len(run) > 1:
            runs.append(run)
        for piece in runs:
            line, = axis.plot([point[0] for point in piece],
                              [point[1] for point in piece],
                              color=SHAPE_COLOUR,
                              linewidth=SHAPE_MAIN_MAP_LINEWIDTH,
                              alpha=SHAPE_MAIN_MAP_ALPHA,
                              linestyle=SHAPE_MAIN_MAP_DASHES,
                              solid_capstyle="round",
                              zorder=SHAPE_MAIN_MAP_ZORDER)
            self.shape_artists.append(line)
        return len(self.shape_artists)

    def clear_trip_map(self) -> None:
        while self.trip_map_artists:
            self.trip_map_artists.pop().remove()
        if self.trip_map_legend is not None:
            try:
                self.trip_map_legend.remove()
            except (AttributeError, ValueError):
                pass
            self.trip_map_legend = None

    def draw_trip_map(self) -> str:
        """The WHOLE planned trip, with the reported position drawn on top.

        Two different statements share these axes and are labelled as such:
        the line is the timetable's planned geometry from static GTFS, and
        the point is a position a vehicle reported in GTFS-Realtime. The
        point is drawn at the coordinate TfNSW supplied - it is never snapped
        to the line, and no part of the line is presented as travelled.
        """
        axis = self.trip_map_axis
        if axis is None:
            return SHAPE_STATUS_NONE
        self.clear_trip_map()
        tracked = self.tracked
        if self.trip_map_text is not None:
            self.trip_map_text.set_text(TRIP_MAP_IDLE_TEXT)
            self.trip_map_text.set_visible(tracked is None)
        if tracked is None:
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            return SHAPE_STATUS_NONE

        handles: List[Any] = []
        labels: List[str] = []
        shape = self.project_shape(tracked.shape_points or [])
        observations = self.positions()
        # The view is fitted BEFORE anything is drawn, because the labels are
        # placed towards the middle of the fitted view and would otherwise be
        # positioned against the previous selection's limits.
        self.fit_trip_map(shape, observations)
        if shape:
            line, = axis.plot([point[0] for point in shape],
                              [point[1] for point in shape],
                              color=SHAPE_COLOUR, linewidth=SHAPE_LINEWIDTH,
                              solid_capstyle="round", zorder=SHAPE_ZORDER)
            self.trip_map_artists.append(line)
            handles.append(line)
            labels.append(TRIP_MAP_SHAPE_LABEL)
            arrow = self.draw_direction(shape)
            if arrow is not None:
                handles.append(arrow)
                labels.append(TRIP_MAP_DIRECTION_LABEL)
            self.draw_endpoints(shape, tracked)

        self.draw_source_stop(tracked)
        live = bool(tracked.current)
        named = False                 # one legend entry, even if AMBIGUOUS
        for observation in observations:
            point = week5.plots.project([observation.longitude],
                                        [observation.latitude])[0]
            position = (float(point[0]), float(point[1]))
            marker = axis.scatter(
                [position[0]], [position[1]],
                s=TRIP_MAP_POSITION_SIZE, marker="o" if live else "X",
                facecolor=TRACK_COLOUR if live else "#ffffff",
                edgecolor=TRACK_COLOUR if not live else "#ffffff",
                linewidth=1.8,
                alpha=1.0 if live else TRACK_LAST_SEEN_ALPHA,
                zorder=TRACK_ZORDER)
            self.trip_map_artists.append(marker)
            self.trip_map_artists.append(axis.annotate(
                self.vehicle_label(observation), position,
                textcoords="offset points", xytext=(0, 12), ha="center",
                va="bottom", fontsize=TRIP_MAP_LABEL_FONTSIZE,
                color=TRACK_COLOUR, zorder=TRACK_ZORDER + 1,
                bbox={"facecolor": "#ffffff", "edgecolor": TRACK_COLOUR,
                      "boxstyle": "round,pad=0.18", "alpha": 0.92}))
            if not named:
                named = True
                handles.append(marker)
                labels.append(TRIP_MAP_LIVE_LABEL if live
                              else TRIP_MAP_LAST_SEEN_LABEL)

        if shape and not observations:
            # The planned route is still worth showing on its own; it just may
            # not be said to be where the vehicle is.
            self.trip_map_artists.append(axis.text(
                0.5, 0.97, TRIP_MAP_NO_POSITION_TEXT,
                transform=axis.transAxes, ha="center", va="top", fontsize=7.0,
                color=week5.plots.COLOUR_MUTED))
        if not shape and self.trip_map_text is not None:
            self.trip_map_text.set_text(
                f"{SHAPE_UNAVAILABLE_TEXT}\n({tracked.shape_status})")
            self.trip_map_text.set_visible(True)
        if handles:
            # UNDER the axes, in one row: an in-axes legend always ended up
            # on top of the route or of the "Trip start" label.
            self.trip_map_legend = axis.legend(
                handles, labels, loc="upper left",
                bbox_to_anchor=(0.0, TRIP_MAP_LEGEND_DROP),
                bbox_transform=axis.transAxes, ncol=len(handles),
                fontsize=6.0, frameon=False, handlelength=1.4,
                handletextpad=0.4, columnspacing=1.2, borderpad=0.2,
                borderaxespad=0.0, scatterpoints=1)
        return tracked.shape_status

    # ----- what makes the route readable ----------------------------------- #
    def draw_endpoints(self, shape: Sequence[Tuple[float, float]],
                       tracked: TrackedTrip) -> int:
        """The trip's START and its DESTINATION, told apart.

        Two identical circles at the two ends said nothing; these are the
        first and last points of the ORDERED static shape, and they are
        labelled with what they are.
        """
        axis = self.trip_map_axis
        if axis is None or not shape:
            return 0
        drawn = 0
        for point, colour, marker, label in (
                (shape[0], TRIP_START_COLOUR, "o", TRIP_START_LABEL),
                (shape[-1], TRIP_END_COLOUR, "s",
                 TRIP_END_PREFIX + short(tracked.destination or "unnamed",
                                         ENDPOINT_LABEL_WRAP))):
            self.trip_map_artists.append(axis.scatter(
                [point[0]], [point[1]], s=TRIP_MAP_ENDPOINT_SIZE,
                marker=marker, facecolor=colour, edgecolor="#ffffff",
                linewidth=1.4, zorder=SHAPE_ZORDER + 2))
            offset, ha, va = self.inward_offset(point)
            self.trip_map_artists.append(axis.annotate(
                label, point, textcoords="offset points", xytext=offset,
                ha=ha, va=va, fontsize=TRIP_MAP_LABEL_FONTSIZE, color=colour,
                zorder=SHAPE_ZORDER + 3))
            drawn += 1
        return drawn

    def inward_offset(self, point: Tuple[float, float]):
        """Put a point's label towards the MIDDLE of the whole-trip panel.

        A trip end sits at the edge of the fitted view by definition, so a
        fixed offset pushed its label off the panel. The marker itself does
        not move; only the side its text sits on does.
        """
        axis = self.trip_map_axis
        if axis is None:
            return ((0, 12), "center", "bottom")
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        if right == left or top == bottom:
            return ((0, 12), "center", "bottom")
        fx = (point[0] - left) / (right - left)
        fy = (point[1] - bottom) / (top - bottom)
        if fy > 0.5:                              # near the top: label below
            return ((0, -12), "right" if fx > 0.6 else
                    ("left" if fx < 0.4 else "center"), "top")
        return ((0, 12), "right" if fx > 0.6 else
                ("left" if fx < 0.4 else "center"), "bottom")

    def source_stop_position(self) -> Optional[Tuple[float, float]]:
        """The SELECTED stop's own static coordinate, projected.

        Week 5 already holds every candidate stop's ``stop_lat``/``stop_lon``
        projected by its own primitive, so this reads that table rather than
        deriving a position - and certainly rather than taking the nearest
        point on the shape.
        """
        tracked = self.tracked
        if tracked is None:
            return None
        stop_ids = list(getattr(self.explorer, "stop_ids", []) or [])
        positions = getattr(self.explorer, "positions", None)
        if positions is None or not len(positions):
            return None
        wanted = str(tracked.source_stop_id)
        if wanted not in stop_ids:
            return None
        row = positions[stop_ids.index(wanted)]
        return (float(row[0]), float(row[1]))

    def draw_source_stop(self, tracked: TrackedTrip) -> bool:
        """The stop the user actually clicked, at its RAW static coordinate."""
        axis = self.trip_map_axis
        position = self.source_stop_position()
        if axis is None or position is None:
            return False
        self.trip_map_artists.append(axis.scatter(
            [position[0]], [position[1]], s=TRIP_MAP_STOP_SIZE, marker="*",
            facecolor=SOURCE_STOP_COLOUR, edgecolor="#ffffff", linewidth=1.2,
            zorder=SHAPE_ZORDER + 4))
        name = tracked.source_stop_name or tracked.source_stop_id
        right = self.side_of(position) < 0.55
        self.trip_map_artists.append(axis.annotate(
            SOURCE_STOP_PREFIX + short(name, ENDPOINT_LABEL_WRAP), position,
            textcoords="offset points", xytext=(11 if right else -11, -2),
            ha="left" if right else "right", va="center",
            fontsize=TRIP_MAP_LABEL_FONTSIZE, color=SOURCE_STOP_COLOUR,
            zorder=SHAPE_ZORDER + 5))
        return True

    def side_of(self, point: Tuple[float, float]) -> float:
        """How far across the whole-trip panel a point sits, 0 to 1."""
        axis = self.trip_map_axis
        if axis is None:
            return 0.5
        left, right = axis.get_xlim()
        if right == left:
            return 0.5
        return (point[0] - left) / (right - left)

    def draw_direction(self, shape: Sequence[Tuple[float, float]]) -> Any:
        """A few arrows showing which way the PLANNED trip runs.

        Their only input is the order of ``shape_pt_sequence``, which is what
        the static timetable says. They are not observed motion, they are not
        a speed, and they are not derived from any VehiclePosition.
        """
        from matplotlib.lines import Line2D

        axis = self.trip_map_axis
        if axis is None or len(shape) < 2:
            return None
        step = max(1, len(shape) // 40)
        drawn = 0
        for fraction in DIRECTION_AT[:DIRECTION_ARROWS]:
            start = int(fraction * (len(shape) - 1))
            end = min(len(shape) - 1, start + step)
            if end <= start:
                continue
            self.trip_map_artists.append(axis.annotate(
                "", xy=shape[end], xytext=shape[start],
                arrowprops={"arrowstyle": "-|>", "color": SHAPE_COLOUR,
                            "linewidth": 1.4, "shrinkA": 0.0, "shrinkB": 0.0,
                            "mutation_scale": 14},
                zorder=SHAPE_ZORDER + 1, annotation_clip=False))
            drawn += 1
        if not drawn:
            return None
        return Line2D([0], [0], color=SHAPE_COLOUR, linewidth=1.4,
                      marker=">", markersize=5,
                      label=TRIP_MAP_DIRECTION_LABEL)

    def vehicle_label(self, observation: ObservedVehicle) -> str:
        """The compact state + age label beside the reported position."""
        tracked = self.tracked
        if tracked is None:
            return ""
        snapshot = (self.overlay.state.snapshot
                    if self.overlay.state.has_snapshot else None)
        state = ("AMBIGUOUS ?" if tracked.state == STATE_AMBIGUOUS
                 else tracked.state)
        return f"{state} · updated {age_phrase(observation, snapshot)}"

    def fit_trip_map(self, shape: Sequence[Tuple[float, float]],
                     observations: Sequence[ObservedVehicle]) -> bool:
        """Fit THESE axes to the whole shape. The Glebe map is not touched.

        With no shape but a reported position, a small box around that single
        point is used instead, so the panel still shows something true rather
        than an empty frame.
        """
        axis = self.trip_map_axis
        if axis is None:
            return False
        xs = [point[0] for point in shape]
        ys = [point[1] for point in shape]
        for observation in observations:
            point = week5.plots.project([observation.longitude],
                                        [observation.latitude])[0]
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        stop = self.source_stop_position()
        if stop is not None:
            xs.append(stop[0])
            ys.append(stop[1])
        if not xs:
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            return False
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span <= 0.0:
            span = NO_POSITION_WINDOW_M
        pad = span * TRIP_MAP_MARGIN
        centre_x = (max(xs) + min(xs)) / 2.0
        centre_y = (max(ys) + min(ys)) / 2.0
        half = span / 2.0 + pad
        axis.set_xlim(centre_x - half, centre_x + half)
        axis.set_ylim(centre_y - half, centre_y + half)
        return True

    # ----- focused tracking mode ------------------------------------------ #
    def lower_layer_map_artists(self) -> List[Any]:
        """Every MAP artist the layers below drew for their own purposes.

        Their state is untouched; only these drawn objects are hidden, and
        the right-hand panels keep reporting the same services in words.
        """
        artists = list(self.overlay.artists)          # V4B current markers
        artists += list(self.controller.artists)      # V4C history points
        artists += list(self.controller.bridge_artists)
        if self.controller.bridge_text is not None:
            artists.append(self.controller.bridge_text)
        return artists

    def apply_focus(self) -> bool:
        """Hide the generic overlays while ONE trip is pinned."""
        focused = self.tracked is not None
        if focused:
            self.hidden = []
            for artist in self.lower_layer_map_artists():
                if artist.get_visible():
                    artist.set_visible(False)
                    self.hidden.append(artist)
        elif self.focused:
            for artist in self.hidden:
                try:
                    artist.set_visible(True)
                except Exception:                     # already removed
                    continue
            self.hidden = []
        if focused != self.focused or focused:
            self.focused = focused
            self.set_map_legend(focused)
        self.focused = focused
        return focused

    def set_map_legend(self, focused: bool) -> None:
        """The map legend says only what can actually be seen right now."""
        from matplotlib.lines import Line2D

        tracked_handle = Line2D([0], [0], marker="o", color="none",
                                markersize=8, markerfacecolor=TRACK_COLOUR,
                                markeredgecolor="#ffffff",
                                label=FOCUS_LEGEND_LABEL)
        if focused:
            handles = [tracked_handle]
            labels = [FOCUS_LEGEND_LABEL]
            if self.shape_artists:
                handles.append(Line2D([0], [0], color=SHAPE_COLOUR,
                                      linewidth=2.0,
                                      label=FOCUS_SHAPE_LEGEND_LABEL))
                labels.append(FOCUS_SHAPE_LEGEND_LABEL)
        else:
            handles = [
                Line2D([0], [0], marker="o", color="none", markersize=7,
                       markerfacecolor=v4c.COLOUR_VEHICLE,
                       markeredgecolor="#ffffff", label=v4b.LEGEND_LABEL),
                Line2D([0], [0], marker="o", color="none", markersize=4,
                       markerfacecolor=v4c.COLOUR_VEHICLE,
                       markeredgecolor="none", alpha=0.5,
                       label=v4c.HISTORY_LEGEND_LABEL),
                Line2D([0], [0], marker="o", color="none", markersize=7,
                       markerfacecolor="none",
                       markeredgecolor=v4c.COLOUR_VEHICLE,
                       markeredgewidth=1.4, label=v4c.BRIDGE_LEGEND_LABEL),
                tracked_handle,
            ]
            labels = [handle.get_label() for handle in handles]
        self.replace_map_legend(handles, labels)

    # ----- text ------------------------------------------------------------ #
    def picker_lines(self) -> List[str]:
        """The HEADING only.

        Every selectable service is its own Text artist, drawn by
        ``draw_rows``, so that a click can be hit-tested against what was
        actually rendered. Repeating the same rows in this block would put
        two copies on the screen and make the hit test meaningless.
        """
        if self.selected_stop is None:
            return ["Click a candidate bus stop on the map.",
                    "Its next services will be listed here,",
                    "and each one can be tracked individually."]
        name = self.panel.state.name_of(self.selected_stop)
        lines = [f"Stop {self.selected_stop}  {name}".rstrip()]
        if not self.rows:
            lines += ["", "No upcoming services are listed for this stop",
                      "in the current Trip Update snapshot."]
        return lines

    def hint_line(self) -> str:
        """The click hint, shown only when there is something to click."""
        return PICK_HINT if self.rows else ""

    def detail_lines(self) -> List[str]:
        """What a reader of the demonstration needs, and nothing more.

        The identifiers this layer actually works with - trip_id, vehicle_id,
        raw latitude and longitude, shape_id, the shape's point count - are
        still reported in full, but by ``--debug-live`` in the terminal. On
        screen they crowded out the four things the panel is FOR: which
        service, when it was due, whether it is running early or late, and
        whether its position is current.
        """
        tracked = self.tracked
        if tracked is None:
            return ["No service is being tracked.", "",
                    "Click one of the services above to pin it.",
                    "Each one is pinned as its own exact trip, so two",
                    "Route 433 services are two different trips, not",
                    "one route."]
        lines = [f"Route {tracked.label} → {tracked.destination or '-'}",
                 ""]
        lines += ["Source stop:",
                  f"  {tracked.source_stop_name or tracked.source_stop_id}",
                  ""]
        # Both times were CAPTURED when the trip was pinned. Neither is
        # recomputed, and neither is an ETA: saying just "Expected" invited
        # exactly that reading once the service had left the stop's list.
        lines.append("  Scheduled: "
                     + (v3.clock(tracked.scheduled_epoch)
                        if tracked.scheduled_epoch is not None else "-"))
        lines.append(f"  {EXPECTED_LABEL} "
                     + (v3.clock(tracked.expected_epoch)
                        if tracked.expected_epoch is not None else "-"))
        lines.append(f"  Running:   {running_phrase(tracked.delay_seconds)}")
        lines += ["", f"Tracking: {tracked.state}"]
        lines += self.vehicle_lines()
        lines += self.presence_lines()
        if not tracked.shape_points and self.want_shape:
            lines += ["", SHAPE_UNAVAILABLE_TEXT + "."]
        return lines

    def vehicle_lines(self) -> List[str]:
        """When the position was reported - never where, in numbers."""
        tracked = self.tracked
        if tracked is None:
            return []
        snapshot = (self.overlay.state.snapshot
                    if self.overlay.state.has_snapshot else None)
        if tracked.state == STATE_UNAVAILABLE:
            return ["No vehicle position has been reported for this",
                    "trip yet. That says nothing about the service."]
        observations = self.positions()
        if not observations:
            return []
        if tracked.state == STATE_AMBIGUOUS:
            lines = [f"{len(observations)} vehicles report this same trip.",
                     "Both are shown; neither is chosen."]
        else:
            lines = []
        heading = ("Vehicle update: " if tracked.current
                   else "Last reported:  ")
        for observation in observations:
            lines.append(heading
                         + f"{v4a.clock(observation.vehicle_timestamp)}"
                         + f" ({age_phrase(observation, snapshot)})")
        return lines

    def presence_lines(self) -> List[str]:
        """The V4D point, in one sentence a reader can act on.

        The precise diagnostic wording is kept, unchanged, in the
        ``--debug-live`` output; this is the same fact, said plainly.
        """
        tracked = self.tracked
        if tracked is None:
            return []
        lines: List[str] = []
        if tracked.in_upcoming is False:
            lines += ["", "No longer listed at source stop;",
                      "tracking continues.", self.separation_note()]
        if tracked.state == STATE_LAST_SEEN:
            if tracked.refresh_failed:
                lines += ["", "Vehicle refresh failed. The position",
                          "above is the last successful report,",
                          "not a new one."]
            else:
                lines += ["", "The current vehicle snapshot does not",
                          "carry this trip.",
                          "That is not a cancellation."]
        return lines

    def separation_note(self) -> str:
        """Why the list above and the trip below can disagree, in one line.

        A LATER Route 433 in the stop's list is a different exact trip from
        the pinned one, and the screen has to say so - without putting a
        trip_id in front of a reader who should not need one. The comparison
        is on what is DISPLAYED (the route number); the identity that is
        actually tracked is untouched by this and is still the trip_id.
        """
        tracked = self.tracked
        if tracked is None:
            return ""
        if tracked.route_short_name and any(
                str(event.route_short_name) == str(tracked.route_short_name)
                for event in self.rows):
            return SAME_ROUTE_NOTE.format(route=tracked.route_short_name)
        return SEPARATE_LIST_NOTE

    def status_line(self) -> str:
        """One short line: what the last cycle did, and what is pinned."""
        if not self.refresh_cycles:
            return (f"'{REFRESH_LABEL}' runs one coordinated arrivals + "
                    "vehicles refresh. No polling.")
        return "  |  ".join([
            f"#{self.refresh_cycles} {v4a.clock(int(time.time()))}",
            f"vehicles {self.controller.last_vehicle_change or '-'}",
            f"tracking {self.tracked.trip_id if self.tracked else 'none'}",
        ])

    def render(self) -> str:
        picker = "\n".join(v3.wrap(self.picker_lines(), PANEL_WRAP + 12))
        detail = "\n".join(v3.wrap(self.detail_lines(), DETAIL_WRAP))
        if self.picker_text is not None:
            self.picker_text.set_text(picker)
        if self.hint_text is not None:
            self.hint_text.set_text(self.hint_line())
        if self.detail_text is not None:
            self.detail_text.set_text(detail)
        if self.status_text is not None:
            self.status_text.set_text(self.status_line())
        return detail

    def redraw(self) -> None:
        for figure in (self.figure, self.map_figure):
            canvas = getattr(figure, "canvas", None)
            if canvas is not None and hasattr(canvas, "draw_idle"):
                canvas.draw_idle()

    # ----- diagnostics ----------------------------------------------------- #
    def debug_lines(self) -> List[str]:
        """The tracked trip, in the terms the feed answered in.

        It holds no key and prints no header: this file never sees either.
        """
        tracked = self.tracked
        lines = ["", "TRACKED TRIP DEBUG"]
        if tracked is None:
            return lines + ["  no service is being tracked"]
        state = self.overlay.state
        snapshot = state.snapshot if state.has_snapshot else None
        records = (snapshot.records_for(tracked.trip_id)
                   if snapshot is not None else [])
        lines += [
            f"  tracked trip_id: {tracked.trip_id}",
            f"  route: {tracked.route}",
            f"  source stop: {tracked.source_stop_id}",
            f"  pinned expected: {v4a.clock(tracked.expected_epoch)}",
            f"  still in SOURCE stop raw events: "
            f"{'yes' if tracked.in_raw_events else 'no'}",
            f"  still in SOURCE stop bounded upcoming: "
            f"{'yes' if tracked.in_upcoming else 'no'}",
            f"  exact VehiclePosition in current snapshot: "
            f"{'yes' if records else 'no'}",
            f"  valid exact positions: {len(tracked.current)}"
            f"   invalid coordinates: {tracked.invalid_coordinates}",
            f"  last vehicle refresh failed: "
            f"{'yes' if tracked.refresh_failed else 'no'}",
            f"  state: {tracked.state}",
            f"  vehicle feed timestamp: "
            f"{v4a.clock(snapshot.feed_timestamp) if snapshot else '-'}",
            f"  age compared against: {age_reference(snapshot)[1] or '-'}",
            f"  source stop presence: {source_presence_text(tracked)}",
            f"  expected basis: {tracked.expected_basis or '-'}",
            f"  delay as V3 reports it: "
            f"{v3.format_delay(tracked.delay_seconds)}",
            f"  static shape_id: {tracked.shape_id or '-'}",
            f"  static shape points: "
            f"{len(tracked.shape_points) if tracked.shape_points else 0}"
            f"   ({tracked.shape_status})",
            f"  shapes.txt scans this session: {TripShapeStore.scans}",
        ]
        extent = self.overlay.extent()
        for observation in self.positions():
            point = week5.plots.project([observation.longitude],
                                        [observation.latitude])[0]
            inside = v4b.inside_extent((float(point[0]), float(point[1])),
                                       extent)
            lines += [
                f"  vehicle_id: {observation.vehicle_id}",
                f"  vehicle timestamp: "
                f"{v4a.clock(observation.vehicle_timestamp)}",
                f"  position age: {age_phrase(observation, snapshot)}",
                f"  raw latitude: {observation.latitude}",
                f"  raw longitude: {observation.longitude}",
                "  shown on map" if inside else "  outside displayed extent",
            ]
        return lines


def attach_tracker(window: Dict[str, Any], debug: bool = False,
                   figure: Any = None,
                   shapes: Optional[TripShapeStore] = None,
                   want_shape: bool = True) -> TripTracker:
    """Build the tracker and tie its lifetime to BOTH figures.

    Matplotlib holds a bound-method callback WEAKLY, so a tracker nothing
    refers to would be collected and its buttons would go dead while still
    looking perfectly normal - the same lifetime rule D-23 records.
    """
    tracker = TripTracker(window, debug=debug, shapes=shapes,
                          want_shape=want_shape)
    tracker.build(figure=figure)
    tracker.connect()
    window["figure"]._week7_trip_tracker = tracker
    tracker.figure._week7_trip_tracker = tracker
    return tracker


# --------------------------------------------------------------------------- #
# start-up
# --------------------------------------------------------------------------- #
def build_tracking_window(context: Dict[str, Any], stops: Any,
                          study_area: Dict[str, Any], realtime_state: Any,
                          vehicle_state: Any,
                          limit: int = v3.NEXT_SERVICES,
                          transit_fetcher: Optional[Callable[[], bytes]] = None,
                          vehicle_fetcher: Optional[Callable[[], bytes]] = None,
                          map_figure: Any = None, tracking_figure: Any = None,
                          debug: bool = False, want_shape: bool = True,
                          shapes: Optional[TripShapeStore] = None
                          ) -> Dict[str, Any]:
    """The V4C window, plus the V4D tracking window beside it."""
    window = v4c.build_window(context, stops, study_area, realtime_state,
                              vehicle_state, limit=limit,
                              transit_fetcher=transit_fetcher,
                              vehicle_fetcher=vehicle_fetcher,
                              figure=map_figure, debug=debug)
    if shapes is None and want_shape:
        shapes = shape_store_for(realtime_state)
    tracker = attach_tracker(window, debug=debug, figure=tracking_figure,
                             shapes=shapes, want_shape=want_shape)
    window["tracker"] = tracker
    window["shapes"] = shapes
    return window


def run_graphical(context: Dict[str, Any], stops: Any,
                  study_area: Dict[str, Any], realtime_state: Any,
                  vehicle_state: Any, limit: int = v3.NEXT_SERVICES,
                  transit_fetcher: Optional[Callable[[], bytes]] = None,
                  vehicle_fetcher: Optional[Callable[[], bytes]] = None,
                  debug: bool = False, want_shape: bool = True) -> int:
    import matplotlib.pyplot as plt

    window = build_tracking_window(
        context, stops, study_area, realtime_state, vehicle_state,
        limit=limit, transit_fetcher=transit_fetcher,
        vehicle_fetcher=vehicle_fetcher, debug=debug, want_shape=want_shape)
    print("")
    print(f"TWO windows are open: the Glebe map, and "
          f"'{TRACKING_WINDOW_TITLE}'.")
    print("They answer two different questions: "
          f"'{STOP_ARRIVALS_TITLE}' is about the STOP, and the tracking "
          "window is about ONE")
    print("selected trip. The stop's next service can be a different route "
          "while the pinned trip is still live.")
    print("Click an origin, then a candidate bus stop on the map. The "
          "tracking window lists that stop's")
    print("next services; click ONE to pin its exact trip_id. "
          f"'{REFRESH_LABEL}' there runs one coordinated")
    print("arrivals + vehicles refresh - nothing polls in the background.")
    print("A pinned trip is kept after it leaves the stop's upcoming list, "
          "and is re-queried by trip_id.")
    print("The WHOLE TRIP panel draws that trip's PLANNED route from the "
          "static GTFS snapshot; the vehicle")
    print("position on it is the raw GTFS-Realtime coordinate and is never "
          "snapped onto the planned line.")
    print("Close the windows to finish.")
    print(f"Tracker attached: "
          f"{window['figure']._week7_trip_tracker is window['tracker']}")
    plt.show()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 V4D.1: pick ONE upcoming service at a Glebe stop "
                    "and keep tracking its exact trip_id after it passes. "
                    "Presentation only - no polling, no prediction, no "
                    "interpolation, no Week 5 method change.")
    parser.add_argument("--next", type=int, default=v3.NEXT_SERVICES,
                        dest="next_services",
                        help=f"services listed per stop "
                             f"(default {v3.NEXT_SERVICES})")
    parser.add_argument("--no-scheduled", action="store_true",
                        help="skip the static scheduled-time join")
    parser.add_argument("--rebuild-extract", action="store_true",
                        help="rebuild the cached Glebe scheduled-time extract")
    parser.add_argument("--parse-file", default=None,
                        help="replay a saved Trip Update .pb payload")
    parser.add_argument("--parse-vehicle-file", default=None,
                        help="replay a saved Vehicle Position .pb payload")
    parser.add_argument("--no-trip-shape", action="store_true",
                        dest="no_trip_shape",
                        help="do not read the static GTFS shape for the "
                             "tracked trip, and leave the whole-trip route "
                             "map empty")
    parser.add_argument("--debug-live", action="store_true", dest="debug_live",
                        help="after each Refresh all, print the V4C "
                             "per-service diagnostic and the tracked-trip "
                             "diagnostic. Never prints the key.")
    parser.add_argument("--graph", default=week5.audit.DEFAULT_GRAPH)
    parser.add_argument("--cost-dir", default=None)
    parser.add_argument("--stops-csv", default=None)
    parser.add_argument("--boundary", default=None)
    options = parser.parse_args(argv)

    graph_path = Path(options.graph)
    if not graph_path.is_absolute():
        graph_path = (week5.CONFIG_ROOT / graph_path).resolve()
    cost_directory = (Path(options.cost_dir).resolve() if options.cost_dir
                      else week5.terrain_routing.cost_dir())
    stops_csv = (Path(options.stops_csv).resolve() if options.stops_csv
                 else week5.stops_path())

    try:
        stops = week5.load_candidate_stops(stops_csv)
        context = week5.build_context(graph_path, cost_directory)
        study_area = week5.load_study_area(
            Path(options.boundary) if options.boundary else None)
    except (week5.DemoError,
            week5.terrain_routing.TerrainRoutingError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Graph        {context['graph'].number_of_nodes()} nodes, "
          f"{context['graph'].number_of_edges()} directed edges")
    print("Week 5 routing is unchanged here. Arrivals, vehicle positions and "
          "trip tracking are presentation layers:")
    print("none of them enters a search, a cost or a route choice.")

    try:
        loaded = v3.load_static_once(want_scheduled=not options.no_scheduled,
                                     rebuild=options.rebuild_extract)
    except v2.GlebeRealtimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    transit_fetcher = (v3.file_fetcher(Path(options.parse_file))
                       if options.parse_file else None)
    vehicle_fetcher = (v4b.file_vehicle_fetcher(
        Path(options.parse_vehicle_file)) if options.parse_vehicle_file
        else None)

    realtime_state = v3.RealtimeState(stops=loaded["stops"],
                                      static=loaded["static"],
                                      scheduled=loaded["scheduled"])
    realtime_state.refresh(transit_fetcher)
    print(v3.describe_start(realtime_state))

    vehicle_state = v4b.VehicleState(static=loaded["static"])
    vehicle_state.refresh(vehicle_fetcher)
    print(v4b.describe_vehicles(vehicle_state))

    return run_graphical(context, stops, study_area, realtime_state,
                         vehicle_state, limit=options.next_services,
                         transit_fetcher=transit_fetcher,
                         vehicle_fetcher=vehicle_fetcher,
                         debug=options.debug_live,
                         want_shape=not options.no_trip_shape)


if __name__ == "__main__":
    sys.exit(main())
