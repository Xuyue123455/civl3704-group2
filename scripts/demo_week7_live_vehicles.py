"""Week 7 V4B - live vehicle markers for the SELECTED stop's services.

WHAT THIS IS
------------
A PRESENTATION layer on top of two finished layers. It opens the existing
Week 7 V3 window - the Week 5 terrain-routing map plus the LIVE TRANSIT panel -
and adds one overlay: for the services V3 is ALREADY showing at the stop the
user clicked, it draws where those vehicles currently are.

    user clicks an origin        ->  Week 5, unchanged
    user clicks a BUS stop       ->  Week 5 routes + V3 live services, unchanged
                                     PLUS a marker for each of those services
                                     that has an exact VehiclePosition

THE ONLY LINK USED
------------------
    ArrivalEvent.trip_id  ==  VehicleRecord.trip_id

Exact string equality, nothing else. No route number, no realtime route_id
parsing, no vehicle-label reading, no nearest-vehicle guess, no bearing or
time-proximity heuristic. A realtime route_id such as "2459_433" is an opaque
token here and is never presented as route 433; the route NUMBER always comes
through the adopted chain trips.txt -> route_id -> routes.txt, which V2 and
V4A already implement.

If a service has no matching VehiclePosition, nothing is drawn and the status
says so. That means only "no matching live update in this snapshot". It does
NOT mean the service is cancelled, unserved or absent.

WHAT IT REUSES, AND THEREFORE DOES NOT REDO
-------------------------------------------
``scripts/demo_week7_realtime.py``      V3: the whole window, the Week 5
                                        explorer, the LIVE TRANSIT panel, the
                                        Trip Update snapshot, the static and
                                        scheduled loading, and its refresh
``scripts/inspect_vehicle_positions.py`` V4A: the vehicle-position registry,
                                        the VehicleRecord reader, coordinate
                                        validation and the route-resolution
                                        chain
``scripts/show_glebe_live_arrivals.py``  V2: ``upcoming`` - the SAME bounded
                                        list of next services V3 displays

There is no second GTFS-Realtime decoder, no second arrival-time rule, no
second definition of "next service" and no second Dijkstra in this file.
V3 is NOT modified: ``python scripts/demo_week7_realtime.py`` still runs
exactly as before.

WHAT IT IS NOT
--------------
Not a trajectory, not tracking history, not continuous tracking and not an
animation. Each refresh replaces one snapshot with another; a marker that
appears somewhere else afterwards is an UPDATED POSITION, not a path. Nothing
is stored. Vehicle coordinates are drawn exactly where TfNSW supplied them:
they are never snapped to a road, interpolated, or moved to make a picture
work, and they never touch routing. See D-28 in docs/ASSUMPTIONS.md.

Usage
-----
    set TFNSW_API_KEY=<key>
    python scripts/demo_week7_live_vehicles.py
    python scripts/demo_week7_live_vehicles.py --next 3
    python scripts/demo_week7_live_vehicles.py --no-scheduled
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The two finished layers. Each of them loads its own copy of the V1/V2
# modules, so this file deliberately takes its DATA from V3's stack (the
# window's own Trip Update snapshot, stops, static GTFS) and takes only
# vehicle PARSING from V4A. Nothing is loaded twice from disk.
v3 = load_script("demo_week7_realtime")
v4a = load_script("inspect_vehicle_positions")

week5 = v3.week5
v2 = v3.v2

# Two module copies mean two RealtimeError classes. Both are caught, always.
REALTIME_ERRORS = (v3.RealtimeError, v4a.RealtimeError)

STATUS_NEVER = v3.STATUS_NEVER
STATUS_OK = v3.STATUS_OK
STATUS_STALE = v3.STATUS_STALE
STATUS_UNAVAILABLE = v3.STATUS_UNAVAILABLE

VEHICLE_FEED = "buses"
BUS_MODE = v2.BUS_MODE
LIGHT_RAIL_MODE = v3.LIGHT_RAIL_MODE

# ----- outcomes, per service ------------------------------------------------ #
OUTCOME_MATCHED = "matched"
OUTCOME_NO_VEHICLE = "no_vehicle"
OUTCOME_INVALID = "invalid_coordinates"

# ----- layout: V3's right column is split, nothing is rearranged ------------ #
PANEL_LEFT = v3.PANEL_RECT[0]
PANEL_WIDTH = v3.PANEL_RECT[2]
TRANSIT_RECT = (PANEL_LEFT, 0.455, PANEL_WIDTH, 0.495)
VEHICLE_RECT = (PANEL_LEFT, 0.115, PANEL_WIDTH, 0.315)
VEHICLE_BUTTON_RECT = (0.885, 0.040, 0.090, 0.045)

VEHICLE_PANEL_TITLE = "LIVE VEHICLES"
VEHICLE_REFRESH_LABEL = "Refresh vehicles"
VEHICLE_REFRESH_KEY = "v"
PANEL_WRAP = v3.PANEL_WRAP
PANEL_FONTSIZE = 8.0

# ----- marker design -------------------------------------------------------- #
# A colour that is not already carrying meaning on this map: the routes are
# blue and orange, the shared sections grey, the origin green and the selected
# stop purple.
COLOUR_VEHICLE = "#e7298a"
VEHICLE_MARKER_SIZE = 110
VEHICLE_ZORDER = 11
VEHICLE_LABEL_OFFSET = (0, 11)
LEGEND_LABEL = "live vehicle (current position)"

ABSENCE_NOTE = ("A service with no VehiclePosition has NOT been shown to be "
                "cancelled or unserved.")
SCOPE_NOTE = "Current-position snapshots, not a trajectory."
UNRESOLVED_NOTE = "A raw realtime route_id is not a route number."
LIGHT_RAIL_NOTE = (
    "Live L1 vehicle positions are available from TfNSW, but selected-service "
    "linkage is not integrated in this prototype yet, so no light-rail "
    "vehicle is tied to this stop.")


# --------------------------------------------------------------------------- #
# vehicle state - one snapshot, replaced whole, never accumulated
# --------------------------------------------------------------------------- #
@dataclass
class VehicleSnapshot:
    """One decoded Vehicle Position feed, indexed by trip_id.

    ``records_by_trip_id`` maps a trip_id to EVERY record carrying it. A list,
    not a single record, because a later feed may report two simultaneous
    positions for one trip and this layer must never pick one silently.
    """

    fetched_at: int
    feed_timestamp: Optional[int]
    observed_at: int
    records_by_trip_id: Dict[str, List[Any]] = field(default_factory=dict)
    record_count: int = 0
    valid_coordinate_count: int = 0

    def records_for(self, trip_id: Optional[str]) -> List[Any]:
        if trip_id is None:
            return []
        return list(self.records_by_trip_id.get(str(trip_id), []))


@dataclass
class VehicleState:
    """Presentation state for the vehicle layer. Not an analytical model.

    The static tables are the ones V3 already loaded; they are used only to
    resolve a route NUMBER through the adopted chain. No history is kept: a
    successful refresh replaces the snapshot outright.
    """

    static: Optional[Any] = None
    snapshot: Optional[VehicleSnapshot] = None
    status: str = STATUS_NEVER
    error: Optional[str] = None
    refreshes: int = 0
    successes: int = 0

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot is not None

    def refresh(self, fetcher: Optional[Callable[[], bytes]] = None) -> str:
        """Fetch ONE vehicle snapshot, decode it with V1, index it by trip_id.

        A failure never raises into the GUI and never destroys a snapshot that
        already succeeded.
        """
        self.refreshes += 1
        source = fetcher if fetcher is not None else fetch_vehicle_payload
        try:
            payload = source()
            message = v4a.v1.parse_feed(payload)
            records = v4a.read_vehicles(message)
            header = v4a.field_or_none(message.header, "timestamp")
        except REALTIME_ERRORS as error:
            self.error = str(error)
            self.status = (STATUS_STALE if self.has_snapshot
                           else STATUS_UNAVAILABLE)
            return self.status

        trips, routes = v4a.static_tables(self.static)
        indexed: Dict[str, List[Any]] = {}
        valid = 0
        for record in records:
            v4a.resolve_route(record, trips, routes)
            if record.coordinate_valid:
                valid += 1
            if record.trip_id is not None:
                indexed.setdefault(str(record.trip_id), []).append(record)

        now = int(time.time())
        self.snapshot = VehicleSnapshot(
            fetched_at=now, feed_timestamp=header, observed_at=now,
            records_by_trip_id=indexed, record_count=len(records),
            valid_coordinate_count=valid)
        self.error = None
        self.status = STATUS_OK
        self.successes += 1
        return STATUS_OK


def fetch_vehicle_payload(timeout: Any = None) -> bytes:
    """One bounded request for the VERIFIED bus vehicle-position feed.

    V1's conventions, unchanged: one attempt, explicit timeouts, no retry
    loop. The key is loaded, used as a header and discarded; anything raised
    is re-raised with the key redacted, so no GUI string can carry it.
    """
    definition = v4a.resolve_vehicle_feed(VEHICLE_FEED)
    api_key = v4a.v1.load_api_key()
    arguments = {} if timeout is None else {"timeout": timeout}
    try:
        return v4a.v1.fetch_feed(definition.url, api_key,
                                 **arguments).payload
    except REALTIME_ERRORS as error:
        raise v4a.RealtimeError(v4a.v1.redact(error, api_key)) from None


def file_vehicle_fetcher(path: Path) -> Callable[[], bytes]:
    """Replay a saved vehicle payload instead of the network, for a dry run."""
    target = Path(path)

    def read() -> bytes:
        if not target.exists():
            raise v4a.RealtimeError(f"{target} does not exist.")
        return target.read_bytes()

    return read


# --------------------------------------------------------------------------- #
# matching - the whole of it
# --------------------------------------------------------------------------- #
@dataclass
class ServiceVehicle:
    """One of the services V3 is showing, and what is known about its vehicle.

    ``records`` holds EVERY VehiclePosition carrying this trip_id. More than
    one is an ambiguity: all of them are kept and marked, and none is chosen.
    """

    trip_id: Optional[str]
    route: str
    outcome: str
    realtime_route_id: Optional[str] = None
    records: List[Any] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    on_map: Optional[bool] = None

    @property
    def ambiguous(self) -> bool:
        return len(self.records) > 1

    @property
    def matched(self) -> bool:
        return self.outcome == OUTCOME_MATCHED


def service_route_label(event: Any, records: Sequence[Any]) -> str:
    """The route line, from the ADOPTED chain only.

    V2 resolved it for the arrival event and V4A resolved it for the vehicle;
    both used trips.txt -> route_id -> routes.txt. If neither resolved a
    number, the raw realtime route_id is shown and labelled as not being one.
    """
    label = v2.route_label(event)
    if "unresolved" not in label:
        return label
    for record in records:
        if record.route_short_name:
            return v4a.route_label(record)
    return label


def match_services(realtime_state: Any, vehicle_state: VehicleState,
                   stop_id: Optional[str],
                   limit: int = v3.NEXT_SERVICES) -> List[ServiceVehicle]:
    """The services V3 shows for this stop, each with its vehicle, if any.

    The service list is NOT recomputed here: it is V2's ``upcoming`` applied
    to V3's own snapshot, with V3's own bound and V3's own observation time,
    so this layer can only ever show vehicles for services the LIVE TRANSIT
    panel is already displaying.
    """
    if stop_id is None or not realtime_state.has_snapshot:
        return []
    snapshot = realtime_state.snapshot
    listed, _unorderable = v2.upcoming(snapshot.events_for(stop_id),
                                       snapshot.observed_at, limit)

    matches: List[ServiceVehicle] = []
    for event in listed:
        records = (vehicle_state.snapshot.records_for(event.trip_id)
                   if vehicle_state.has_snapshot else [])
        usable = [record for record in records if record.coordinate_valid]
        if not records:
            outcome = OUTCOME_NO_VEHICLE
        elif not usable:
            # A position was supplied but its coordinates are out of range.
            # It is reported and NEVER drawn, and never clamped into range.
            outcome = OUTCOME_INVALID
        else:
            outcome = OUTCOME_MATCHED
        matches.append(ServiceVehicle(
            trip_id=event.trip_id,
            route=service_route_label(event, usable or records),
            outcome=outcome, records=usable if usable else records,
            realtime_route_id=event.realtime_route_id))
    return matches


def project_record(record: Any) -> Tuple[float, float]:
    """One vehicle coordinate in the map's own metric CRS.

    The projection is Week 5's own helper. The coordinate itself is untouched:
    nothing is snapped to a road, rounded onto a network or interpolated.
    """
    point = week5.plots.project([float(record.longitude)],
                                [float(record.latitude)])[0]
    return float(point[0]), float(point[1])


def inside_extent(point: Tuple[float, float],
                  extent: Optional[Tuple[float, float, float, float]]
                  ) -> Optional[bool]:
    """Whether a projected point falls in the CURRENT displayed extent.

    The extent is read from the axes; it is never changed to fit a vehicle.
    """
    if extent is None:
        return None
    left, right, bottom, top = extent
    return bool(left <= point[0] <= right and bottom <= point[1] <= top)


def locate_matches(matches: Sequence[ServiceVehicle],
                   extent: Optional[Tuple[float, float, float, float]] = None
                   ) -> List[ServiceVehicle]:
    """Fill in projected positions and the on-map flag. Draws nothing."""
    for match in matches:
        if not match.matched:
            continue
        match.positions = [project_record(record) for record in match.records]
        flags = [inside_extent(point, extent) for point in match.positions]
        match.on_map = (None if extent is None
                        else any(flag for flag in flags))
    return matches


# --------------------------------------------------------------------------- #
# the status text - pure, so it can be tested without a canvas
# --------------------------------------------------------------------------- #
def age_text(record: Any, observed_at: Optional[int]) -> str:
    if record.timestamp is None or observed_at is None:
        return "position age not supplied"
    return f"updated {int(observed_at) - int(record.timestamp)} s ago"


def vehicle_line(record: Any) -> str:
    """The vehicle's own identity, secondary to the route. Never the plate."""
    if record.vehicle_id:
        return f"vehicle {record.vehicle_id}"
    if record.vehicle_label:
        return f"vehicle label {record.vehicle_label}"
    return "vehicle id not supplied"


def short_route(match: ServiceVehicle) -> str:
    """The panel's route line: a resolved number, or the raw id in brackets.

    The raw realtime route_id is shown for traceability and is never dressed
    up as a route number; the panel footer says so in words whenever one
    appears.
    """
    if "unresolved" not in match.route:
        return match.route
    raw = match.realtime_route_id
    if raw is None:
        for record in match.records:
            raw = raw or record.realtime_route_id
    return f"Route unresolved ({raw})" if raw else "Route unresolved"


def match_lines(match: ServiceVehicle,
                observed_at: Optional[int]) -> List[str]:
    """One service, in at most three panel lines."""
    head = short_route(match)
    if match.outcome == OUTCOME_NO_VEHICLE:
        return [head, "  no live VehiclePosition in this snapshot"]
    if match.outcome == OUTCOME_INVALID:
        return [head,
                "  coordinates out of range - reported, not drawn, not "
                "clamped"]
    first = match.records[0]
    lines = [f"{head}  {vehicle_line(first)}"]
    where = ("outside displayed extent" if match.on_map is False
             else "shown on map" if match.on_map else "position known")
    lines.append(f"  {age_text(first, observed_at)} - {where}")
    if match.ambiguous:
        lines.append(f"  AMBIGUOUS: {len(match.records)} simultaneous "
                     "positions, all shown, none is chosen")
    return lines


def vehicle_feed_lines(vehicle_state: VehicleState) -> List[str]:
    snapshot = vehicle_state.snapshot
    lines: List[str] = []
    if vehicle_state.status == STATUS_STALE and snapshot is not None:
        lines += ["Vehicle refresh failed",
                  f"showing the previous snapshot from "
                  f"{v4a.clock(snapshot.fetched_at)}"]
        if vehicle_state.error:
            lines.append(vehicle_state.error)
        lines.append("")
    elif vehicle_state.status == STATUS_UNAVAILABLE:
        lines += ["Live vehicle positions unavailable"]
        if vehicle_state.error:
            lines.append(vehicle_state.error)
        lines += ["", "Walking analysis and live arrivals are unaffected."]
        return lines
    elif vehicle_state.status == STATUS_NEVER:
        return ["Vehicle positions not fetched yet.", "",
                "Walking analysis and live arrivals are unaffected."]
    if snapshot is not None:
        lines.append(f"Vehicle feed {v4a.clock(snapshot.feed_timestamp)} - "
                     f"{snapshot.record_count} vehicles, bus only")
    return lines


def vehicle_panel_lines(realtime_state: Any, vehicle_state: VehicleState,
                        stop_id: Optional[str],
                        matches: Sequence[ServiceVehicle],
                        mode: Optional[str] = None) -> List[str]:
    """The whole LIVE VEHICLES body for the currently selected stop."""
    if stop_id is None:
        return (["Click a candidate bus stop to see the live",
                 "vehicles for its next services.", ""]
                + vehicle_feed_lines(vehicle_state))
    if mode == LIGHT_RAIL_MODE:
        return [f"Stop {stop_id}", "", LIGHT_RAIL_NOTE]
    if mode is not None and mode != BUS_MODE:
        return [f"Stop {stop_id}", "",
                f"Mode {mode!r} has no integrated vehicle linkage in this "
                "prototype."]

    lines = [f"Stop {stop_id}", ""]
    if not vehicle_state.has_snapshot:
        return lines + vehicle_feed_lines(vehicle_state)
    if not matches:
        lines.append("No upcoming services are listed for this stop in")
        lines.append("the current Trip Update snapshot, so there is")
        lines.append("nothing to match a vehicle to.")
        return lines + [""] + vehicle_feed_lines(vehicle_state)

    observed = (vehicle_state.snapshot.observed_at
                if vehicle_state.has_snapshot else None)
    for match in matches:
        lines += match_lines(match, observed) + [""]
    if any(match.outcome == OUTCOME_NO_VEHICLE for match in matches):
        lines += [ABSENCE_NOTE]
    if any("unresolved" in match.route for match in matches):
        lines += [UNRESOLVED_NOTE]
    return lines + vehicle_feed_lines(vehicle_state) + [SCOPE_NOTE]


def vehicle_panel_text(realtime_state: Any, vehicle_state: VehicleState,
                       stop_id: Optional[str],
                       matches: Sequence[ServiceVehicle],
                       mode: Optional[str] = None) -> str:
    return "\n".join(v3.wrap(vehicle_panel_lines(
        realtime_state, vehicle_state, stop_id, matches, mode), PANEL_WRAP))


# --------------------------------------------------------------------------- #
# the overlay
# --------------------------------------------------------------------------- #
def split_right_column(panel: Any) -> None:
    """Give the LIVE TRANSIT panel the top of its column and keep the bottom.

    Only the V3 panel's own axes is moved, and only within the column V3
    already occupies. No V3 code is changed and nothing in the Week 5 layout
    is touched.
    """
    panel.axis.set_position(TRANSIT_RECT)


class LiveVehicleOverlay:
    """The vehicle layer: markers on the map, a status panel, one button.

    It owns ONLY the artists it creates. It never clears the map, never
    redraws Week 5, never touches the V3 panel's text, and never calls a
    search, a solve or a reroute.
    """

    def __init__(self, panel: Any, vehicle_state: VehicleState,
                 fetcher: Optional[Callable[[], bytes]] = None):
        self.panel = panel                      # the V3 LiveTransitPanel
        self.explorer = panel.explorer          # the Week 5 explorer
        self.state = vehicle_state
        self.realtime_state = panel.state       # V3's RealtimeState
        self.figure = panel.figure
        self.fetcher = fetcher

        self.axis = None
        self.text = None
        self.button = None
        self.button_axis = None
        self.legend = None
        self.artists: List[Any] = []
        self.selected_stop: Optional[str] = None
        self.matches: List[ServiceVehicle] = []
        self.renders = 0
        self.draws = 0

    # ----- construction -------------------------------------------------- #
    def build(self):
        from matplotlib.lines import Line2D
        from matplotlib.legend import Legend
        from matplotlib.widgets import Button

        split_right_column(self.panel)
        self.axis = self.figure.add_axes(VEHICLE_RECT)
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        for spine in self.axis.spines.values():
            spine.set_edgecolor("#bbbbbb")
        self.axis.set_title(VEHICLE_PANEL_TITLE, fontsize=11,
                            color=week5.plots.COLOUR_INK, pad=6.0)
        self.text = self.axis.text(
            0.045, 0.955, "", transform=self.axis.transAxes, ha="left",
            va="top", fontsize=PANEL_FONTSIZE, family="monospace",
            color=week5.plots.COLOUR_INK)

        self.button_axis = self.figure.add_axes(VEHICLE_BUTTON_RECT)
        self.button = Button(self.button_axis, VEHICLE_REFRESH_LABEL)
        self.button.label.set_fontsize(8)
        self.button.on_clicked(self.on_refresh)

        # A SECOND legend, added as an artist so the Week 5 legend keeps its
        # place and its handles. It is created once and never recreated.
        self.legend = Legend(
            self.explorer.map_axis,
            [Line2D([0], [0], marker="o", color="none", markersize=7,
                    markerfacecolor=COLOUR_VEHICLE,
                    markeredgecolor="#ffffff", label=LEGEND_LABEL)],
            [LEGEND_LABEL], loc="upper left",
            bbox_to_anchor=(0.012, 0.845),
            bbox_transform=self.explorer.map_axis.transAxes, fontsize=6.4,
            framealpha=0.92, handlelength=1.6, handletextpad=0.5,
            borderpad=0.4, borderaxespad=0.0)
        self.explorer.map_axis.add_artist(self.legend)
        self.render()
        return self

    def connect(self):
        """Observe clicks and keys, independently of dispatch order.

        The stop under the pointer is hit-tested with the Week 5 explorer's
        own primitive, exactly as V3 does, so this overlay does not care
        whether the Week 5 handler or the V3 handler ran first.
        """
        canvas = self.figure.canvas
        canvas.mpl_connect("button_press_event", self.on_click)
        canvas.mpl_connect("key_press_event", self.on_key)
        return self

    # ----- events --------------------------------------------------------- #
    def on_click(self, event) -> Optional[str]:
        if getattr(event, "inaxes", None) is not self.explorer.map_axis:
            return None
        if getattr(event, "button", 1) != 1:
            return None
        stop_id = self.explorer.hit_test_stop(event)
        if stop_id is None:
            return None                      # an origin click selects nothing
        self.selected_stop = str(stop_id)
        self.update()
        return self.selected_stop

    def on_key(self, event) -> Optional[str]:
        if getattr(event, "key", None) != VEHICLE_REFRESH_KEY:
            return None
        return self.on_refresh(event)

    def on_refresh(self, _event=None) -> str:
        """One vehicle request. It reroutes nothing and refetches nothing else.

        It does not touch the origin, the selected stop, the walking routes,
        the static GTFS, the scheduled extract or V3's Trip Update snapshot.
        """
        if self.text is not None:
            self.text.set_text("Requesting live vehicle positions ...")
            self.redraw()
        status = self.state.refresh(self.fetcher)
        self.update()
        return status

    # ----- drawing -------------------------------------------------------- #
    def extent(self) -> Optional[Tuple[float, float, float, float]]:
        """The CURRENT displayed extent. Read only - never widened."""
        axis = self.explorer.map_axis
        if axis is None:
            return None
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        return (float(left), float(right), float(bottom), float(top))

    def clear_artists(self) -> None:
        """Remove ONLY this overlay's markers and labels."""
        while self.artists:
            self.artists.pop().remove()

    def mode_of_selection(self) -> Optional[str]:
        if self.selected_stop is None:
            return None
        return self.realtime_state.mode_of(self.selected_stop)

    def update(self) -> List[ServiceVehicle]:
        """Recompute the matches, redraw the markers, rewrite the status."""
        mode = self.mode_of_selection()
        if mode is not None and mode != BUS_MODE:
            self.matches = []
        else:
            self.matches = locate_matches(
                match_services(self.realtime_state, self.state,
                               self.selected_stop, self.panel.limit),
                self.extent())
        self.draw_markers()
        self.render()
        self.redraw()
        return self.matches

    def draw_markers(self) -> None:
        """Replace this overlay's markers. Nothing else on the map is touched.

        A vehicle is drawn only when the service matched exactly, its
        coordinates are valid, and the point already falls inside the
        displayed extent. The extent is never changed to make one visible.
        """
        self.clear_artists()
        self.draws += 1
        axis = self.explorer.map_axis
        if axis is None:
            return
        extent = self.extent()
        for match in self.matches:
            if not match.matched:
                continue
            for record, point in zip(match.records, match.positions):
                if not inside_extent(point, extent):
                    continue
                self.artists.append(axis.scatter(
                    [point[0]], [point[1]], marker=self.marker(record),
                    s=VEHICLE_MARKER_SIZE, facecolor=COLOUR_VEHICLE,
                    edgecolor="#ffffff", linewidth=1.2,
                    zorder=VEHICLE_ZORDER))
                self.artists.append(axis.annotate(
                    self.marker_label(match), point,
                    textcoords="offset points", xytext=VEHICLE_LABEL_OFFSET,
                    ha="center", va="bottom", fontsize=8,
                    color=COLOUR_VEHICLE, zorder=VEHICLE_ZORDER + 1,
                    bbox={"facecolor": "#ffffff", "edgecolor": COLOUR_VEHICLE,
                          "boxstyle": "round,pad=0.18", "alpha": 0.9}))

    def marker(self, record: Any):
        """A triangle pointing along the reported bearing, else a circle.

        Bearing is a coarse orientation cue taken straight from the feed. It
        is not a claim about movement, and no custom icon is rotated.
        """
        if record.bearing is None:
            return "o"
        # Matplotlib rotates a marker anticlockwise from "up"; a compass
        # bearing is clockwise from north.
        return (3, 0, -float(record.bearing))

    def marker_label(self, match: ServiceVehicle) -> str:
        """The route NUMBER where one was safely resolved, never a raw id."""
        route = match.route
        if route.startswith("Route ") and "unresolved" not in route:
            label = route[len("Route "):]
        else:
            label = "route unresolved"
        return f"{label} ?" if match.ambiguous else label

    def render(self) -> str:
        body = vehicle_panel_text(self.realtime_state, self.state,
                                  self.selected_stop, self.matches,
                                  self.mode_of_selection())
        if self.text is not None:
            self.text.set_text(body)
        self.renders += 1
        return body

    def redraw(self) -> None:
        canvas = getattr(self.figure, "canvas", None)
        if canvas is not None and hasattr(canvas, "draw_idle"):
            canvas.draw_idle()


def attach_overlay(panel: Any, vehicle_state: VehicleState,
                   fetcher: Optional[Callable[[], bytes]] = None
                   ) -> LiveVehicleOverlay:
    """Build the overlay and tie its lifetime to the Figure.

    Matplotlib holds a bound-method callback WEAKLY, so an overlay nothing
    refers to would be collected and its button would go dead while still
    looking perfectly normal. This is the same lifetime rule the Week 5
    explorer and the V3 panel already document.
    """
    overlay = LiveVehicleOverlay(panel, vehicle_state, fetcher=fetcher)
    overlay.build()
    overlay.connect()
    panel.figure._week7_vehicle_overlay = overlay
    return overlay


# --------------------------------------------------------------------------- #
# start-up
# --------------------------------------------------------------------------- #
def describe_vehicles(state: VehicleState) -> str:
    if state.status == STATUS_OK and state.snapshot is not None:
        return (f"Vehicle feed updated "
                f"{v4a.clock(state.snapshot.feed_timestamp)}  "
                f"({state.snapshot.record_count} vehicles, "
                f"{state.snapshot.valid_coordinate_count} with valid "
                f"coordinates, fetched "
                f"{v4a.clock(state.snapshot.fetched_at)})")
    if state.status == STATUS_UNAVAILABLE:
        return ("Live vehicle positions unavailable at start-up. The walking "
                f"demo and the live arrivals still work. Reason: "
                f"{state.error}")
    return "Live vehicle positions have not been fetched."


def run_graphical(context: Dict[str, Any], stops: Any,
                  study_area: Dict[str, Any], realtime_state: Any,
                  vehicle_state: VehicleState,
                  limit: int = v3.NEXT_SERVICES,
                  transit_fetcher: Optional[Callable[[], bytes]] = None,
                  vehicle_fetcher: Optional[Callable[[], bytes]] = None
                  ) -> int:
    """The V3 window, with the vehicle overlay on it."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=v3.FIGURE_SIZE)
    explorer = week5.build_explorer(context, stops, study_area, figure=figure)
    panel = v3.attach_panel(explorer, realtime_state, limit=limit,
                            fetcher=transit_fetcher)
    overlay = attach_overlay(panel, vehicle_state, fetcher=vehicle_fetcher)
    print("")
    print("A map window is open. "
          + week5.GUI_INSTRUCTIONS.replace("\n", " "))
    print(f"LIVE TRANSIT follows the stop you click ({panel.limit} services). "
          f"LIVE VEHICLES shows where those services' vehicles are.")
    print(f"'{v3.REFRESH_LABEL}' re-requests the arrivals feed; "
          f"'{VEHICLE_REFRESH_LABEL}' (or '{VEHICLE_REFRESH_KEY}') "
          "re-requests only the vehicle feed.")
    print("Close the window to finish.")
    # explorer, panel and overlay all stay live locals across the event loop,
    # and each is also attached to the Figure, so none can be collected while
    # the window is open - a collected overlay would leave a visible but dead
    # button.
    print(f"Vehicle overlay attached to figure "
          f"{'yes' if figure._week7_vehicle_overlay is overlay else 'no'}.")
    plt.show()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 V4B: the Week 7 window with live vehicle markers "
                    "for the selected stop's next services. Presentation and "
                    "integration only - bus vehicle positions, exact trip_id "
                    "matching, one snapshot at a time.")
    parser.add_argument("--next", type=int, default=v3.NEXT_SERVICES,
                        dest="next_services",
                        help=f"services to consider for the selected stop "
                             f"(default {v3.NEXT_SERVICES}, "
                             f"max {v3.MAX_NEXT_SERVICES})")
    parser.add_argument("--no-scheduled", action="store_true",
                        help="skip the static scheduled-time join")
    parser.add_argument("--rebuild-extract", action="store_true",
                        help="rebuild the cached Glebe scheduled-time extract")
    parser.add_argument("--parse-file", default=None,
                        help="replay a saved Trip Update .pb payload")
    parser.add_argument("--parse-vehicle-file", default=None,
                        help="replay a saved Vehicle Position .pb payload")
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
    print("Week 5 routing is unchanged here. Live arrivals and live vehicle "
          "positions are presentation layers:")
    print("neither enters a search, a cost or a route choice.")

    try:
        loaded = v3.load_static_once(want_scheduled=not options.no_scheduled,
                                     rebuild=options.rebuild_extract)
    except v2.GlebeRealtimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    transit_fetcher = (v3.file_fetcher(Path(options.parse_file))
                       if options.parse_file else None)
    vehicle_fetcher = (file_vehicle_fetcher(Path(options.parse_vehicle_file))
                       if options.parse_vehicle_file else None)

    realtime_state = v3.RealtimeState(stops=loaded["stops"],
                                      static=loaded["static"],
                                      scheduled=loaded["scheduled"])
    realtime_state.refresh(transit_fetcher)
    print(v3.describe_start(realtime_state))

    # A vehicle failure must never stop the window: the Week 5 comparison and
    # the V3 arrivals are the parts that already work.
    vehicle_state = VehicleState(static=loaded["static"])
    vehicle_state.refresh(vehicle_fetcher)
    print(describe_vehicles(vehicle_state))

    return run_graphical(context, stops, study_area, realtime_state,
                         vehicle_state, limit=options.next_services,
                         transit_fetcher=transit_fetcher,
                         vehicle_fetcher=vehicle_fetcher)


if __name__ == "__main__":
    sys.exit(main())
