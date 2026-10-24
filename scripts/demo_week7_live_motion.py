"""Week 7 V4C - one coordinated refresh, and recent observed vehicle positions.

WHAT THIS IS
------------
A PRESENTATION refinement on top of three finished layers. It opens the same
integrated window V4B opens - the Week 5 terrain map, the V3 LIVE TRANSIT
panel and the V4B live-vehicle overlay - and changes only how the user drives
it and how much of what the feed already said is kept on screen:

1. ONE user-facing control, **Refresh all**, performing one manual Trip
   Update + Vehicle Position cycle per press;
2. up to FOUR recent observations per vehicle identity, held in memory only,
   drawn as small subordinate points behind the current marker;
3. (V4C.1) a service that a LATER SUCCESSFUL Trip Update snapshot no longer
   carries is retained, visibly, as LAST SEEN - never silently as live;
4. (V4C.1) a successful vehicle request is classified against the feed's OWN
   header timestamp, so a repeated or older snapshot is never reported as new
   movement, and an older one never replaces the newer one already held.

THE OMISSION BRIDGE, AND WHY IT EXISTS
--------------------------------------
The real feed showed a service that had been displayed with a future expected
arrival vanish from the next successful Trip Update snapshot BEFORE that time
arrived, while the vehicle feed's own timestamp was minutes older. The two
feeds are not synchronised, and an absence is not a cancellation.

So a previously DISPLAYED service that is absent from EVERY event in the new
snapshot for that stop is kept on screen under an explicit
LAST SEEN - CURRENT TRIP UPDATE MISSING state, hollow and labelled, until its
OWN last reported effective arrival passes. There is no grace period: the only
clock is the service's own last known arrival time. Falling out of the bounded
top-N is not an omission, a failed request is not an omission, and a service
with no defensible effective arrival is never bridged. V3's LIVE TRANSIT panel
is untouched and keeps saying exactly what the current snapshot holds.

WHAT THE SMALL POINTS ARE, AND ARE NOT
--------------------------------------
Each refresh REQUESTS the realtime feeds; the status reports whether the
Vehicle Position snapshot actually advanced. The current marker is the latest
reported vehicle position, while the smaller points show PREVIOUS OBSERVED
POSITIONS. **No path is interpolated between them.**

They are therefore never called a trajectory, a continuous track, a GPS path
or an interpolated path: successive VehiclePosition snapshots are separate
reports, and what the vehicle did between two of them was not observed. For
the same reason the points are NEVER joined by a line - a line would draw a
route that nobody measured.

WHAT IT REUSES, AND THEREFORE DOES NOT REDO
-------------------------------------------
``scripts/demo_week7_live_vehicles.py``  V4B: the vehicle overlay, its exact
                                         ``trip_id`` matching, its current
                                         marker, its ``VehicleState.refresh``
``scripts/demo_week7_realtime.py``       V3: the window, the LIVE TRANSIT
                                         panel and ``RealtimeState.refresh``
``scripts/inspect_vehicle_positions.py`` V4A: the vehicle-position reader
``scripts/show_glebe_live_arrivals.py``  V2: the bounded service list

None of those files is modified. ``python scripts/demo_week7_live_vehicles.py``
and ``python scripts/demo_week7_realtime.py`` still run exactly as before, with
their own refresh buttons.

WHAT IT DOES NOT DO
-------------------
No automatic polling, no timer, no thread, no animation loop, no interpolation,
no road snapping, no route-shape fitting, no infinite history, nothing written
to disk. Vehicle coordinates are drawn exactly where TfNSW supplied them and
never reach a search, a cost, a route choice or a stop ranking. See D-29 in
docs/ASSUMPTIONS.md.

Usage
-----
    set TFNSW_API_KEY=<key>
    python scripts/demo_week7_live_motion.py
    python scripts/demo_week7_live_motion.py --next 3
    python scripts/demo_week7_live_motion.py --debug-live
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


# ONE import chain: V4B already carries V3, V4A, V2, V1 and the Week 5
# explorer, so nothing is loaded twice and there is only one of each class.
v4b = load_script("demo_week7_live_vehicles")
v3 = v4b.v3
v4a = v4b.v4a
v2 = v4b.v2
week5 = v4b.week5

STATUS_OK = v4b.STATUS_OK
STATUS_STALE = v4b.STATUS_STALE
STATUS_UNAVAILABLE = v4b.STATUS_UNAVAILABLE

# ----- history rules -------------------------------------------------------- #
MAX_OBSERVATIONS = 4          # per (trip_id, vehicle_id), in memory only

# ----- layout: one control, where V4B's vehicle button used to sit ---------- #
PANEL_LEFT = v3.PANEL_RECT[0]
REFRESH_ALL_RECT = (0.810, 0.040, 0.115, 0.045)
STATUS_XY = (PANEL_LEFT, 0.094)
EXPLANATION_XY = (PANEL_LEFT, 0.006)

REFRESH_ALL_LABEL = "Refresh all"
HIDDEN_HINT_PREFIX = "or press"

EXPLANATION = (
    "Each refresh requests the realtime feeds; the status reports whether the "
    "Vehicle Position snapshot\nactually advanced. The current marker is the "
    "latest reported vehicle position; the smaller points\nare previous "
    "observed positions. No path is interpolated.")

# ----- the subordinate points ----------------------------------------------- #
COLOUR_VEHICLE = v4b.COLOUR_VEHICLE
HISTORY_ZORDER = v4b.VEHICLE_ZORDER - 1
# oldest -> newest. Three sizes, because four retained observations leave at
# most three PREVIOUS ones once the newest is the current marker.
HISTORY_SIZES = (9.0, 14.0, 20.0)
HISTORY_ALPHAS = (0.28, 0.42, 0.58)
HISTORY_LEGEND_LABEL = "previous observed positions"

# ----- the omission bridge (V4C.1) ------------------------------------------ #
# A service V3 was displaying can be absent from the NEXT successful Trip
# Update snapshot while its own last reported expected arrival is still in the
# future. That is a feed observation, NOT a cancellation, so the service is
# retained under an explicit, visibly different state instead of vanishing.
STATE_CURRENT = "CURRENT"
STATE_LAST_SEEN = "LAST_SEEN_CURRENT_UPDATE_MISSING"

BRIDGE_MARKER_SIZE = 95
BRIDGE_ALPHA = 0.75
BRIDGE_LAST_POSITION_SIZE = 55
BRIDGE_LAST_POSITION_ALPHA = 0.45
BRIDGE_LABEL_SUFFIX = " last seen"
BRIDGE_LEGEND_LABEL = "last seen service (current Trip Update missing)"
# Map axes fraction: the empty right-middle band, clear of the instruction
# box, both legends, the scale bar and the divergence inset.
BRIDGE_TEXT_XY = (0.985, 0.280)

# ----- vehicle-snapshot advancement (V4C.1) --------------------------------- #
# A successful HTTP request is not the same thing as a NEWER snapshot. The real
# bus feed normally advances (a 00:59:30 snapshot followed by a 01:00:25 one,
# with a vehicle whose coordinates had changed), but a repeat or even an older
# header timestamp is possible, and neither may be reported as new movement.
SNAPSHOT_INITIAL = "initial"       # nothing to compare against
SNAPSHOT_NEW = "new"               # the feed advanced
SNAPSHOT_SAME = "same"             # the same feed snapshot came back
SNAPSHOT_OLDER = "older"           # an EARLIER feed snapshot came back
SNAPSHOT_UNKNOWN = "unknown"       # a header carried no timestamp to compare

# ----- label de-collision (annotation offsets only) ------------------------- #
LABEL_COLLISION_PX = 42.0
LABEL_OFFSETS = ((0, 11), (0, -13), (16, 7), (16, -11))
# The Week 5 instruction box sits in the map's top-left corner. A label whose
# anchor is under it is pushed below its marker instead.
INSTRUCTION_BOX_AXES = (0.0, 0.55, 0.82, 1.0)     # x0, x1, y0, y1 in axes


# --------------------------------------------------------------------------- #
# one observation, and the rolling in-memory buffer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservedPosition:
    """One VehiclePosition, as reported by ONE snapshot.

    It is a report, not a sample of a path: nothing here says where the
    vehicle was between this observation and the next one.
    """

    trip_id: str
    vehicle_id: str
    latitude: float
    longitude: float
    vehicle_timestamp: Optional[int]
    observed_at: int
    route_short_name: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str]:
        return (self.trip_id, self.vehicle_id)

    def same_report_as(self, other: "ObservedPosition") -> bool:
        """Whether two entries are the SAME report rather than two reports.

        Same vehicle timestamp AND same coordinates means the feed repeated
        itself, so pressing Refresh all again must not add a second dot. A NEW
        timestamp at the same coordinates IS a new report - TfNSW has said the
        vehicle is still there - and is kept. Movement is never inferred from
        a timestamp alone.

        When neither report carries a timestamp there is nothing to tell them
        apart, so identical coordinates are treated as one report.
        """
        return (self.vehicle_timestamp == other.vehicle_timestamp
                and self.latitude == other.latitude
                and self.longitude == other.longitude)


@dataclass
class ObservationHistory:
    """At most ``limit`` recent observations per (trip_id, vehicle_id).

    THE KEY IS THE PAIR, never the vehicle_id alone: the same physical bus may
    later work a different trip, and two different trips must never be joined
    into one history.

    A record with NO vehicle_id is deliberately not kept. Without a stable
    vehicle identity, two snapshots cannot be attributed to the same vehicle,
    and substituting ``vehicle_label`` would invent a key whose stability this
    project has not verified. Such a vehicle still gets its current V4B
    marker; it simply accumulates no previous observations.

    Everything here lives in memory for the life of the window. Nothing is
    written to disk and no cache is created.
    """

    limit: int = MAX_OBSERVATIONS
    entries: Dict[Tuple[str, str], List[ObservedPosition]] = field(
        default_factory=dict)
    skipped_without_vehicle_id: int = 0
    appended: int = 0

    def clear(self) -> None:
        self.entries.clear()
        self.skipped_without_vehicle_id = 0

    def observation_from(self, match: Any, record: Any,
                         observed_at: int) -> Optional[ObservedPosition]:
        """A recordable observation, or None with the reason left implicit."""
        if match.trip_id is None or not record.vehicle_id:
            return None
        if not record.coordinate_valid:
            return None
        return ObservedPosition(
            trip_id=str(match.trip_id), vehicle_id=str(record.vehicle_id),
            latitude=float(record.latitude), longitude=float(record.longitude),
            vehicle_timestamp=record.timestamp, observed_at=int(observed_at),
            route_short_name=record.route_short_name)

    def append(self, observation: ObservedPosition) -> bool:
        """Append unless the newest entry is the same report. Rolls at limit."""
        bucket = self.entries.setdefault(observation.key, [])
        if bucket and bucket[-1].same_report_as(observation):
            return False
        bucket.append(observation)
        while len(bucket) > self.limit:
            bucket.pop(0)
        self.appended += 1
        return True

    def record(self, matches: Sequence[Any], observed_at: int) -> int:
        """Record every currently matched, valid, identified vehicle."""
        added = 0
        self.skipped_without_vehicle_id = 0
        for match in matches:
            if not match.matched:
                continue
            for record in match.records:
                observation = self.observation_from(match, record, observed_at)
                if observation is None:
                    if record.coordinate_valid and not record.vehicle_id:
                        self.skipped_without_vehicle_id += 1
                    continue
                added += int(self.append(observation))
        return added

    def keep_only(self, trip_ids: Sequence[Optional[str]]) -> int:
        """Drop the history of any trip that is no longer relevant.

        The history exists to explain the services CURRENTLY shown for the
        selected stop. A trip that has dropped out of V3's upcoming list is no
        longer one of them, so its points are removed rather than left on the
        map as orphans.
        """
        wanted = {str(trip_id) for trip_id in trip_ids
                  if trip_id is not None}
        stale = [key for key in self.entries if key[0] not in wanted]
        for key in stale:
            del self.entries[key]
        return len(stale)

    def previous(self, match: Any, record: Any) -> List[ObservedPosition]:
        """Every retained observation EXCEPT the one drawn as current.

        The newest retained entry is normally the very report V4B is drawing
        as the current marker, so it is excluded here and can never appear
        twice.
        """
        if match.trip_id is None or not record.vehicle_id:
            return []
        bucket = self.entries.get((str(match.trip_id), str(record.vehicle_id)),
                                  [])
        if not bucket:
            return []
        current = self.observation_from(match, record, 0)
        if current is None:
            return list(bucket)
        return [entry for entry in bucket
                if not entry.same_report_as(current)]

    def total(self) -> int:
        return sum(len(bucket) for bucket in self.entries.values())

    def newest(self, trip_id: Optional[str]) -> Optional[ObservedPosition]:
        """The most recent retained observation of a trip, whatever vehicle.

        Used only to show a LAST REPORTED position for a bridged service that
        has no current VehiclePosition. It is never presented as current.
        """
        candidates = [bucket[-1] for key, bucket in self.entries.items()
                      if trip_id is not None and key[0] == str(trip_id)
                      and bucket]
        if not candidates:
            return None
        return max(candidates, key=lambda entry: (entry.observed_at,
                                                  entry.vehicle_timestamp or 0))


# --------------------------------------------------------------------------- #
# the omission bridge - a previously DISPLAYED service that a later successful
# snapshot did not carry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LastSeenService:
    """What was displayed for this stop, the last time the feed carried it.

    ``last_effective_arrival`` is V2's own ordering/display time for that
    arrival event - ``ArrivalEvent.sort_epoch``, the very value ``upcoming``
    ordered by and the LIVE TRANSIT panel printed as "Expected". No second
    arrival-time rule is computed here.
    """

    stop_id: str
    trip_id: str
    route: str
    last_effective_arrival: int
    last_effective_basis: Optional[str]
    last_trip_update_feed_timestamp: Optional[int]
    last_trip_update_observed_at: int
    realtime_route_id: Optional[str] = None

    @property
    def short_route(self) -> str:
        """The compact route line: a resolved number, or the raw id in
        brackets - never the raw id dressed up as a route number."""
        if "unresolved" not in self.route:
            return self.route
        if self.realtime_route_id:
            return f"Route unresolved ({self.realtime_route_id})"
        return "Route unresolved"

    @property
    def key(self) -> Tuple[str, str]:
        return (self.stop_id, self.trip_id)

    def expired(self, observed_at: int) -> bool:
        """Whether the service's OWN last reported arrival time has passed.

        There is no grace period here, and no invented number of seconds,
        minutes or refresh cycles: the only clock this bridge answers to is
        the service's last known effective arrival.
        """
        return int(observed_at) > int(self.last_effective_arrival)


def classify_vehicle_snapshot(previous: Any, received: Any) -> str:
    """Compare two VehicleSnapshots by the feed's OWN header timestamp.

    This is the difference between "the request worked" and "the feed moved
    on". It invents no freshness threshold: it only reports which of the two
    header timestamps is later, and says so plainly when there is nothing to
    compare.
    """
    if previous is None:
        return SNAPSHOT_INITIAL
    before, after = previous.feed_timestamp, getattr(received,
                                                     "feed_timestamp", None)
    if before is None or after is None:
        return SNAPSHOT_UNKNOWN
    if int(after) > int(before):
        return SNAPSHOT_NEW
    if int(after) == int(before):
        return SNAPSHOT_SAME
    return SNAPSHOT_OLDER


def effective_arrival(event: Any) -> Optional[int]:
    """V2's canonical effective time for an arrival event.

    ``sort_epoch`` is what V2's ``upcoming`` ordered by and what V3 prints as
    "Expected": an absolute realtime time where the feed supplied one, and V2's
    own safely derived time otherwise. An event without one has no defensible
    effective arrival at all - and is therefore never bridged.
    """
    return getattr(event, "sort_epoch", None)


def displayed_services(realtime_state: Any, stop_id: Optional[str],
                       limit: int) -> List[Any]:
    """The services V3 is displaying for this stop, from V2's own function.

    This is a call into ``v2.upcoming`` with V3's own snapshot, bound and
    observation time - the same call V4B makes. There is no second ordering,
    filtering or arrival-time rule in this file.
    """
    if stop_id is None or not realtime_state.has_snapshot:
        return []
    snapshot = realtime_state.snapshot
    listed, _unorderable = v2.upcoming(snapshot.events_for(stop_id),
                                       snapshot.observed_at, limit)
    return list(listed)


def all_stop_trip_ids(realtime_state: Any,
                      stop_id: Optional[str]) -> set:
    """EVERY trip in the current snapshot for this stop, listed or not.

    A trip that is still here but has merely fallen outside the bounded top-N
    has not been omitted by the feed, so it must never be bridged.
    """
    if stop_id is None or not realtime_state.has_snapshot:
        return set()
    return {str(event.trip_id)
            for event in realtime_state.snapshot.events_for(stop_id)
            if event.trip_id is not None}


# --------------------------------------------------------------------------- #
# the controller
# --------------------------------------------------------------------------- #
def hide_button(button: Any) -> None:
    """Hide a lower layer's refresh control in THIS figure only.

    The Button object and its handler are untouched, and the script that owns
    it still shows it normally when it is run on its own.
    """
    axis = getattr(button, "ax", None)
    if axis is not None:
        axis.set_visible(False)


def hide_hint_texts(figure: Any, prefix: str = HIDDEN_HINT_PREFIX) -> int:
    """Hide the lower layers' 'or press ...' hints on the V4C screen.

    V4C binds no single-feed shortcut, so those hints would describe controls
    this window does not offer.
    """
    hidden = 0
    for text in list(figure.texts):
        if text.get_text().strip().startswith(prefix):
            text.set_visible(False)
            hidden += 1
    return hidden


class LiveMotionController:
    """One coordinated refresh, plus the recent-observation presentation.

    It owns only its own artists, its own button and its own status text. It
    never reroutes, never moves a vehicle, never widens the map and never
    edits what V3 or V4B drew.
    """

    def __init__(self, panel: Any, overlay: Any,
                 history: Optional[ObservationHistory] = None):
        self.panel = panel                       # V3 LiveTransitPanel
        self.overlay = overlay                   # V4B LiveVehicleOverlay
        self.explorer = overlay.explorer         # Week 5 explorer
        self.figure = overlay.figure
        self.history = history or ObservationHistory()

        self.button = None
        self.button_axis = None
        self.status_text = None
        self.bridge_text = None
        self.legend = None
        self.artists: List[Any] = []
        self.bridge_artists: List[Any] = []
        self.selected_stop: Optional[str] = None
        self.refresh_cycles = 0
        self.last_transit_status: Optional[str] = None
        self.last_vehicle_status: Optional[str] = None
        self.last_vehicle_change: Optional[str] = None
        self.previous_vehicle_feed_timestamp: Optional[int] = None
        self.received_vehicle_feed_timestamp: Optional[int] = None
        self.draws = 0

        # (stop_id, trip_id) -> LastSeenService, in memory only.
        self.bridges: Dict[Tuple[str, str], LastSeenService] = {}
        self.debug = False
        self.debug_output: List[str] = []

    # ----- construction -------------------------------------------------- #
    def build(self):
        from matplotlib.lines import Line2D
        from matplotlib.legend import Legend
        from matplotlib.widgets import Button

        hide_button(self.panel.button)
        hide_button(self.overlay.button)
        hide_hint_texts(self.figure)

        self.button_axis = self.figure.add_axes(REFRESH_ALL_RECT)
        self.button = Button(self.button_axis, REFRESH_ALL_LABEL)
        self.button.label.set_fontsize(9)
        self.button.on_clicked(self.on_refresh_all)

        self.status_text = self.figure.text(
            STATUS_XY[0], STATUS_XY[1], "", ha="left", va="bottom",
            fontsize=7.5, color=week5.plots.COLOUR_MUTED)
        self.figure.text(EXPLANATION_XY[0], EXPLANATION_XY[1], EXPLANATION,
                         ha="left", va="bottom", fontsize=6.4,
                         color=week5.plots.COLOUR_MUTED)

        # One legend for the vehicle layer, replacing V4B's single-entry one
        # in THIS figure so the two point kinds are explained together.
        if self.overlay.legend is not None:
            self.overlay.legend.remove()
            self.overlay.legend = None
        handles = [
            Line2D([0], [0], marker="o", color="none", markersize=7,
                   markerfacecolor=COLOUR_VEHICLE, markeredgecolor="#ffffff",
                   label=v4b.LEGEND_LABEL),
            Line2D([0], [0], marker="o", color="none", markersize=4,
                   markerfacecolor=COLOUR_VEHICLE, markeredgecolor="none",
                   alpha=0.5, label=HISTORY_LEGEND_LABEL),
            Line2D([0], [0], marker="o", color="none", markersize=7,
                   markerfacecolor="none", markeredgecolor=COLOUR_VEHICLE,
                   markeredgewidth=1.4, label=BRIDGE_LEGEND_LABEL),
        ]
        self.legend = Legend(
            self.explorer.map_axis, handles,
            [handle.get_label() for handle in handles], loc="upper left",
            bbox_to_anchor=(0.012, 0.845),
            bbox_transform=self.explorer.map_axis.transAxes, fontsize=6.4,
            framealpha=0.92, handlelength=1.6, handletextpad=0.5,
            borderpad=0.4, borderaxespad=0.0)
        self.explorer.map_axis.add_artist(self.legend)

        # The bridge block lives in the map's empty bottom-right corner, next
        # to the markers it explains. It is empty - and invisible - whenever
        # no service is being retained.
        self.bridge_text = self.explorer.map_axis.text(
            BRIDGE_TEXT_XY[0], BRIDGE_TEXT_XY[1], "",
            transform=self.explorer.map_axis.transAxes, ha="right",
            va="bottom", fontsize=7.0, family="monospace",
            color=COLOUR_VEHICLE, visible=False, zorder=12,
            bbox={"facecolor": "#ffffff", "edgecolor": COLOUR_VEHICLE,
                  "boxstyle": "round,pad=0.35", "alpha": 0.93})

        self.render_status()
        return self

    def connect(self):
        """ONE click handler for the whole V4C screen, in a fixed order.

        V4C composes the lower layers itself and drives their handlers in a
        deterministic sequence, so nothing here depends on the order in which
        Matplotlib happens to dispatch callbacks. The stop under the pointer
        is hit-tested with the Week 5 explorer's own primitive.
        """
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        return self

    # ----- events --------------------------------------------------------- #
    def on_click(self, event) -> Optional[str]:
        if getattr(event, "inaxes", None) is not self.explorer.map_axis:
            return None
        if getattr(event, "button", 1) != 1:
            return None
        self.panel.on_click(event)
        stop_id = self.explorer.hit_test_stop(event)
        if stop_id is None:
            return None                      # an origin click fabricates no
        stop_id = str(stop_id)               # history and selects no vehicle
        if stop_id != self.selected_stop:
            # A different stop is a different question. The history explains
            # the services shown for the stop in front of the user, so it
            # starts empty again rather than accumulating a city-wide record.
            self.history.clear()
            self.bridges.clear()
            self.clear_artists()
            self.clear_bridge_artists()
            self.selected_stop = stop_id
        self.overlay.on_click(event)
        self.after_update(record=True)
        return stop_id

    def on_refresh_all(self, _event=None) -> Tuple[str, str]:
        """ONE manual cycle: Trip Updates, then Vehicle Positions, then draw.

        Exactly one request of each per press. Nothing repeats afterwards:
        there is no timer, no thread and no loop anywhere in this file.

        It reroutes nothing, changes neither the origin nor the selected stop,
        and touches neither the static GTFS nor the cached scheduled extract.
        """
        self.refresh_cycles += 1
        if self.status_text is not None:
            self.status_text.set_text("Refreshing arrivals and vehicles ...")
            self.redraw()

        # What V3 is displaying RIGHT NOW, captured before the snapshot it
        # came from is replaced. Only these services can ever be bridged.
        previously_displayed = self.capture_displayed()

        transit_status = self.panel.state.refresh(self.panel.fetcher)
        previous_vehicles = self.overlay.state.snapshot
        vehicle_status = self.overlay.state.refresh(self.overlay.fetcher)
        self.last_transit_status = transit_status
        self.last_vehicle_status = vehicle_status
        self.classify_vehicles(previous_vehicles, vehicle_status)

        self.panel.render()
        self.overlay.update()
        self.update_bridges(previously_displayed, transit_status)
        # An observation is recorded only from a snapshot that SUCCEEDED. A
        # failed vehicle refresh leaves the previous snapshot in place and
        # invents nothing.
        self.after_update(record=vehicle_status == STATUS_OK)
        if self.debug:
            self.debug_output = self.debug_lines()
            print("\n".join(self.debug_output))
        return transit_status, vehicle_status

    def classify_vehicles(self, previous: Any, vehicle_status: str) -> str:
        """Say whether the vehicle feed actually advanced, and keep the newer.

        A snapshot whose header timestamp is EARLIER than the one already held
        is not an update: the newer snapshot is put back, so nothing on screen
        goes backwards and no movement is fabricated. This rearranges only
        V4C's own view of V4B's state field; no V4B code is changed and its
        own script keeps its own behaviour.
        """
        self.previous_vehicle_feed_timestamp = (
            previous.feed_timestamp if previous is not None else None)
        if vehicle_status != STATUS_OK:
            self.last_vehicle_change = None
            self.received_vehicle_feed_timestamp = None
            return self.last_vehicle_change
        received = self.overlay.state.snapshot
        self.received_vehicle_feed_timestamp = (
            received.feed_timestamp if received is not None else None)
        self.last_vehicle_change = classify_vehicle_snapshot(previous,
                                                             received)
        if self.last_vehicle_change == SNAPSHOT_OLDER and previous is not None:
            self.overlay.state.snapshot = previous
        return self.last_vehicle_change

    # ----- the omission bridge -------------------------------------------- #
    def capture_displayed(self) -> Dict[str, LastSeenService]:
        """The services V3 is displaying for the selected stop, right now."""
        state = self.panel.state
        if self.selected_stop is None or not state.has_snapshot:
            return {}
        snapshot = state.snapshot
        captured: Dict[str, LastSeenService] = {}
        for event in displayed_services(state, self.selected_stop,
                                        self.panel.limit):
            arrival = effective_arrival(event)
            if event.trip_id is None or arrival is None:
                continue
            captured[str(event.trip_id)] = LastSeenService(
                stop_id=str(self.selected_stop), trip_id=str(event.trip_id),
                route=v2.route_label(event),
                last_effective_arrival=int(arrival),
                last_effective_basis=event.sort_basis,
                last_trip_update_feed_timestamp=snapshot.feed_timestamp,
                last_trip_update_observed_at=int(snapshot.observed_at),
                realtime_route_id=event.realtime_route_id)
        return captured

    def update_bridges(self, previously_displayed: Dict[str, LastSeenService],
                       transit_status: str) -> Dict[Tuple[str, str],
                                                    LastSeenService]:
        """Create, keep, or drop last-seen entries after a Trip Update refresh.

        A bridge is created ONLY when a service that was being displayed is
        absent from EVERY event the new successful snapshot carries for this
        stop, and only while its own last reported effective arrival has not
        passed. Falling out of the bounded top-N is not an omission. A failed
        request is not an omission either: V3 keeps its previous snapshot, and
        nothing here reinterprets a network error as a missing service.
        """
        if transit_status != STATUS_OK:
            return self.bridges
        state = self.panel.state
        if self.selected_stop is None or not state.has_snapshot:
            return self.bridges
        observed_at = int(state.snapshot.observed_at)
        present = all_stop_trip_ids(state, self.selected_stop)

        for trip_id, entry in previously_displayed.items():
            if trip_id in present:
                continue                       # still carried: nothing to do
            if entry.expired(observed_at):
                continue                       # its own arrival time has gone
            self.bridges[entry.key] = entry

        for key in list(self.bridges):
            entry = self.bridges[key]
            if entry.stop_id != str(self.selected_stop):
                del self.bridges[key]          # belongs to another question
            elif entry.trip_id in present:
                del self.bridges[key]          # reappeared: back to LIVE
            elif entry.expired(observed_at):
                del self.bridges[key]          # its arrival time has passed
        return self.bridges

    def bridged_services(self) -> List[LastSeenService]:
        """The retained services, in a deterministic order."""
        return [self.bridges[key] for key in sorted(self.bridges)]

    def bridge_vehicles(self, entry: LastSeenService
                        ) -> Tuple[List[Any], Optional[ObservedPosition]]:
        """EVERY current valid position for the bridged trip, else the last one.

        The key stays exactly ``trip_id`` - the same one V4B uses - and every
        record carrying it is returned. More than one is an AMBIGUITY: all of
        them are kept and none is chosen, exactly as V4B already refuses to
        choose. Nothing here matches on a route number, a label, a bearing, a
        time or a distance.
        """
        state = self.overlay.state
        records = (state.snapshot.records_for(entry.trip_id)
                   if state.has_snapshot else [])
        usable = [record for record in records if record.coordinate_valid]
        if usable:
            return usable, None
        return [], self.history.newest(entry.trip_id)

    # ----- presentation --------------------------------------------------- #
    def after_update(self, record: bool = True) -> int:
        """Prune, record, redraw the small points, and rewrite the status."""
        # A bridged service keeps its history while its own arrival time has
        # not passed; everything else is pruned to what V3 currently shows.
        self.history.keep_only(
            [match.trip_id for match in self.overlay.matches]
            + [entry.trip_id for entry in self.bridges.values()])
        added = 0
        if record and self.overlay.state.has_snapshot:
            added = self.history.record(
                self.overlay.matches,
                self.overlay.state.snapshot.observed_at)
        self.draw_history()
        self.draw_bridges()
        self.adjust_labels()
        self.render_status()
        self.redraw()
        return added

    def clear_artists(self) -> None:
        """Remove ONLY this controller's small points."""
        while self.artists:
            self.artists.pop().remove()

    def clear_bridge_artists(self) -> None:
        """Remove ONLY this controller's last-seen markers."""
        while self.bridge_artists:
            self.bridge_artists.pop().remove()
        if self.bridge_text is not None:
            self.bridge_text.set_text("")
            self.bridge_text.set_visible(False)

    def draw_history(self) -> None:
        """Draw the previous observed positions, and nothing else.

        Small, subordinate, unlabelled, and NEVER joined by a line: a line
        would assert a travelled path that was not observed. A point outside
        the current displayed extent is not drawn, and the extent is not
        changed to bring it in.
        """
        self.clear_artists()
        self.draws += 1
        axis = self.explorer.map_axis
        if axis is None:
            return
        extent = self.overlay.extent()
        for match in self.overlay.matches:
            if not match.matched:
                continue
            for record in match.records:
                previous = self.history.previous(match, record)
                if not previous:
                    continue
                # oldest first, so the newest previous point is the strongest
                start = max(0, len(HISTORY_SIZES) - len(previous))
                for index, observation in enumerate(previous[-len(
                        HISTORY_SIZES):]):
                    point = week5.plots.project([observation.longitude],
                                                [observation.latitude])[0]
                    position = (float(point[0]), float(point[1]))
                    if not v4b.inside_extent(position, extent):
                        continue
                    step = min(start + index, len(HISTORY_SIZES) - 1)
                    self.artists.append(axis.scatter(
                        [position[0]], [position[1]], marker="o",
                        s=HISTORY_SIZES[step], facecolor=COLOUR_VEHICLE,
                        edgecolor="none", alpha=HISTORY_ALPHAS[step],
                        zorder=HISTORY_ZORDER))

    def draw_bridges(self) -> None:
        """Draw the retained services, visibly DIFFERENT from a live one.

        A bridged marker is hollow and faded, and its label says "last seen",
        so it can never be read as a service the current snapshot confirmed.
        The coordinate is whatever the vehicle feed supplied - nothing is
        moved, snapped or interpolated - and a point outside the displayed
        extent is not drawn.
        """
        self.clear_bridge_artists()
        axis = self.explorer.map_axis
        if axis is None or not self.bridges:
            return
        extent = self.overlay.extent()
        for entry in self.bridged_services():
            records, observation = self.bridge_vehicles(entry)
            if records:
                # EVERY exact match is drawn. Two simultaneous positions for
                # one trip_id is an ambiguity, and none of them is chosen.
                points = [(v4b.project_record(record),
                           BRIDGE_MARKER_SIZE, BRIDGE_ALPHA)
                          for record in records]
            elif observation is not None:
                point = week5.plots.project([observation.longitude],
                                            [observation.latitude])[0]
                points = [((float(point[0]), float(point[1])),
                           BRIDGE_LAST_POSITION_SIZE,
                           BRIDGE_LAST_POSITION_ALPHA)]
            else:
                continue                       # text only: nothing to draw
            for point, size, alpha in points:
                if not v4b.inside_extent(point, extent):
                    continue
                self.bridge_artists.append(axis.scatter(
                    [point[0]], [point[1]], marker="o", s=size,
                    facecolor="none", edgecolor=COLOUR_VEHICLE,
                    linewidth=1.6, alpha=alpha, zorder=v4b.VEHICLE_ZORDER))
                self.bridge_artists.append(axis.annotate(
                    self.bridge_label(entry, ambiguous=len(records) > 1),
                    point, textcoords="offset points",
                    xytext=(0, -13), ha="center", va="top", fontsize=7.5,
                    color=COLOUR_VEHICLE, alpha=alpha,
                    zorder=v4b.VEHICLE_ZORDER + 1,
                    bbox={"facecolor": "#ffffff", "edgecolor": COLOUR_VEHICLE,
                          "boxstyle": "round,pad=0.18", "alpha": 0.85,
                          "linestyle": "--"}))
        lines = self.bridge_lines()
        if lines and self.bridge_text is not None:
            self.bridge_text.set_text("\n".join(lines))
            self.bridge_text.set_visible(True)

    def bridge_label(self, entry: LastSeenService,
                     ambiguous: bool = False) -> str:
        """The map label for a retained service - never a bare route number."""
        route = entry.route
        number = (route[len("Route "):] if route.startswith("Route ")
                  and "unresolved" not in route else "route unresolved")
        label = f"{number}{BRIDGE_LABEL_SUFFIX}"
        return f"{label} ?" if ambiguous else label

    BRIDGE_BLOCK_LIMIT = 1

    def bridge_lines(self) -> List[str]:
        """The block that says, in words, why a service is still on screen."""
        if not self.bridges:
            return []
        retained = self.bridged_services()
        lines = ["LAST SEEN - CURRENT TRIP UPDATE MISSING"]
        for entry in retained[:self.BRIDGE_BLOCK_LIMIT]:
            records, observation = self.bridge_vehicles(entry)
            lines.append(f"{entry.short_route} - last expected "
                         f"{v4a.clock(entry.last_effective_arrival)}")
            if len(records) > 1:
                lines.append(f"  AMBIGUOUS: {len(records)} exact positions, "
                             "all shown, none chosen")
            elif records:
                record = records[0]
                age = (self.overlay.state.snapshot.observed_at
                       - int(record.timestamp)
                       if record.timestamp is not None
                       and self.overlay.state.has_snapshot else None)
                lines.append("  vehicle position reported "
                             + (f"{age} s ago" if age is not None
                                else "at an unstated time"))
            elif observation is not None:
                lines.append("  last reported vehicle position "
                             f"{v4a.clock(observation.vehicle_timestamp)}")
            else:
                lines.append("  no vehicle position for this trip")
        if len(retained) > self.BRIDGE_BLOCK_LIMIT:
            lines.append(f"(+{len(retained) - self.BRIDGE_BLOCK_LIMIT} more "
                         "retained service(s))")
        lines.append("The current snapshot does not carry these services.")
        lines.append("That is not a cancellation.")
        return lines

    def label_anchors(self) -> List[Any]:
        """V4B's current-vehicle text labels, in a deterministic order."""
        from matplotlib.text import Annotation

        labels = [artist for artist in self.overlay.artists
                  if isinstance(artist, Annotation)]
        return sorted(labels, key=lambda item: (round(item.xy[0], 3),
                                                round(item.xy[1], 3)))

    def adjust_labels(self) -> int:
        """Deterministically offset TEXT that would otherwise overlap.

        This moves the annotation OFFSET only. The marker, the projected
        point and the TfNSW coordinate behind it are never touched, no
        position is snapped to a road and no bearing is altered.
        """
        axis = self.explorer.map_axis
        labels = self.label_anchors()
        if axis is None or not labels:
            return 0
        moved = 0
        placed: List[Tuple[float, float]] = []
        for label in labels:
            screen = axis.transData.transform(label.xy)
            crowded = sum(1 for other in placed
                          if abs(other[0] - screen[0]) < LABEL_COLLISION_PX
                          and abs(other[1] - screen[1]) < LABEL_COLLISION_PX)
            index = crowded % len(LABEL_OFFSETS)
            if index == 0 and self.under_instruction_box(label):
                index = 1                     # push it below its own marker
            offset = LABEL_OFFSETS[index]
            if tuple(label.get_position()) != offset:
                label.set_position(offset)
                label.set_va("bottom" if offset[1] >= 0 else "top")
                moved += 1
            placed.append((float(screen[0]), float(screen[1])))
        return moved

    def under_instruction_box(self, label: Any) -> bool:
        """Whether a label's anchor sits under the Week 5 instruction box."""
        axis = self.explorer.map_axis
        if axis is None:
            return False
        point = axis.transLimits.transform(label.xy)
        left, right, bottom, top = INSTRUCTION_BOX_AXES
        return bool(left <= point[0] <= right and bottom <= point[1] <= top)

    def status_line(self) -> str:
        """One short line. Every critical state still has to fit in it.

        The detail lives where it belongs - the retained services are
        explained on the map, and the per-service wording in the LIVE VEHICLES
        panel - so this line stays short enough not to be clipped.
        """
        if not self.refresh_cycles:
            return (f"'{REFRESH_ALL_LABEL}': one manual arrivals + vehicles "
                    "refresh. No polling.")
        words = {STATUS_OK: "ok", STATUS_STALE: "FAILED, prev kept",
                 STATUS_UNAVAILABLE: "FAILED, none kept"}
        # A successful request only earns the word "new" when the feed's own
        # header timestamp actually advanced.
        vehicles = words.get(self.last_vehicle_status, "-")
        if self.last_vehicle_status == STATUS_OK:
            vehicles = {SNAPSHOT_NEW: "new snapshot",
                        SNAPSHOT_INITIAL: "first snapshot",
                        SNAPSHOT_SAME: "ok, SAME feed snapshot",
                        SNAPSHOT_OLDER: "OLDER feed, newer kept",
                        SNAPSHOT_UNKNOWN: "ok, no feed time"}.get(
                            self.last_vehicle_change, "ok")
        parts = [f"#{self.refresh_cycles} {v4a.clock(int(time.time()))}",
                 f"arrivals {words.get(self.last_transit_status, '-')}",
                 f"vehicles {vehicles}",
                 f"{self.history.total()} obs"]
        if self.bridges:
            parts.append(f"{len(self.bridges)} last-seen")
        if self.history.skipped_without_vehicle_id:
            parts.append(f"{self.history.skipped_without_vehicle_id} "
                         "no vehicle_id")
        return "  |  ".join(parts)

    # ----- diagnostics ---------------------------------------------------- #
    def debug_lines(self) -> List[str]:
        """A concise per-service diagnostic for the selected stop.

        It answers one question the real feed kept raising: was the bus really
        outside the map, or was the projection/extent decision wrong? So it
        prints the raw coordinate, the projected point, the current axes
        limits and the inside/outside result together.

        It holds no key and no header: this file never sees either.
        """
        transit = self.panel.state
        vehicle = self.overlay.state
        extent = self.overlay.extent()
        lines = ["", "LIVE DEBUG", f"stop_id: {self.selected_stop}", ""]
        lines.append("Trip Update:")
        if transit.has_snapshot:
            lines.append(f"  feed timestamp: "
                         f"{v4a.clock(transit.snapshot.feed_timestamp)}")
            lines.append(f"  observed_at:    "
                         f"{v4a.clock(transit.snapshot.observed_at)}")
        else:
            lines.append("  no successful snapshot")
        lines.append(f"  status: {self.last_transit_status or '-'}")
        lines.append("Vehicle Position:")
        if vehicle.has_snapshot:
            lines.append(f"  feed timestamp: "
                         f"{v4a.clock(vehicle.snapshot.feed_timestamp)}")
            lines.append(f"  observed_at:    "
                         f"{v4a.clock(vehicle.snapshot.observed_at)}")
        else:
            lines.append("  no successful snapshot")
        lines.append(f"  status: {self.last_vehicle_status or '-'}")
        lines.append(f"  previous feed timestamp: "
                     f"{v4a.clock(self.previous_vehicle_feed_timestamp)}")
        lines.append(f"  received feed timestamp: "
                     f"{v4a.clock(self.received_vehicle_feed_timestamp)}")
        lines.append(f"  snapshot advancement: "
                     f"{self.last_vehicle_change or '-'}")
        lines.append("  (a successful request is not the same thing as a "
                     "NEWER snapshot)")
        lines.append("  (the two feeds carry their own timestamps and are "
                     "NOT synchronised)")

        present = all_stop_trip_ids(transit, self.selected_stop)
        listed = {str(event.trip_id) for event
                  in displayed_services(transit, self.selected_stop,
                                        self.panel.limit)
                  if event.trip_id is not None}
        for match in self.overlay.matches:
            lines += self.debug_service(
                str(match.trip_id), match.route, STATE_CURRENT, None,
                present, listed, extent,
                match.records[0] if match.records else None)
        for entry in self.bridged_services():
            records, observation = self.bridge_vehicles(entry)
            if len(records) > 1:
                lines.append("")
                lines.append(f"  trip_id: {entry.trip_id}")
                lines.append(f"  route: {entry.route}")
                lines.append(f"  state: {STATE_LAST_SEEN}")
                lines.append(f"  AMBIGUOUS: {len(records)} exact "
                             "VehiclePositions, none chosen")
            for record in records or [None]:
                lines += self.debug_service(
                    entry.trip_id, entry.route, STATE_LAST_SEEN,
                    entry.last_effective_arrival, present, listed, extent,
                    record, observation if record is None else None)
        return lines

    def debug_service(self, trip_id: Optional[str], route: str, state: str,
                      arrival: Optional[int], present: set, listed: set,
                      extent: Any, record: Any = None,
                      observation: Optional[ObservedPosition] = None
                      ) -> List[str]:
        lines = ["", f"  trip_id: {trip_id}",
                 f"  route: {route}",
                 f"  state: {state}",
                 f"  last effective arrival: "
                 f"{v4a.clock(arrival) if arrival is not None else '-'}",
                 f"  exists in current raw stop events: "
                 f"{'yes' if trip_id in present else 'no'}",
                 f"  exists in current bounded upcoming: "
                 f"{'yes' if trip_id in listed else 'no'}"]
        if record is None and observation is None:
            lines.append("  vehicle: no exact trip_id match")
            return lines
        if record is not None:
            latitude, longitude = record.latitude, record.longitude
            lines.append(f"  vehicle_id: {record.vehicle_id}")
            lines.append(f"  vehicle timestamp: "
                         f"{v4a.clock(record.timestamp)}")
            if (record.timestamp is not None
                    and self.overlay.state.has_snapshot):
                lines.append(
                    f"  age: {self.overlay.state.snapshot.observed_at - int(record.timestamp)} s")
        else:
            latitude, longitude = observation.latitude, observation.longitude
            lines.append(f"  vehicle_id: {observation.vehicle_id} "
                         "(last reported, not current)")
            lines.append(f"  vehicle timestamp: "
                         f"{v4a.clock(observation.vehicle_timestamp)}")
        point = week5.plots.project([longitude], [latitude])[0]
        position = (float(point[0]), float(point[1]))
        lines.append(f"  latitude: {latitude}   longitude: {longitude}")
        lines.append(f"  projected x/y: {position[0]:.1f}, {position[1]:.1f}")
        if extent is not None:
            lines.append(f"  current xlim: {extent[0]:.1f}, {extent[1]:.1f}")
            lines.append(f"  current ylim: {extent[2]:.1f}, {extent[3]:.1f}")
        lines.append(f"  inside displayed extent: "
                     f"{v4b.inside_extent(position, extent)}")
        return lines

    def render_status(self) -> str:
        line = self.status_line()
        if self.status_text is not None:
            self.status_text.set_text(line)
        return line

    def redraw(self) -> None:
        canvas = getattr(self.figure, "canvas", None)
        if canvas is not None and hasattr(canvas, "draw_idle"):
            canvas.draw_idle()


def attach_controller(panel: Any, overlay: Any,
                      history: Optional[ObservationHistory] = None
                      ) -> LiveMotionController:
    """Build the controller and tie its lifetime to the Figure.

    Matplotlib holds a bound-method callback WEAKLY, so a controller nothing
    refers to would be collected and Refresh all would go dead while still
    looking perfectly normal - the same lifetime rule D-23 records.
    """
    controller = LiveMotionController(panel, overlay, history=history)
    controller.build()
    controller.connect()
    overlay.figure._week7_motion_controller = controller
    return controller


# --------------------------------------------------------------------------- #
# start-up
# --------------------------------------------------------------------------- #
def build_window(context: Dict[str, Any], stops: Any,
                 study_area: Dict[str, Any], realtime_state: Any,
                 vehicle_state: Any, limit: int = v3.NEXT_SERVICES,
                 transit_fetcher: Optional[Callable[[], bytes]] = None,
                 vehicle_fetcher: Optional[Callable[[], bytes]] = None,
                 figure: Any = None, debug: bool = False) -> Dict[str, Any]:
    """The V4B window, composed by V4C so callback order is fixed.

    The V3 panel and the V4B overlay are built exactly as they build
    themselves, but their own click subscriptions are not registered here:
    V4C drives them, in order, from its single handler. That is composition,
    not modification - both files are untouched and both still subscribe
    normally when run on their own.
    """
    import matplotlib.pyplot as plt

    figure = figure if figure is not None else plt.figure(
        figsize=v3.FIGURE_SIZE)
    explorer = week5.build_explorer(context, stops, study_area, figure=figure)

    panel = v3.LiveTransitPanel(explorer, realtime_state, limit=limit,
                                fetcher=transit_fetcher)
    panel.build()
    figure._week7_live_panel = panel

    overlay = v4b.LiveVehicleOverlay(panel, vehicle_state,
                                     fetcher=vehicle_fetcher)
    overlay.build()
    figure._week7_vehicle_overlay = overlay

    controller = attach_controller(panel, overlay)
    controller.debug = bool(debug)
    return {"figure": figure, "explorer": explorer, "panel": panel,
            "overlay": overlay, "controller": controller}


def run_graphical(context: Dict[str, Any], stops: Any,
                  study_area: Dict[str, Any], realtime_state: Any,
                  vehicle_state: Any, limit: int = v3.NEXT_SERVICES,
                  transit_fetcher: Optional[Callable[[], bytes]] = None,
                  vehicle_fetcher: Optional[Callable[[], bytes]] = None,
                  debug: bool = False) -> int:
    import matplotlib.pyplot as plt

    window = build_window(context, stops, study_area, realtime_state,
                          vehicle_state, limit=limit,
                          transit_fetcher=transit_fetcher,
                          vehicle_fetcher=vehicle_fetcher, debug=debug)
    print("")
    print("A map window is open. "
          + week5.GUI_INSTRUCTIONS.replace("\n", " "))
    print(f"'{REFRESH_ALL_LABEL}' performs ONE manual arrivals + vehicles "
          "refresh. Nothing polls in the background.")
    print("The small points behind a vehicle are its previous observed "
          "positions. No path is interpolated.")
    print("Close the window to finish.")
    # window, and every layer in it, stays a live local across the event loop,
    # and each layer is also attached to the Figure.
    print(f"Controller attached: "
          f"{window['figure']._week7_motion_controller is window['controller']}"
          )
    plt.show()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 V4C: the Week 7 live window with ONE coordinated "
                    "manual refresh and up to four recent observed vehicle "
                    "positions. Presentation only - no polling, no "
                    "interpolation, no Week 5 method change.")
    parser.add_argument("--next", type=int, default=v3.NEXT_SERVICES,
                        dest="next_services",
                        help=f"services to consider for the selected stop "
                             f"(default {v3.NEXT_SERVICES})")
    parser.add_argument("--no-scheduled", action="store_true",
                        help="skip the static scheduled-time join")
    parser.add_argument("--rebuild-extract", action="store_true",
                        help="rebuild the cached Glebe scheduled-time extract")
    parser.add_argument("--parse-file", default=None,
                        help="replay a saved Trip Update .pb payload")
    parser.add_argument("--parse-vehicle-file", default=None,
                        help="replay a saved Vehicle Position .pb payload")
    parser.add_argument("--debug-live", action="store_true",
                        dest="debug_live",
                        help="after each Refresh all, print a per-service "
                             "diagnostic for the selected stop: both feed "
                             "timestamps, the raw coordinate, the projected "
                             "point, the current axes limits and the "
                             "inside/outside decision. Never prints the key.")
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
          "recent observations are presentation layers:")
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
    print("Recent observed positions start empty: the first snapshot is the "
          "CURRENT position, not a previous one.")

    return run_graphical(context, stops, study_area, realtime_state,
                         vehicle_state, limit=options.next_services,
                         transit_fetcher=transit_fetcher,
                         vehicle_fetcher=vehicle_fetcher,
                         debug=options.debug_live)


if __name__ == "__main__":
    sys.exit(main())
