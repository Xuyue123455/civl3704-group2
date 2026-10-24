"""Week 7 V3 - live bus information beside the Week 5 walking comparison.

WHAT THIS IS
------------
A PRESENTATION AND INTEGRATION layer, and nothing else. It opens the EXISTING
Week 5 interactive terrain-routing window, unmodified, and adds one extra
panel - LIVE TRANSIT - that shows what the already-validated Week 7 V2 layer
says about the stop you clicked.

    user clicks an origin        ->  Week 5, unchanged
    user clicks a candidate stop ->  Week 5 shortest / least-effort routes
                                     PLUS this panel's live bus information

WHY IT IS A SEPARATE FILE
-------------------------
``scripts/demo_terrain_route.py`` is NOT modified by this work. The Week 5
command still runs exactly as before and still needs no API key. This wrapper
imports the Week 5 explorer, asks it to build itself into a wider figure, and
then attaches its own axes, its own click observer and its own refresh button.
The routing code therefore cannot see a bus, and the realtime code cannot see
a graph.

WHAT IT REUSES, AND THEREFORE DOES NOT REDO
-------------------------------------------
``scripts/demo_terrain_route.py``       the whole Week 5 explorer, as-is
``scripts/show_glebe_live_arrivals.py`` V2: candidate stops, static GTFS join,
                                        ``build_events``, ``upcoming``,
                                        ``route_label``, the scheduled-time
                                        extract and its cache
``scripts/inspect_gtfs_realtime.py``    V1: fetching, decoding, summarising,
                                        Sydney local time, key redaction

There is no second GTFS-Realtime decoder, no second arrival-time rule and no
second Dijkstra in this file.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not change any Week 5 number. Realtime information never reaches a
search, a cost, a snap or a route choice: for a given origin and stop, the
distance route, the effort route and their four numbers are bit-for-bit the
ones Week 5 already produced. There is no catchability, no journey planning,
no transfer, no Tobler walking time, no stop ranking and no combined
accessibility/realtime score. See D-26 in docs/ASSUMPTIONS.md.

TIME
----
Every displayed time is one V2 already computed. This file never adds a delay
to a scheduled time itself: it reads ``sort_epoch`` / ``sort_basis``, which V2
set from an absolute realtime time where the feed supplied one and from its own
safely derived time otherwise. A missing delay stays missing and is never
rendered as "on time"; a delay of 0 is a real zero.

SCOPE
-----
Bus only, because the bus Trip Update feed is the verified one (D-24). A
light-rail candidate stop gets an explicit statement that realtime is not
available in this prototype - never a bus answer and never a silent blank.

Usage
-----
    set TFNSW_API_KEY=<key>
    python scripts/demo_week7_realtime.py
    python scripts/demo_week7_realtime.py --no-scheduled
    python scripts/demo_week7_realtime.py --parse-file <saved>.pb
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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


# The two finished layers, imported and used as they stand.
week5 = load_script("demo_terrain_route")
v2 = load_script("show_glebe_live_arrivals")
v1 = v2.v1

RealtimeError = v1.RealtimeError
MissingApiKey = v1.MissingApiKey

BUS_FEED = v2.BUS_FEED
BUS_MODE = v2.BUS_MODE
LIGHT_RAIL_MODE = "light_rail"

# How many services to show for the selected stop. Small on purpose: this is a
# projected demonstration panel, not a departure board.
NEXT_SERVICES = 3
MAX_NEXT_SERVICES = 6

# Bounded network timeouts. The real requests have taken about 0.2 s, so a
# refresh that is still waiting after this is a failure, not slow success.
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 10.0

STATUS_NEVER = "never"          # no refresh has been attempted yet
STATUS_OK = "ok"                # the current snapshot is this session's latest
STATUS_STALE = "stale"          # a refresh failed; an older snapshot is shown
STATUS_UNAVAILABLE = "unavailable"   # a refresh failed and there is nothing

PANEL_TITLE = "LIVE TRANSIT"
PANEL_WRAP = 46
PANEL_FONTSIZE = 9.0

# The one-line description of the whole window, for a projected demonstration.
WINDOW_TITLE = ("Week 7 prototype - terrain-aware walking access "
                "+ live bus information")

REFRESH_LABEL = "Refresh live data"
REFRESH_KEY = "r"

# The Week 5 axes are squeezed into this share of the figure width, so the
# panel gets the rest. Nothing inside the Week 5 layout is rearranged.
MAP_SHARE = 0.74
PANEL_RECT = (MAP_SHARE + 0.020, 0.115, 0.975 - MAP_SHARE - 0.020, 0.835)
BUTTON_RECT = (MAP_SHARE + 0.020, 0.040, 0.115, 0.045)
FIGURE_SIZE = (18.0, 8.0)

LIGHT_RAIL_LINES = [
    "Inner West Light Rail",
    "",
    "Realtime not yet available in this prototype.",
    "The current TfNSW v2 endpoint has not been",
    "verified, so no light-rail feed is requested",
    "and no bus answer is substituted.",
    "",
    "Walking analysis remains available.",
]

SNAPSHOT_ABSENCE_LINES = [
    "No matching live update for this stop",
    "in the current feed snapshot.",
]

SCOPE_FOOTER = "Bus GTFS-R only - no light-rail realtime."

WALKING_STILL_WORKS = "Walking analysis remains available."


class Week7DemoError(RuntimeError):
    """Raised with an actionable message when the Week 7 demo cannot start."""


# --------------------------------------------------------------------------- #
# small display helpers - they format, they never compute a time
# --------------------------------------------------------------------------- #
def clock(epoch: Optional[int], seconds: bool = True) -> str:
    """A wall-clock ``HH:MM:SS`` taken from V1's Sydney rendering.

    This SHORTENS a string V1 already produced. It does not convert a
    timezone, and if V1 could not produce its usual format the full string is
    returned rather than a guess.
    """
    if epoch is None:
        return "-"
    text = v1.to_sydney(int(epoch))
    if not text:
        return "-"
    parts = text.split(" ")
    if len(parts) >= 2 and parts[1].count(":") == 2:
        return parts[1] if seconds else parts[1][:5]
    return text


def format_delay(seconds: Optional[int]) -> str:
    """A delay, honestly.

    An ABSENT delay is absent - never zero and never "on time". A delay of
    exactly zero is a real measurement and is shown as one.
    """
    if seconds is None:
        return "not supplied by the feed"
    value = int(seconds)
    if value == 0:
        return "0 s (reported on time)"
    sign = "+" if value > 0 else "-"
    size = abs(value)
    if size < 60:
        return f"{sign}{size} s"
    return f"{sign}{size // 60} min {size % 60:02d} s  ({sign}{size} s)"


def event_delay(event: Any) -> Optional[int]:
    """The delay V2 itself would use: arrival's, else departure's."""
    arrival = event.arrival.get("delay")
    return arrival if arrival is not None else event.departure.get("delay")


def expected_time_text(event: Any) -> str:
    """The expected time, taken from what V2 already decided.

    ``sort_epoch`` and ``sort_basis`` are V2's validated result: an absolute
    realtime time when the feed supplied one, otherwise V2's own safely
    derived time, otherwise nothing at all. No arithmetic happens here.
    """
    if event.sort_epoch is None:
        return "-  (no orderable time supplied)"
    return f"{clock(event.sort_epoch)}  ({event.sort_basis})"


def scheduled_time_text(event: Any) -> str:
    if event.scheduled_epoch is not None:
        return clock(event.scheduled_epoch)
    if event.trip_in_static is False:
        return "-  (trip not in the static snapshot)"
    return "-  (no matched scheduled time)"


def destination_text(event: Any) -> Optional[str]:
    if event.trip_headsign:
        return f"To: {event.trip_headsign}"
    if event.route_long_name:
        return f"To: {event.route_long_name}"
    return None


def wrap(lines: Sequence[str], width: int = PANEL_WRAP) -> List[str]:
    """Wrap for a fixed-width panel, keeping blank lines as separators."""
    import textwrap

    wrapped: List[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped += textwrap.wrap(line, width=width,
                                 subsequent_indent="  ") or [""]
    return wrapped


# --------------------------------------------------------------------------- #
# the realtime application state - presentation state, not a model
# --------------------------------------------------------------------------- #
@dataclass
class LiveSnapshot:
    """One decoded feed, reduced to what the panel shows.

    ``events`` are V2 ``ArrivalEvent`` objects, unmodified.
    """

    feed_timestamp: Optional[int]
    observed_at: int
    fetched_at: int
    entity_count: int
    trip_update_count: int
    events_by_stop: Dict[str, List[Any]] = field(default_factory=dict)

    def events_for(self, stop_id: str) -> List[Any]:
        return list(self.events_by_stop.get(str(stop_id), []))


@dataclass
class RealtimeState:
    """Everything the LIVE TRANSIT panel needs, and nothing analytical.

    The static GTFS tables and the Glebe scheduled-time extract are loaded
    ONCE, before the window opens. A refresh fetches only the small live
    protobuf payload and rebuilds the in-memory events from it.

    The API key is never stored here. It exists only inside the fetcher, and
    any error text that leaves the fetcher has already been redacted.
    """

    stops: pd.DataFrame
    static: Optional[Any] = None
    scheduled: Optional[pd.DataFrame] = None
    snapshot: Optional[LiveSnapshot] = None
    status: str = STATUS_NEVER
    error: Optional[str] = None
    refreshes: int = 0
    successes: int = 0

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot is not None

    def bus_stop_ids(self) -> List[str]:
        return [str(value)
                for value in v2.bus_candidates(self.stops)["stop_id"]]

    def mode_of(self, stop_id: str) -> Optional[str]:
        rows = self.stops[self.stops["stop_id"].astype(str) == str(stop_id)]
        if rows.empty:
            return None
        return str(rows.iloc[0]["mode"])

    def name_of(self, stop_id: str) -> str:
        rows = self.stops[self.stops["stop_id"].astype(str) == str(stop_id)]
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("stop_name", ""))

    def refresh(self, fetcher: Optional[Callable[[], bytes]] = None) -> str:
        """Fetch once, decode with V1, rebuild the events with V2.

        A failure NEVER destroys a snapshot that already succeeded, and never
        raises into the GUI: the outcome is a status string.
        """
        self.refreshes += 1
        source = fetcher if fetcher is not None else fetch_payload
        try:
            payload = source()
            message = v1.parse_feed(payload)
            summary = v1.summarise_feed(message, sample=0)
            events = v2.build_events(message, self.stops, self.static,
                                     self.scheduled)
        except RealtimeError as error:
            self.error = str(error)
            self.status = (STATUS_STALE if self.has_snapshot
                           else STATUS_UNAVAILABLE)
            return self.status

        grouped: Dict[str, List[Any]] = {}
        for event in events:
            grouped.setdefault(str(event.stop_id), []).append(event)
        self.snapshot = LiveSnapshot(
            feed_timestamp=summary.header_timestamp,
            observed_at=int(summary.observed_at or time.time()),
            fetched_at=int(time.time()),
            entity_count=summary.entity_count,
            trip_update_count=summary.trip_update_count,
            events_by_stop=grouped)
        self.error = None
        self.status = STATUS_OK
        self.successes += 1
        return self.status


def fetch_payload(timeout: Any = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) -> bytes:
    """One bounded request for the verified bus feed.

    The key is loaded, used as a header by V1 and discarded. Anything V1
    raises is re-raised with the key redacted, so no GUI string can ever carry
    it.
    """
    definition = v1.resolve_feed(BUS_FEED)
    api_key = v1.load_api_key()
    try:
        return v1.fetch_feed(definition.url, api_key, timeout=timeout).payload
    except RealtimeError as error:
        raise RealtimeError(v1.redact(error, api_key)) from None


def file_fetcher(path: Path) -> Callable[[], bytes]:
    """Replay a saved payload instead of the network, for a dry run."""
    target = Path(path)

    def read() -> bytes:
        if not target.exists():
            raise RealtimeError(f"{target} does not exist.")
        return target.read_bytes()

    return read


# --------------------------------------------------------------------------- #
# the presenter - pure text, so it can be tested without a canvas
# --------------------------------------------------------------------------- #
def feed_lines(state: RealtimeState) -> List[str]:
    """The status block: where the shown information came from, and when."""
    lines: List[str] = []
    snapshot = state.snapshot
    if state.status == STATUS_STALE and snapshot is not None:
        lines += ["Refresh failed",
                  f"showing previous snapshot from "
                  f"{clock(snapshot.fetched_at)}"]
        if state.error:
            lines.append(state.error)
        lines.append("")
    elif state.status == STATUS_UNAVAILABLE:
        lines += ["Live transit unavailable"]
        if state.error:
            lines.append(state.error)
        lines.append("")
        lines.append(WALKING_STILL_WORKS)
        return lines
    elif state.status == STATUS_NEVER:
        lines += ["Live transit not fetched yet.", "",
                  WALKING_STILL_WORKS]
        return lines

    if snapshot is not None:
        lines += [f"Feed updated: {clock(snapshot.feed_timestamp)}",
                  f"Fetched:      {clock(snapshot.fetched_at)}",
                  f"{snapshot.trip_update_count} live TripUpdates in this "
                  f"snapshot"]
    return lines


def service_block(event: Any) -> List[str]:
    """The selected stop's next service, in full."""
    lines = [v2.route_label(event)]
    destination = destination_text(event)
    if destination:
        lines.append(destination)
    lines += [
        "",
        f"Scheduled: {scheduled_time_text(event)}",
        f"Expected:  {expected_time_text(event)}",
        f"Delay:     {format_delay(event_delay(event))}",
    ]
    return lines


def service_line(event: Any) -> str:
    """One later service, compactly."""
    label = (f"Route {event.route_short_name}" if event.route_short_name
             else "Route unresolved")
    return f"  {label:<18}{clock(event.sort_epoch, seconds=False)}"


def stop_lines(state: RealtimeState, stop_id: Optional[str],
               limit: int = NEXT_SERVICES) -> List[str]:
    """The whole LIVE TRANSIT body for the currently selected stop."""
    if stop_id is None:
        return ["Click a candidate stop on the map to see",
                "its live bus information."] + [""] + feed_lines(state)

    name = state.name_of(stop_id)
    header = [f"Stop {stop_id}", name] if name else [f"Stop {stop_id}"]
    mode = state.mode_of(stop_id)

    if mode == LIGHT_RAIL_MODE:
        return header + [""] + LIGHT_RAIL_LINES
    if mode != BUS_MODE:
        return header + ["", f"Mode {mode!r} has no verified realtime feed",
                         "in this prototype.", "", WALKING_STILL_WORKS]

    if not state.has_snapshot:
        return header + [""] + feed_lines(state)

    snapshot = state.snapshot
    events = snapshot.events_for(stop_id)
    listed, unorderable = v2.upcoming(events, snapshot.observed_at,
                                      max(1, min(int(limit),
                                                 MAX_NEXT_SERVICES)))
    body: List[str] = []
    if not listed:
        body += SNAPSHOT_ABSENCE_LINES if not events else [
            "This snapshot carried updates for this stop,",
            "but none of them supplied a usable time."]
        body += ["", v2.ABSENCE_NOTE]
        if unorderable:
            body.append(f"({unorderable} update(s) here carried no orderable "
                        "time.)")
    else:
        body += service_block(listed[0])
        if len(listed) > 1:
            body += ["", "Next:"] + [service_line(event)
                                     for event in listed[1:]]
        if unorderable:
            body += ["", f"({unorderable} further update(s) here carried no "
                         "orderable time and are not listed.)"]
    return header + [""] + body + ["", ""] + feed_lines(state) + \
        ["", SCOPE_FOOTER]


def panel_text(state: RealtimeState, stop_id: Optional[str],
               limit: int = NEXT_SERVICES) -> str:
    return "\n".join(wrap(stop_lines(state, stop_id, limit)))


# --------------------------------------------------------------------------- #
# the panel itself - one axes, one button, one click observer
# --------------------------------------------------------------------------- #
def reserve_panel_space(figure, share: float = MAP_SHARE) -> None:
    """Squeeze the finished Week 5 layout into the left ``share``.

    The Week 5 layout is not rearranged: every axes and every figure-level
    label keeps its relative place, and only the horizontal scale changes.
    Nothing inside ``demo_terrain_route.py`` is touched to make this work.
    """
    suptitle = getattr(figure, "_suptitle", None)
    for axis in list(figure.axes):
        box = axis.get_position()
        axis.set_position([box.x0 * share, box.y0,
                           box.width * share, box.height])
    for text in list(figure.texts):
        if suptitle is not None and text is suptitle:
            continue
        x, y = text.get_position()
        text.set_position((x * share, y))
    if suptitle is not None:
        x, y = suptitle.get_position()
        suptitle.set_position((x * share, y))


class LiveTransitPanel:
    """The realtime presenter, bolted onto - never into - the Week 5 window.

    It owns exactly three things: an axes that draws text, a refresh button,
    and a click observer that notices which stop the user clicked. It never
    calls a search, a snap, a solve or a redraw of the walking routes.
    """

    def __init__(self, explorer, state: RealtimeState,
                 limit: int = NEXT_SERVICES,
                 fetcher: Optional[Callable[[], bytes]] = None):
        self.explorer = explorer
        self.state = state
        self.limit = limit
        self.fetcher = fetcher
        self.figure = explorer.figure
        self.axis = None
        self.text = None
        self.button = None
        self.button_axis = None
        self.selected_stop: Optional[str] = None
        self.renders = 0

    # ----- construction ------------------------------------------------- #
    def build(self):
        from matplotlib.widgets import Button

        reserve_panel_space(self.figure)
        # The window now shows two things, so it says so. The Week 5 title
        # described a window that held only the walking comparison.
        self.figure.suptitle(WINDOW_TITLE, fontsize=12,
                             color=week5.plots.COLOUR_INK)
        self.axis = self.figure.add_axes(PANEL_RECT)
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        for spine in self.axis.spines.values():
            spine.set_edgecolor("#bbbbbb")
        self.axis.set_title(PANEL_TITLE, fontsize=12,
                            color=week5.plots.COLOUR_INK, pad=8.0)
        self.text = self.axis.text(
            0.045, 0.975, "", transform=self.axis.transAxes, ha="left",
            va="top", fontsize=PANEL_FONTSIZE, family="monospace",
            color=week5.plots.COLOUR_INK)

        self.button_axis = self.figure.add_axes(BUTTON_RECT)
        self.button = Button(self.button_axis, REFRESH_LABEL)
        self.button.label.set_fontsize(9)
        self.button.on_clicked(self.on_refresh)
        self.figure.text(PANEL_RECT[0], BUTTON_RECT[1] + BUTTON_RECT[3]
                         + 0.008,
                         f"or press '{REFRESH_KEY}'  -  one request per "
                         "press, no background polling", ha="left",
                         va="bottom", fontsize=7,
                         color=week5.plots.COLOUR_MUTED)
        self.render()
        return self

    def connect(self):
        """Observe clicks; the Week 5 handler keeps its own subscription.

        The stop under the pointer is hit-tested here with the explorer's own
        primitive, so this panel does not depend on the order in which
        Matplotlib happens to dispatch the two handlers.
        """
        canvas = self.figure.canvas
        canvas.mpl_connect("button_press_event", self.on_click)
        canvas.mpl_connect("key_press_event", self.on_key)
        return self

    # ----- events -------------------------------------------------------- #
    def on_click(self, event) -> Optional[str]:
        if getattr(event, "inaxes", None) is not self.explorer.map_axis:
            return None
        if getattr(event, "button", 1) != 1:
            return None
        stop_id = self.explorer.hit_test_stop(event)
        if stop_id is None:
            return None
        self.selected_stop = str(stop_id)
        self.render()
        self.redraw()
        return self.selected_stop

    def on_key(self, event) -> Optional[str]:
        if getattr(event, "key", None) != REFRESH_KEY:
            return None
        return self.on_refresh(event)

    def on_refresh(self, _event=None) -> str:
        """One synchronous request. It never touches the walking routes."""
        self.show_busy()
        status = self.state.refresh(self.fetcher)
        self.render()
        self.redraw()
        return status

    def show_busy(self) -> None:
        if self.text is not None:
            self.text.set_text("Requesting the live bus feed ...")
        self.redraw()

    # ----- drawing ------------------------------------------------------- #
    def render(self) -> str:
        body = panel_text(self.state, self.selected_stop, self.limit)
        if self.text is not None:
            self.text.set_text(body)
        self.renders += 1
        return body

    def redraw(self) -> None:
        canvas = getattr(self.figure, "canvas", None)
        if canvas is not None and hasattr(canvas, "draw_idle"):
            canvas.draw_idle()


def attach_panel(explorer, state: RealtimeState, limit: int = NEXT_SERVICES,
                 fetcher: Optional[Callable[[], bytes]] = None
                 ) -> LiveTransitPanel:
    """Build the panel and tie its lifetime to the Figure.

    Matplotlib holds a bound-method callback WEAKLY, so a panel nothing refers
    to would be collected and its button would go dead while still looking
    perfectly normal. This is the same lifetime rule the Week 5 explorer
    documents, applied to the same Figure.
    """
    panel = LiveTransitPanel(explorer, state, limit=limit, fetcher=fetcher)
    panel.build()
    panel.connect()
    explorer.figure._week7_live_panel = panel
    return panel


# --------------------------------------------------------------------------- #
# start-up: static once, live once, then the window
# --------------------------------------------------------------------------- #
def scheduled_cache_path(stop_ids: Sequence[str], static: Any) -> Path:
    """Where V2 would already have cached the Glebe scheduled-time extract.

    The key comes from V2's own ``extract_key``, so this cannot drift from
    what V2 actually writes; only the file name pattern is repeated, and it is
    used solely to decide whether to WARN before a first build.
    """
    key = v2.extract_key(set(str(value) for value in stop_ids),
                         static.provenance if static else {})
    return v2.extract_dir() / f"glebe_stop_times_{key}.csv"


def load_static_once(want_scheduled: bool = True,
                     rebuild: bool = False) -> Dict[str, Any]:
    """The static side of the demo, loaded exactly once.

    ``stop_times.txt`` is ~400 MB. It is streamed at most ONCE, only when the
    cached Glebe-only extract is absent, and never again - a refresh only ever
    fetches the small live payload.
    """
    stops = v2.load_candidate_stops()
    static = v2.load_static_gtfs()
    scheduled = None
    if want_scheduled:
        # Week 11: the extract covers every candidate stop of a realtime mode (bus AND the light-rail
        # platforms), so a delay-only light-rail update can be timed exactly as a bus one; the bus rows are
        # the same rows as before (the extract is keyed by the stop-id set, so it is rebuilt once)
        bus_ids = set(str(value) for value in v2.realtime_candidates(stops)
                      ["stop_id"])
        cache = scheduled_cache_path(sorted(bus_ids), static)
        if not cache.exists() or rebuild:
            print("Building the Glebe scheduled-time extract once. This "
                  "streams stop_times.txt (~400 MB) a single time and caches "
                  "a Glebe-only slice; later runs and every refresh reuse the "
                  "cache.")
        scheduled = v2.load_scheduled(static, bus_ids, rebuild=rebuild)
        if scheduled is None:
            print("No stop_times.txt in the static snapshot: scheduled times "
                  "will be shown as unavailable. Realtime still works.")
    return {"stops": stops, "static": static, "scheduled": scheduled}


def describe_start(state: RealtimeState) -> str:
    if state.status == STATUS_OK and state.snapshot is not None:
        return (f"Live feed updated {clock(state.snapshot.feed_timestamp)}  "
                f"({state.snapshot.trip_update_count} TripUpdates, fetched "
                f"{clock(state.snapshot.fetched_at)})")
    if state.status == STATUS_UNAVAILABLE:
        return ("Live transit unavailable at start-up. The walking demo still "
                f"works. Reason: {state.error}")
    return "Live transit has not been fetched."


def run_graphical(context: Dict[str, Any], stops: pd.DataFrame,
                  study_area: Dict[str, Any], state: RealtimeState,
                  limit: int = NEXT_SERVICES,
                  fetcher: Optional[Callable[[], bytes]] = None) -> int:
    """Open the Week 5 window, widened, with the LIVE TRANSIT panel on it."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=FIGURE_SIZE)
    explorer = week5.build_explorer(context, stops, study_area, figure=figure)
    panel = attach_panel(explorer, state, limit=limit, fetcher=fetcher)
    print("")
    print("A map window is open. "
          + week5.GUI_INSTRUCTIONS.replace("\n", " "))
    print(f"The LIVE TRANSIT panel follows the stop you click "
          f"({panel.limit} services). '{REFRESH_LABEL}' re-requests the feed "
          "once.")
    print("Close the window to finish.")
    # Both objects stay live locals across the event loop, and both are also
    # attached to the Figure, so neither can be collected while the window is
    # open - a collected panel would leave a visible but dead button.
    plt.show()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 V3: the Week 5 terrain-routing window with a live "
                    "bus LIVE TRANSIT panel. Presentation and integration "
                    "only - no Week 5 method changes, bus realtime only.")
    parser.add_argument("--next", type=int, default=NEXT_SERVICES,
                        dest="next_services",
                        help=f"services to show for the selected stop "
                             f"(default {NEXT_SERVICES}, "
                             f"max {MAX_NEXT_SERVICES})")
    parser.add_argument("--no-scheduled", action="store_true",
                        help="skip the static scheduled-time join")
    parser.add_argument("--rebuild-extract", action="store_true",
                        help="rebuild the cached Glebe scheduled-time extract")
    parser.add_argument("--parse-file", default=None,
                        help="replay a saved .pb payload instead of "
                             "requesting the live feed (no API key needed)")
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
    print(f"Trusted cost {context['provenance']['trusted_edges']} directed "
          "edges carry an adopted terrain cost")
    print("Week 5 routing is unchanged here. Realtime is a presentation "
          "layer: it never enters a search, a cost or a route choice.")

    try:
        loaded = load_static_once(want_scheduled=not options.no_scheduled,
                                  rebuild=options.rebuild_extract)
    except v2.GlebeRealtimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    fetcher = (file_fetcher(Path(options.parse_file)) if options.parse_file
               else None)
    state = RealtimeState(stops=loaded["stops"], static=loaded["static"],
                          scheduled=loaded["scheduled"])
    # A missing key or a failed request is NOT fatal here. The Week 5 walking
    # comparison is the research result and must still open without one;
    # ``refresh`` reports the failure as a status instead of raising.
    state.refresh(fetcher)
    print(describe_start(state))

    return run_graphical(context, stops, study_area, state,
                         limit=options.next_services, fetcher=fetcher)


if __name__ == "__main__":
    sys.exit(main())
