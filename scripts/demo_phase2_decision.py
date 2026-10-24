"""Week 9 Phase 2 - interactive prototype: where am I, which service, which stop.

WHAT THIS IS
------------
A Matplotlib desktop window (the same toolkit as the Week 5 and Week 7
prototypes) that walks a first-time viewer through one question:

    1. Where am I?                 click the map (or load the demo preset)
    2. Which route do I want?      Route selector
    3. Which way am I travelling?  Direction selector
    4. Which exact trip?           the NEXT FEASIBLE TRIP, chosen for you,
                                   or any listed upcoming trip you click
    5. Which stops can I reach     catchable / not proven / timetable-only,
       in time for THAT trip?      per stop, with the slack in minutes
    6. Among the catchable stops,  lowest effort / fastest walking /
       which walking trade-off?    shortest walk / Pareto shortlist, and the
                                   three walking routes to the selected stop

Catchability is a constraint, not an objective: no stop is ever ranked by
boarding time (boarding the same trip upstream is not an earlier arrival).

THE WINDOW IS A VIEW, NOT A CALCULATOR
--------------------------------------
Every number and every state on screen is read from the scene that
``scripts/phase2_decision.py`` produces - the same object it writes to
``results/current_scene.json`` and ``results/current_scene.geojson`` on
every run. No routing, Minetti, Tobler, ranking or catchability logic lives
here; choosing another trip calls the engine's ``apply_trip_selection`` on
the same scene (no re-routing, no feed access). The map is drawn from the
scene's route edge keys and the graph's stored geometry. Three route
objectives that share one path are drawn ONCE and said to be identical,
from the scene's ``route_identity``.

MODES (never mislabelled)
-------------------------
    --mode analytical   Terrain + walking analysis only. No realtime.
    --mode live         Live GTFS-Realtime (one request per run / refresh)
    --mode replay       Recorded realtime snapshot (--replay PATH.pb), with
                        its captured timestamp; never shown as live

Usage
-----
    python scripts/demo_phase2_decision.py --mode analytical            # empty map; this is the OPTIONAL
                                                                        # 'Compare nearby stops' view of
                                                                        # demo_week9_primary.py
    python scripts/demo_phase2_decision.py --mode analytical --demo     # load the documented demo preset
    python scripts/demo_phase2_decision.py --mode live --origin -33.8795 151.1875 --service 433/1
    python scripts/demo_phase2_decision.py --mode replay --replay data/cache/gtfs_realtime/snapshots/buses_....pb
    python scripts/demo_phase2_decision.py --mode analytical --save results/phase2_prototype_preview.png
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402


def load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase2 = load_script("phase2_decision")
week5 = load_script("demo_terrain_route")       # study area, click guard, CRS helpers
plots = week5.plots                              # drawing helpers (projection, context)
intent_mod = phase2.intent_mod
tor = phase2.tor

# --------------------------------------------------------------------------- #
# the demo preset - a REAL, already-computed case, nothing invented
# --------------------------------------------------------------------------- #
# Origin O5 of the Week 9 representative-origin study (rule-selected:
# results/candidate_ranking/representative_origins/origins.json), service
# 433 towards Balmain, stop 203732 Glebe Point Rd At St Johns Rd. From here
# the shortest, lowest-effort and fastest routes to 203732 are three DIFFERENT
# paths (each differs from the others over ~350 m of a ~370 m walk): effort
# saving 2.8 % for 1.2 % extra distance. Documented in
# docs/week9_phase2_ui_redesign.md.
DEMO_PRESET = {
    "label": "Demo preset",
    "origin_label": "Week 9 origin O5 (Mt Vernon St / Glebe Point Rd area)",
    "lat": -33.880897, "lon": 151.184860,
    "service": "433/0",
    "stop": "203732",
    "why": "shortest, lowest-effort and fastest walking routes are three different paths",
}

WINDOW_TITLE = "Accessible transit stop selection - Phase 2 prototype"
FIGURE_SIZE = (17.5, 9.6)

# layout (figure fractions)
MAP_RECT = (0.015, 0.035, 0.595, 0.925)
RIGHT_X0, RIGHT_X1 = 0.635, 0.985
BANNER_Y = 0.955
ROUTE_RECT = (RIGHT_X0, 0.70, 0.075, 0.21)
DIRECTION_RECT = (0.725, 0.835, 0.26, 0.075)
RUN_RECT = (0.725, 0.785, 0.115, 0.038)
RESET_RECT = (0.85, 0.785, 0.135, 0.038)
DEMO_RECT = (0.725, 0.74, 0.115, 0.038)
TECH_RECT = (0.85, 0.74, 0.135, 0.038)
ORIGIN_TEXT_XY = (0.725, 0.728)
TRIP_PANEL_RECT = (RIGHT_X0, 0.515, RIGHT_X1 - RIGHT_X0, 0.17)
AUTO_RECT = (0.862, 0.655, 0.118, 0.027)
STOP_PANEL_RECT = (RIGHT_X0, 0.195, RIGHT_X1 - RIGHT_X0, 0.305)
LIST_PANEL_RECT = (RIGHT_X0, 0.035, RIGHT_X1 - RIGHT_X0, 0.145)
TECH_PANEL_RECT = (0.02, 0.04, 0.585, 0.46)

# colours (distinct hues AND distinct line styles - never colour alone)
COLOUR_DISTANCE = "#2171b5"      # blue, dashed
COLOUR_EFFORT = "#d95f02"        # orange, solid
COLOUR_TIME = "#1b9e77"          # green, dotted
COLOUR_SHARED = "#333333"        # one path shared by every objective
STYLE_DISTANCE = (0, (7, 4))
STYLE_EFFORT = "-"
STYLE_TIME = (0, (1.6, 2.2))
COLOUR_ORIGIN = "#7570b3"
COLOUR_STOP_IN = "#1a1a1a"
COLOUR_STOP_OUT = "#d4d4d4"
COLOUR_UNAVAILABLE = "#b05c8a"
COLOUR_LOWEST_EFFORT = "#d95f02"
COLOUR_FASTEST = "#1b9e77"
COLOUR_SHORTEST = "#2171b5"
COLOUR_PARETO = "#7570b3"
COLOUR_INK = "#1a1a1a"
COLOUR_MUTED = "#666666"
COLOUR_PANEL = "#f7f7f7"
MODE_COLOURS = {phase2.MODE_ANALYTICAL: "#e4e4e4", phase2.MODE_LIVE: "#d5efd2",
                phase2.MODE_REPLAY: "#fbe8c3"}
# catchability of a stop for the ONE selected trip: fill + edge + hatch, never colour alone
FILL_CATCHABLE = "#a1d99b"       # green fill, black edge
FILL_SCHEDULE_ONLY = "#fde98a"   # amber fill, hatched (timetable only)
FILL_NOT_PROVEN = "#ffffff"      # white, thin grey edge
FILL_NO_TIME = "#eeeeee"         # grey, dotted edge: the trip has no time here

MARKER_BY_MODE = {"bus": "o", "light_rail": "^"}
STOP_PICK_RADIUS_PX = 14.0
TOP_N = 5
TRIP_ROWS = 4

OBJECTIVE_LABELS = {"distance": "Shortest route", "effort": "Lowest-effort route",
                    "time": "Fastest route"}
OBJECTIVE_STYLE = {"distance": (COLOUR_DISTANCE, STYLE_DISTANCE, 3.0),
                   "effort": (COLOUR_EFFORT, STYLE_EFFORT, 2.6),
                   "time": (COLOUR_TIME, STYLE_TIME, 2.0)}
# objective -> (colour, linestyle, linewidth, zorder): widest at the bottom,
# thinnest on top, as in demo_week9_primary.py, so coinciding routes all stay visible
ROUTE_LAYERS = {"distance": (COLOUR_DISTANCE, STYLE_DISTANCE, 6.0, 4),
                "effort": (COLOUR_EFFORT, STYLE_EFFORT, 3.8, 5),
                "time": (COLOUR_TIME, STYLE_TIME, 2.2, 6)}
HIGHLIGHT_LABELS = {"lowest_terrain_effort": "Lowest effort",
                    "lowest_modelled_walking_time": "Fastest walking",
                    "shortest_walk": "Shortest walk",
                    "pareto_non_dominated": "Pareto shortlist"}
# Wording is provenance-aware: "catchable" is reserved for realtime-confirmed
# feasibility; a timetable-only stop is "timetable-feasible", never catchable.
CATCH_LABELS = {phase2.CATCHABLE: "catchable under the current model (realtime)",
                phase2.SCHEDULED_AFTER: "timetable-feasible only",
                phase2.NOT_PROVEN: "not proven catchable",
                phase2.NO_SERVICE_FOUND: "no service timing found",
                phase2.TRIP_TIME_UNAVAILABLE: "no timing for this trip at this stop"}
SOURCE_LABELS = {phase2.SOURCE_REALTIME: "realtime",
                 phase2.SOURCE_DERIVED: "realtime (scheduled + delay)",
                 phase2.SOURCE_SCHEDULE: "timetable, not realtime",
                 phase2.BASIS_REALTIME: "realtime",
                 phase2.BASIS_SCHEDULE: "timetable only, not realtime",
                 phase2.BASIS_MIXED: "realtime at some stops, timetable at others"}
WINNER_MARKERS = {"lowest_terrain_effort": ("*", COLOUR_LOWEST_EFFORT, 230),
                  "lowest_modelled_walking_time": ("D", COLOUR_FASTEST, 80),
                  "shortest_walk": ("s", COLOUR_SHORTEST, 70)}


# --------------------------------------------------------------------------- #
# presentation helpers (pure, testable; read scene fields only)
# --------------------------------------------------------------------------- #
def mode_banner(scene: Optional[Dict[str, Any]], mode: str, replay_path: Optional[Path]) -> str:
    """Short, human text for the active mode; REPLAY is never 'live'."""
    mode = mode.upper()
    rt = (scene or {}).get("realtime") or {}
    if mode == phase2.MODE_ANALYTICAL:
        return "ANALYTICAL  -  Terrain + walking analysis only. No realtime."
    if mode == phase2.MODE_LIVE:
        if rt.get("status") == "ok":
            return f"LIVE  -  Live GTFS-Realtime, fetched {rt.get('fetched_at') and phase2.sydney_text(rt['fetched_at'])}"
        if scene is None:
            return "LIVE  -  Live GTFS-Realtime (requested when you run)"
        return "LIVE requested  -  feed unavailable; showing terrain + walking analysis only"
    tag = " (synthetic test)" if rt.get("is_synthetic") else ""
    when = phase2.sydney_text(rt.get("observed_at")) if rt.get("observed_at") else None
    if scene is None or rt.get("status") != "ok":
        name = Path(replay_path).name if replay_path else "?"
        return f"REPLAY  -  Recorded realtime snapshot {name} (not live)"
    return f"REPLAY  -  Recorded realtime snapshot captured {when[:16] if when else '?'}{tag}. Not live."


def route_identity_text(entry: Dict[str, Any]) -> str:
    """One sentence saying which route objectives share a path."""
    identity = entry.get("route_identity")
    if not identity:
        return ""
    if identity.get("all_same"):
        return "All three route objectives produce the same path."
    names = {"distance": "distance", "effort": "effort", "time": "time"}
    parts = []
    for group in identity["groups"]:
        if len(group) > 1:
            parts.append(" and ".join(names[g] for g in group) + " routes are the same")
    if parts:
        others = [g[0] for g in identity["groups"] if len(g) == 1]
        tail = (f"; the {names[others[0]]} route differs" if len(others) == 1 else "")
        text = "; ".join(parts) + tail
        return text[0].upper() + text[1:] + "."
    return "The three route objectives produce three different paths."


def distinct_route_groups(entry: Dict[str, Any]) -> List[Tuple[List[str], Dict[str, Any]]]:
    """[(objectives sharing the path, route dict)] - one item per distinct path."""
    identity = entry.get("route_identity")
    routes = entry.get("routes") or {}
    if not identity:
        return [([obj], routes[obj]) for obj in ("distance", "effort", "time") if routes.get(obj)]
    return [(group, routes[group[0]]) for group in identity["groups"] if routes.get(group[0])]


def fmt_m(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:,.0f} m"


def fmt_effort(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:,.0f} J/kg"


def fmt_min(seconds: Optional[float]) -> str:
    return "-" if seconds is None else f"{seconds / 60.0:.1f} min"


def fmt_pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f} %"


def fmt_slack(seconds: Optional[int]) -> str:
    """Signed slack: '+1 min 00 s', '-2 min 00 s', '+45 s'."""
    if seconds is None:
        return "-"
    sign = "-" if seconds < 0 else "+"
    minutes, rest = divmod(abs(int(seconds)), 60)
    return f"{sign}{minutes} min {rest:02d} s" if minutes else f"{sign}{rest} s"


def clock(sydney_text: Optional[str]) -> str:
    """'2026-10-08 22:10:00 AEDT' -> '22:10'."""
    return sydney_text[11:16] if sydney_text else "-"


def route_table_lines(entry: Dict[str, Any]) -> List[str]:
    """The compact three-row comparison for the selected stop."""
    routes = entry.get("routes") or {}
    lines = [f"{'':21s}{'Distance':>10s}{'Effort':>12s}{'Walk time':>11s}"]
    for obj in ("distance", "effort", "time"):
        r = routes.get(obj)
        if r is None:
            lines.append(f"{OBJECTIVE_LABELS[obj]:21s}{'-':>10s}{'-':>12s}{'-':>11s}")
            continue
        lines.append(f"{OBJECTIVE_LABELS[obj]:21s}{fmt_m(r['distance_m']):>10s}"
                     f"{fmt_effort(r['effort_j_per_kg']):>12s}{fmt_min(r['walking_time_s']):>11s}")
    return lines


def comparison_marks(entry: Dict[str, Any]) -> List[Tuple[bool, str]]:
    highlights = set(entry.get("highlights") or [])
    return [(key in highlights, label) for key, label in HIGHLIGHT_LABELS.items()]


def winner_marker(highlights) -> Optional[Tuple[str, str, float]]:
    """The ONE marker a stop gets for its walking wins: star if it is the
    lowest-effort stop, else diamond if fastest, else square if shortest."""
    for key in ("lowest_terrain_effort", "lowest_modelled_walking_time", "shortest_walk"):
        if key in highlights:
            return WINNER_MARKERS[key]
    return None


def legend_handles() -> List[Any]:
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], marker="o", color="none", markersize=7, markerfacecolor=FILL_CATCHABLE,
               markeredgecolor=COLOUR_STOP_IN, label="Realtime-catchable for the selected trip"),
        Line2D([0], [0], marker="o", color="none", markersize=7, markerfacecolor=FILL_SCHEDULE_ONLY,
               markeredgecolor=COLOUR_STOP_IN, label="Timetable-feasible only (hatched)"),
        Line2D([0], [0], marker="o", color="none", markersize=6, markerfacecolor=FILL_NOT_PROVEN,
               markeredgecolor="#8a8a8a", label="Not proven catchable / no timing"),
        Line2D([0], [0], marker="o", color="none", markersize=4, markerfacecolor=COLOUR_STOP_OUT,
               markeredgecolor="none", label="Other candidate stop (context)"),
        Line2D([0], [0], marker="*", color="none", markersize=12, markerfacecolor=COLOUR_LOWEST_EFFORT,
               markeredgecolor="#000000", label="Walking winner: \u2605 lowest effort, \u25c6 fastest, \u25a0 shortest"),
        Line2D([0], [0], marker="o", color="none", markersize=12, markerfacecolor="none",
               markeredgecolor=COLOUR_PARETO, markeredgewidth=1.2, label="Pareto shortlist (ring)"),
        Line2D([0], [0], color=COLOUR_DISTANCE, linewidth=2.4, linestyle=STYLE_DISTANCE,
               label="Shortest route (dashed)"),
        Line2D([0], [0], color=COLOUR_EFFORT, linewidth=2.6, linestyle=STYLE_EFFORT,
               label="Lowest-effort route (solid)"),
        Line2D([0], [0], color=COLOUR_TIME, linewidth=2.6, linestyle=STYLE_TIME,
               label="Fastest-walking route (dotted)"),
    ]


def feasibility_count_text(candidate: Dict[str, Any]) -> str:
    """Provenance-aware count: realtime-catchable vs timetable-feasible, never mixed up."""
    rt, sched = candidate.get("n_catchable_stops", 0), candidate.get("n_schedule_only_stops", 0)
    if rt == 0 and sched == 0:
        return "no stop proven catchable"
    parts = []
    if rt:
        parts.append(f"{rt} realtime-catchable stop{'s' if rt != 1 else ''}")
    if sched:
        parts.append(f"{sched} timetable-feasible stop{'s' if sched != 1 else ''}")
    return " + ".join(parts)


def trip_basis(candidate: Dict[str, Any]) -> str:
    """realtime / schedule_only / mixed, from the trip's feasible stops (not its listing)."""
    rt, sched = candidate.get("n_catchable_stops", 0), candidate.get("n_schedule_only_stops", 0)
    if rt and sched:
        return phase2.BASIS_MIXED
    if rt:
        return phase2.BASIS_REALTIME
    if sched:
        return phase2.BASIS_SCHEDULE
    return candidate.get("basis") or phase2.BASIS_SCHEDULE


def trip_display_labels(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """{trip_id: (time text, suffix)} - identical-looking exact trips told apart.

    Rule (deterministic over the given list, which is chronological):
    1. rows are normally 'HH:MM  route -> headsign';
    2. rows that collide on that text show the reference time with seconds
       ('HH:MM:SS'), a real, human-readable field;
    3. rows that still collide get ' Trip A', ' Trip B', ... in list order.
    The suffix is a display aid only; identity stays the exact trip_id.
    """
    def visible(c, with_seconds):
        ref = c.get("reference_sydney") or ""
        t = ref[11:19] if with_seconds else ref[11:16]
        return (t, c["route_short_name"], str(c.get("trip_headsign") or ""))
    labels: Dict[str, Tuple[str, str]] = {}
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for c in candidates:
        groups.setdefault(visible(c, False), []).append(c)
    for key, members in groups.items():
        if len(members) == 1:
            labels[members[0]["trip_id"]] = (key[0], "")
            continue
        second: Dict[Any, List[Dict[str, Any]]] = {}
        for c in members:
            second.setdefault(visible(c, True), []).append(c)
        for key2, members2 in second.items():
            if len(members2) == 1:
                labels[members2[0]["trip_id"]] = (key2[0], "")
            else:
                for index, c in enumerate(members2):
                    labels[c["trip_id"]] = (key2[0], f" Trip {chr(ord('A') + index)}")
    return labels


def trip_row_text(candidate: Dict[str, Any], label: Optional[Tuple[str, str]] = None) -> str:
    """'10:56  370 -> Glebe Point   4 timetable-feasible stops' (no raw trip_id here)."""
    time_text, suffix = label if label else (clock(candidate.get("reference_sydney")), "")
    return (f"{time_text:8s} {candidate['route_short_name']} -> "
            f"{str(candidate.get('trip_headsign') or '')[:20]:20s}{suffix:7s} {feasibility_count_text(candidate)}")


def trip_rows(scene: Dict[str, Any], max_rows: int = TRIP_ROWS) -> List[Dict[str, Any]]:
    """The listed upcoming exact trips, chronological; the selected one is always shown."""
    candidates = list(scene.get("trip_candidates") or [])
    rows = candidates[:max_rows]
    selected = next((c for c in candidates if c.get("is_selected")), None)
    if selected is not None and selected not in rows:
        rows = rows[:-1] + [selected]
    return rows


def selected_trip_text(scene: Dict[str, Any], mode: str) -> str:
    """One sentence on which exact trip the stops are judged against."""
    chosen = scene.get("selected_trip")
    if chosen is None:
        if scene.get("trip_candidates"):
            return "No listed trip is proven catchable from here (see note below)."
        return "No upcoming trip could be timed for this service."
    counts = {"n_catchable_stops": len(chosen.get("catchable_stop_ids") or []),
              "n_schedule_only_stops": len(chosen.get("schedule_only_stop_ids") or [])}
    basis = trip_basis(counts)
    if chosen["selection"] != phase2.SELECTION_AUTO:
        how = "Trip chosen by you"
    elif basis == phase2.BASIS_REALTIME or basis == phase2.BASIS_MIXED:
        how = "Next feasible trip (automatic)"
    else:
        how = "Next timetable-feasible trip (automatic)"
    labels = trip_display_labels(scene.get("trip_candidates") or [])
    time_text, suffix = labels.get(chosen["trip_id"], (clock(chosen.get("reference_sydney")), ""))
    rt = (scene.get("realtime") or {})
    note = ""
    if basis == phase2.BASIS_SCHEDULE and rt.get("status") == "ok" and mode.upper() != phase2.MODE_ANALYTICAL:
        note = "  ·  feed loaded, but this trip has timetable timing only"
    elif basis == phase2.BASIS_SCHEDULE and mode.upper() != phase2.MODE_ANALYTICAL:
        note = "  ·  no realtime confirmation"
    return (f"{how}: {time_text}  {chosen['route_short_name']} -> "
            f"{str(chosen.get('trip_headsign') or '')[:22]}{suffix}  ·  {feasibility_count_text(counts)}{note}")


def trip_at_stop_lines(entry: Dict[str, Any], scene: Dict[str, Any], mode: str) -> List[str]:
    """This stop against the ONE selected trip; never looks live when it is not."""
    mode = mode.upper()
    catch = entry.get("catchability") or {}
    chosen = scene.get("selected_trip")
    lines: List[str] = []
    if not entry.get("serves_selected_service"):
        return ["This stop is not served by the selected route and direction."]
    if mode == phase2.MODE_ANALYTICAL:
        lines.append("Realtime not used in this mode.")
    feed_ok = (scene.get("realtime") or {}).get("status") == "ok" and mode != phase2.MODE_ANALYTICAL
    if mode == phase2.MODE_ANALYTICAL and chosen is not None \
            and (catch.get("status") == phase2.SCHEDULED_AFTER):
        lines = []      # said on the confirmation line instead, to keep the block to four lines
    if chosen is None:
        if catch.get("trips"):
            worst = max((t.get("slack_s") for t in catch["trips"] if t.get("slack_s") is not None), default=None)
            lines.append("No listed trip is proven catchable at this stop"
                         + (f" (best slack {fmt_slack(worst)})." if worst is not None else "."))
        else:
            lines.append(no_timing_sentence(catch, feed_ok))
        return lines
    status = catch.get("status")
    source = catch.get("expected_time_source")
    source_text = ("timetable, not realtime" if mode == phase2.MODE_ANALYTICAL
                   else SOURCE_LABELS.get(source, source or "-"))
    if catch.get("trip_expected_arrival_epoch") is not None:
        lines.append(f"Trip {clock(chosen.get('reference_sydney'))} {chosen['route_short_name']} -> "
                     f"{chosen.get('trip_headsign')}: expected at this stop "
                     f"{clock(catch.get('trip_expected_arrival_sydney'))}  ({source_text})")
        lines.append(f"You arrive {clock(catch.get('pedestrian_arrival_sydney'))} (modelled)  ·  "
                     f"slack {fmt_slack(catch.get('slack_s'))}")
        lines.append(f"Status: {CATCH_LABELS.get(status, status)}")
        if status == phase2.SCHEDULED_AFTER:
            lines.append("Realtime confirmation unavailable" + (
                " - the feed is loaded, but this exact trip has no realtime timing here." if feed_ok
                else ". Realtime not used in this mode." if mode == phase2.MODE_ANALYTICAL
                else " - timetable only."))
    elif status == phase2.TRIP_TIME_UNAVAILABLE:
        lines.append(f"Status: {CATCH_LABELS.get(status, status)}")
        lines.append("Realtime feed loaded, but this exact trip has no realtime timing at this stop."
                     if feed_ok else "No timing for this exact trip at this stop in the timetable used.")
    else:
        lines.append(no_timing_sentence(catch, feed_ok))
    return lines


def no_timing_sentence(catch: Dict[str, Any], feed_ok: bool) -> str:
    """One precise sentence instead of 'No service found: no service found'."""
    reason = catch.get("reason") or catch.get("inventory_reason") or ""
    if reason == phase2.REALTIME_LIGHT_RAIL:
        return "Light rail: no realtime feed in this project; no timetable match either."
    if reason == phase2.REALTIME_UNAVAILABLE:
        return "Realtime feed unavailable; no timetable fallback found."
    if reason == phase2.REALTIME_NO_MATCH:
        return "No matching realtime service found for this stop."
    if reason == phase2.SCHEDULE_NO_MATCH:
        return ("Realtime feed loaded, but no service of this route and direction was found here; "
                "timetable has none either." if feed_ok else "No service of this route and direction in the timetable for today.")
    if reason == phase2.SCHEDULE_DATE_NOT_COVERED:
        return "The static timetable does not cover this date."
    if reason == phase2.SCHEDULE_DISABLED:
        return "Timetable fallback is switched off; no realtime timing here."
    if reason == phase2.SCHEDULE_UNAVAILABLE:
        return "No calendar in the static snapshot, so no timetable fallback."
    if reason == phase2.STOP_NOT_ROUTABLE:
        return "No trusted walking route, so no arrival could be modelled."
    return "No service timing found for this stop."


def feasible_pool_name(scene: Dict[str, Any]) -> str:
    """'realtime-catchable' / 'timetable-feasible' / 'feasible (realtime + timetable)' for this trip."""
    summary = scene.get("summary") or {}
    rt, sched = summary.get("stops_realtime_catchable") or [], summary.get("stops_scheduled_only") or []
    if rt and sched:
        return "feasible (realtime + timetable)"
    if sched and not rt:
        return "timetable-feasible"
    return "realtime-catchable"


def rank_lines(scene: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    """Explicit labels: 'Overall effort rank: 2 / 34' and
    'Among realtime-catchable stops for this trip: 1 / 6'."""
    if entry.get("effort_rank") is None:
        return ["Overall effort rank: not ranked (no trusted walking route)"]
    summary = scene.get("summary") or {}
    pool_ids = set(summary.get("feasible_stops") or [])
    basis = summary.get("comparison_basis")
    ranked_all = sum(1 for s in scene["stops"] if s["effort_rank"] is not None)
    first = f"Overall effort rank: {entry['effort_rank']} / {ranked_all}"
    if entry.get("near_tie_group") is not None:
        first += f"   (near-tied within {scene['parameters']['near_tie_pct']:g} %)"
    if basis == phase2.COMPARISON_FEASIBLE and pool_ids:
        pool = sorted(s["effort_rank"] for s in scene["stops"]
                      if s["stop_id"] in pool_ids and s["effort_rank"] is not None)
        name = feasible_pool_name(scene)
        if entry["effort_rank"] in pool:
            second = f"Among {name} stops for this trip: {pool.index(entry['effort_rank']) + 1} / {len(pool)}"
        else:
            second = f"Among {name} stops for this trip: not included (not proven catchable)"
    elif basis == phase2.COMPARISON_NO_FEASIBLE_TRIP:
        served = sorted(s["effort_rank"] for s in scene["stops"]
                        if s["serves_selected_service"] and s["effort_rank"] is not None)
        second = (f"Among all served stops (no trip proven catchable): "
                  f"{served.index(entry['effort_rank']) + 1} / {len(served)}")
    elif basis == phase2.COMPARISON_EMPTY:
        second = "Among feasible stops for this trip: none (no stop feasible for the chosen trip)"
    else:
        second = ""
    return [first, second] if second else [first]


def rank_sentence(scene: Dict[str, Any], entry: Dict[str, Any]) -> str:
    """Kept for callers that want one line; the panel uses rank_lines."""
    return "  ·  ".join(rank_lines(scene, entry))


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def candidate_rows(scene: Dict[str, Any], top_n: int = TOP_N) -> List[Dict[str, Any]]:
    """Catchable stops of the selected trip first (by effort rank), then the
    other served stops with their status, then served-but-unavailable ones."""
    served = [s for s in scene["stops"] if s["serves_selected_service"]]
    ranked = sorted([s for s in served if s["effort_rank"] is not None], key=lambda s: s["effort_rank"])
    feasible = [s for s in ranked if (s.get("catchability") or {}).get("status") in phase2.FEASIBLE_STATUSES]
    others = [s for s in ranked if s not in feasible]
    unavailable = [s for s in served if s["effort_rank"] is None]
    if (scene.get("summary") or {}).get("comparison_basis") == phase2.COMPARISON_NO_FEASIBLE_TRIP:
        return ranked[:top_n] + unavailable
    return feasible[:top_n] + others[:max(0, top_n - len(feasible[:top_n]))] + unavailable


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
class Phase2Demo:
    """One window; state lives here, every number comes from the scene."""

    def __init__(self, context: Dict[str, Any], tables: Any, cost_table: pd.DataFrame,
                 study_area: Dict[str, Any], mode: str, replay_path: Optional[Path],
                 margin_s: float, schedule_fallback: bool, out_dir: Optional[Path],
                 figure=None) -> None:
        import matplotlib.pyplot as plt

        self.context = context
        self.tables = tables
        self.cost_table = cost_table
        self.study_area = study_area
        self.mode = mode.upper()
        self.replay_path = replay_path
        self.margin_s = margin_s
        self.schedule_fallback = schedule_fallback
        self.out_dir = out_dir
        self.choices = tables.choices()
        self.routes_available: List[str] = []
        for choice in self.choices:
            if choice.route_short_name not in self.routes_available:
                self.routes_available.append(choice.route_short_name)
        self.route: Optional[str] = self.routes_available[0] if self.routes_available else None
        self.direction_id: Optional[str] = None
        if self.route:
            self.direction_id = self.directions_for(self.route)[0].direction_id
        self.origin: Optional[Dict[str, float]] = None
        self.origin_label: str = ""
        self.scene: Optional[Dict[str, Any]] = None
        self.selected_stop: Optional[str] = None
        self.loaded_static: Optional[Dict[str, Any]] = None
        self.realtime: Optional[phase2.RealtimeContext] = None
        self.tech_visible = False
        self.stops = tor.load_stops()
        self.positions = plots.project(self.stops["stop_lon"].to_numpy(float),
                                       self.stops["stop_lat"].to_numpy(float))
        self.dynamic_artists: List[Any] = []
        self.route_artists: List[Any] = []
        self.panel_artists: List[Any] = []
        self.list_rows: List[Tuple[float, float, str]] = []     # (y_top, y_bottom, stop_id)
        self.trip_row_boxes: List[Tuple[float, float, str]] = []   # (y_top, y_bottom, trip_id)
        self.manual_trip_id: Optional[str] = None               # None = next feasible trip (automatic)
        self.figure = figure if figure is not None else plt.figure(figsize=FIGURE_SIZE)
        try:
            self.figure.canvas.manager.set_window_title(WINDOW_TITLE)
        except Exception:
            pass
        self.build()

    # ----- route + direction selectors ------------------------------------- #
    def directions_for(self, route_short_name: str) -> List[Any]:
        """Valid directions of one route, as service choices (human labels)."""
        return self.tables.find(route_short_name)

    @property
    def choice_key(self) -> Optional[str]:
        if self.route is None or self.direction_id is None:
            return None
        return f"{self.route}/{self.direction_id}"

    @choice_key.setter
    def choice_key(self, key: Optional[str]) -> None:
        if key is None:
            self.route = self.direction_id = None
            return
        choice = self.tables.choice(key)
        self.route, self.direction_id = choice.route_short_name, choice.direction_id

    def set_route(self, route_short_name: str) -> None:
        self.route = route_short_name
        directions = self.directions_for(route_short_name)
        if not directions:
            self.direction_id = None
        elif self.direction_id not in {d.direction_id for d in directions}:
            self.direction_id = directions[0].direction_id
        self.rebuild_direction_selector()

    def set_direction(self, direction_id: str) -> None:
        self.direction_id = str(direction_id)

    @staticmethod
    def direction_label(choice: Any) -> str:
        text = choice.route_direction or " / ".join(choice.headsigns)
        return text.replace(" to ", " -> ", 1) if " to " in text else text

    def current_choice(self) -> Optional[Any]:
        return self.tables.choice(self.choice_key) if self.choice_key else None

    # ----- realtime ---------------------------------------------------------- #
    def load_realtime(self) -> phase2.RealtimeContext:
        if self.mode != phase2.MODE_ANALYTICAL and self.loaded_static is None:
            v3 = load_script("demo_week7_realtime")
            self.loaded_static = v3.load_static_once(want_scheduled=True)
        self.realtime = phase2.load_realtime(self.mode, self.replay_path, self.loaded_static)
        return self.realtime

    # ----- layout ------------------------------------------------------------ #
    def build(self) -> None:
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.widgets import Button, RadioButtons

        fig = self.figure
        self.map_axis = fig.add_axes(MAP_RECT)
        self.draw_static_map()

        # mode banner
        self.banner_patch = FancyBboxPatch((RIGHT_X0, BANNER_Y - 0.012), RIGHT_X1 - RIGHT_X0, 0.034,
                                           boxstyle="round,pad=0.004,rounding_size=0.006",
                                           transform=fig.transFigure, facecolor=MODE_COLOURS[self.mode],
                                           edgecolor="none", zorder=1)
        fig.patches.append(self.banner_patch)
        self.mode_text = fig.text(RIGHT_X0 + 0.008, BANNER_Y + 0.005, "", fontsize=8.8, fontweight="bold",
                                  ha="left", va="center", color=COLOUR_INK, zorder=2)

        # trip intent
        fig.text(RIGHT_X0, 0.932, "TRIP INTENT", fontsize=9, fontweight="bold", color=COLOUR_INK, va="center")
        fig.text(RIGHT_X0 + 0.075, 0.932, "Route = which service.   Direction = which way it travels.",
                 fontsize=7.6, color=COLOUR_MUTED, va="center")
        self.route_axis = fig.add_axes(ROUTE_RECT)
        self.route_axis.set_title("Route", fontsize=8.4, loc="left", pad=3)
        self.route_radio = RadioButtons(self.route_axis, self.routes_available,
                                        active=self.routes_available.index(self.route) if self.route else 0)
        for text in self.route_radio.labels:
            text.set_fontsize(7.8)
        self.route_radio.on_clicked(self.on_route)
        self.direction_axis = fig.add_axes(DIRECTION_RECT)
        self.direction_radio = None
        self.rebuild_direction_selector()

        self.run_axis = fig.add_axes(RUN_RECT)
        self.run_button = Button(self.run_axis, "Run comparison")
        self.run_button.on_clicked(lambda _e: self.run(refresh=self.mode == phase2.MODE_LIVE))
        self.reset_axis = fig.add_axes(RESET_RECT)
        self.reset_button = Button(self.reset_axis, "Reset origin")
        self.reset_button.on_clicked(lambda _e: self.reset_origin())
        self.demo_axis = fig.add_axes(DEMO_RECT)
        self.demo_button = Button(self.demo_axis, "Demo preset")
        self.demo_button.on_clicked(lambda _e: self.load_demo_preset())
        self.tech_axis = fig.add_axes(TECH_RECT)
        self.tech_button = Button(self.tech_axis, "Technical details")
        self.tech_button.on_clicked(lambda _e: self.toggle_technical())
        for button in (self.run_button, self.reset_button, self.demo_button, self.tech_button):
            button.label.set_fontsize(8.2)
        self.origin_text = fig.text(ORIGIN_TEXT_XY[0], ORIGIN_TEXT_XY[1], "", fontsize=8.2,
                                    ha="left", va="top", color=COLOUR_INK)

        # trip panel (exact trips), selected stop panel and candidate list
        self.trip_axis = fig.add_axes(TRIP_PANEL_RECT)
        self.stop_axis = fig.add_axes(STOP_PANEL_RECT)
        self.list_axis = fig.add_axes(LIST_PANEL_RECT)
        self.auto_axis = fig.add_axes(AUTO_RECT)
        self.auto_button = Button(self.auto_axis, "Next feasible trip")
        self.auto_button.on_clicked(lambda _e: self.select_trip(None))
        self.auto_button.label.set_fontsize(7.6)
        for axis in (self.trip_axis, self.stop_axis, self.list_axis):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor(COLOUR_PANEL)
            for spine in axis.spines.values():
                spine.set_edgecolor("#dddddd")
        self.tech_panel_axis = fig.add_axes(TECH_PANEL_RECT)
        self.tech_panel_axis.set_axis_off()
        self.tech_panel_axis.set_visible(False)
        self.tech_text = self.tech_panel_axis.text(
            0.01, 0.98, "", ha="left", va="top", fontsize=7.2, family="monospace",
            transform=self.tech_panel_axis.transAxes, zorder=30,
            bbox={"facecolor": "#ffffff", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.5", "alpha": 0.97})

        fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.show_mode_text()
        self.show_origin_text()
        self.draw_panels()

    def rebuild_direction_selector(self) -> None:
        from matplotlib.widgets import RadioButtons

        self.direction_axis.clear()
        self.direction_axis.set_title("Direction", fontsize=8.4, loc="left", pad=3)
        directions = self.directions_for(self.route) if self.route else []
        labels = [self.direction_label(d) for d in directions] or ["(no direction)"]
        active = 0
        for index, d in enumerate(directions):
            if d.direction_id == self.direction_id:
                active = index
        self.direction_radio = RadioButtons(self.direction_axis, labels, active=active)
        for text in self.direction_radio.labels:
            text.set_fontsize(7.8)
        self.direction_radio.on_clicked(self.on_direction)
        self.figure.canvas.draw_idle()

    def draw_static_map(self) -> None:
        from matplotlib.collections import LineCollection

        axis = self.map_axis
        left, bottom, right, top = self.study_area["bounds_metric"]
        margin = plots.MAP_MARGIN_M
        segments = plots.context_polylines(self.context["graph"], (left, bottom, right, top), margin)
        axis.add_collection(LineCollection(segments, colors="#e9e9e9", linewidths=0.55, zorder=1))
        polygon = self.study_area["metric"]
        for ring in [polygon.exterior] + list(polygon.interiors):
            xs, ys = ring.xy
            axis.plot(xs, ys, color="#a8a8a8", linewidth=1.0, linestyle=(0, (6, 3)), zorder=2)
        axis.set_xlim(left - margin, right + margin)
        axis.set_ylim(bottom - margin, top + margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#cccccc")
        span = (right + margin) - (left - margin)
        length = plots.nice_scale_length(span)
        bar_y = (bottom - margin) + 0.04 * ((top + margin) - (bottom - margin))
        bar_right = (right + margin) - 0.05 * span
        axis.plot([bar_right - length, bar_right], [bar_y, bar_y], color=COLOUR_INK,
                  linewidth=2.2, solid_capstyle="butt", zorder=9)
        axis.text(bar_right - length / 2.0, bar_y + 0.012 * ((top + margin) - (bottom - margin)),
                  f"{length:g} m", ha="center", va="bottom", fontsize=8, color=COLOUR_INK)
        self.hint_text = axis.text(0.012, 0.988, "Click the map to set your origin", transform=axis.transAxes,
                                   ha="left", va="top", fontsize=9, fontweight="bold", color=COLOUR_INK,
                                   bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc",
                                         "boxstyle": "round,pad=0.4", "alpha": 0.93}, zorder=10)
        self.legend_artist = None
        self.draw_stops(None)

    # ----- map drawing from the scene -------------------------------------- #
    def clear_dynamic(self) -> None:
        for artist in self.dynamic_artists + self.route_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self.dynamic_artists = []
        self.route_artists = []
        if self.legend_artist is not None:
            try:
                self.legend_artist.remove()
            except (ValueError, AttributeError):
                pass
            self.legend_artist = None

    def stop_entry(self, stop_id: Any) -> Optional[Dict[str, Any]]:
        if self.scene is None:
            return None
        return next((s for s in self.scene["stops"] if s["stop_id"] == str(stop_id)), None)

    def draw_stops(self, scene: Optional[Dict[str, Any]]) -> None:
        from matplotlib.lines import Line2D

        axis = self.map_axis
        self.clear_dynamic()
        handles: List[Any] = []
        for index, stop in self.stops.iterrows():
            stop_id = str(stop["stop_id"])
            marker = MARKER_BY_MODE.get(str(stop["mode"]), "o")
            x, y = self.positions[index]
            entry = next((s for s in scene["stops"] if s["stop_id"] == stop_id), None) if scene else None
            if entry is None:
                # no scene yet: every candidate stop, quietly
                self.dynamic_artists.append(axis.scatter(
                    [x], [y], marker=marker, s=30, facecolor="#ffffff", edgecolor="#8a8a8a",
                    linewidth=0.9, zorder=6))
                continue
            if not entry["serves_selected_service"]:
                # context only - very faint, never competing
                self.dynamic_artists.append(axis.scatter(
                    [x], [y], marker=marker, s=14, facecolor=COLOUR_STOP_OUT, edgecolor="none", zorder=4))
                continue
            if entry["eligibility"] != tor.STATUS_COMPARED:
                self.dynamic_artists.append(axis.scatter(
                    [x], [y], marker="x", s=55, color=COLOUR_UNAVAILABLE, linewidth=1.5, zorder=7))
                self.dynamic_artists.append(axis.annotate(
                    "no trusted route", (x, y), xytext=(7, -3), textcoords="offset points",
                    fontsize=6.6, color=COLOUR_UNAVAILABLE, zorder=10))
                continue
            highlights = set(entry["highlights"])
            status = (entry.get("catchability") or {}).get("status")
            # the stop itself: its catchability for the ONE selected trip (fill + edge + hatch)
            if status == phase2.CATCHABLE:
                style = {"facecolor": FILL_CATCHABLE, "edgecolor": COLOUR_STOP_IN, "linewidth": 1.5, "s": 70}
            elif status == phase2.SCHEDULED_AFTER:
                style = {"facecolor": FILL_SCHEDULE_ONLY, "edgecolor": COLOUR_STOP_IN, "linewidth": 1.2,
                         "s": 70, "hatch": "////"}
            elif status == phase2.NOT_PROVEN:
                style = {"facecolor": FILL_NOT_PROVEN, "edgecolor": "#8a8a8a", "linewidth": 1.0, "s": 52}
            else:
                style = {"facecolor": FILL_NO_TIME, "edgecolor": "#8a8a8a", "linewidth": 1.0, "s": 52,
                         "linestyle": ":"}
            self.dynamic_artists.append(axis.scatter([x], [y], marker=marker, zorder=7, **style))
            # Pareto: a subtle outer ring only
            if "pareto_non_dominated" in highlights:
                self.dynamic_artists.append(axis.scatter(
                    [x], [y], marker="o", s=260, facecolor="none", edgecolor=COLOUR_PARETO,
                    linewidth=1.2, zorder=6))
            # at most ONE walking-winner marker per stop (effort > time > distance);
            # the stop panel lists every objective the stop wins
            winner = winner_marker(highlights)
            if winner is not None:
                marker_, colour, size = winner
                self.dynamic_artists.append(axis.scatter(
                    [x], [y], marker=marker_, s=size, facecolor=colour, edgecolor="#000000",
                    linewidth=0.7, zorder=8))
        # origin
        origin = (scene or {}).get("origin") or self.origin
        if origin:
            ox, oy = plots.project([origin["lon"]], [origin["lat"]])[0]
            self.dynamic_artists.append(axis.scatter(
                [ox], [oy], marker="P", s=170, facecolor=COLOUR_ORIGIN, edgecolor="#000000",
                linewidth=0.9, zorder=11))
            self.dynamic_artists.append(axis.annotate(
                "Origin", (ox, oy), xytext=(8, -12), textcoords="offset points", fontsize=7.6,
                fontweight="bold", color=COLOUR_ORIGIN, zorder=11))
        self.draw_selected_routes()
        # legend: feasibility (fill), one walking-winner marker, the Pareto ring,
        # the three routes. Origin and the selected stop are labelled on the map itself.
        self.legend_artist = axis.legend(handles=legend_handles(), loc="lower left", fontsize=6.9,
                                         framealpha=0.94, handlelength=2.2, handletextpad=0.6,
                                         labelspacing=0.34, borderpad=0.5, title="Map key",
                                         title_fontsize=7.6)

    def draw_selected_routes(self) -> None:
        """All three routes, layered (shortest widest and lowest, fastest thinnest
        and on top) - the same convention as the primary window, so a shared path
        stays visible as nested lines and needs no extra legend entry. Which
        objectives share a path is still said in the panel (``route_identity``)."""
        if self.scene is None or self.selected_stop is None:
            return
        entry = self.stop_entry(self.selected_stop)
        if entry is None or entry["eligibility"] != tor.STATUS_COMPARED:
            return
        graph = self.context["graph"]
        axis = self.map_axis
        routes = entry.get("routes") or {}
        for objective, (colour, style, width, zorder) in ROUTE_LAYERS.items():
            route = routes.get(objective)
            if route is None:
                continue
            coords = phase2.route_linestring(graph, route["edge_keys"])
            xy = plots.project([c[0] for c in coords], [c[1] for c in coords])
            line, = axis.plot(xy[:, 0], xy[:, 1], color=colour, linestyle=style, linewidth=width,
                              zorder=zorder, solid_capstyle="round", solid_joinstyle="round",
                              label=OBJECTIVE_LABELS[objective])
            self.route_artists.append(line)
        index = self.stops.index[self.stops.stop_id.astype(str) == self.selected_stop][0]
        x, y = self.positions[index]
        self.route_artists.append(axis.scatter(
            [x], [y], marker="o", s=360, facecolor="none", edgecolor="#000000", linewidth=1.6,
            linestyle="--", zorder=12))
        self.route_artists.append(axis.annotate(
            f"Selected: {entry['stop_name']}", (x, y), xytext=(10, -16), textcoords="offset points",
            fontsize=8.0, fontweight="bold", color="#000000", zorder=12,
            bbox={"facecolor": "#ffffff", "edgecolor": "#000000", "boxstyle": "round,pad=0.3", "alpha": 0.95}))

    # ----- text panels ----------------------------------------------------- #
    def show_mode_text(self) -> None:
        self.mode_text.set_text(mode_banner(self.scene, self.mode, self.replay_path))
        self.banner_patch.set_facecolor(MODE_COLOURS[self.mode])

    def show_origin_text(self) -> None:
        if self.origin is None:
            text = "Origin: not set - click the map"
        elif self.origin_label:
            text = f"Origin selected: {self.origin_label}"
        else:
            text = "Origin selected (click the map again to move it)"
        self.origin_text.set_text(text)

    def clear_panels(self) -> None:
        for artist in self.panel_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self.panel_artists = []
        self.list_rows = []

    def _text(self, axis, x: float, y: float, text: str, **kwargs) -> Any:
        defaults = {"ha": "left", "va": "top", "fontsize": 8.2, "color": COLOUR_INK,
                    "transform": axis.transAxes}
        defaults.update(kwargs)
        artist = axis.text(x, y, text, **defaults)
        self.panel_artists.append(artist)
        return artist

    def draw_panels(self) -> None:
        self.clear_panels()
        self.draw_trip_panel()
        self.draw_stop_panel()
        self.draw_list_panel()
        if self.tech_visible:
            self.tech_text.set_text(self.technical_text())
        self.figure.canvas.draw_idle()

    def draw_trip_panel(self) -> None:
        """Which exact trip the stops are judged against; click a row to choose another."""
        axis = self.trip_axis
        self.trip_row_boxes = []
        self._text(axis, 0.02, 0.95, "EXACT TRIP", fontsize=8.6, fontweight="bold")
        self._text(axis, 0.19, 0.95, "catchability is a constraint, not an objective",
                   fontsize=6.9, color=COLOUR_MUTED)
        if self.scene is None:
            self._text(axis, 0.02, 0.74, "Upcoming exact trips of the chosen route + direction appear here after "
                                         "you run the comparison.", color=COLOUR_MUTED)
            return
        self._text(axis, 0.02, 0.77, selected_trip_text(self.scene, self.mode), fontsize=7.8,
                   fontweight="bold")
        rows = trip_rows(self.scene)
        labels = trip_display_labels(self.scene.get("trip_candidates") or [])
        y = 0.56
        step = 0.135
        for row in rows:
            marker = "▶ " if row.get("is_selected") else "  "
            if row.get("is_next_feasible"):
                tag = ("   (next feasible)" if trip_basis(row) != phase2.BASIS_SCHEDULE
                       else "   (next timetable-feasible)")
            else:
                tag = ""
            self._text(axis, 0.02, y, marker + trip_row_text(row, labels.get(row["trip_id"])) + tag,
                       family="monospace", fontsize=7.2,
                       fontweight="bold" if row.get("is_selected") else "normal",
                       color=COLOUR_INK if row["n_feasible_stops"] else COLOUR_MUTED)
            self.trip_row_boxes.append((y, y - step, row["trip_id"]))
            y -= step
        info = self.scene.get("trip_selection") or {}
        more = info.get("trips_beyond_horizon") or 0
        hidden = max(0, (info.get("trips_listed") or 0) - len(rows))
        foot = "click a trip to judge the stops against it"
        if hidden or more:
            foot += f"  ·  {hidden + more} later trip(s) not shown"
        self._text(axis, 0.02, 0.02, foot, fontsize=6.8, color=COLOUR_MUTED, va="bottom")

    def draw_stop_panel(self) -> None:
        axis = self.stop_axis
        self._text(axis, 0.02, 0.965, "SELECTED STOP", fontsize=8.6, fontweight="bold")
        if self.scene is None:
            self._text(axis, 0.02, 0.86, "Set an origin, choose a route and direction, then run the comparison.",
                       fontsize=8.6, color=COLOUR_MUTED, wrap=True)
            self._text(axis, 0.02, 0.72, "Or press  Demo preset  to see a real case where the shortest,\n"
                                         "lowest-effort and fastest walking routes are three different paths.",
                       fontsize=8.2, color=COLOUR_MUTED)
            return
        entry = self.stop_entry(self.selected_stop) if self.selected_stop else None
        if entry is None:
            self._text(axis, 0.02, 0.86, "Click a stop on the map or a row below.", color=COLOUR_MUTED)
            return
        self._text(axis, 0.02, 0.91, f"{entry['stop_name']}", fontsize=10.5, fontweight="bold")
        self._text(axis, 0.02, 0.845, f"stop {entry['stop_id']}  ·  {'light rail' if entry['mode'] == 'light_rail' else 'bus'}",
                   fontsize=7.4, color=COLOUR_MUTED)
        if entry["eligibility"] != tor.STATUS_COMPARED:
            self._text(axis, 0.02, 0.72, "No trusted walking route from this origin.", fontsize=9)
            self._text(axis, 0.02, 0.62, "A network-coverage fact, not a statement about the stop.",
                       fontsize=8, color=COLOUR_MUTED)
            return
        # this stop against the ONE selected trip (the constraint) - first, because it decides the rest
        y = 0.785
        self._text(axis, 0.02, y, "THIS TRIP AT THIS STOP", fontsize=7.8, fontweight="bold")
        y -= 0.052
        for line in trip_at_stop_lines(entry, self.scene, self.mode)[:4]:
            self._text(axis, 0.02, y, line, fontsize=7.5)
            y -= 0.047
        y -= 0.008
        # three-row walking comparison
        for index, line in enumerate(route_table_lines(entry)):
            self._text(axis, 0.02, y, line, family="monospace", fontsize=7.5,
                       fontweight="bold" if index == 0 else "normal",
                       color=COLOUR_MUTED if index == 0 else COLOUR_INK)
            y -= 0.047
        saving, extra = entry.get("effort_saving_vs_shortest_pct"), entry.get("extra_distance_vs_shortest_pct")
        if saving is not None and saving > 0.05:
            self._text(axis, 0.02, y, f"Effort saving vs shortest route: {fmt_pct(saving)}      "
                                      f"Extra distance: {fmt_pct(extra)}", fontsize=7.5)
            y -= 0.045
        self._text(axis, 0.02, y, route_identity_text(entry), fontsize=7.4, color=COLOUR_MUTED)
        y -= 0.056
        # stop comparison (among the feasible stops of the trip)
        self._text(axis, 0.02, y, "STOP COMPARISON", fontsize=7.8, fontweight="bold")
        y -= 0.05
        for line in rank_lines(self.scene, entry):
            self._text(axis, 0.02, y, line, fontsize=7.5)
            y -= 0.046
        marks = comparison_marks(entry)
        self._text(axis, 0.02, y, "   ".join(("✓ " if on else "○ ") + label for on, label in marks),
                   fontsize=7.8)

    def draw_list_panel(self) -> None:
        axis = self.list_axis
        if self.scene is None:
            self._text(axis, 0.02, 0.94, "STOPS FOR THIS TRIP", fontsize=8.6, fontweight="bold")
            self._text(axis, 0.02, 0.74, "Run a comparison to list the stops served by the chosen service.",
                       color=COLOUR_MUTED)
            return
        intent = self.scene["intent"]
        summary = self.scene["summary"]
        served = summary["candidates_serving_selected_service"]
        chosen = self.scene.get("selected_trip")
        if chosen is not None:
            counts = {"n_catchable_stops": len(summary.get("stops_realtime_catchable") or []),
                      "n_schedule_only_stops": len(summary.get("stops_scheduled_only") or [])}
            title = (f"STOPS FOR THE {clock(chosen.get('reference_sydney'))} TRIP   "
                     f"({feasibility_count_text(counts)} of {served} served; lowest effort first)")
        else:
            title = (f"STOPS SERVED BY {intent['route_short_name']} -> {(intent['headsigns'] or [''])[0]}   "
                     f"({served} stops; no trip proven catchable)")
        self._text(axis, 0.02, 0.93, title, fontsize=8.0, fontweight="bold")
        header = f"{'Overall rank':>12s}  {'Stop':26s}{'Effort':>10s}{'Walk':>8s}{'Dist':>8s}{'Slack':>15s}"
        self._text(axis, 0.02, 0.77, header, family="monospace", fontsize=7.4, color=COLOUR_MUTED)
        y = 0.62
        step = 0.135
        for row in candidate_rows(self.scene)[:4]:
            selected = row["stop_id"] == self.selected_stop
            catch = row.get("catchability") or {}
            feasible = catch.get("status") in phase2.FEASIBLE_STATUSES
            if row["effort_rank"] is None:
                line = (f"{'-':>12s}  {(row['stop_id'] + ' ' + row['stop_name'])[:26]:26s}"
                        f"{'no trusted route':>41s}")
            else:
                tag = " *" if "lowest_terrain_effort" in row["highlights"] else ""
                slack = catch.get("slack_s")
                slack_text = fmt_slack(slack) if slack is not None else "-"
                if catch.get("status") == phase2.SCHEDULED_AFTER:
                    slack_text = "tt " + fmt_slack(slack)
                elif not feasible:
                    slack_text = ("✗ " + fmt_slack(slack)) if slack is not None else "no timing"
                line = (f"{row['effort_rank']:>12d}  {(row['stop_id'] + ' ' + row['stop_name'])[:24] + tag:26s}"
                        f"{fmt_effort(row['effort_j_per_kg']):>10s}{fmt_min(row['time_route_walking_time_s']):>8s}"
                        f"{fmt_m(row['distance_m']):>8s}{slack_text:>15s}")
            self._text(axis, 0.02, y, ("▶ " if selected else "  ") + line, family="monospace", fontsize=7.4,
                       fontweight="bold" if selected else "normal",
                       color=COLOUR_INK if (feasible or row["effort_rank"] is None
                                            or summary.get("comparison_basis") == phase2.COMPARISON_NO_FEASIBLE_TRIP)
                       else COLOUR_MUTED)
            self.list_rows.append((y, y - step, row["stop_id"]))
            y -= step
        ranked_all = sum(1 for s in self.scene["stops"] if s["effort_rank"] is not None)
        self._text(axis, 0.02, 0.02, f"Overall rank = Phase 1 effort rank among all {ranked_all} candidates · "
                                     f"* lowest effort among feasible · slack = trip arrival - your modelled "
                                     f"arrival · tt = timetable only · ✗ = not proven",
                   fontsize=6.2, color=COLOUR_MUTED, va="bottom")

    def technical_text(self) -> str:
        if self.scene is None:
            return "No scene yet."
        s = self.scene
        o = s["origin"]
        chosen = s.get("selected_trip") or {}
        lines = [
            "TECHNICAL DETAILS (scene = results/current_scene.json)",
            f"mode {s['mode']}   realtime status {s['realtime'].get('status')}   "
            f"clock anchor {s['realtime'].get('clock_anchor')}   observed_at {s['realtime'].get('observed_at')} "
            f"= {s['realtime'].get('observed_at_sydney')}",
            f"origin {o['lat']:.6f}, {o['lon']:.6f}  snapped to node {o['nearest_node']} "
            f"({o['snap_distance_m']} m off the pedestrian network)",
            f"intent key {s['intent']['key']}  route_id {s['intent']['route_id']}  "
            f"direction_id {s['intent']['direction_id']}",
            f"margin {s['parameters']['margin_s']} s   near-tie {s['parameters']['near_tie_pct']} %   "
            f"service date {s['parameters']['service_date']}   trip horizon {s['parameters'].get('trip_horizon')}",
            f"timed trusted edges {s['provenance']['routing']['timed_subgraph_edges']} / "
            f"{s['provenance']['routing']['trusted_subgraph_edges']}",
            "",
            f"selected trip_id {chosen.get('trip_id') or '-'}  selection {chosen.get('selection') or '-'}  "
            f"reference {chosen.get('reference_sydney') or '-'} at stop {chosen.get('reference_stop_id') or '-'}  "
            f"times {chosen.get('expected_time_source') or '-'}",
            f"comparison basis {s['summary'].get('comparison_basis')}   feasible stops "
            f"{', '.join(s['summary'].get('feasible_stops') or []) or '-'}",
            "upcoming exact trips (chronological; > selected, n next feasible):",
        ]
        labels = trip_display_labels(s.get("trip_candidates") or [])
        for c in (s.get("trip_candidates") or [])[:8]:
            mark = ">" if c["is_selected"] else ("n" if c["is_next_feasible"] else " ")
            shown = "".join(labels.get(c["trip_id"], ("", "")))
            lines.append(f"  {mark} {c['trip_id']:28s} shown as {shown:16s} ref {clock(c['reference_sydney'])} @ {c['reference_stop_id']}  "
                         f"feasible {c['n_feasible_stops']} (rt {c['n_catchable_stops']}, sched {c['n_schedule_only_stops']}) "
                         f"not proven {c['n_not_proven_stops']}  [{c['basis']}]")
        entry = self.stop_entry(self.selected_stop) if self.selected_stop else None
        if entry:
            c = entry.get("catchability") or {}
            lines += [
                "",
                f"selected stop {entry['stop_id']}  eligibility {entry['eligibility']} "
                f"{entry['status_reason']}".rstrip(),
                f"  route edges: distance {entry['routes']['distance'] and entry['routes']['distance']['edges']}, "
                f"effort {entry['routes']['effort'] and entry['routes']['effort']['edges']}, "
                f"time {entry['routes']['time'] and entry['routes']['time']['edges']}",
                f"  catchability status {c.get('status')}  reason {c.get('reason') or '-'}",
                f"  realtime status {c.get('realtime_status') or '-'}  source {c.get('expected_time_source') or '-'}",
                "  manual check (epoch seconds):",
                f"    observed_at                 {c.get('observed_at')}  = {c.get('observed_at_sydney') or '-'}",
                f"    + time-route walking time   {c.get('time_route_walking_time_s')} s   + margin {c.get('margin_s')} s",
                f"    = pedestrian arrival        {c.get('pedestrian_arrival_epoch')}  = {c.get('pedestrian_arrival_sydney') or '-'}",
                f"    trip expected arrival       {c.get('trip_expected_arrival_epoch')}  = {c.get('trip_expected_arrival_sydney') or '-'}",
                f"    slack = trip - pedestrian   {c.get('slack_s')} s",
                f"  pareto {entry['pareto_status']}  dominated_by {entry['dominated_by'] or '-'}",
            ]
            if c.get("trips"):
                lines.append("  trips with a time at this stop: " + "; ".join(
                    f"{t['trip_id']} {clock(t['expected_arrival_sydney'])} slack {t['slack_s']:+d} s {t['status']}"
                    for t in c["trips"][:6]))
        lines += ["", "Model notes:"] + [textwrap.fill(m, 95, initial_indent="  - ", subsequent_indent="    ")
                                         for m in s["model_statements"]]
        return "\n".join(lines)

    def toggle_technical(self) -> None:
        self.tech_visible = not self.tech_visible
        self.tech_panel_axis.set_visible(self.tech_visible)
        self.tech_text.set_text(self.technical_text() if self.tech_visible else "")
        self.tech_button.label.set_text("Hide details" if self.tech_visible else "Technical details")
        self.figure.canvas.draw_idle()

    # ----- events ------------------------------------------------------------ #
    def hit_test_stop(self, event) -> Optional[str]:
        if event.inaxes is not self.map_axis:
            return None
        pixels = self.map_axis.transData.transform(self.positions)
        distances = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        index = int(np.argmin(distances))
        if distances[index] <= STOP_PICK_RADIUS_PX:
            return str(self.stops.iloc[index]["stop_id"])
        return None

    def hit_test_row(self, event) -> Optional[str]:
        if event.inaxes is not self.list_axis or self.scene is None:
            return None
        _x, y = self.list_axis.transAxes.inverted().transform((event.x, event.y))
        for top, bottom, stop_id in self.list_rows:
            if bottom <= y <= top:
                return stop_id
        return None

    def hit_test_trip_row(self, event) -> Optional[str]:
        if event.inaxes is not self.trip_axis or self.scene is None:
            return None
        _x, y = self.trip_axis.transAxes.inverted().transform((event.x, event.y))
        for top, bottom, trip_id in self.trip_row_boxes:
            if bottom <= y <= top:
                return trip_id
        return None

    def on_click(self, event) -> None:
        if event.button != 1:
            return
        row = self.hit_test_row(event)
        if row is not None:
            self.select_stop(row)
            return
        trip = self.hit_test_trip_row(event)
        if trip is not None:
            self.select_trip(trip)
            return
        if event.inaxes is not self.map_axis:
            return
        stop_id = self.hit_test_stop(event)
        if stop_id is not None:
            self.select_stop(stop_id)
            return
        latitude, longitude = week5.metric_to_wgs84(event.xdata, event.ydata)
        if not week5.inside_study_area(self.study_area, latitude, longitude):
            self.hint_text.set_text("Please click inside the Glebe study area")
            self.figure.canvas.draw_idle()
            return
        self.set_origin(latitude, longitude)
        if self.choice_key is not None:
            self.run()

    def on_route(self, label: str) -> None:
        self.set_route(str(label).strip())
        self.manual_trip_id = None
        if self.origin is not None:
            self.run()

    def on_direction(self, label: str) -> None:
        for d in self.directions_for(self.route) if self.route else []:
            if self.direction_label(d) == label:
                self.set_direction(d.direction_id)
                break
        self.manual_trip_id = None
        if self.origin is not None:
            self.run()

    def set_origin(self, latitude: float, longitude: float, label: str = "") -> None:
        self.origin = {"lat": float(latitude), "lon": float(longitude)}
        self.origin_label = label
        self.scene = None
        self.selected_stop = None
        self.hint_text.set_text("Origin set - choose route + direction and run, or click again to move it")
        self.show_origin_text()
        self.draw_stops(None)
        self.draw_panels()

    def reset_origin(self) -> None:
        self.origin = None
        self.origin_label = ""
        self.scene = None
        self.selected_stop = None
        self.hint_text.set_text("Click the map to set your origin")
        self.show_mode_text()
        self.show_origin_text()
        self.draw_stops(None)
        self.draw_panels()

    def select_stop(self, stop_id: str) -> None:
        self.selected_stop = str(stop_id)
        self.draw_stops(self.scene)
        self.draw_panels()

    def select_trip(self, trip_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Judge the stops against another exact trip (None = next feasible trip).

        The engine re-derives catchability, highlights and Pareto from the
        scene's existing inventory; nothing is re-routed and no feed is read.
        """
        self.manual_trip_id = str(trip_id) if trip_id is not None else None
        if self.scene is None:
            return None
        phase2.apply_trip_selection(self.scene, self.manual_trip_id)
        geojson = phase2.scene_to_geojson(self.scene, self.context["graph"])
        phase2.write_scene(self.scene, geojson, self.out_dir)
        self.default_selected_stop()
        self.draw_stops(self.scene)
        self.draw_panels()
        return self.scene

    def default_selected_stop(self) -> None:
        """Keep the viewer's stop; otherwise the lowest-effort catchable stop, else the first served."""
        if self.selected_stop is not None and self.stop_entry(self.selected_stop) is not None:
            return
        summary = self.scene["summary"]
        self.selected_stop = summary.get("lowest_terrain_effort_stop")
        if self.selected_stop is None:
            rows = candidate_rows(self.scene)
            self.selected_stop = rows[0]["stop_id"] if rows else None

    def load_demo_preset(self) -> Optional[Dict[str, Any]]:
        preset = DEMO_PRESET
        self.choice_key = preset["service"]
        self.route_radio.set_active(self.routes_available.index(self.route))
        self.rebuild_direction_selector()
        self.set_origin(preset["lat"], preset["lon"], f"{preset['label']} - {preset['origin_label']}")
        self.manual_trip_id = None
        scene = self.run()
        if scene is not None:
            self.select_stop(preset["stop"])
        return scene

    # ----- the run: engine -> scene -> export -> draw ------------------------ #
    def run(self, refresh: bool = False) -> Optional[Dict[str, Any]]:
        if self.origin is None:
            self.hint_text.set_text("Set an origin first: click the map")
            self.figure.canvas.draw_idle()
            return None
        if self.choice_key is None:
            return None
        if self.realtime is None or refresh:
            self.load_realtime()
        trip_id = self.manual_trip_id
        try:
            scene = phase2.run_phase2(
                self.context, self.tables, self.origin["lat"], self.origin["lon"], self.choice_key,
                self.realtime, self.cost_table, margin_s=self.margin_s,
                schedule_fallback=self.schedule_fallback, stops=self.stops,
                origin_label=self.origin_label, trip_id=trip_id)
        except phase2.Phase2Error:
            # the manually chosen trip is no longer listed (new origin / snapshot): back to automatic
            self.manual_trip_id = None
            scene = phase2.run_phase2(
                self.context, self.tables, self.origin["lat"], self.origin["lon"], self.choice_key,
                self.realtime, self.cost_table, margin_s=self.margin_s,
                schedule_fallback=self.schedule_fallback, stops=self.stops,
                origin_label=self.origin_label, trip_id=None)
        geojson = phase2.scene_to_geojson(scene, self.context["graph"])
        phase2.write_scene(scene, geojson, self.out_dir)
        self.scene = scene
        self.default_selected_stop()
        self.hint_text.set_text("Click a stop to compare its three walking routes · click a trip to change the trip · "
                                "click elsewhere to move the origin")
        self.show_mode_text()
        self.show_origin_text()
        self.draw_stops(scene)
        self.draw_panels()
        return scene


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default="analytical",
                        choices=["live", "replay", "analytical", "LIVE", "REPLAY", "ANALYTICAL"])
    parser.add_argument("--replay", default=None, help="saved snapshot .pb for REPLAY mode")
    parser.add_argument("--origin", nargs=2, type=float, metavar=("LAT", "LON"), default=None)
    parser.add_argument("--service", default=None, help="route/direction key, e.g. 433/1")
    parser.add_argument("--stop", default=None, help="initial selected stop_id")
    parser.add_argument("--demo", action="store_true",
                        help="load the documented demo preset at start (never automatic)")
    parser.add_argument("--no-demo", action="store_true", help=argparse.SUPPRESS)   # kept for old commands
    parser.add_argument("--margin", type=float, default=phase2.DEFAULT_MARGIN_S)
    parser.add_argument("--no-schedule-fallback", action="store_true")
    parser.add_argument("--trip", default=None, help="exact trip_id to judge the stops against (default: next feasible)")
    parser.add_argument("--cost-dir", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--out", default=None, help="scene output directory (default results/)")
    parser.add_argument("--save", default=None, help="render once and save a PNG (no window)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = args.mode.upper()
    if mode == phase2.MODE_REPLAY and not args.replay:
        print("--mode replay needs --replay PATH.pb (see scripts/capture_realtime_snapshot.py)",
              file=sys.stderr)
        return 2
    tables = intent_mod.ServiceIntentTables.load()
    cost_dir = Path(args.cost_dir) if args.cost_dir else tor.cmp.cost_dir()
    cost_table = tor.cmp.load_cost_table(cost_dir)
    context = tor.build_context(cost_table, Path(args.graph) if args.graph else None)
    study_area = week5.load_study_area()
    demo = Phase2Demo(context, tables, cost_table, study_area, mode,
                      Path(args.replay) if args.replay else None, args.margin,
                      not args.no_schedule_fallback, Path(args.out) if args.out else None)
    if args.service:
        demo.choice_key = args.service          # validates the key
        demo.route_radio.set_active(demo.routes_available.index(demo.route))
        demo.rebuild_direction_selector()
    if args.trip:
        demo.manual_trip_id = args.trip
    if args.origin:
        demo.set_origin(args.origin[0], args.origin[1])
        demo.run()
        if args.stop:
            demo.select_stop(args.stop)
    elif args.demo and not args.no_demo:
        demo.load_demo_preset()
        if args.stop:
            demo.select_stop(args.stop)
    if args.save:
        demo.figure.savefig(args.save, dpi=130)
        print(f"saved {args.save}")
        return 0
    print("A window is open: click the map to set an origin, pick Route and Direction, "
          "run the comparison, then click a stop (or another exact trip). Close the window to finish.")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
