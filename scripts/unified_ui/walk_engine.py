"""The adapter between the unified window and the validated engine.

Reader's summary (cleanup pass 2)
---------------------------------
Responsibility  WALK: snap an origin, run the three route searches, solve one stop,
                and hand the window a result it can only format.
Inputs          ``data/cache/terrain_cost/edge_terrain_cost.csv`` (the trusted cost table),
                ``data/cache/glebe_walk.graphml`` (the routing graph), ``data/cache/
                glebe_stops_snapped.csv`` (the 37 candidate stops), an origin (lat, lon),
                a ``stop_id``.
Outputs         ``solve()`` -> one dict per stop: ``routes`` (distance / effort / time, each
                with its own distance, walking time, effort and edge geometry),
                ``route_identity``, the two "vs shortest" percentages; ``scene_entry()``
                for the sandbox scene.
Depends on      ``scripts/time_optimal_route.py`` (build_context / search_all / solve_stop),
                ``scripts/demo_terrain_route.py`` (snap_origin_coordinate -> src.snapping),
                ``scripts/phase2_decision.py`` (route_identity / route_linestring) - all
                loaded by file path through ``load_script``; the Standard profile is a
                second context in ``standard_hybrid.py`` built on top of this one.
Do not infer    that any number is computed here (none is), that the three routes share
                a metric (each reports its own triplet), or that the walking time is
                measured (it is Tobler, modelled). No combined score exists anywhere.

Nothing here routes, snaps, integrates Minetti or Tobler, or classifies
paths. Every number comes from:

* ``demo_terrain_route.snap_origin_coordinate``  (Week 5; src.snapping)
* ``time_optimal_route.build_context / search_all / solve_stop``
  (Week 9; the comparison script's own lexicographic Dijkstra for the
  distance and effort routes, the time route on the same trusted subgraph)
* ``phase2_decision.route_identity / route_linestring`` (Week 9)

``solve()`` returns the same result shape ``demo_week9_primary.ToblerLayer``
builds (routes, route_identity, the two "vs shortest" percentages), so the
regression tests can compare the two views field by field.

The text helpers at the bottom are pure functions over that result; they
format, they never compute a new analytical quantity.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_script(name: str):
    """Load ``scripts/<name>.py`` once, sharing the instance other callers made."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tor = load_script("time_optimal_route")        # three routes (imports compare_terrain_routes)
phase2 = load_script("phase2_decision")        # route_identity, route_linestring, route_summary
week5 = load_script("demo_terrain_route")      # snapping, study area, click guard
plots = week5.plots                            # projection, context polylines, scale bar

OBJECTIVES = ("distance", "effort", "time")
HUMAN_NAMES = {"distance": "Shortest", "effort": "Gentlest", "time": "Fastest"}
TECHNICAL_NAMES = {"distance": "distance-optimal", "effort": "effort-optimal", "time": "time-optimal"}
NOTE_TIME = "Walking time is modelled, not measured."
NOTE_EFFORT = "Terrain effort is a modelled terrain-related walking cost."
MODEL_CAVEATS = [
    "Walking time: Tobler (1993) hiking function integrated over 5 m DEM gradient cells "
    "along the time-optimal route on the trusted subgraph - a modelled minimum.",
    "Terrain effort: Minetti et al. (2002) cost of walking integrated the same way along "
    "the effort-optimal route (J/kg). No accessibility validation is claimed.",
    "Distance, effort and time are three separate objectives; nothing combines them.",
    "The origin is snapped to the nearest graph node (src.snapping); the snap gap is not "
    "part of any route distance.",
]


class WalkEngine:
    """Build once, then: set_origin() per origin click, solve() per stop click."""

    def __init__(self, context: Dict[str, Any], stops: pd.DataFrame,
                 study_area: Dict[str, Any], cost_table: pd.DataFrame) -> None:
        self.context = context
        self.stops = stops.copy()
        self.stops["stop_id"] = self.stops["stop_id"].astype(str)
        self.study_area = study_area
        self.cost_table = cost_table
        self.positions = plots.project(self.stops["stop_lon"].to_numpy(float),
                                       self.stops["stop_lat"].to_numpy(float))
        self.stop_ids: List[str] = list(self.stops["stop_id"])
        self.load_seconds: Optional[float] = None

    # ----- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, cost_dir: Optional[Path] = None, graph_path: Optional[Path] = None,
             stops_path: Optional[Path] = None) -> "WalkEngine":
        started = time.perf_counter()
        cost_table = tor.cmp.load_cost_table(Path(cost_dir) if cost_dir else tor.cmp.cost_dir())
        context = tor.build_context(cost_table, Path(graph_path) if graph_path else None)
        stops = tor.load_stops(Path(stops_path) if stops_path else None)
        study_area = week5.load_study_area()
        engine = cls(context, stops, study_area, cost_table)
        engine.load_seconds = time.perf_counter() - started
        return engine

    @property
    def graph(self):
        return self.context["graph"]

    # ----- the validated calls ---------------------------------------------- #
    def inside_study_area(self, latitude: float, longitude: float) -> bool:
        return week5.inside_study_area(self.study_area, latitude, longitude)

    def snap_origin(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Week 5 snapping (src.snapping.snap_point), unchanged."""
        return week5.snap_origin_coordinate(self.graph, latitude, longitude)

    def search_from(self, origin_node: int) -> Dict[str, Any]:
        """The three single-source searches (Week 9 time_optimal_route.search_all)."""
        return tor.search_all(self.context, int(origin_node))

    def stop_row(self, stop_id: str) -> Optional[pd.Series]:
        hit = self.stops[self.stops["stop_id"] == str(stop_id)]
        return None if hit.empty else hit.iloc[0]

    def solve(self, search: Dict[str, Any], stop_id: str,
              origin_component: Optional[str] = None) -> Dict[str, Any]:
        """All three routes to one exact stop, in the ToblerLayer result shape."""
        row = self.stop_row(stop_id)
        if row is None:
            raise KeyError(f"unknown stop_id {stop_id!r}")
        base = {"stop_id": str(stop_id), "stop_name": str(row["stop_name"]),
                "mode": str(row.get("mode", "")), "stop_node": int(row["nearest_node"])}
        component = row.get("component_id", None)
        if origin_component is not None and component not in (None, "", origin_component):
            return {**base, "available": False, "reason": tor.cmp.REASON_RAW_COMPONENT_GAP,
                    "status": tor.STATUS_RAW_GAP, "routes": {}, "route_identity": None}
        solved = tor.solve_stop(self.context, search, int(row["nearest_node"]), f"stop {stop_id}")
        if solved["status"] != tor.STATUS_COMPARED:
            return {**base, "available": False, "reason": solved["status_reason"],
                    "status": solved["status"], "routes": solved["routes"], "route_identity": None}
        routes = solved["routes"]
        d, e, t = routes["distance"], routes["effort"], routes["time"]
        return {
            **base, "available": True, "reason": "", "status": solved["status"],
            "routes": routes,
            "route_identity": phase2.route_identity(routes),
            "effort_saving_vs_shortest_pct": (100.0 * (d["effort"] - e["effort"]) / d["effort"]) if d["effort"] else None,
            "extra_distance_vs_shortest_pct": (100.0 * (e["distance"] - d["distance"]) / d["distance"]) if d["distance"] else None,
            "time_saving_vs_shortest_s": ((d["walking_time_s"] - t["walking_time_s"])
                                          if d.get("walking_time_s") is not None else None),
        }

    def origin_component(self, origin_node: int) -> Optional[str]:
        return self.context["component_labels"].get(int(origin_node))

    def route_xy(self, route: Dict[str, Any]) -> np.ndarray:
        """Metric drawing coordinates of one route, from the stored OSM geometry."""
        coords = phase2.route_linestring(self.graph, route["edges"])
        return plots.project([c[0] for c in coords], [c[1] for c in coords])

    def scene_entry(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """The selected stop in the schema-1.x ``routes`` shape (route_summary), for export/tests."""
        return {obj: phase2.route_summary(result["routes"].get(obj)) for obj in OBJECTIVES}


# --------------------------------------------------------------------------- #
# human-facing text (pure formatting)
# --------------------------------------------------------------------------- #
def fmt_m(metres: Optional[float]) -> str:
    return "-" if metres is None else f"{metres:,.0f} m"


def fmt_min(seconds: Optional[float]) -> str:
    return "-" if seconds is None else f"{seconds / 60.0:.1f} min"


def fmt_effort(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:,.0f} J/kg"


def option_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Shortest / Gentlest / Fastest with distance, modelled time, effort."""
    rows = []
    for obj in OBJECTIVES:
        route = (result.get("routes") or {}).get(obj)
        rows.append({
            "objective": obj, "name": HUMAN_NAMES[obj], "technical": TECHNICAL_NAMES[obj],
            "distance_m": None if route is None else float(route["distance"]),
            "walking_time_s": None if route is None else route.get("walking_time_s"),
            "effort_j_per_kg": None if route is None else float(route["effort"]),
            "edges": None if route is None else len(route["edges"]),
        })
    return rows


def comparison_sentences(result: Dict[str, Any]) -> List[str]:
    """One or two plain-language comparisons against the shortest route."""
    routes = result.get("routes") or {}
    d, e, t = routes.get("distance"), routes.get("effort"), routes.get("time")
    identity = result.get("route_identity") or {}
    lines: List[str] = []
    if d is None or e is None:
        return lines
    if identity.get("all_same"):
        return ["All three options use the same path."]
    if identity.get("distance_equals_effort"):
        lines.append("Shortest and Gentlest use the same path.")
    else:
        extra = e["distance"] - d["distance"]
        saving = result.get("effort_saving_vs_shortest_pct")
        parts = [f"{extra:,.0f} m longer" if extra >= 0.5 else
                 ("the same length" if abs(extra) < 0.5 else f"{-extra:,.0f} m shorter")]
        if saving is not None:
            parts.append(f"{saving:.1f}% less terrain effort" if saving >= 0 else
                         f"{-saving:.1f}% more terrain effort")
        lines.append("Gentlest route: " + ", ".join(parts))
    if t is not None:
        if identity.get("time_equals_effort") and identity.get("time_equals_distance"):
            pass
        elif identity.get("time_equals_effort"):
            lines.append("Fastest uses the same path as Gentlest.")
        elif identity.get("time_equals_distance"):
            lines.append("Fastest uses the same path as Shortest.")
        else:
            saving_s = result.get("time_saving_vs_shortest_s")
            if saving_s is not None:
                lines.append(f"Fastest route: {abs(saving_s) / 60.0:.1f} min "
                             f"{'quicker' if saving_s >= 0 else 'slower'} than Shortest (modelled)")
    return lines[:2]


def identity_note(result: Dict[str, Any]) -> str:
    identity = result.get("route_identity") or {}
    if not identity:
        return ""
    if identity.get("all_same"):
        return "All three options use the same path."
    if identity.get("distinct_paths") == 3:
        return "The three options follow three different paths."
    names = {"distance": "Shortest", "effort": "Gentlest", "time": "Fastest"}
    for group in identity.get("groups", []):
        if len(group) == 2:
            return f"{names[group[0]]} and {names[group[1]]} use the same path."
    return ""


def step_text(phase: str) -> Tuple[str, str, str]:
    """(step label, title, body) for the instruction block."""
    if phase == "no_origin":
        return ("Step 1", "Choose your starting point", "Click the map to set your origin.")
    if phase == "origin_selected":
        return ("Step 2", "Choose a transit stop", "Click a stop to compare walking routes.")
    return ("Step 3", "Compare your walking options",
            "Shortest, Gentlest and Fastest are drawn on the map.")


def technical_lines(state_summary: Dict[str, Any], result: Optional[Dict[str, Any]],
                    engine: Optional[WalkEngine] = None) -> List[str]:
    """The Technical details box: internals the primary view keeps out of sight."""
    origin = state_summary.get("origin") or {}
    lines = ["TECHNICAL DETAILS",
             f"mode {state_summary.get('mode')}   run mode {state_summary.get('run_mode')}   "
             f"phase {state_summary.get('phase')}"]
    if origin:
        lines.append(f"origin {origin['input_latitude']:.6f}, {origin['input_longitude']:.6f}  "
                     f"snapped to node {origin['nearest_node']} ({origin['snap_distance_m']:.1f} m gap, "
                     "not part of any route distance)")
    if engine is not None:
        lines.append(f"trusted subgraph edges {engine.context['subgraph'].number_of_edges()}   "
                     f"timed edges {engine.context['timed'].number_of_edges()}   "
                     f"candidate stops {len(engine.stop_ids)}")
    if result:
        lines.append(f"selected stop_id {result['stop_id']}  node {result.get('stop_node')}  "
                     f"status {result.get('status')}" + (f"  reason {result['reason']}" if result.get("reason") else ""))
        for row in option_rows(result):
            if row["distance_m"] is None:
                continue
            seconds = row["walking_time_s"]
            lines.append(f"  {row['technical']:17s} {row['distance_m']:9.3f} m  {row['effort_j_per_kg']:10.4f} J/kg  "
                         f"{(f'{seconds:9.3f}' if seconds is not None else '        -')} s (Tobler)  "
                         f"{row['edges']:4d} edges")
        identity = result.get("route_identity")
        if identity:
            lines.append(f"  route identity: distinct paths {identity['distinct_paths']}, groups {identity['groups']}")
        if result.get("effort_saving_vs_shortest_pct") is not None:
            lines.append(f"  effort-optimal vs distance-optimal: {result['effort_saving_vs_shortest_pct']:.4f} % less effort, "
                         f"{result['extra_distance_vs_shortest_pct']:.4f} % extra distance")
    timings = state_summary.get("timings") or {}
    if timings:
        lines.append("  timings (s): " + ", ".join(f"{k} {v:.3f}" for k, v in timings.items()))
    lines.append("Model caveats:")
    lines += [f"  - {c}" for c in MODEL_CAVEATS]
    return lines
