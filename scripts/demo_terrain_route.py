"""Week 5 interactive demo - distance route against terrain-effort route.

WHAT THIS IS
------------
A PRESENTATION LAYER over the finished Week 5C-2B method, in the same spirit as
the Week 4 ``scripts/demo_route.py``. ``--interactive`` opens ONE persistent
Matplotlib window showing the whole official Glebe study area: you CLICK a
point inside the boundary to set an origin and CLICK a candidate transit stop
to compare the two routes to it. Nothing has to be typed. The click is snapped
with the Stage 4 snapping code and the ADOPTED lexicographic search runs from
that snapped node.

The direction of the question never changes: the ORIGIN is an arbitrary
clicked point, the DESTINATION is always one of the existing Stage 4 candidate
transit stops. A destination is never a free-space point - the project is about
access to transit stops.

It is NOT a new research method. Nothing here samples a DEM, derives a
gradient, evaluates the Minetti polynomial, ranks a stop, or invents a combined
distance-effort score. Every number on the screen is either read from
``edge_terrain_cost.csv`` or summed from the edges the adopted search chose.
See D-23 in docs/ASSUMPTIONS.md.

WHAT IT REUSES
--------------
``scripts/compare_terrain_routes.py`` is imported and used as-is:
``build_trusted_subgraph``, ``collapse_parallel_edges``,
``lexicographic_dijkstra``, ``reconstruct_nodes``, ``route_hops``,
``solve_route``, ``verify_route``, ``verify_optimality``,
``verify_cost_provenance``, and its constants. There is no second Dijkstra in
this file. ``src.snapping.snap_point`` is the only snapping primitive.
``scripts/plot_terrain_routes.py`` supplies the low-level drawing helpers.

The routing rules are therefore unchanged::

    distance route: lexicographic minimum of (total distance, total effort)
    effort route:   lexicographic minimum of (total effort, total distance)

The second metric is a TIE-BREAK ONLY. Both routes run on exactly the same
trusted-cost directed subgraph.

WHAT IT DELIBERATELY DOES NOT SHOW
----------------------------------
``restriction_effect_m``. That quantity is the gap between the Week 4 RAW-graph
distance and the trusted-subgraph distance, and Week 4 measured its baseline
from the A-02 origin only. An arbitrary interactive origin has no such
baseline, so the demo compares the two objectives on the trusted subgraph and
says nothing about a raw-graph gap. ``route_comparison.csv`` is likewise not
consulted: it is historical A-02 evidence, not a result for your coordinate.

The snap gap is reported separately and is NEVER added to a route distance: it
is an off-network straight-line gap from your coordinate to a graph node.

When no route exists, that is reported as a topology or model-coverage fact.
A stop is never called inaccessible, unwalkable or impossible on this evidence.

Cost, per click
---------------
The graph, the cost table, the provenance check, the trusted subgraph, the two
collapsed objective graphs, the official boundary and the candidate stops are
built ONCE, before the window opens. A NEW ORIGIN costs exactly two
single-source lexicographic searches. A NEW STOP at the same origin costs no
search at all: both searches are single-source, so every reachable stop is
already solved and only the path is reconstructed and verified.

Usage
-----
    python scripts/demo_terrain_route.py --interactive
    python scripts/demo_terrain_route.py --lat -33.8795 --lon 151.1875 \
        --stop-id 203732
    python scripts/demo_terrain_route.py --interactive --text-prompts
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import Point  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402

from src import boundary, snapping  # noqa: E402
from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from src.osm_network import load_graph  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The adopted Week 5C-2B implementation, used unchanged. Nothing it does is
# re-implemented here.
terrain_routing = load_script("compare_terrain_routes")
plots = load_script("plot_terrain_routes")
audit = terrain_routing.audit

OBJECTIVE_DISTANCE = terrain_routing.OBJECTIVE_DISTANCE
OBJECTIVE_EFFORT = terrain_routing.OBJECTIVE_EFFORT
OBJECTIVES: Tuple[str, ...] = terrain_routing.OBJECTIVES
ATTR_LENGTH = terrain_routing.ATTR_LENGTH
ATTR_EFFORT = terrain_routing.ATTR_EFFORT
TRUSTED_STATUSES: Tuple[str, ...] = terrain_routing.TRUSTED_STATUSES
OBJECTIVE_ATTRIBUTES = terrain_routing.OBJECTIVE_ATTRIBUTES

week5c1 = terrain_routing.terrain_cost.week5c1

# Cw(0) from the ALREADY-ADOPTED Week 5C-1 Minetti model, reused - not a new
# parameter and not a second evaluation of the polynomial. The demo never
# evaluates Minetti; it only divides an existing J/kg difference by this.
LEVEL_WALKING_COST_J_PER_KG_M = week5c1.LEVEL_COST_J_PER_KG_PER_M   # 2.5

# A unit conversion, nothing more.
JOULES_PER_KCAL = 4184.0

STOPS_CSV = "glebe_stops_snapped.csv"

# The map is drawn in the project's metric CRS, so a click carries metric
# coordinates and has to be converted back before it can be snapped.
METRIC_CRS = plots.METRIC_CRS                              # EPSG:7856
WGS84 = plots.WGS84                                        # EPSG:4326
TO_METRIC = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
TO_WGS84 = Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True)

# Reporting precision, the same as the Week 5C-2B tables.
DISTANCE_DECIMALS = 3
EFFORT_DECIMALS = 4

ORIGIN_LABEL = "user input (snapped origin)"

SNAP_GAP_NOTE = (
    "Snap distance is an off-network coordinate-to-node gap and is not "
    "included in the route distance or terrain effort."
)

MENU_ORDER_NOTE = (
    "Menu order is by stop_id, for finding a stop. It is not a ranking and "
    "carries no analytical meaning."
)

NO_ROUTE_HEADLINE = (
    "Terrain-aware route comparison is unavailable from this snapped origin "
    "to this stop under the current trusted-cost model coverage."
)

# Every one of these is a statement about topology or about which edges have an
# adopted terrain cost. None of them is a statement about the stop.
REASON_NOTES: Dict[str, str] = {
    "stop_node_has_no_trusted_cost_edge": (
        "No edge touching the stop's snapped node has an adopted terrain "
        "cost, so no route on the trusted-cost subgraph can reach it. This is "
        "model coverage, not a terrain effect and not a property of the stop."),
    "origin_node_has_no_trusted_cost_edge": (
        "No edge touching your snapped origin node has an adopted terrain "
        "cost, so no route on the trusted-cost subgraph can leave it. This is "
        "model coverage, not a terrain effect."),
    "raw_topology_gap": (
        "The raw OSM pedestrian graph itself has no origin-to-stop path, so "
        "the two are in different components of the downloaded network. This "
        "is an upstream topology artefact (see I-04); OSM-FIX-001 remains "
        "proposed and is not applied."),
    "no_trusted_cost_path": (
        "A path exists in the raw OSM graph, but not one made only of edges "
        "with an adopted terrain cost. This is a trusted-cost SUBGRAPH "
        "restriction, i.e. model coverage - not a terrain effect."),
    "origin_and_stop_share_a_node": (
        "Your click and this stop snap to the SAME graph node, so there is no "
        "walking route between them to compare. This is a snapping fact about "
        "the network, not a statement about the stop; both snap gaps are "
        "off-network and are reported separately."),
}

REASON_SAME_NODE = "origin_and_stop_share_a_node"

OUTSIDE_STUDY_AREA_MESSAGE = "Please click inside the Glebe study area."

GUI_INSTRUCTIONS = (
    "1. Click anywhere inside the Glebe boundary to choose an origin.\n"
    "2. Click a transit stop marker to compare the two routes.\n"
    "3. Click another stop, or another point, to keep exploring.")

GUI_TITLE = ("Week 5 demo - click an origin in Glebe, then a transit stop "
             "(presentation layer)")

INSET_TITLE = "Local route divergence"

BODY_MASS_LABEL = "Body mass (kg)  "
BODY_MASS_PLACEHOLDER = "Enter body mass to personalise"
BODY_MASS_INVALID = ("Body mass must be a positive number in kilograms. "
                     "The routes are unchanged.")
LEVEL_EQUIVALENT_NOTE = ("(energy equivalent only; not extra\n"
                         " distance or perceived exertion)")
BODY_MASS_SCOPE_NOTE = (
    "Body mass scales the result only. Both routes are multiplied by the same "
    "positive number, so it cannot change which route wins.")
BODY_MASS_AXES_RECT = (0.660, 0.052, 0.055, 0.028)   # figure fraction

# Interpretation only. It changes nothing about how anything is routed.
OSM_REPRESENTATION_NOTE = (
    "OSM pedestrian graph edges are network representations; a roadway "
    "centreline edge does not necessarily mean walking in the traffic lane.")

IDENTICAL_ROUTE_PANEL_NOTE = "Both objectives selected the same route."

# The inset's own geographic padding, as a share of the divergent extent.
INSET_PADDING_SHARE = 0.18
INSET_PADDING_FLOOR_M = 15.0
INSET_RECTANGLE = (0.615, 0.575, 0.375, 0.40)   # axes fraction on the map

# A click within this many screen pixels of a stop marker selects that stop.
STOP_PICK_RADIUS_PX = 14.0

COLOUR_BOUNDARY = "#6a6a6a"
COLOUR_STOP = "#4d4d4d"
COLOUR_STOP_SELECTED = "#7570b3"
COLOUR_ORIGIN = "#1b9e77"
MARKER_BY_MODE = {"bus": "o", "light_rail": "^"}
DEFAULT_STOP_MARKER = "o"

IDENTICAL_ROUTE_NOTE = (
    "Both objectives choose exactly the same directed edges, so there is no "
    "trade-off to show: the shortest route is already the least-effort route "
    "on this subgraph."
)


class DemoError(RuntimeError):
    """Raised with an actionable message when the demo cannot proceed."""


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def stops_path(root: Path = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache" / STOPS_CSV).resolve()


def load_candidate_stops(path: Path) -> pd.DataFrame:
    """The Stage 4 snapped candidate stops. Nothing is re-snapped here."""
    path = Path(path)
    if not path.exists():
        raise DemoError(
            f"{path} is missing. Run the Week 4 foundation first:\n"
            "    python scripts/run_week4_foundation.py")
    frame = pd.read_csv(path)
    for column in ("stop_id", "stop_name", "nearest_node"):
        if column not in frame.columns:
            raise DemoError(f"{path} has no {column!r} column.")
    frame = frame.copy()
    frame["stop_id"] = frame["stop_id"].astype(str).str.strip()
    return frame


def menu_rows(stops: pd.DataFrame) -> pd.DataFrame:
    """Candidate stops in a stable presentation order.

    Sorted by stop_id purely so the menu is reproducible and easy to scan.
    This is NOT a ranking: nothing here scores, orders by distance, or selects
    a "best" stop.
    """
    return stops.sort_values("stop_id", kind="mergesort").reset_index(drop=True)


def find_stop(stops: pd.DataFrame, stop_id: str) -> pd.Series:
    wanted = str(stop_id).strip()
    match = stops[stops["stop_id"] == wanted]
    if match.empty:
        raise DemoError(
            f"No candidate stop has stop_id {wanted!r}. Run with "
            "--interactive to see the list.")
    return match.iloc[0]


def load_study_area(path: Path = None) -> Dict[str, Any]:
    """The official Glebe suburb polygon, in WGS84 and in the metric CRS.

    Loaded through ``src.boundary.load_candidate_polygon`` - the same official
    file Stage 2 selected candidate stops with. Nothing is downloaded and no
    rectangle is invented: the click guard tests the real polygon.
    """
    geometry, properties = boundary.load_candidate_polygon(path)
    metric = shapely_transform(
        lambda xs, ys, z=None: TO_METRIC.transform(xs, ys), geometry)
    return {"wgs84": geometry, "metric": metric, "properties": properties,
            "bounds_metric": tuple(float(v) for v in metric.bounds)}


def metric_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """One metric map coordinate back to (latitude, longitude).

    The map axis is EPSG:7856, so ``event.xdata`` / ``event.ydata`` are
    eastings and northings. They are converted with an explicit pyproj
    transformer before anything is snapped.
    """
    longitude, latitude = TO_WGS84.transform(float(x), float(y))
    return float(latitude), float(longitude)


def inside_study_area(study_area: Dict[str, Any], latitude: float,
                      longitude: float) -> bool:
    """True when the coordinate is inside or exactly on the Glebe polygon."""
    return bool(study_area["wgs84"].covers(Point(float(longitude),
                                                 float(latitude))))


def snap_origin_coordinate(
    graph: nx.MultiDiGraph, latitude: float, longitude: float
) -> Dict[str, Any]:
    """Snap one arbitrary coordinate with the Stage 4 snapping primitive.

    ``src.snapping.snap_point`` is the project's only nearest-node
    implementation and measures the gap in EPSG:7856. No second nearest-node
    search exists, and no maximum snap distance is imposed here.
    """
    node, gap_m = snapping.snap_point(graph, float(latitude), float(longitude))
    node = int(node)
    return {
        "input_latitude": float(latitude),
        "input_longitude": float(longitude),
        "nearest_node": node,
        "snap_distance_m": float(gap_m),
        "node_lat": float(graph.nodes[node]["y"]),
        "node_lon": float(graph.nodes[node]["x"]),
        "role": ORIGIN_LABEL,
        "snap_gap_note": SNAP_GAP_NOTE,
    }


# --------------------------------------------------------------------------- #
# the routing context, built once from the adopted implementation
# --------------------------------------------------------------------------- #
def build_context(graph_path: Path, cost_directory: Path) -> Dict[str, Any]:
    """Load the graph and costs, verify provenance, build the routing graphs.

    Every step is the Week 5C-2B one. The raw graph is read-only and its hash
    is recorded so the caller can prove it was not touched.
    """
    graph_path = Path(graph_path)
    cost_table = terrain_routing.load_cost_table(cost_directory)
    cost_summary = terrain_routing.load_cost_summary(cost_directory)
    graph_sha256 = audit.file_sha256(graph_path)
    graph = load_graph(graph_path)
    provenance = terrain_routing.verify_cost_provenance(
        cost_summary, cost_table, graph, graph_path, graph_sha256)

    subgraph = terrain_routing.build_trusted_subgraph(graph, cost_table)
    trusted_keys = {(int(u), int(v), int(k))
                    for u, v, k in subgraph.edges(keys=True)}
    collapsed = {}
    for objective in OBJECTIVES:
        collapsed[objective], _counters = \
            terrain_routing.collapse_parallel_edges(subgraph, objective)

    return {
        "graph": graph,
        "graph_path": graph_path,
        "graph_sha256": graph_sha256,
        "cost_table": cost_table,
        "provenance": provenance,
        "subgraph": subgraph,
        "trusted_keys": trusted_keys,
        "collapsed": collapsed,
    }


def unavailable_reason(
    context: Dict[str, Any], origin_node: int, stop_node: int
) -> str:
    """Why no trusted-cost route exists - topology, or model coverage.

    ``networkx.has_path`` is a plain reachability question about the raw graph;
    it produces no route and no distance, and its only use is to tell an
    upstream topology gap apart from a trusted-cost subgraph restriction.
    """
    subgraph = context["subgraph"]
    if subgraph.degree(stop_node) == 0:
        return "stop_node_has_no_trusted_cost_edge"
    if subgraph.degree(origin_node) == 0:
        return "origin_node_has_no_trusted_cost_edge"
    if not nx.has_path(context["graph"], origin_node, stop_node):
        return "raw_topology_gap"
    return "no_trusted_cost_path"


def search_from_origin(
    context: Dict[str, Any], origin_node: int
) -> Dict[str, Any]:
    """The two adopted single-source searches, run ONCE for this origin.

    Both are single-source, so every stop reachable from this origin is
    already solved by these two runs. Selecting a different stop afterwards
    only reconstructs a path from the stored predecessors - it must never
    re-run a search, which is what keeps the GUI responsive.
    """
    if origin_node not in context["graph"]:
        raise DemoError(f"Node {origin_node} is not in the graph.")
    labels: Dict[str, Any] = {}
    predecessors: Dict[str, Any] = {}
    for objective in OBJECTIVES:
        primary, secondary = OBJECTIVE_ATTRIBUTES[objective]
        labels[objective], predecessors[objective] = \
            terrain_routing.lexicographic_dijkstra(
                context["collapsed"][objective], origin_node, primary,
                secondary)
    return {"origin_node": int(origin_node), "labels": labels,
            "predecessors": predecessors}


def solve_selected_stop(
    context: Dict[str, Any], search: Dict[str, Any], stop: pd.Series,
) -> Dict[str, Any]:
    """Both optima to one stop, from an already-completed origin search.

    Reconstruction and verification only - no search runs here. No restriction
    effect is computed either: there is no Week 4 raw baseline for an
    arbitrary origin, and inventing one would be a new result, not a demo.
    """
    graph = context["graph"]
    origin_node = int(search["origin_node"])
    labels = search["labels"]
    predecessors = search["predecessors"]
    stop_node = int(stop["nearest_node"])
    if stop_node not in graph:
        raise DemoError(f"The stop's snapped node {stop_node} is not in the "
                        "graph.")

    if stop_node == origin_node or any(
            stop_node not in labels[objective] for objective in OBJECTIVES):
        reason = (REASON_SAME_NODE if stop_node == origin_node
                  else unavailable_reason(context, origin_node, stop_node))
        return {
            "available": False,
            "stop_id": stop["stop_id"],
            "stop_name": stop.get("stop_name", ""),
            "origin_node": origin_node,
            "stop_node": stop_node,
            "reason": reason,
            "headline": NO_ROUTE_HEADLINE,
            "note": REASON_NOTES[reason],
        }

    routes = {
        objective: terrain_routing.solve_route(
            context["collapsed"][objective], labels[objective],
            predecessors[objective], objective, origin_node, stop_node)
        for objective in OBJECTIVES
    }
    for objective, route in routes.items():
        terrain_routing.verify_route(
            route["hops"], route["nodes"], origin_node, stop_node,
            route["distance"], route["effort"], graph,
            context["trusted_keys"],
            f"demo stop {stop['stop_id']} {objective} route")
    terrain_routing.verify_optimality(stop["stop_id"],
                                      routes[OBJECTIVE_DISTANCE],
                                      routes[OBJECTIVE_EFFORT])

    distance_route = routes[OBJECTIVE_DISTANCE]
    effort_route = routes[OBJECTIVE_EFFORT]
    shared = set(distance_route["edges"]) & set(effort_route["edges"])
    saving = distance_route["effort"] - effort_route["effort"]
    detour = effort_route["distance"] - distance_route["distance"]

    return {
        "available": True,
        "stop_id": stop["stop_id"],
        "stop_name": stop.get("stop_name", ""),
        "origin_node": origin_node,
        "stop_node": stop_node,
        "routes": routes,
        "shared_edges": shared,
        "identical_routes": distance_route["edges"] == effort_route["edges"],
        "distance_route_distance_m": round(distance_route["distance"],
                                           DISTANCE_DECIMALS),
        "distance_route_effort_j_per_kg": round(distance_route["effort"],
                                                EFFORT_DECIMALS),
        "effort_route_distance_m": round(effort_route["distance"],
                                         DISTANCE_DECIMALS),
        "effort_route_effort_j_per_kg": round(effort_route["effort"],
                                              EFFORT_DECIMALS),
        "distance_detour_m": round(detour, DISTANCE_DECIMALS),
        "effort_saving_j_per_kg": round(saving, EFFORT_DECIMALS),
        "effort_saving_pct": round(100.0 * saving / distance_route["effort"],
                                   EFFORT_DECIMALS)
        if distance_route["effort"] else None,
        "verified": True,
    }


def compare_from_origin(
    context: Dict[str, Any], origin_node: int, stop: pd.Series,
) -> Dict[str, Any]:
    """One origin, one stop, in one call - the non-interactive path.

    The GUI keeps the two halves apart so a new stop reuses the search; a
    single command-line comparison has nothing to reuse, so it does both.
    """
    return solve_selected_stop(
        context, search_from_origin(context, origin_node), stop)


# --------------------------------------------------------------------------- #
# terminal report
# --------------------------------------------------------------------------- #
RULE = "-" * 60


def format_report(origin: Dict[str, Any], result: Dict[str, Any]) -> str:
    lines: List[str] = [
        RULE,
        "Week 5 terrain-aware walking demo",
        RULE,
        "Origin",
        f"  input:          {origin['input_latitude']:.6f}, "
        f"{origin['input_longitude']:.6f}",
        f"  snapped node:   {origin['nearest_node']}",
        f"  snap gap:       {origin['snap_distance_m']:.2f} m",
        "  (off-network gap; not included below)",
        "",
        "Destination",
        f"  {result['stop_id']} - {result['stop_name']}",
        "",
    ]

    if not result["available"]:
        lines += [result["headline"], "", f"  reason: {result['reason']}",
                  f"  {result['note']}", RULE]
        return "\n".join(lines)

    lines += [
        "Shortest route",
        f"  distance:       {result['distance_route_distance_m']:.3f} m",
        f"  terrain effort: "
        f"{result['distance_route_effort_j_per_kg']:.4f} J/kg",
        "",
        "Least-effort route",
        f"  distance:       {result['effort_route_distance_m']:.3f} m",
        f"  terrain effort: {result['effort_route_effort_j_per_kg']:.4f} J/kg",
        "",
    ]
    if result["identical_routes"]:
        lines += ["Trade-off", f"  {IDENTICAL_ROUTE_NOTE}"]
    else:
        saving_pct = result["effort_saving_pct"]
        lines += [
            "Trade-off",
            f"  extra distance: {result['distance_detour_m']:+.3f} m",
            f"  effort saved:   {result['effort_saving_j_per_kg']:.4f} J/kg",
            f"  effort saved:   "
            f"{saving_pct:.4f} %" if saving_pct is not None else
            "  effort saved:   n/a",
        ]
    lines += ["", SNAP_GAP_NOTE, RULE]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the runtime figure
# --------------------------------------------------------------------------- #
def hops_frame(hops: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """The plotting helpers read a table of hops, so give them one.

    The columns are named even for an empty route, so a caller never gets a
    frame the drawing helpers cannot index.
    """
    return pd.DataFrame([
        {"hop_index": hop["hop_index"], "u": hop["u"], "v": hop["v"],
         "key": hop["key"], "length_m": hop["length_m"],
         "terrain_effort_j_per_kg": hop["terrain_effort_j_per_kg"]}
        for hop in hops
    ], columns=["hop_index", "u", "v", "key", "length_m",
                "terrain_effort_j_per_kg"])


def demo_comparison_text(result: Dict[str, Any]) -> str:
    """The four totals and the trade-off. No restriction effect.

    The restriction effect compares against a Week 4 RAW-graph baseline that
    exists only for the A-02 origin, so it has no meaning for a coordinate the
    user just typed and is deliberately absent.
    """
    text = (
        f"shortest        {result['distance_route_distance_m']:.3f} m"
        f"    {result['distance_route_effort_j_per_kg']:.3f} J/kg\n"
        f"least effort    {result['effort_route_distance_m']:.3f} m"
        f"    {result['effort_route_effort_j_per_kg']:.3f} J/kg\n"
    )
    if result["identical_routes"]:
        return text + "\nboth objectives choose the same edges"
    saving_pct = result["effort_saving_pct"]
    text += (f"\nextra distance     {result['distance_detour_m']:+.3f} m\n"
             f"effort saved       {result['effort_saving_j_per_kg']:.3f} J/kg")
    if saving_pct is not None:
        text += f"  ({saving_pct:.3f} %)"
    return text


def route_overlap(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How much of the two SELECTED routes is the same directed edge.

    Purely DESCRIPTIVE, and only about the two routes on screen. It is not a
    similarity measure, not an accessibility measure, not a score and not a
    ranking: it just says how much of each route is made of edges the other
    route also uses, so a reader can see at a glance whether a small numeric
    difference comes from a small local detour or from two different ways
    across Glebe.

    Every value is summed from the hop lengths Week 5C-2B already recorded.
    Nothing is re-routed, no new terrain quantity is derived, and the snap gap
    - which is off-network and belongs to no edge - is not part of any of it.
    """
    if not result.get("available"):
        return None
    routes = result["routes"]
    shared = set(result["shared_edges"])
    hops = {objective: {(h["u"], h["v"], h["key"]): h
                        for h in routes[objective]["hops"]}
            for objective in OBJECTIVES}

    lengths = {}
    divergent = {}
    for objective in OBJECTIVES:
        edges = routes[objective]["edges"]
        lengths[objective] = sum(hops[objective][e]["length_m"] for e in edges)
        divergent[objective] = sum(hops[objective][e]["length_m"]
                                   for e in edges if e not in shared)
    # A shared edge has one length; take it from the distance route's hops.
    shared_length = sum(hops[OBJECTIVE_DISTANCE][e]["length_m"]
                        for e in routes[OBJECTIVE_DISTANCE]["edges"]
                        if e in shared)

    def share(total):
        return round(100.0 * shared_length / total, 1) if total else None

    return {
        "identical_routes": bool(result["identical_routes"]),
        "shared_edge_count": len(shared),
        "shared_length_m": round(shared_length, 3),
        "shared_pct_of_distance_route": share(lengths[OBJECTIVE_DISTANCE]),
        "shared_pct_of_effort_route": share(lengths[OBJECTIVE_EFFORT]),
        "divergent_distance_route_length_m": round(
            divergent[OBJECTIVE_DISTANCE], 3),
        "divergent_effort_route_length_m": round(
            divergent[OBJECTIVE_EFFORT], 3),
        "distance_only_edges": sum(
            1 for e in routes[OBJECTIVE_DISTANCE]["edges"] if e not in shared),
        "effort_only_edges": sum(
            1 for e in routes[OBJECTIVE_EFFORT]["edges"] if e not in shared),
        "basis": ("exact shared directed (u, v, key) edges; descriptive only, "
                  "not a score, ranking or accessibility measure"),
    }


def parse_body_mass(text: Any) -> Optional[float]:
    """A body mass in kilograms, or None if the entry is not usable.

    Accepts ANY finite positive value. Nothing is clamped to a "reasonable"
    human range: that would be an assumption about the user, and this quantity
    is a display scale, not a claim about anybody.
    """
    try:
        mass = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(mass) or mass <= 0.0:
        return None
    return mass


def level_walking_equivalent_m(effort_saving_j_per_kg: float) -> float:
    """The saving expressed as a length of LEVEL walking of equal energy.

        metres = effort saving [J/kg] / Cw(0) [J/kg/m]

    ``Cw(0)`` is the adopted Week 5C-1 level-walking cost, imported, not
    recomputed - the Minetti polynomial is never evaluated in this file.

    The result is an ENERGY equivalence and only that. It is NOT additional
    route distance, NOT perceived exertion, NOT subjective fatigue, not an
    accessibility measure and not a health claim. Cumulative metabolic energy
    is related to physical demand but does not measure how hard a route feels:
    a short steep segment can add very little to a route's total energy while
    still being locally demanding. No fatigue model is introduced here.
    """
    return float(effort_saving_j_per_kg) / LEVEL_WALKING_COST_J_PER_KG_M


def personal_energy(result: Dict[str, Any],
                    body_mass_kg: Optional[float]) -> Optional[Dict[str, Any]]:
    """The two route efforts as absolute metabolic energy, for one body mass.

        energy [J] = terrain effort [J/kg] * body mass [kg]

    Presentation only. Body mass never enters an edge weight, never reaches
    the search, and cannot reorder the two routes: both are multiplied by the
    same positive number. Returns None when no body mass has been entered, so
    the caller can say so instead of inventing one.
    """
    if body_mass_kg is None or not result.get("available"):
        return None
    mass = float(body_mass_kg)
    distance_j = float(result["distance_route_effort_j_per_kg"]) * mass
    effort_j = float(result["effort_route_effort_j_per_kg"]) * mass
    saving_j = float(result["effort_saving_j_per_kg"]) * mass
    return {
        "body_mass_kg": mass,
        "distance_route_j": distance_j,
        "effort_route_j": effort_j,
        "saving_j": saving_j,
        "distance_route_kj": distance_j / 1000.0,
        "effort_route_kj": effort_j / 1000.0,
        "saving_kj": saving_j / 1000.0,
        "saving_kcal": saving_j / JOULES_PER_KCAL,
        "basis": ("terrain effort [J/kg] x body mass [kg]; a display scale "
                  "only, applied identically to both routes"),
    }


def significant_decimals(value: float, minimum: int = 2,
                         maximum: int = 6) -> int:
    """Enough decimals that a small non-zero value never prints as 0.00."""
    if not value or not np.isfinite(value):
        return minimum
    decimals = minimum
    while decimals < maximum and abs(round(value, decimals)) * 10 ** decimals \
            < 10:
        decimals += 1
    return decimals


def format_amount(value: float, minimum: int = 2, maximum: int = 6) -> str:
    return f"{value:.{significant_decimals(value, minimum, maximum)}f}"


def energy_text(result: Dict[str, Any],
                body_mass_kg: Optional[float]) -> str:
    """The energy block: model result first, then translations.

    J/kg stays the primary quantity. The kilojoule and kilocalorie lines are
    a personalisation of it, printed after it and never instead of it.
    """
    saving = float(result["effort_saving_j_per_kg"])
    lines = [
        "MODEL RESULT",
        f"  effort saved   {saving:.4f} J/kg",
        "INTUITIVE TRANSLATION",
        f"  = energy of "
        f"{format_amount(level_walking_equivalent_m(saving))} m level walking",
        LEVEL_EQUIVALENT_NOTE,
    ]
    personal = personal_energy(result, body_mass_kg)
    if personal is None:
        lines += ["PERSONAL", f"  {BODY_MASS_PLACEHOLDER}"]
        return "\n".join(lines)
    lines += [
        f"PERSONAL ({format_amount(personal['body_mass_kg'], 0, 3)} kg)",
        f"  shortest       {format_amount(personal['distance_route_kj'])} kJ",
        f"  least effort   {format_amount(personal['effort_route_kj'])} kJ",
        f"  saving         {format_amount(personal['saving_kj'])} kJ",
        f"                 {format_amount(personal['saving_kcal'])} kcal",
    ]
    return "\n".join(lines)


def overlap_text(overlap: Dict[str, Any]) -> str:
    """The overlap block as it appears under the route totals."""
    if overlap["identical_routes"]:
        return IDENTICAL_ROUTE_PANEL_NOTE
    return (
        f"exact shared route\n"
        f"    {overlap['shared_pct_of_distance_route']:.1f} % of shortest\n"
        f"    {overlap['shared_pct_of_effort_route']:.1f} % of least effort\n"
        f"divergent portions\n"
        f"    shortest     {overlap['divergent_distance_route_length_m']:.1f} m"
        f"\n    least effort "
        f"{overlap['divergent_effort_route_length_m']:.1f} m")


def divergent_edges(result: Dict[str, Any]) -> Dict[str, List[Any]]:
    """The exact directed edges each route does NOT share with the other."""
    shared = set(result["shared_edges"])
    return {objective: [edge for edge in result["routes"][objective]["edges"]
                        if edge not in shared]
            for objective in OBJECTIVES}


def draw_demo_map(axis, graph, polylines, shared, origin, result) -> None:
    """The map panel, with the USER's origin labelled as such.

    The C-2C map helper hard-labels the A-02 provisional test origin, which
    would be a false caption for a coordinate the user typed, so the panel is
    drawn here from the same low-level helpers instead.
    """
    from matplotlib.lines import Line2D

    bounds = plots.route_bounds(polylines[OBJECTIVE_DISTANCE]
                                + polylines[OBJECTIVE_EFFORT])
    margin = plots.MAP_MARGIN_M
    padded = (bounds[0] - margin, bounds[1] - margin,
              bounds[2] + margin, bounds[3] + margin)

    for points in plots.context_polylines(graph, bounds, margin):
        axis.plot(points[:, 0], points[:, 1], color=plots.COLOUR_CONTEXT,
                  linewidth=1.0, zorder=1, solid_capstyle="round")

    for line in polylines[OBJECTIVE_DISTANCE]:
        if line["edge"] in shared:
            axis.plot(line["points"][:, 0], line["points"][:, 1],
                      color=plots.COLOUR_SHARED, linewidth=5.5, zorder=2,
                      solid_capstyle="round")
    for objective, colour, style, order in (
            (OBJECTIVE_DISTANCE, plots.COLOUR_DISTANCE, plots.STYLE_DISTANCE, 3),
            (OBJECTIVE_EFFORT, plots.COLOUR_EFFORT, plots.STYLE_EFFORT, 4)):
        for line in polylines[objective]:
            if line["edge"] in shared:
                continue
            axis.plot(line["points"][:, 0], line["points"][:, 1], color=colour,
                      linewidth=3.0, linestyle=style, zorder=order,
                      solid_capstyle="round")

    origin_node = int(origin["nearest_node"])
    stop_node = int(result["stop_node"])
    origin_xy = plots.project([graph.nodes[origin_node]["x"]],
                              [graph.nodes[origin_node]["y"]])[0]
    stop_xy = plots.project([graph.nodes[stop_node]["x"]],
                            [graph.nodes[stop_node]["y"]])[0]
    axis.scatter([origin_xy[0]], [origin_xy[1]], marker="s", s=90,
                 facecolor="#ffffff", edgecolor=plots.COLOUR_INK, linewidth=1.6,
                 zorder=7)
    axis.scatter([stop_xy[0]], [stop_xy[1]], marker="^", s=130,
                 facecolor="#ffffff", edgecolor=plots.COLOUR_INK, linewidth=1.6,
                 zorder=7)
    axis.annotate(
        f"{ORIGIN_LABEL}\nsnap gap {origin['snap_distance_m']:.2f} m "
        "(not in the route)",
        origin_xy, textcoords="offset points", xytext=(10, -8), va="top",
        fontsize=8.5, color=plots.COLOUR_INK, zorder=8)
    axis.annotate(f"stop {result['stop_id']}\n{result['stop_name']}", stop_xy,
                  textcoords="offset points", xytext=(10, -18), fontsize=8.5,
                  color=plots.COLOUR_INK, zorder=8)

    axis.set_xlim(padded[0], padded[2])
    axis.set_ylim(padded[1], padded[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#cccccc")
    plots.draw_scale_bar(axis, padded)
    axis.set_title(f"Routes from your snapped origin to stop "
                   f"{result['stop_id']}", fontsize=10,
                   color=plots.COLOUR_INK)
    axis.legend(handles=[
        Line2D([0], [0], color=plots.COLOUR_SHARED, linewidth=5.5,
               label=plots.LABEL_SHARED),
        Line2D([0], [0], color=plots.COLOUR_DISTANCE, linewidth=3.0,
               linestyle=plots.STYLE_DISTANCE, label=plots.LABEL_DISTANCE),
        Line2D([0], [0], color=plots.COLOUR_EFFORT, linewidth=3.0,
               linestyle=plots.STYLE_EFFORT, label=plots.LABEL_EFFORT),
        Line2D([0], [0], color=plots.COLOUR_CONTEXT, linewidth=1.0,
               label=plots.LABEL_CONTEXT),
    ], loc="upper left", fontsize=6.6, framealpha=0.92, handlelength=1.5,
        handletextpad=0.5, labelspacing=0.3, borderpad=0.4, borderaxespad=0.4)


def build_figure(context: Dict[str, Any], origin: Dict[str, Any],
                 result: Dict[str, Any]):
    """The two-panel runtime figure. Returns the Matplotlib figure."""
    import matplotlib.pyplot as plt

    graph = context["graph"]
    frames = {objective: hops_frame(result["routes"][objective]["hops"])
              for objective in OBJECTIVES}
    polylines = {objective: plots.route_polylines(graph, frames[objective])
                 for objective in OBJECTIVES}
    series = {objective: plots.cumulative_series(frames[objective])
              for objective in OBJECTIVES}

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.4),
                               gridspec_kw={"width_ratios": [1.05, 0.95]})
    draw_demo_map(axes[0], graph, polylines, result["shared_edges"], origin,
                  result)
    # draw_cumulative reads only effort_saving_j_per_kg from this row.
    plots.draw_cumulative(
        axes[1], series,
        pd.Series({"effort_saving_j_per_kg":
                   result["effort_saving_j_per_kg"]}), compact=True)
    axes[1].text(0.985, 0.02, demo_comparison_text(result),
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=8.5, family="monospace", color=plots.COLOUR_INK,
                 bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc",
                       "boxstyle": "round,pad=0.5", "alpha": 0.95})
    figure.suptitle(
        "Week 5 demo - distance-optimal and effort-optimal walking routes on "
        "the same trusted-cost subgraph", fontsize=12, color=plots.COLOUR_INK)
    figure.text(0.5, 0.008,
                SNAP_GAP_NOTE + "  " + plots.CHORD_NOTE
                + "  Metric drawing in EPSG:7856.",
                ha="center", fontsize=7.5, color=plots.COLOUR_MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    return figure


def show_or_save(figure, save_path: Optional[Path]) -> Optional[Path]:
    """Display the runtime figure, or write it where the caller asked.

    Nothing is written unless ``--save`` names a file: running the demo must
    not create a research output dataset.
    """
    import matplotlib.pyplot as plt

    if save_path is None:
        plt.show()
        plt.close(figure)
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return save_path


# --------------------------------------------------------------------------- #
# the graphical explorer - one window, one small state machine
# --------------------------------------------------------------------------- #
def stop_positions(stops: pd.DataFrame) -> np.ndarray:
    """Every candidate stop's own coordinate, projected for the map."""
    return plots.project(stops["stop_lon"].to_numpy(dtype=float),
                         stops["stop_lat"].to_numpy(dtype=float))


class TerrainRouteExplorer:
    """A click-driven explorer over the adopted Week 5C-2B comparison.

    State machine
    -------------
    The whole interaction is four fields - ``selected_origin``,
    ``selected_stop``, ``current_result`` and ``route_artists`` - and three
    transitions:

    ==============================  =========================================
    click                           effect
    ==============================  =========================================
    inside Glebe, not on a stop     snap a new origin, re-run the two
                                    single-source searches, and recompute to
                                    the already-selected stop if there is one
    on a stop marker                reuse this origin's finished searches and
                                    reconstruct the two routes to that stop
    outside Glebe                   nothing changes; a message is shown
    ==============================  =========================================

    Cost
    ----
    The graph, the cost table, the provenance check, the trusted subgraph, the
    two collapsed objective graphs, the boundary and the stops are built ONCE,
    before the window opens. A new origin costs exactly two single-source
    searches. A new stop at the same origin costs no search at all - both
    searches are single-source, so every reachable stop is already solved.
    """

    def __init__(self, context: Dict[str, Any], stops: pd.DataFrame,
                 study_area: Dict[str, Any], figure=None):
        self.context = context
        self.stops = menu_rows(stops)
        self.study_area = study_area
        self.figure = figure
        self.map_axis = None
        self.panel_axis = None

        self.positions = stop_positions(self.stops)
        self.stop_ids: List[str] = [str(value)
                                    for value in self.stops["stop_id"]]

        # ----- the state machine's whole state -------------------------- #
        self.selected_origin: Optional[Dict[str, Any]] = None
        self.selected_stop: Optional[str] = None
        self.current_result: Optional[Dict[str, Any]] = None
        self.search: Optional[Dict[str, Any]] = None
        self.route_artists: List[Any] = []
        self.origin_artists: List[Any] = []
        self.highlight_artists: List[Any] = []
        self.inset_axis = None
        self.message_text = None

        # Presentation only. It scales what is displayed and nothing else:
        # it never reaches a weight, a search, a snap or a route identity.
        self.body_mass_kg: Optional[float] = None
        self.body_mass_box = None
        self.body_mass_axis = None

        # Counters, so a test can prove a stop change costs no search.
        self.searches_run = 0
        self.solves_run = 0

    # ----- construction ------------------------------------------------- #
    def build(self):
        """Draw every static layer once and return the figure."""
        import matplotlib.pyplot as plt

        if self.figure is None:
            self.figure = plt.figure(figsize=(15.0, 8.0))
        axes = self.figure.subplots(
            1, 2, gridspec_kw={"width_ratios": [1.15, 0.85]})
        self.map_axis, self.panel_axis = axes[0], axes[1]
        self.draw_static_map()
        self.draw_panel()
        self.figure.suptitle(GUI_TITLE, fontsize=12, color=plots.COLOUR_INK)
        self.message_text = self.figure.text(
            0.5, 0.030, SNAP_GAP_NOTE, ha="center", fontsize=8,
            color=plots.COLOUR_MUTED)
        # Interpretation, not a warning: it explains what an edge is, and
        # changes nothing about how anything is routed.
        self.figure.text(0.5, 0.008, OSM_REPRESENTATION_NOTE, ha="center",
                         fontsize=7, color=plots.COLOUR_MUTED)
        self.figure.tight_layout(rect=(0, 0.115, 1, 0.95))
        self.build_body_mass_input()
        return self.figure

    def build_body_mass_input(self):
        """An optional body-mass box, under the right-hand panel.

        It is a display control. Typing in it re-renders the panel text and
        nothing else - no search, no snapping, no change of origin, stop or
        route. Its axes is added AFTER tight_layout so the validated layout
        above it is untouched.
        """
        from matplotlib.widgets import TextBox

        self.body_mass_axis = self.figure.add_axes(BODY_MASS_AXES_RECT)
        self.body_mass_box = TextBox(self.body_mass_axis, BODY_MASS_LABEL,
                                     initial="", textalignment="center")
        self.body_mass_box.label.set_fontsize(8)
        self.body_mass_box.text_disp.set_fontsize(8)
        self.body_mass_box.on_submit(self.on_body_mass)
        self.figure.text(BODY_MASS_AXES_RECT[0] + BODY_MASS_AXES_RECT[2] + 0.01,
                         BODY_MASS_AXES_RECT[1] + 0.013,
                         "optional - scales the displayed energy only",
                         ha="left", va="center", fontsize=7,
                         color=plots.COLOUR_MUTED)
        return self.body_mass_box

    def draw_static_map(self) -> None:
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        axis = self.map_axis
        left, bottom, right, top = self.study_area["bounds_metric"]
        margin = plots.MAP_MARGIN_M

        segments = plots.context_polylines(
            self.context["graph"], (left, bottom, right, top), margin)
        axis.add_collection(LineCollection(
            segments, colors=plots.COLOUR_CONTEXT, linewidths=0.8, zorder=1))

        polygon = self.study_area["metric"]
        for ring in [polygon.exterior] + list(polygon.interiors):
            xs, ys = ring.xy
            axis.plot(xs, ys, color=COLOUR_BOUNDARY, linewidth=1.6,
                      linestyle=(0, (5, 2)), zorder=2)

        for mode, marker in MARKER_BY_MODE.items():
            mask = (self.stops["mode"].astype(str) == mode).to_numpy()
            if not mask.any():
                continue
            axis.scatter(self.positions[mask, 0], self.positions[mask, 1],
                         marker=marker, s=44, facecolor="#ffffff",
                         edgecolor=COLOUR_STOP, linewidth=1.2, zorder=6)
        other = ~self.stops["mode"].astype(str).isin(MARKER_BY_MODE).to_numpy()
        if other.any():
            axis.scatter(self.positions[other, 0], self.positions[other, 1],
                         marker=DEFAULT_STOP_MARKER, s=44,
                         facecolor="#ffffff", edgecolor=COLOUR_STOP,
                         linewidth=1.2, zorder=6)

        axis.set_xlim(left - margin, right + margin)
        axis.set_ylim(bottom - margin, top + margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#cccccc")
        # The committed helper puts its bar bottom-LEFT, where this map's
        # legend has to live, so the bar is placed bottom-right here. The
        # round length still comes from the committed helper.
        span = (right + margin) - (left - margin)
        length = plots.nice_scale_length(span)
        bar_y = (bottom - margin) + 0.04 * ((top + margin) - (bottom - margin))
        bar_right = (right + margin) - 0.05 * span
        axis.plot([bar_right - length, bar_right], [bar_y, bar_y],
                  color=plots.COLOUR_INK, linewidth=2.4,
                  solid_capstyle="butt", zorder=9)
        axis.text(bar_right - length / 2.0,
                  bar_y + 0.012 * ((top + margin) - (bottom - margin)),
                  f"{length:g} m", ha="center", va="bottom", fontsize=8,
                  color=plots.COLOUR_INK, zorder=9)
        axis.set_title("Official Glebe study area - click to explore",
                       fontsize=10, color=plots.COLOUR_INK)
        axis.text(0.015, 0.985, GUI_INSTRUCTIONS, transform=axis.transAxes,
                  ha="left", va="top", fontsize=8, color=plots.COLOUR_INK,
                  bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.4", "alpha": 0.93},
                  zorder=10)
        axis.legend(handles=[
            Line2D([0], [0], color=COLOUR_BOUNDARY, linewidth=1.6,
                   linestyle=(0, (5, 2)), label="official Glebe boundary"),
            Line2D([0], [0], marker="o", color="none", markersize=6,
                   markerfacecolor="#ffffff", markeredgecolor=COLOUR_STOP,
                   label="candidate bus stop"),
            Line2D([0], [0], marker="^", color="none", markersize=7,
                   markerfacecolor="#ffffff", markeredgecolor=COLOUR_STOP,
                   label="candidate light-rail stop"),
            Line2D([0], [0], color=plots.COLOUR_SHARED, linewidth=4.0,
                   label=plots.LABEL_SHARED),
            Line2D([0], [0], color=plots.COLOUR_DISTANCE, linewidth=2.4,
                   linestyle=plots.STYLE_DISTANCE,
                   label=plots.LABEL_DISTANCE),
            Line2D([0], [0], color=plots.COLOUR_EFFORT, linewidth=2.4,
                   linestyle=plots.STYLE_EFFORT, label=plots.LABEL_EFFORT),
            Line2D([0], [0], color=plots.COLOUR_CONTEXT, linewidth=1.0,
                   label="raw OSM pedestrian network (context only)"),
        ], loc="lower left", fontsize=6.4, framealpha=0.92, handlelength=1.6,
            handletextpad=0.5, labelspacing=0.3, borderpad=0.4,
            borderaxespad=0.4)

    # ----- events ------------------------------------------------------- #
    def connect(self):
        """One handler. Picking is done explicitly, in screen pixels.

        Matplotlib fires ``button_press_event`` BEFORE it fires the pick
        events for the same click, so a handler pair would have to guess
        whether a pick is still coming. One handler that hit-tests the stop
        markers itself is deterministic, testable with a synthetic event, and
        keeps each marker tied to its own stop_id by index.
        """
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        return self

    def hit_test_stop(self, event) -> Optional[str]:
        """The stop_id of the marker under the click, or None."""
        if getattr(event, "x", None) is None or event.y is None:
            return None
        if not len(self.positions):
            return None
        screen = self.map_axis.transData.transform(self.positions)
        gaps = np.hypot(screen[:, 0] - float(event.x),
                        screen[:, 1] - float(event.y))
        nearest = int(np.argmin(gaps))
        if gaps[nearest] <= STOP_PICK_RADIUS_PX:
            return self.stop_ids[nearest]
        return None

    def on_click(self, event) -> Optional[str]:
        """The whole state machine. Returns what the click did, for testing."""
        if getattr(event, "inaxes", None) is not self.map_axis:
            return None
        if getattr(event, "button", 1) != 1:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        stop_id = self.hit_test_stop(event)
        if stop_id is not None:
            if self.selected_origin is None:
                self.show_message(
                    "Choose an origin first: click anywhere inside the Glebe "
                    "boundary.")
                self.redraw()
                return "stop_without_origin"
            self.select_stop(stop_id)
            self.redraw()
            return "stop"

        latitude, longitude = metric_to_wgs84(event.xdata, event.ydata)
        if not inside_study_area(self.study_area, latitude, longitude):
            self.show_message(OUTSIDE_STUDY_AREA_MESSAGE)
            self.redraw()
            return "outside"
        self.set_origin(latitude, longitude)
        self.redraw()
        return "origin"

    # ----- transitions -------------------------------------------------- #
    def set_origin(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """A new origin: snap it, run the two searches once, then recompute.

        If a stop is already selected the routes are recomputed to that same
        stop straight away, so changing origin stays a single click.
        """
        self.selected_origin = snap_origin_coordinate(
            self.context["graph"], latitude, longitude)
        self.search = search_from_origin(self.context,
                                         self.selected_origin["nearest_node"])
        self.searches_run += 1
        self.current_result = None
        self.show_message(
            f"Origin snapped to node {self.selected_origin['nearest_node']}, "
            f"{self.selected_origin['snap_distance_m']:.1f} m away. "
            + SNAP_GAP_NOTE)
        if self.selected_stop is not None:
            self.recompute()
        self.draw_origin()
        self.draw_panel()
        return self.selected_origin

    def on_body_mass(self, text: Any) -> Optional[float]:
        """A new body mass: redraw the panel text, and nothing else.

        This handler deliberately touches no routing state. A bad entry is
        reported and ignored - the previously accepted mass, the origin, the
        stop and the routes all stay exactly as they were.
        """
        mass = parse_body_mass(text)
        if mass is None:
            self.show_message(BODY_MASS_INVALID)
            self.redraw()
            return None
        self.body_mass_kg = mass
        self.show_message(
            f"Body mass {format_amount(mass, 0, 3)} kg. "
            + BODY_MASS_SCOPE_NOTE)
        self.draw_panel()
        self.redraw()
        return mass

    def select_stop(self, stop_id: str) -> Optional[Dict[str, Any]]:
        """A new destination at the same origin - no search re-runs."""
        self.selected_stop = str(stop_id)
        self.recompute()
        self.draw_panel()
        return self.current_result

    def recompute(self) -> Optional[Dict[str, Any]]:
        if self.selected_origin is None or self.selected_stop is None:
            return None
        stop = find_stop(self.stops, self.selected_stop)
        self.current_result = solve_selected_stop(self.context, self.search,
                                                  stop)
        self.solves_run += 1
        self.draw_routes()
        if not self.current_result["available"]:
            self.show_message(self.current_result["headline"])
        else:
            self.show_message(
                f"Stop {self.current_result['stop_id']} - "
                f"{self.current_result['stop_name']}.  The snap gap is "
                "off-network and is not part of these numbers.")
        return self.current_result

    # ----- drawing ------------------------------------------------------ #
    def clear_artists(self, artists: List[Any]) -> None:
        while artists:
            artists.pop().remove()

    def draw_origin(self) -> None:
        self.clear_artists(self.origin_artists)
        if self.selected_origin is None:
            return
        node = self.selected_origin["nearest_node"]
        graph = self.context["graph"]
        clicked = plots.project([self.selected_origin["input_longitude"]],
                                [self.selected_origin["input_latitude"]])[0]
        snapped = plots.project([graph.nodes[node]["x"]],
                                [graph.nodes[node]["y"]])[0]
        self.origin_artists.append(self.map_axis.plot(
            [clicked[0], snapped[0]], [clicked[1], snapped[1]],
            color=COLOUR_ORIGIN, linewidth=1.0, linestyle=":", zorder=7)[0])
        self.origin_artists.append(self.map_axis.scatter(
            [snapped[0]], [snapped[1]], marker="s", s=95,
            facecolor="#ffffff", edgecolor=COLOUR_ORIGIN, linewidth=1.8,
            zorder=8))
        self.origin_artists.append(self.map_axis.annotate(
            ORIGIN_LABEL, snapped, textcoords="offset points",
            xytext=(9, -6), va="top", fontsize=8, color=COLOUR_ORIGIN,
            zorder=9))

    def clear_inset(self) -> None:
        """Remove the divergence inset, if one is on the map.

        Called before every redraw, so a new selection can never leave the
        previous inset behind or stack a second axis on top of it.
        """
        if self.inset_axis is not None:
            self.inset_axis.remove()
            self.inset_axis = None

    def draw_inset(self) -> None:
        """A small map of ONLY the sections the two routes do not share.

        The main map stays fixed on the whole study area - a divergence of a
        few metres is smaller than the line width at that scale, so it needs
        its own frame rather than a zoom of the main one.

        Nothing is routed, measured or derived here: the same edges the
        comparison already chose are redrawn with the same drawing convention
        - stored OSM geometry where present, an endpoint chord for drawing
        only where it is absent.
        """
        self.clear_inset()
        result = self.current_result
        if result is None or not result["available"] \
                or result["identical_routes"]:
            return

        divergent = divergent_edges(result)
        polylines = {
            objective: [plots.edge_drawing_geometry(self.context["graph"], *e)
                        for e in divergent[objective]]
            for objective in OBJECTIVES}
        points = np.vstack([p for objective in OBJECTIVES
                            for p, _stored in polylines[objective]])
        left, bottom = points[:, 0].min(), points[:, 1].min()
        right, top = points[:, 0].max(), points[:, 1].max()
        pad = max(INSET_PADDING_SHARE * max(right - left, top - bottom),
                  INSET_PADDING_FLOOR_M)

        axis = self.map_axis.inset_axes(INSET_RECTANGLE)
        self.inset_axis = axis
        for line in plots.context_polylines(
                self.context["graph"], (left, bottom, right, top), pad):
            axis.plot(line[:, 0], line[:, 1], color=plots.COLOUR_CONTEXT,
                      linewidth=0.9, zorder=1)
        # The shared edges that touch this window, lightly, so the reader can
        # see where the two alternatives leave and rejoin the common route.
        for edge in result["shared_edges"]:
            line, _stored = plots.edge_drawing_geometry(
                self.context["graph"], *edge)
            if (line[:, 0].max() < left - pad or line[:, 0].min() > right + pad
                    or line[:, 1].max() < bottom - pad
                    or line[:, 1].min() > top + pad):
                continue
            axis.plot(line[:, 0], line[:, 1], color=plots.COLOUR_SHARED,
                      linewidth=2.2, alpha=0.35, zorder=2,
                      solid_capstyle="round")
        for objective, colour, style, width, order in (
                (OBJECTIVE_DISTANCE, plots.COLOUR_DISTANCE,
                 plots.STYLE_DISTANCE, 3.2, 3),
                (OBJECTIVE_EFFORT, plots.COLOUR_EFFORT, plots.STYLE_EFFORT,
                 2.6, 4)):
            for line, _stored in polylines[objective]:
                axis.plot(line[:, 0], line[:, 1], color=colour,
                          linewidth=width, linestyle=style, zorder=order,
                          solid_capstyle="round")

        axis.set_xlim(left - pad, right + pad)
        axis.set_ylim(bottom - pad, top + pad)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#9a9a9a")
        axis.patch.set_alpha(0.96)
        axis.set_title(INSET_TITLE, fontsize=8, color=plots.COLOUR_INK,
                       pad=3.0)

    def draw_routes(self) -> None:
        """Replace the previous overlay - nothing is allowed to accumulate."""
        self.clear_artists(self.route_artists)
        self.clear_artists(self.highlight_artists)
        self.clear_inset()
        result = self.current_result
        if result is None:
            return

        index = self.stop_ids.index(str(result["stop_id"]))
        position = self.positions[index]
        self.highlight_artists.append(self.map_axis.scatter(
            [position[0]], [position[1]], marker="o", s=170,
            facecolor="none", edgecolor=COLOUR_STOP_SELECTED, linewidth=2.0,
            zorder=9))
        self.highlight_artists.append(self.map_axis.annotate(
            f"{result['stop_id']}\n{result['stop_name']}", position,
            textcoords="offset points", xytext=(10, -6), va="top",
            fontsize=8, color=COLOUR_STOP_SELECTED, zorder=9))
        if not result["available"]:
            return

        shared = result["shared_edges"]
        polylines = {
            objective: plots.route_polylines(
                self.context["graph"],
                hops_frame(result["routes"][objective]["hops"]))
            for objective in OBJECTIVES}
        for line in polylines[OBJECTIVE_DISTANCE]:
            if line["edge"] in shared:
                self.route_artists.append(self.map_axis.plot(
                    line["points"][:, 0], line["points"][:, 1],
                    color=plots.COLOUR_SHARED, linewidth=5.0, zorder=3,
                    solid_capstyle="round")[0])
        for objective, colour, style, order in (
                (OBJECTIVE_DISTANCE, plots.COLOUR_DISTANCE,
                 plots.STYLE_DISTANCE, 4),
                (OBJECTIVE_EFFORT, plots.COLOUR_EFFORT, plots.STYLE_EFFORT,
                 5)):
            for line in polylines[objective]:
                if line["edge"] in shared:
                    continue
                self.route_artists.append(self.map_axis.plot(
                    line["points"][:, 0], line["points"][:, 1], color=colour,
                    linewidth=2.8, linestyle=style, zorder=order,
                    solid_capstyle="round")[0])
        self.draw_inset()

    def draw_panel(self) -> None:
        """The right-hand panel, redrawn from the current state."""
        axis = self.panel_axis
        axis.clear()
        result = self.current_result

        if result is None:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_edgecolor("#cccccc")
            axis.text(0.5, 0.62, GUI_INSTRUCTIONS, ha="center", va="center",
                      fontsize=10, color=plots.COLOUR_INK,
                      transform=axis.transAxes)
            axis.text(0.5, 0.3, self.origin_summary(), ha="center",
                      va="center", fontsize=9, color=plots.COLOUR_MUTED,
                      transform=axis.transAxes)
            axis.set_title("Route comparison", fontsize=10,
                           color=plots.COLOUR_INK)
            return

        if not result["available"]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_edgecolor("#cccccc")
            axis.text(0.5, 0.62, result["headline"], ha="center", va="center",
                      fontsize=10, color=plots.COLOUR_INK, wrap=True,
                      transform=axis.transAxes)
            axis.text(0.5, 0.32, f"reason: {result['reason']}\n\n"
                      + result["note"], ha="center", va="center", fontsize=8,
                      color=plots.COLOUR_MUTED, wrap=True,
                      transform=axis.transAxes)
            axis.set_title(
                f"Stop {result['stop_id']} - {result['stop_name']}",
                fontsize=10, color=plots.COLOUR_INK)
            return

        series = {objective: plots.cumulative_series(
            hops_frame(result["routes"][objective]["hops"]))
            for objective in OBJECTIVES}
        plots.draw_cumulative(
            axis, series,
            pd.Series({"effort_saving_j_per_kg":
                       result["effort_saving_j_per_kg"]}), compact=True)
        overlap = route_overlap(result)
        axis.text(0.985, 0.02,
                  demo_comparison_text(result) + "\n\n" + overlap_text(overlap),
                  transform=axis.transAxes, ha="right", va="bottom",
                  fontsize=8.0, family="monospace", color=plots.COLOUR_INK,
                  bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.5", "alpha": 0.95})
        # The energy block sits in the empty wedge above the curves, so the
        # J/kg comparison box below keeps its place and its precedence.
        axis.text(0.025, 0.745, energy_text(result, self.body_mass_kg),
                  transform=axis.transAxes, ha="left", va="top",
                  fontsize=7.8, family="monospace", color=plots.COLOUR_INK,
                  bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.4", "alpha": 0.95})
        axis.set_title(
            f"Stop {result['stop_id']} - {result['stop_name']}",
            fontsize=10, color=plots.COLOUR_INK)

    def origin_summary(self) -> str:
        if self.selected_origin is None:
            return "No origin chosen yet."
        origin = self.selected_origin
        return (f"clicked {origin['input_latitude']:.6f}, "
                f"{origin['input_longitude']:.6f}\n"
                f"snapped node {origin['nearest_node']}, "
                f"snap gap {origin['snap_distance_m']:.2f} m\n"
                "(off-network gap; never part of a route)")

    def show_message(self, text: str) -> None:
        if self.message_text is not None:
            self.message_text.set_text(text)

    def redraw(self) -> None:
        canvas = getattr(self.figure, "canvas", None)
        if canvas is not None and hasattr(canvas, "draw_idle"):
            canvas.draw_idle()


def build_explorer(context: Dict[str, Any], stops: pd.DataFrame,
                   study_area: Dict[str, Any], figure=None
                   ) -> TerrainRouteExplorer:
    """Build the explorer, and tie its lifetime to its Figure.

    The returned object MUST be kept by the caller. Matplotlib holds the
    ``on_click`` bound method weakly, so an explorer nothing refers to is
    collected and the window goes dead-but-visible. The Figure attribute is
    the belt to the caller's braces.
    """
    explorer = TerrainRouteExplorer(context, stops, study_area, figure=figure)
    explorer.build()
    explorer.connect()
    explorer.figure._terrain_route_explorer = explorer
    return explorer


def run_graphical(context: Dict[str, Any], stops: pd.DataFrame,
                  study_area: Dict[str, Any]) -> int:
    """Open the one persistent window and hand control to the user.

    LIFETIME, and why this is not cosmetic
    --------------------------------------
    Matplotlib's callback registry holds a bound method WEAKLY. The click
    handler is ``explorer.on_click``, so if nothing else refers to the
    explorer, it is garbage-collected the moment this function's frame stops
    holding it - and the window then stays open, looking perfectly normal,
    while every click silently does nothing.

    Discarding the result of ``build_explorer`` is therefore a real bug, not a
    style choice. The explorer is bound to a local name that lives across
    ``plt.show()``, and also attached to the Figure, so it stays reachable for
    as long as the window it belongs to.
    """
    import matplotlib.pyplot as plt

    explorer = build_explorer(context, stops, study_area)
    explorer.figure._terrain_route_explorer = explorer
    print("")
    print("A map window is open. " + GUI_INSTRUCTIONS.replace("\n", " "))
    print("Close the window to finish.")
    # `explorer` is a live local for the whole of this call, so the frame
    # itself holds a strong reference across the event loop below.
    plt.show()
    return 0


# --------------------------------------------------------------------------- #
# terminal fallback session
# --------------------------------------------------------------------------- #
def format_menu(stops: pd.DataFrame) -> str:
    lines = [f"  [{index + 1}] {row.stop_id}  {row.stop_name}"
             for index, row in enumerate(menu_rows(stops).itertuples())]
    return "\n".join(lines)


def ask_float(prompt: str, reader) -> float:
    while True:
        raw = reader(prompt)
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            print("  Please enter a number, for example -33.8795.")


def ask_stop(stops: pd.DataFrame, reader) -> pd.Series:
    ordered = menu_rows(stops)
    print("")
    print("Candidate stops (existing Stage 4 snapped stops):")
    print(format_menu(stops))
    print(f"  {MENU_ORDER_NOTE}")
    while True:
        raw = str(reader("Choose a stop by number or stop_id: ")).strip()
        # Menu indices are 1..N and stop_ids are six digits, so the two
        # cannot collide; the index is tried first.
        if raw.isdigit() and 1 <= int(raw) <= len(ordered):
            return ordered.iloc[int(raw) - 1]
        try:
            return find_stop(stops, raw)
        except DemoError:
            print("  Not a listed number or stop_id. Try again.")


def run_interactive(context: Dict[str, Any], stops: pd.DataFrame,
                    reader=input, save_path: Optional[Path] = None) -> int:
    print("Week 5 terrain-aware walking demo (presentation layer).")
    print("Enter a coordinate in or near Glebe. It is snapped with the Stage 4")
    print("snapping code; the snap gap is reported separately and is never")
    print("added to a route distance.")
    print("")
    latitude = ask_float("Latitude:  ", reader)
    longitude = ask_float("Longitude: ", reader)
    stop = ask_stop(stops, reader)
    return run_once(context, stops, latitude, longitude, str(stop["stop_id"]),
                    save_path=save_path)


# --------------------------------------------------------------------------- #
def run_once(context: Dict[str, Any], stops: pd.DataFrame, latitude: float,
             longitude: float, stop_id: str, plot: bool = True,
             save_path: Optional[Path] = None) -> int:
    """One demo comparison, printed and optionally drawn."""
    stop = find_stop(stops, stop_id)
    origin = snap_origin_coordinate(context["graph"], latitude, longitude)
    result = compare_from_origin(context, origin["nearest_node"], stop)
    print("")
    print(format_report(origin, result))

    after = audit.file_sha256(context["graph_path"])
    if after != context["graph_sha256"]:
        raise DemoError(f"{context['graph_path']} changed while the demo ran. "
                        "It is read-only.")

    if result["available"] and (plot or save_path is not None):
        figure = build_figure(context, origin, result)
        written = show_or_save(figure, save_path)
        if written is not None:
            print(f"Figure written to {written}")
    return 0 if result["available"] else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 5 demo: distance route against terrain-effort route "
                    "from a coordinate you choose. Presentation layer only.")
    parser.add_argument("--interactive", action="store_true",
                        help="open the clickable Glebe map")
    parser.add_argument("--text-prompts", action="store_true",
                        help="fallback for a machine with no display: type a "
                             "latitude, longitude and stop instead of "
                             "clicking")
    parser.add_argument("--lat", type=float, default=None,
                        help="origin latitude (WGS84)")
    parser.add_argument("--lon", type=float, default=None,
                        help="origin longitude (WGS84)")
    parser.add_argument("--stop-id", default=None,
                        help="a stop_id from data/cache/glebe_stops_snapped.csv")
    parser.add_argument("--graph", default=audit.DEFAULT_GRAPH)
    parser.add_argument("--cost-dir", default=None)
    parser.add_argument("--stops-csv", default=None)
    parser.add_argument("--boundary", default=None,
                        help="override the official Glebe polygon file")
    parser.add_argument("--no-plot", action="store_true",
                        help="print the comparison without opening a window")
    parser.add_argument("--save", default=None,
                        help="write the figure to this path instead of "
                             "showing it")
    options = parser.parse_args(argv)

    non_interactive = (options.lat is not None and options.lon is not None
                       and options.stop_id is not None)
    if options.interactive == non_interactive:
        parser.error("pass either --interactive, or all of --lat, --lon and "
                     "--stop-id")

    graph_path = Path(options.graph)
    if not graph_path.is_absolute():
        graph_path = (CONFIG_ROOT / graph_path).resolve()
    cost_directory = Path(options.cost_dir).resolve() if options.cost_dir \
        else terrain_routing.cost_dir()
    stops_csv = Path(options.stops_csv).resolve() if options.stops_csv \
        else stops_path()
    save_path = Path(options.save).resolve() if options.save else None

    try:
        stops = load_candidate_stops(stops_csv)
        context = build_context(graph_path, cost_directory)
    except (DemoError, terrain_routing.TerrainRoutingError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Graph        {context['graph'].number_of_nodes()} nodes, "
          f"{context['graph'].number_of_edges()} directed edges")
    print(f"             sha256 {context['graph_sha256']}")
    print(f"Trusted cost {context['provenance']['trusted_edges']} directed "
          "edges carry an adopted terrain cost")
    print("Both objectives run on exactly this subgraph. The second metric is "
          "a tie-break only.")
    print("This is a demo layer: no DEM sampling, no new terrain cost, no stop "
          "ranking, no research output written.")

    try:
        if options.interactive and options.text_prompts:
            return run_interactive(context, stops, save_path=save_path)
        if options.interactive:
            study_area = load_study_area(
                Path(options.boundary) if options.boundary else None)
            return run_graphical(context, stops, study_area)
        return run_once(context, stops, options.lat, options.lon,
                        options.stop_id, plot=not options.no_plot,
                        save_path=save_path)
    except (DemoError, terrain_routing.TerrainRoutingError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
