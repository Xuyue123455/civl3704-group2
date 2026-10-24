"""Week 5C-2C - static figures for the Week 5C-2B route comparison.

What these figures have to make obvious
---------------------------------------
That minimising walking EFFORT picks a different way through Glebe than
minimising DISTANCE, and that the swap costs almost nothing in metres. For stop
203732 the two routes differ by 0.185 m of walking and 10.470 J/kg of effort,
so the map alone cannot carry the argument - 0.185 m is invisible at this
scale. The cumulative panel beside it is what shows the difference.

Nothing is recomputed
---------------------
The routes are read back from ``route_edges.csv`` as exact ``(u, v, key)``
sequences. No Dijkstra runs here, no DEM is sampled, and no terrain cost is
derived. GraphML geometry is used FOR DRAWING ONLY, and where an edge has none
its two endpoint nodes are joined by a straight line - again for drawing only.
Nothing is written back to the graph.

Provenance
----------
The gradient evidence on the figure comes from ``edge_terrain_cost.csv``, so it
has to be the SAME cost table Week 5C-2B routed on. ``terrain_routing_summary``
``.json`` is required, and both of its recorded hashes must still hold: the raw
GraphML's, and the cost table's. A stale table is a hard failure before any PNG
is written - it is NOT quietly dropped from the figure, because an annotation
that silently disappears is indistinguishable from an edge with no gradient.

The "maximum effort saving" label on the secondary figure is a DATA CLAIM, so
it is re-derived from ``route_comparison.csv`` every run: the comparable count
is counted, the maximum is taken, and stop 203717 must still hold it. The count
in the title is that number, never a literal.

Precision
---------
``route_edges.csv`` holds the exact hop values. ``route_comparison.csv`` is a
ROUNDED reporting table - distances to 3 decimals, efforts to 4 - so the totals
are checked the way that table was written:

    round(sum(hop length_m), 3)                == *_route_distance_m
    round(sum(hop terrain_effort_j_per_kg), 4) == *_route_effort_j_per_kg

Comparing raw sums to the rounded table with a 1e-6 tolerance would fail on
arithmetic that is in fact correct.

Records versus unique edges
---------------------------
An edge shared by both routes appears once in each route's records. Stop 203732
has 15 route-edge RECORDS over 14 UNIQUE directed edges, because exactly one
edge is shared. Both counts are reported separately, including for the
stored-geometry breakdown, so neither can be mistaken for the other.

Two things the figures must not say
-----------------------------------
Both routes have a mean cost per metre above the 2.5 J/kg/m level-walking
baseline. That means only what it says: the mean metabolic cost per metre is
above the level baseline. It does NOT establish that either route is net
uphill, because steep enough descent also costs more than level walking. A
specific edge is described as climbing only where its own stored signed
gradient supports it.

And the annotated edge is the one with the highest terrain-to-level COST RATIO,
which is not necessarily the one with the largest gradient. It is labelled
"highest-cost edge", never "steepest".

And stop 203769 is not drawn. Its distance route and effort route are the same
21 edges with zero effort saving; its +17.498 m is a trusted-cost model-coverage
restriction, not terrain optimisation. It appears only as a caveat record.

Usage
-----
    python scripts/plot_terrain_routes.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from src.osm_network import load_graph  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The terrain-cost stage carries the trusted-status list, the metric CRS and the
# small file helpers, so none of them is restated here.
terrain_cost = load_script("build_terrain_cost")
audit = terrain_cost.audit
week5a = audit.week5a

TRUSTED_STATUSES: Tuple[str, ...] = terrain_cost.TRUSTED_STATUSES
METRIC_CRS = week5a.METRIC_CRS                   # EPSG:7856
WGS84 = "EPSG:4326"

OUTPUT_SUBDIR = "terrain_visualisation"
ROUTING_SUBDIR = "terrain_routing"
COST_SUBDIR = terrain_cost.OUTPUT_SUBDIR

COMPARISON_FILENAME = "route_comparison.csv"
EDGES_FILENAME = "route_edges.csv"
ROUTING_SUMMARY_FILENAME = "terrain_routing_summary.json"
COST_FILENAME = "edge_terrain_cost.csv"

COMPARED_STATUS = "compared"

PRIMARY_STOP = 203732
SECONDARY_STOP = 203717
CAVEAT_STOP = 203769

OBJECTIVE_DISTANCE = "distance"
OBJECTIVE_EFFORT = "effort"
OBJECTIVES: Tuple[str, ...] = (OBJECTIVE_DISTANCE, OBJECTIVE_EFFORT)

# route_comparison.csv is a rounded reporting table.
DISTANCE_DECIMALS = 3
EFFORT_DECIMALS = 4
ROUNDING_EPSILON = 1e-9

# Level walking, from the Minetti polynomial at zero gradient.
LEVEL_COST_J_PER_KG_PER_M = 2.5

MAP_MARGIN_M = 45.0

# Validated with the dataviz palette checker (light surface, categorical):
# CVD separation dE 22.4 protan / 31.5 tritan, normal-vision dE 31.2, all five
# checks PASS. Colour is never the only encoding - line style and direct end
# labels carry the same distinction in print and for colour-vision deficiency.
COLOUR_DISTANCE = "#2171b5"
COLOUR_EFFORT = "#d95f02"
COLOUR_SHARED = "#525252"
COLOUR_CONTEXT = "#d4d4d4"
COLOUR_INK = "#1a1a1a"
COLOUR_MUTED = "#555555"

STYLE_DISTANCE = "-"
STYLE_EFFORT = (0, (6, 3))

LABEL_DISTANCE = "distance-optimal route"
LABEL_EFFORT = "effort-optimal route"
LABEL_SHARED = "shared by both routes"
LABEL_CONTEXT = "raw OSM pedestrian network (context only, not the routing graph)"

# Short direct labels on the cumulative terminal markers. The numeric totals
# live in the comparison box and are deliberately not repeated here.
TERMINAL_LABEL_DISTANCE = "shortest"
TERMINAL_LABEL_EFFORT = "least effort"

# The effort-difference bracket, placed clear of both curves.
BRACKET_LABEL_PREFIX = "Δ effort = "
BRACKET_X_FACTOR = 1.085              # of the longer route's total distance
BRACKET_CAP_HALF_WIDTH_FACTOR = 0.009
BRACKET_LABEL_GAP_PT = 9.0            # points between the cap and the text
X_HEADROOM_FACTOR = 1.60              # room for the bracket and its label
Y_HEADROOM_FACTOR = 1.09              # 9 % above the highest plotted value

# Parked in the empty wedge above the curves and below the legend, with a
# leader line down to its edge. Below the curve is where the comparison box
# lives, and the two collided there at every offset that still fit the axes.
HIGHEST_COST_LABEL_AXES_XY = (0.03, 0.86)

CHORD_NOTE = (
    "Edges without stored OSM geometry are drawn as a straight line between "
    "their endpoint nodes, for drawing only."
)

CAVEAT_203769 = (
    "Stop 203769 (Parramatta Rd Opp University Of Sydney Footbridge) has a "
    "restriction effect of +17.498 m, but its distance route and effort route "
    "are the SAME 21 directed edges and its effort saving is 0.0 J/kg. That "
    "figure is trusted-cost model coverage, not terrain optimisation, so no "
    "route-comparison figure is drawn for it."
)


class VisualisationError(RuntimeError):
    """Raised with an actionable message when a figure cannot be trusted."""


# --------------------------------------------------------------------------- #
# reading the Week 5C-2B outputs
# --------------------------------------------------------------------------- #
def routing_dir(root: Path = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache"
            / ROUTING_SUBDIR).resolve()


def load_routing_outputs(source_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparison = Path(source_dir) / COMPARISON_FILENAME
    edges = Path(source_dir) / EDGES_FILENAME
    for path in (comparison, edges):
        if not path.exists():
            raise VisualisationError(
                f"{path} is missing. Run the route comparison first:\n"
                "    python scripts/compare_terrain_routes.py")
    return pd.read_csv(comparison), pd.read_csv(edges)


def load_routing_summary(source_dir: Path) -> Dict[str, Any]:
    """The Week 5C-2B provenance record. Required, not optional."""
    path = Path(source_dir) / ROUTING_SUMMARY_FILENAME
    if not path.exists():
        raise VisualisationError(
            f"{path} is missing, so the figures cannot prove which cost table "
            "the routes were derived from. Run the route comparison first:\n"
            "    python scripts/compare_terrain_routes.py")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_cost_provenance(
    routing_summary: Dict[str, Any], graph_sha256: str, cost_path: Path,
) -> Dict[str, str]:
    """The cost table on the figure must be the one Week 5C-2B routed on.

    The annotation quotes a terrain-to-level ratio and a signed gradient range.
    Both are properties of a particular cost table, so a table that has been
    rebuilt since the routing run would put numbers on the figure that do not
    belong to the drawn routes.

    A mismatch is a HARD FAILURE. The annotation is not silently dropped: a
    missing annotation looks exactly like an edge that had no gradient
    evidence, so it would hide the problem rather than report it.

    Raises:
        VisualisationError: if the summary is missing either hash, if the raw
            graph has changed since the routing run, or if the cost table has.
    """
    recorded_graph = routing_summary.get("graph", {}).get("sha256")
    recorded_cost = routing_summary.get("edge_terrain_cost_sha256")
    if not recorded_graph or not recorded_cost:
        raise VisualisationError(
            f"{ROUTING_SUMMARY_FILENAME} does not record both "
            "graph.sha256 and edge_terrain_cost_sha256, so the cost table "
            "cannot be tied to the routes. Re-run the route comparison.")

    if recorded_graph != graph_sha256:
        raise VisualisationError(
            "The raw graph has changed since the routing run: "
            f"{ROUTING_SUMMARY_FILENAME} records {recorded_graph}, the graph "
            f"on disk hashes to {graph_sha256}. The stored routes may no "
            "longer describe this graph, so no figure is drawn.")

    if not cost_path.exists():
        raise VisualisationError(
            f"{cost_path} is missing, but the figure quotes terrain-cost "
            "evidence from it. Run the terrain-cost build first:\n"
            "    python scripts/build_terrain_cost.py")
    cost_sha256 = audit.file_sha256(cost_path)
    if recorded_cost != cost_sha256:
        raise VisualisationError(
            "The terrain-cost table has changed since the routing run: "
            f"{ROUTING_SUMMARY_FILENAME} records {recorded_cost}, "
            f"{cost_path.name} on disk hashes to {cost_sha256}. The gradient "
            "evidence on the figure would not belong to the drawn routes, so "
            "no figure is drawn.")

    return {"graph_sha256": graph_sha256,
            "edge_terrain_cost_sha256": cost_sha256}


def verify_secondary_case(
    comparison: pd.DataFrame, stop_id: int
) -> Dict[str, Any]:
    """Re-derive the "maximum effort saving" claim from the current table.

    The secondary figure's title asserts something about the whole result set,
    so it is checked against the result set every run rather than trusted from
    the run that first produced it. The comparable count it prints is counted
    here too - it is never a literal in the drawing code.

    Raises:
        VisualisationError: if the stop is not comparable, or no longer holds
            the maximum. A false "maximum" label is worse than no figure.
    """
    compared = comparison[comparison["comparison_status"] == COMPARED_STATUS]
    if compared.empty:
        raise VisualisationError(
            "route_comparison.csv has no comparable stop, so no maximum "
            "effort saving exists.")
    savings = compared["effort_saving_j_per_kg"].astype(float)
    maximum = float(savings.max())

    target = compared[compared["stop_id"] == stop_id]
    if target.empty:
        raise VisualisationError(
            f"Stop {stop_id} is not a comparable stop, so it cannot hold the "
            "maximum effort saving.")
    saving = float(target.iloc[0]["effort_saving_j_per_kg"])

    if round(saving, EFFORT_DECIMALS) != round(maximum, EFFORT_DECIMALS):
        holder = compared.loc[savings.idxmax()]
        raise VisualisationError(
            f"Stop {stop_id} saves {saving} J/kg, but the maximum among the "
            f"{len(compared)} comparable stops is {maximum} J/kg at stop "
            f"{int(holder['stop_id'])}. The figure would carry a false "
            "'maximum' label, so it is not drawn.")

    return {
        "comparable_count": int(len(compared)),
        "max_effort_saving_j_per_kg": maximum,
        "effort_saving_j_per_kg": saving,
        "holds_the_maximum": True,
    }


def route_records(
    edge_table: pd.DataFrame, stop_id: int, metric: str
) -> pd.DataFrame:
    """One route's hops, in hop order, exactly as Week 5C-2B recorded them."""
    part = edge_table[(edge_table["stop_id"] == stop_id)
                      & (edge_table["metric"] == metric)]
    if part.empty:
        raise VisualisationError(
            f"route_edges.csv has no {metric!r} hops for stop {stop_id}.")
    ordered = part.sort_values("hop_index").reset_index(drop=True)
    if list(ordered["hop_index"]) != list(range(len(ordered))):
        raise VisualisationError(
            f"Stop {stop_id} {metric}: hop_index is not 0..{len(ordered) - 1}.")
    return ordered


def comparison_row(comparison: pd.DataFrame, stop_id: int) -> pd.Series:
    part = comparison[comparison["stop_id"] == stop_id]
    if part.empty:
        raise VisualisationError(
            f"route_comparison.csv has no row for stop {stop_id}.")
    return part.iloc[0]


# --------------------------------------------------------------------------- #
# verification, precision-aware
# --------------------------------------------------------------------------- #
def verify_route(
    records: pd.DataFrame, row: pd.Series, metric: str, graph,
) -> Dict[str, Any]:
    """Every check a plotted route must pass before a single line is drawn.

    The totals are compared the way ``route_comparison.csv`` was written -
    rounded to its own precision - rather than against the raw sums, because
    that table stores distances to 3 decimals and efforts to 4. A universal
    1e-6 tolerance would reject arithmetic that is in fact exact.

    Raises:
        VisualisationError: on any failure. A route that cannot be reproduced
            from its own edge list is not drawn.
    """
    label = f"stop {row['stop_id']} {metric} route"

    for hop in records.itertuples():
        edge = (int(hop.u), int(hop.v), int(hop.key))
        if hop.terrain_cost_status not in TRUSTED_STATUSES:
            raise VisualisationError(
                f"{label}: hop {hop.hop_index} uses {edge} with status "
                f"{hop.terrain_cost_status!r}, which is not a trusted terrain "
                "cost. Untrusted edges must never be plotted as a route.")
        if not graph.has_edge(*edge):
            raise VisualisationError(
                f"{label}: hop {hop.hop_index} uses {edge}, which is not in "
                "the raw graph.")

    nodes = [int(records.iloc[0]["u"])] + [int(v) for v in records["v"]]
    for index, hop in enumerate(records.itertuples()):
        if int(hop.u) != nodes[index] or int(hop.v) != nodes[index + 1]:
            raise VisualisationError(
                f"{label}: hop {index} does not join the node sequence.")

    reported_distance = float(row[f"{metric}_route_distance_m"])
    reported_effort = float(row[f"{metric}_route_effort_j_per_kg"])
    summed_distance = float(records["length_m"].sum())
    summed_effort = float(records["terrain_effort_j_per_kg"].sum())

    if abs(round(summed_distance, DISTANCE_DECIMALS)
           - reported_distance) > ROUNDING_EPSILON:
        raise VisualisationError(
            f"{label}: the hops sum to {summed_distance!r} m, which rounds to "
            f"{round(summed_distance, DISTANCE_DECIMALS)!r}, but the "
            f"comparison table reports {reported_distance!r} m.")
    if abs(round(summed_effort, EFFORT_DECIMALS)
           - reported_effort) > ROUNDING_EPSILON:
        raise VisualisationError(
            f"{label}: the hops sum to {summed_effort!r} J/kg, which rounds "
            f"to {round(summed_effort, EFFORT_DECIMALS)!r}, but the "
            f"comparison table reports {reported_effort!r} J/kg.")

    if int(row["nearest_node"]) != nodes[-1]:
        raise VisualisationError(
            f"{label}: the route ends at node {nodes[-1]}, not at the snapped "
            f"stop node {int(row['nearest_node'])}.")

    return {
        "metric": metric,
        "hops": int(len(records)),
        "summed_distance_m": summed_distance,
        "summed_effort_j_per_kg": summed_effort,
        "reported_distance_m": reported_distance,
        "reported_effort_j_per_kg": reported_effort,
        "origin_node": nodes[0],
        "destination_node": nodes[-1],
    }


def edge_keys(records: pd.DataFrame) -> List[Tuple[int, int, int]]:
    return [(int(r.u), int(r.v), int(r.key)) for r in records.itertuples()]


def geometry_counts(
    graph, routes: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Stored-geometry breakdown, reported for records AND for unique edges.

    A shared edge occurs once in each route's records, so the two bases differ
    and each must be named wherever it is quoted.
    """
    records = [key for metric in OBJECTIVES for key in edge_keys(routes[metric])]
    unique = sorted(set(records))
    stored_records = sum(
        1 for key in records if graph.edges[key].get("geometry") is not None)
    stored_unique = sum(
        1 for key in unique if graph.edges[key].get("geometry") is not None)
    return {
        "route_edge_records": len(records),
        "unique_directed_edges": len(unique),
        "shared_directed_edges": len(records) - len(unique),
        "records_with_stored_geometry": stored_records,
        "records_drawn_as_endpoint_chord": len(records) - stored_records,
        "unique_edges_with_stored_geometry": stored_unique,
        "unique_edges_drawn_as_endpoint_chord": len(unique) - stored_unique,
    }


# --------------------------------------------------------------------------- #
# geometry, projected for drawing only
# --------------------------------------------------------------------------- #
_TO_METRIC = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
_TO_WGS84 = Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True)


def project(longitudes: Sequence[float], latitudes: Sequence[float]) -> np.ndarray:
    """Project lon/lat to the project's metric CRS, for drawing only.

    EPSG:7856 is the CRS the project already uses for every metric operation,
    so a metre on the page is a metre on the ground and a scale bar can be
    honest. Plotting raw degrees at this latitude would compress the map by
    about a fifth horizontally.
    """
    xs, ys = _TO_METRIC.transform(np.asarray(longitudes, dtype=float),
                                  np.asarray(latitudes, dtype=float))
    return np.column_stack((np.asarray(xs, dtype=float),
                            np.asarray(ys, dtype=float)))


def edge_drawing_geometry(
    graph, u: int, v: int, key: int
) -> Tuple[np.ndarray, bool]:
    """One edge's drawing polyline in the metric CRS.

    Returns:
        Tuple of (projected coordinates, whether OSM stored a geometry). When
        it did not, the two endpoint nodes are joined by a straight line - a
        DRAWING fallback only. Nothing is written back to the graph and no
        topology changes.
    """
    data = graph.edges[u, v, key]
    geometry = data.get("geometry")
    if geometry is not None:
        coordinates = list(geometry.coords)
        return project([x for x, _y in coordinates],
                       [y for _x, y in coordinates]), True
    return project([graph.nodes[u]["x"], graph.nodes[v]["x"]],
                   [graph.nodes[u]["y"], graph.nodes[v]["y"]]), False


def route_polylines(
    graph, records: pd.DataFrame
) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for hop in records.itertuples():
        key = (int(hop.u), int(hop.v), int(hop.key))
        points, stored = edge_drawing_geometry(graph, *key)
        lines.append({"edge": key, "points": points,
                      "has_stored_geometry": stored})
    return lines


def context_polylines(
    graph, bounds: Tuple[float, float, float, float], margin_m: float,
) -> List[np.ndarray]:
    """Raw-graph edges near the routes, for background context only.

    Selected by node position so the whole graph does not have to be projected.
    """
    left, bottom, right, top = bounds
    corners = _TO_WGS84.transform(
        [left - margin_m, right + margin_m, left - margin_m, right + margin_m],
        [bottom - margin_m, bottom - margin_m, top + margin_m, top + margin_m])
    west, east = min(corners[0]), max(corners[0])
    south, north = min(corners[1]), max(corners[1])

    inside = {
        node for node, data in graph.nodes(data=True)
        if west <= float(data["x"]) <= east and south <= float(data["y"]) <= north
    }
    lines: List[np.ndarray] = []
    for u, v, key in graph.edges(keys=True):
        if u not in inside and v not in inside:
            continue
        points, _stored = edge_drawing_geometry(graph, u, v, key)
        lines.append(points)
    return lines


def route_bounds(
    polylines: Sequence[Dict[str, Any]]
) -> Tuple[float, float, float, float]:
    stacked = np.vstack([line["points"] for line in polylines])
    return (float(stacked[:, 0].min()), float(stacked[:, 1].min()),
            float(stacked[:, 0].max()), float(stacked[:, 1].max()))


# --------------------------------------------------------------------------- #
# cumulative effort against cumulative distance
# --------------------------------------------------------------------------- #
def cumulative_series(records: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative distance and effort along a route, starting at (0, 0).

    Each straight segment's slope is that edge's mean terrain cost per metre,
    so a steep segment is a costly edge. Nothing new is computed: the values
    are the hop costs Week 5C-2B recorded.
    """
    distance = np.concatenate(
        ([0.0], np.cumsum(records["length_m"].to_numpy(dtype=float))))
    effort = np.concatenate(
        ([0.0], np.cumsum(
            records["terrain_effort_j_per_kg"].to_numpy(dtype=float))))
    return distance, effort


def highest_cost_segment(
    records: pd.DataFrame, cost_table: Optional[pd.DataFrame]
) -> Optional[Dict[str, Any]]:
    """The route's most expensive edge per metre, from the existing cost table.

    Selection is by ``terrain_to_level_ratio`` - the highest METABOLIC COST per
    metre relative to level walking. That is not the same edge as the one with
    the largest signed or absolute gradient: a long gentle climb can cost more
    per metre than a short sharp one, and a steep descent can cost less than a
    mild ascent. The figure therefore calls this the highest-cost edge, never
    the steepest. Its gradient range is quoted as evidence, not as the
    selection criterion.

    Returns None when no cost table is supplied, so a caller that has decided
    not to use cost-table evidence at all gets one annotation fewer rather than
    an invented number. A caller that DOES use it must verify the table's
    provenance first - see ``verify_cost_provenance``.
    """
    if cost_table is None:
        return None
    ratios = {(int(r.u), int(r.v), int(r.key)): r
              for r in cost_table.itertuples()}
    best = None
    for index, hop in enumerate(records.itertuples()):
        entry = ratios.get((int(hop.u), int(hop.v), int(hop.key)))
        if entry is None or pd.isna(entry.terrain_to_level_ratio):
            continue
        if best is None or entry.terrain_to_level_ratio > best["ratio"]:
            best = {
                "hop_index": index,
                "edge": (int(hop.u), int(hop.v), int(hop.key)),
                "ratio": float(entry.terrain_to_level_ratio),
                "length_m": float(hop.length_m),
                "min_gradient_pct": (float(entry.min_gradient_pct)
                                     if pd.notna(entry.min_gradient_pct)
                                     else None),
                "max_gradient_pct": (float(entry.max_gradient_pct)
                                     if pd.notna(entry.max_gradient_pct)
                                     else None),
            }
    return best


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def nice_scale_length(span_m: float) -> float:
    """A round scale-bar length of roughly a quarter of the map's width."""
    target = max(span_m / 4.0, 1.0)
    for candidate in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if candidate >= target:
            return float(candidate)
    return float(candidate)


def draw_scale_bar(axis, bounds, colour=COLOUR_INK) -> float:
    left, bottom, right, _top = bounds
    length = nice_scale_length(right - left)
    x0 = left + 0.06 * (right - left)
    y0 = bottom + 0.05 * (bounds[3] - bottom)
    axis.plot([x0, x0 + length], [y0, y0], color=colour, linewidth=2.4,
              solid_capstyle="butt", zorder=6)
    axis.text(x0 + length / 2.0, y0 + 0.012 * (bounds[3] - bottom),
              f"{length:g} m", ha="center", va="bottom", fontsize=8,
              color=colour, zorder=6)
    return length


def draw_map(axis, graph, routes, polylines, shared, row, bounds) -> None:
    from matplotlib.lines import Line2D

    padded = (bounds[0] - MAP_MARGIN_M, bounds[1] - MAP_MARGIN_M,
              bounds[2] + MAP_MARGIN_M, bounds[3] + MAP_MARGIN_M)
    for points in context_polylines(graph, bounds, MAP_MARGIN_M):
        axis.plot(points[:, 0], points[:, 1], color=COLOUR_CONTEXT,
                  linewidth=1.0, zorder=1, solid_capstyle="round")

    for line in polylines[OBJECTIVE_DISTANCE]:
        if line["edge"] in shared:
            axis.plot(line["points"][:, 0], line["points"][:, 1],
                      color=COLOUR_SHARED, linewidth=5.5, zorder=2,
                      solid_capstyle="round")
    for metric, colour, style, width, order in (
            (OBJECTIVE_DISTANCE, COLOUR_DISTANCE, STYLE_DISTANCE, 3.0, 3),
            (OBJECTIVE_EFFORT, COLOUR_EFFORT, STYLE_EFFORT, 3.0, 4)):
        for line in polylines[metric]:
            if line["edge"] in shared:
                continue
            axis.plot(line["points"][:, 0], line["points"][:, 1],
                      color=colour, linewidth=width, linestyle=style,
                      zorder=order, solid_capstyle="round")

    origin_node = int(routes[OBJECTIVE_DISTANCE].iloc[0]["u"])
    stop_node = int(row["nearest_node"])
    origin_xy = project([graph.nodes[origin_node]["x"]],
                        [graph.nodes[origin_node]["y"]])[0]
    stop_xy = project([graph.nodes[stop_node]["x"]],
                      [graph.nodes[stop_node]["y"]])[0]
    axis.scatter([origin_xy[0]], [origin_xy[1]], marker="s", s=90,
                 facecolor="#ffffff", edgecolor=COLOUR_INK, linewidth=1.6,
                 zorder=7)
    axis.scatter([stop_xy[0]], [stop_xy[1]], marker="^", s=130,
                 facecolor="#ffffff", edgecolor=COLOUR_INK, linewidth=1.6,
                 zorder=7)
    axis.annotate("A-02 provisional test origin", origin_xy,
                  textcoords="offset points", xytext=(10, 6), fontsize=8.5,
                  color=COLOUR_INK, zorder=8)
    axis.annotate(f"stop {int(row['stop_id'])}\n{row['stop_name']}", stop_xy,
                  textcoords="offset points", xytext=(10, -18), fontsize=8.5,
                  color=COLOUR_INK, zorder=8)

    axis.set_xlim(padded[0], padded[2])
    axis.set_ylim(padded[1], padded[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#cccccc")
    draw_scale_bar(axis, padded)
    axis.set_title(f"Routes from the A-02 origin to stop {int(row['stop_id'])}",
                   fontsize=10, color=COLOUR_INK)

    axis.legend(handles=[
        Line2D([0], [0], color=COLOUR_SHARED, linewidth=5.5,
               label=LABEL_SHARED),
        Line2D([0], [0], color=COLOUR_DISTANCE, linewidth=3.0,
               linestyle=STYLE_DISTANCE, label=LABEL_DISTANCE),
        Line2D([0], [0], color=COLOUR_EFFORT, linewidth=3.0,
               linestyle=STYLE_EFFORT, label=LABEL_EFFORT),
        Line2D([0], [0], color=COLOUR_CONTEXT, linewidth=1.0,
               label=LABEL_CONTEXT),
    # Same four entries, same wording - only smaller and tighter, so it covers
    # less of the map. The scale bar sits bottom-left, so this stays top-left.
    ], loc="upper left", fontsize=6.6, framealpha=0.92, handlelength=1.5,
        handletextpad=0.5, labelspacing=0.3, borderpad=0.4,
        borderaxespad=0.4)


def draw_cumulative(
    axis, series, row, highest_cost=None, compact=False,
) -> None:
    """Cumulative effort against cumulative distance, both routes.

    The two routes end at DIFFERENT cumulative distances (206.916 m against
    207.101 m for stop 203732), so the vertical gap between the terminal points
    is not the effort difference. The difference is therefore stated separately
    by a bracket to the right of both curves, which does not depend on x.

    The terminal markers carry only a short name - "shortest" and "least
    effort". Their numeric totals are in the comparison box, and printing them
    twice both crowded the terminal markers and invited the reader to subtract
    two numbers that sit at different x.
    """
    for metric, colour, style, label in (
            (OBJECTIVE_DISTANCE, COLOUR_DISTANCE, STYLE_DISTANCE,
             LABEL_DISTANCE),
            (OBJECTIVE_EFFORT, COLOUR_EFFORT, STYLE_EFFORT, LABEL_EFFORT)):
        distance, effort = series[metric]
        axis.plot(distance, effort, color=colour, linewidth=2.0,
                  linestyle=style, label=label, zorder=3)
        axis.scatter([distance[-1]], [effort[-1]], s=46, facecolor="#ffffff",
                     edgecolor=colour, linewidth=1.8, zorder=5)

    axis.scatter([0.0], [0.0], s=40, facecolor="#ffffff",
                 edgecolor=COLOUR_INK, linewidth=1.4, zorder=5)

    ends = {metric: (series[metric][0][-1], series[metric][1][-1])
            for metric in OBJECTIVES}
    # Short names only. Every number they would have repeated is in the
    # comparison box.
    axis.annotate(TERMINAL_LABEL_DISTANCE, ends[OBJECTIVE_DISTANCE],
                  textcoords="offset points", xytext=(-8, 9), ha="right",
                  fontsize=8, color=COLOUR_DISTANCE)
    axis.annotate(TERMINAL_LABEL_EFFORT, ends[OBJECTIVE_EFFORT],
                  textcoords="offset points", xytext=(10, -30), ha="left",
                  va="top", fontsize=8, color=COLOUR_EFFORT)

    # A bracket clear of both curves: the effort difference, independent of
    # where each route happens to end on the x axis.
    lower = min(ends[OBJECTIVE_DISTANCE][1], ends[OBJECTIVE_EFFORT][1])
    upper = max(ends[OBJECTIVE_DISTANCE][1], ends[OBJECTIVE_EFFORT][1])
    span = max(ends[OBJECTIVE_DISTANCE][0], ends[OBJECTIVE_EFFORT][0])
    bracket_x = span * BRACKET_X_FACTOR
    cap_half_width = span * BRACKET_CAP_HALF_WIDTH_FACTOR
    axis.plot([bracket_x, bracket_x], [lower, upper], color=COLOUR_INK,
              linewidth=1.2, zorder=6)
    for height in (lower, upper):
        axis.plot([bracket_x - cap_half_width, bracket_x + cap_half_width],
                  [height, height], color=COLOUR_INK, linewidth=1.2, zorder=6)
    # The caps stop well short of the text, so a cap and the "=" cannot read
    # as "==".
    axis.annotate(
        f"{BRACKET_LABEL_PREFIX}"
        f"{float(row['effort_saving_j_per_kg']):.3f} J/kg",
        (bracket_x + cap_half_width, (lower + upper) / 2.0),
        textcoords="offset points", xytext=(BRACKET_LABEL_GAP_PT, 0),
        va="center", ha="left", fontsize=8, color=COLOUR_INK)
    axis.set_xlim(-span * 0.03, span * X_HEADROOM_FACTOR)
    # Deterministic top padding, so a terminal marker never touches the title.
    axis.set_ylim(-upper * 0.025, upper * Y_HEADROOM_FACTOR)

    # Selected by cost ratio, so it is named by cost ratio. The gradient range
    # below it is supporting evidence, not what picked the edge.
    if highest_cost is not None and not compact:
        distance, effort = series[OBJECTIVE_DISTANCE]
        index = highest_cost["hop_index"]
        midpoint = ((distance[index] + distance[index + 1]) / 2.0,
                    (effort[index] + effort[index + 1]) / 2.0)
        gradient = ""
        if highest_cost["min_gradient_pct"] is not None:
            gradient = (f"\n{highest_cost['min_gradient_pct']:+.1f} to "
                        f"{highest_cost['max_gradient_pct']:+.1f} % local "
                        "gradient")
        axis.annotate(
            f"highest-cost edge on the distance route\n"
            f"{highest_cost['ratio']:.2f} x level walking{gradient}",
            midpoint, textcoords="axes fraction",
            xytext=HIGHEST_COST_LABEL_AXES_XY, ha="left", va="top",
            fontsize=7.5, color=COLOUR_MUTED,
            arrowprops={"arrowstyle": "-", "color": COLOUR_MUTED,
                        "linewidth": 0.8})

    axis.set_xlabel("cumulative route distance (m)", fontsize=9)
    axis.set_ylabel("cumulative terrain effort (J/kg)", fontsize=9)
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend(loc="upper left", fontsize=8, framealpha=0.92)
    axis.set_title("Cumulative terrain effort against distance walked",
                   fontsize=10, color=COLOUR_INK)


def comparison_text(row: pd.Series) -> str:
    return (
        f"shortest        {float(row['distance_route_distance_m']):.3f} m"
        f"    {float(row['distance_route_effort_j_per_kg']):.3f} J/kg\n"
        f"least effort    {float(row['effort_route_distance_m']):.3f} m"
        f"    {float(row['effort_route_effort_j_per_kg']):.3f} J/kg\n"
        f"\n"
        f"distance detour    {float(row['distance_detour_m']):+.3f} m\n"
        f"effort saving      {-float(row['effort_saving_j_per_kg']):+.3f} J/kg"
        f"  ({-float(row['effort_saving_pct']):+.3f} %)\n"
        f"restriction effect {float(row['restriction_effect_m']):.3f} m"
    )


def plot_primary(
    graph, routes, row, series, shared, highest_cost, out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    polylines = {metric: route_polylines(graph, routes[metric])
                 for metric in OBJECTIVES}
    bounds = route_bounds(polylines[OBJECTIVE_DISTANCE]
                          + polylines[OBJECTIVE_EFFORT])

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.4),
                                gridspec_kw={"width_ratios": [1.05, 0.95]})
    draw_map(axes[0], graph, routes, polylines, shared, row, bounds)
    draw_cumulative(axes[1], series, row, highest_cost)
    axes[1].text(0.985, 0.02, comparison_text(row), transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=8.5, family="monospace",
                 color=COLOUR_INK,
                 bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc",
                       "boxstyle": "round,pad=0.5", "alpha": 0.95})

    figure.suptitle(
        "Distance-optimal and effort-optimal walking routes on the same "
        "trusted-cost subgraph", fontsize=12, color=COLOUR_INK)
    figure.text(0.5, 0.008, CHORD_NOTE + "  Metric drawing in EPSG:7856.",
                ha="center", fontsize=7.5, color=COLOUR_MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_secondary(row, series, out_path: Path, comparable_count: int) -> None:
    """The secondary figure.

    ``comparable_count`` is counted from ``route_comparison.csv`` by
    ``verify_secondary_case`` and passed in, so the title cannot outlive the
    result it describes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    draw_cumulative(axis, series, row, compact=True)
    axis.text(0.985, 0.02, comparison_text(row), transform=axis.transAxes,
              ha="right", va="bottom", fontsize=8.5, family="monospace",
              color=COLOUR_INK,
              bbox={"facecolor": "#ffffff", "edgecolor": "#cccccc",
                    "boxstyle": "round,pad=0.5", "alpha": 0.95})
    axis.set_title(
        f"Stop {int(row['stop_id'])} - {row['stop_name']}\n"
        f"maximum absolute effort saving among the {int(comparable_count)} "
        "comparable stops",
        fontsize=10, color=COLOUR_INK)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
def figure_record(
    row: pd.Series, verified: Dict[str, Dict[str, Any]],
    counts: Optional[Dict[str, Any]], filename: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "stop_id": int(row["stop_id"]),
        "stop_name": row["stop_name"],
        "figure": filename,
        "comparison_status": row["comparison_status"],
        "distance_route_distance_m": float(row["distance_route_distance_m"]),
        "distance_route_effort_j_per_kg": float(
            row["distance_route_effort_j_per_kg"]),
        "effort_route_distance_m": float(row["effort_route_distance_m"]),
        "effort_route_effort_j_per_kg": float(
            row["effort_route_effort_j_per_kg"]),
        "distance_detour_m": float(row["distance_detour_m"]),
        "effort_saving_j_per_kg": float(row["effort_saving_j_per_kg"]),
        "effort_saving_pct": float(row["effort_saving_pct"]),
        "restriction_effect_m": float(row["restriction_effect_m"]),
        "edge_route_changed": bool(row["edge_route_changed"]),
        "node_route_changed": bool(row["node_route_changed"]),
        "routes": {metric: {
            "hops": verified[metric]["hops"],
            "reported_distance_m": verified[metric]["reported_distance_m"],
            "reported_effort_j_per_kg":
                verified[metric]["reported_effort_j_per_kg"],
        } for metric in OBJECTIVES},
    }
    if counts is not None:
        record["edge_counts"] = counts
    return record


def build_summary(
    figures: List[Dict[str, Any]], caveat: Optional[Dict[str, Any]],
    graph_path: Path, graph_sha256: str, input_hashes: Dict[str, str],
    provenance: Optional[Dict[str, Any]] = None,
    secondary_case: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "stage": "Week 5C-2C static terrain-routing visualisation",
        "purpose": (
            "Static figures for the Week 5C-2B comparison. Routes are read "
            "back from route_edges.csv as exact (u, v, key) sequences; no "
            "routing, no DEM sampling and no terrain cost is recomputed here."),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "inputs": {"graph": {"path": str(graph_path), "sha256": graph_sha256},
                   "sha256": input_hashes},
        "provenance": provenance,
        "secondary_case": secondary_case,
        "method": {
            "route_identity": "the exact (u, v, key) sequence in "
                              "route_edges.csv; Dijkstra is not re-run",
            "total_verification": (
                f"round(sum(hop length_m), {DISTANCE_DECIMALS}) and "
                f"round(sum(hop effort), {EFFORT_DECIMALS}) must equal the "
                "rounded values in route_comparison.csv"),
            "drawing_crs": METRIC_CRS,
            "missing_geometry": (
                "an edge with no stored OSM geometry is drawn as a straight "
                "line between its endpoint nodes, for drawing only; the graph "
                "is never modified"),
            "cumulative_panel": (
                "cumulative terrain effort against cumulative route distance, "
                "from the hop costs already recorded; each segment's slope is "
                "that edge's mean cost per metre"),
            "terminal_points": (
                "the two routes end at different cumulative distances, so the "
                "vertical gap between the terminal points is NOT the effort "
                "difference; each terminal marker carries only a short name "
                "('shortest', 'least effort'), the totals are printed once in "
                "the comparison box, and the difference is stated by a "
                "labelled bracket clear of both curves"),
            "annotated_edge": (
                "the annotated edge is selected by the highest "
                "terrain_to_level_ratio - metabolic cost per metre relative to "
                "level walking - which is not necessarily the largest signed "
                "or absolute gradient. It is labelled 'highest-cost edge', "
                "never 'steepest'; the gradient range beside it is supporting "
                "evidence, not the selection criterion."),
            "cost_table_provenance": (
                "edge_terrain_cost.csv must be the table Week 5C-2B routed "
                "on: both hashes recorded in terrain_routing_summary.json are "
                "re-checked before any figure is written, and a mismatch is a "
                "hard failure rather than a quietly omitted annotation"),
            "secondary_maximum_claim": (
                "the 'maximum effort saving' label is re-derived from "
                "route_comparison.csv every run - the comparable stops are "
                "counted and the maximum taken - and the printed count is that "
                "number, never a literal"),
            "level_walking_baseline_j_per_kg_per_m": LEVEL_COST_J_PER_KG_PER_M,
            "mean_cost_note": (
                "a route mean above the level baseline means only that its "
                "mean metabolic cost per metre exceeds level walking. It does "
                "not establish that the route is net uphill, since steep "
                "descent also costs more than level walking."),
            "palette": {
                "distance_route": COLOUR_DISTANCE,
                "effort_route": COLOUR_EFFORT,
                "shared": COLOUR_SHARED,
                "validated": "dataviz palette checker, light surface: CVD dE "
                             "22.4 protan / 31.5 tritan, normal-vision dE "
                             "31.2, all checks pass",
                "secondary_encoding": "line style and direct end labels, so "
                                      "colour is never the only encoding",
            },
        },
        "figures": figures,
        "model_coverage_caveat": caveat,
        "not_done": [
            "no routing", "no DEM sampling", "no new terrain cost",
            "no stop ranking", "no stairs penalty", "OSM-FIX-001 not applied",
            "interactive demo unchanged", "no AR calibration",
            "raw GraphML not modified",
            "no route-comparison figure for stop 203769",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 5C-2C static terrain-routing figures.")
    parser.add_argument("--graph", default=audit.DEFAULT_GRAPH)
    parser.add_argument("--routing-dir", default=None)
    options = parser.parse_args(argv)

    graph_path = Path(options.graph)
    if not graph_path.is_absolute():
        graph_path = (CONFIG_ROOT / graph_path).resolve()
    source_dir = Path(options.routing_dir).resolve() if options.routing_dir \
        else routing_dir()

    print("Week 5C-2C - static figures for the terrain-routing comparison")
    print("Routes are read back from route_edges.csv as exact (u, v, key) "
          "sequences.")
    print("No routing is re-run, no DEM is sampled and no terrain cost is "
          "recomputed.")
    print("")

    try:
        comparison, edge_table = load_routing_outputs(source_dir)
        routing_summary = load_routing_summary(source_dir)
    except VisualisationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    graph_sha256_before = audit.file_sha256(graph_path)
    cost_path = (CONFIG_ROOT / "data" / "cache" / COST_SUBDIR / COST_FILENAME)
    try:
        hashes = verify_cost_provenance(
            routing_summary, graph_sha256_before, cost_path)
        secondary_case = verify_secondary_case(comparison, SECONDARY_STOP)
    except VisualisationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    routing_summary_sha256 = audit.file_sha256(
        source_dir / ROUTING_SUMMARY_FILENAME)
    provenance = {
        "terrain_routing_summary_json_sha256": routing_summary_sha256,
        "edge_terrain_cost_csv_sha256": hashes["edge_terrain_cost_sha256"],
        "graph_sha256": hashes["graph_sha256"],
        "verified_against_routing_summary": True,
        "checks": [
            "terrain_routing_summary.json graph.sha256 == the raw GraphML on "
            "disk",
            "terrain_routing_summary.json edge_terrain_cost_sha256 == "
            "edge_terrain_cost.csv on disk",
        ],
    }
    print(f"Cost-table provenance verified against {ROUTING_SUMMARY_FILENAME}:")
    print(f"  graph               {hashes['graph_sha256']}")
    print(f"  edge_terrain_cost   {hashes['edge_terrain_cost_sha256']}")
    print(f"Stop {SECONDARY_STOP} still holds the maximum effort saving "
          f"({secondary_case['max_effort_saving_j_per_kg']} J/kg) among the "
          f"{secondary_case['comparable_count']} comparable stops.")
    print("")

    graph = load_graph(graph_path)
    cost_table = pd.read_csv(cost_path)

    out_dir = audit.ensure_directory(audit.cache_dir(OUTPUT_SUBDIR))
    figures: List[Dict[str, Any]] = []

    # ----- primary: stop 203732 ------------------------------------------- #
    row = comparison_row(comparison, PRIMARY_STOP)
    routes = {metric: route_records(edge_table, PRIMARY_STOP, metric)
              for metric in OBJECTIVES}
    verified = {metric: verify_route(routes[metric], row, metric, graph)
                for metric in OBJECTIVES}
    shared = (set(edge_keys(routes[OBJECTIVE_DISTANCE]))
              & set(edge_keys(routes[OBJECTIVE_EFFORT])))
    counts = geometry_counts(graph, routes)
    if counts["shared_directed_edges"] != int(row["shared_edge_count"]):
        raise VisualisationError(
            f"Stop {PRIMARY_STOP}: {counts['shared_directed_edges']} shared "
            f"edges reconstructed but the comparison table reports "
            f"{int(row['shared_edge_count'])}.")
    series = {metric: cumulative_series(routes[metric])
              for metric in OBJECTIVES}
    highest_cost = highest_cost_segment(routes[OBJECTIVE_DISTANCE], cost_table)
    primary_name = f"route_{PRIMARY_STOP}_comparison.png"
    plot_primary(graph, routes, row, series, shared, highest_cost,
                 out_dir / primary_name)
    figures.append(figure_record(row, verified, counts, primary_name))
    print(f"  {primary_name}")
    print(f"    {counts['route_edge_records']} route-edge records over "
          f"{counts['unique_directed_edges']} unique directed edges "
          f"({counts['shared_directed_edges']} shared)")
    print(f"    stored geometry: "
          f"{counts['records_with_stored_geometry']} of "
          f"{counts['route_edge_records']} records, "
          f"{counts['unique_edges_with_stored_geometry']} of "
          f"{counts['unique_directed_edges']} unique edges; the rest are drawn "
          "as endpoint chords")

    # ----- secondary: stop 203717 ----------------------------------------- #
    second = comparison_row(comparison, SECONDARY_STOP)
    second_routes = {metric: route_records(edge_table, SECONDARY_STOP, metric)
                     for metric in OBJECTIVES}
    second_verified = {
        metric: verify_route(second_routes[metric], second, metric, graph)
        for metric in OBJECTIVES}
    second_series = {metric: cumulative_series(second_routes[metric])
                     for metric in OBJECTIVES}
    second_name = f"route_{SECONDARY_STOP}_effort_profile.png"
    plot_secondary(second, second_series, out_dir / second_name,
                   secondary_case["comparable_count"])
    second_record = figure_record(second, second_verified,
                                  geometry_counts(graph, second_routes),
                                  second_name)
    second_record["maximum_effort_saving_case"] = secondary_case
    figures.append(second_record)
    print(f"  {second_name}")

    # ----- the caveat stop is recorded, never drawn ------------------------ #
    caveat_row = comparison_row(comparison, CAVEAT_STOP)
    caveat = {
        "stop_id": CAVEAT_STOP,
        "stop_name": caveat_row["stop_name"],
        "figure": None,
        "restriction_effect_m": float(caveat_row["restriction_effect_m"]),
        "effort_saving_j_per_kg": float(caveat_row["effort_saving_j_per_kg"]),
        "edge_route_changed": bool(caveat_row["edge_route_changed"]),
        "shared_edge_count": int(caveat_row["shared_edge_count"]),
        "distance_route_edges": int(caveat_row["distance_route_edges"]),
        "note": CAVEAT_203769,
    }

    graph_sha256_after = audit.file_sha256(graph_path)
    if graph_sha256_after != graph_sha256_before:
        raise VisualisationError(
            f"{graph_path} changed while plotting. It is read-only.")

    summary = build_summary(
        figures, caveat, graph_path, graph_sha256_after,
        {COMPARISON_FILENAME:
            audit.file_sha256(source_dir / COMPARISON_FILENAME),
         EDGES_FILENAME:
            audit.file_sha256(source_dir / EDGES_FILENAME),
         ROUTING_SUMMARY_FILENAME: routing_summary_sha256,
         COST_FILENAME: hashes["edge_terrain_cost_sha256"]},
        provenance=provenance, secondary_case=secondary_case)
    with open(out_dir / "terrain_visualisation_summary.json", "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("  terrain_visualisation_summary.json")
    print("")
    print(f"Wrote data/cache/{OUTPUT_SUBDIR}/")
    print(f"The Stage 3 graph is unchanged: sha256 {graph_sha256_after}")
    print("")
    print(f"Stop {CAVEAT_STOP} is recorded as a model-coverage caveat and is "
          "deliberately not drawn:")
    print("  its two routes are the same edges and its effort saving is 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
