"""Week 5C-2B - distance route against terrain-effort route, same subgraph.

The question
-----------
For the A-02 provisional test origin, does minimising walking EFFORT choose a
different way to each candidate stop than minimising DISTANCE - and what does
the swap cost in metres and save in joules per kilogram?

The comparison principle that decides the whole design
------------------------------------------------------
Both metrics are evaluated on EXACTLY the same directed subgraph: the 6,294
directed edges that Week 5C-2A.2 gave a trusted terrain cost
(``ordinary_5m`` or ``ordinary_short_endpoint_fallback``). Comparing a
distance route on the full raw graph against an effort route on a smaller one
would attribute the missing edges to terrain, which would be wrong.

The historical Week 4 raw-graph distance stays a SEPARATE baseline. The gap
between it and the distance route on the trusted subgraph is reported as
``restriction_effect_m``: it measures model coverage and topology, not terrain,
and is never folded into the effort comparison.

Lexicographic routing, not a combined score
-------------------------------------------
Two entirely different node paths can share a total distance and differ in
effort, so resolving ties only among parallel edges is not enough. Each route
is therefore the lexicographic minimum of a PAIR:

    distance route: minimise (total_distance_m, total_effort_j_per_kg)
    effort route:   minimise (total_effort_j_per_kg, total_distance_m)

The second component is ONLY a tie-break. This is not a weighted score, there
is no ``alpha * distance + beta * effort`` anywhere, and no conversion between
metres and joules per kilogram is invented. Pairs add componentwise and compare
lexicographically; every trusted weight is finite and positive, so adding an
edge strictly increases a label and Dijkstra remains valid.

What this does NOT do
---------------------
No stop is ranked. No cost is invented for an untrusted edge. No stairs
penalty. OSM-FIX-001 is not applied. The raw GraphML is opened read-only and
hashed before and after. Week 4 outputs are read and re-verified, never
rewritten. The interactive demo is untouched.

Direction: origin -> stop only, matching Week 4. Minetti cost is directional -
walking a street uphill costs more than walking it down - so the reverse leg
can differ and is deliberately left as future analysis rather than assumed
symmetric (D-21).

Usage
-----
    python scripts/compare_terrain_routes.py
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from src.osm_network import load_graph  # noqa: E402
from src.routing import (  # noqa: E402
    STATUS_DIFFERENT_COMPONENT,
    STATUS_ROUTED,
    classify_route,
    load_json,
    resolve_origin_component,
    single_source_distances,
)


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# One import brings the whole adopted chain: Week 5A sampling, the Week 5B
# gradient, Week 5C-1 Minetti, the Week 5C-2A audit and the Week 5C-2A.2 cost
# construction, each with a single implementation.
terrain_cost = load_script("build_terrain_cost")
audit = terrain_cost.audit


OUTPUT_SUBDIR = "terrain_routing"
COST_SUBDIR = terrain_cost.OUTPUT_SUBDIR                 # terrain_cost
TRUSTED_STATUSES: Tuple[str, ...] = terrain_cost.TRUSTED_STATUSES

# Network distance is the raw GraphML edge attribute. sampled_length_m is a
# DEM-sampling artefact of the cost construction and is NOT a network distance.
ATTR_LENGTH = "length"
ATTR_EFFORT = "terrain_effort_j_per_kg"

OBJECTIVE_DISTANCE = "distance"
OBJECTIVE_EFFORT = "effort"
OBJECTIVES: Tuple[str, ...] = (OBJECTIVE_DISTANCE, OBJECTIVE_EFFORT)

# (primary attribute, secondary attribute) for each objective. The secondary is
# only ever a tie-break.
OBJECTIVE_ATTRIBUTES: Dict[str, Tuple[str, str]] = {
    OBJECTIVE_DISTANCE: (ATTR_LENGTH, ATTR_EFFORT),
    OBJECTIVE_EFFORT: (ATTR_EFFORT, ATTR_LENGTH),
}

# The cost table stores length_m rounded to millimetres, so it may differ from
# the GraphML attribute by up to half a millimetre.
LENGTH_CROSSCHECK_TOLERANCE_M = 1e-3

# Stage 5 stored its distances rounded to 2 dp.
BASELINE_TOLERANCE_M = 0.05

# Route sums are recomputed from the same floats, so they must agree closely.
SUM_TOLERANCE = 1e-6

STATUS_COMPARED = "compared"
STATUS_NO_TRUSTED_PATH = "no_trusted_cost_path"
STATUS_RAW_GAP = "raw_topology_component_gap"
STATUS_NO_RAW_BASELINE = "raw_baseline_unavailable"

STATUS_ORDER: Tuple[str, ...] = (
    STATUS_COMPARED, STATUS_NO_TRUSTED_PATH, STATUS_RAW_GAP,
    STATUS_NO_RAW_BASELINE,
)

REASON_NO_TRUSTED_EDGE_AT_STOP = "stop_node_has_no_trusted_cost_edge"
REASON_NO_TRUSTED_PATH = "no_trusted_cost_path_from_origin"
REASON_RAW_COMPONENT_GAP = "raw_topology_component_gap_I04"
REASON_NOT_IN_BASELINE = "stop_absent_from_the_week4_baseline"

# The condition this note explains is only ``degree == 0`` on the trusted
# subgraph. An incident edge can be untrusted for any of several reasons - a
# special structure, an out-of-domain gradient, a low-confidence short edge, an
# unresolved DEM - so the generic wording must not name one of them.
SUBGRAPH_RESTRICTION_NOTE = (
    "No incident edge at the snapped node has an adopted terrain cost, so no "
    "comparison route exists on the trusted-cost subgraph. This is model "
    "coverage: NOT a terrain effect and NOT a statement about the stop. The "
    "terrain_cost_status of those incident edges says why each one is "
    "untrusted."
)

JUBILEE_NOTE = (
    "No path exists in the raw OSM topology, so there is no Week 4 baseline to "
    "compare against either. This is I-04, a known upstream data artefact; "
    "OSM-FIX-001 remains proposed and is NOT applied."
)

HOP_UNIQUE = "unique"
HOP_DUAL_METRIC_TIE = "dual_metric_equivalent_tie"


class TerrainRoutingError(RuntimeError):
    """Raised with an actionable message when a comparison cannot be trusted."""


class ProvenanceMismatch(TerrainRoutingError):
    """Raised when the terrain-cost table does not belong to this graph."""


class BaselineMismatch(TerrainRoutingError):
    """Raised when the Week 4 baseline cannot be reproduced."""


# --------------------------------------------------------------------------- #
# lexicographic single-source Dijkstra
# --------------------------------------------------------------------------- #
def lexicographic_dijkstra(
    graph: nx.DiGraph, source: int, primary: str, secondary: str,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, int]]:
    """Shortest paths under lexicographic comparison of a PAIR of totals.

    A label is ``(primary total, secondary total)``. Labels add componentwise
    and compare lexicographically, so the secondary quantity separates paths
    that tie on the primary one and does nothing else. It is not a score and
    the two are never combined arithmetically.

    Dijkstra is valid here because every trusted weight is finite and strictly
    positive, so appending an edge strictly increases a label in the
    lexicographic order.

    Ordering is total and deterministic: the heap key is
    ``(primary, secondary, node)`` and node ids are integers, so two runs over
    the same graph return the same paths even when both metrics tie exactly.

    Args:
        graph: a DiGraph whose edges carry both attributes.
        source: the node to search from.
        primary: edge attribute minimised first.
        secondary: edge attribute minimised only among primary ties.

    Returns:
        Tuple of (node -> label, node -> predecessor). Unreachable nodes are
        absent from both. The source has label ``(0.0, 0.0)`` and no
        predecessor.

    Raises:
        TerrainRoutingError: if the source is not in the graph, or an edge
            carries a non-finite or non-positive weight.
    """
    if source not in graph:
        raise TerrainRoutingError(f"Source node {source} is not in the graph.")

    best: Dict[int, Tuple[float, float]] = {source: (0.0, 0.0)}
    predecessor: Dict[int, int] = {}
    settled: set = set()
    heap: List[Tuple[float, float, int]] = [(0.0, 0.0, source)]

    while heap:
        first, second, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        for neighbour, data in graph[node].items():
            # A self-loop reaches an already settled node and is skipped here,
            # which is why a positive self-loop can never enter a route.
            if neighbour in settled:
                continue
            step_primary = float(data[primary])
            step_secondary = float(data[secondary])
            if not (np.isfinite(step_primary) and np.isfinite(step_secondary)):
                raise TerrainRoutingError(
                    f"Edge {node} -> {neighbour} has a non-finite weight "
                    f"({step_primary}, {step_secondary}).")
            if step_primary <= 0 or step_secondary <= 0:
                raise TerrainRoutingError(
                    f"Edge {node} -> {neighbour} has a non-positive weight "
                    f"({step_primary}, {step_secondary}). Lexicographic "
                    "Dijkstra needs strictly positive steps.")
            candidate = (first + step_primary, second + step_secondary)
            if neighbour not in best or candidate < best[neighbour]:
                best[neighbour] = candidate
                predecessor[neighbour] = node
                heapq.heappush(heap, (candidate[0], candidate[1], neighbour))
    return best, predecessor


def reconstruct_nodes(
    predecessor: Dict[int, int], source: int, target: int
) -> List[int]:
    """The node sequence from source to target, following predecessors."""
    if target == source:
        return [source]
    if target not in predecessor:
        raise TerrainRoutingError(
            f"Node {target} has no predecessor, so no path can be rebuilt.")
    nodes = [target]
    while nodes[-1] != source:
        nodes.append(predecessor[nodes[-1]])
        if len(nodes) > len(predecessor) + 2:
            raise TerrainRoutingError("The predecessor chain does not close.")
    nodes.reverse()
    return nodes


# --------------------------------------------------------------------------- #
# building the trusted subgraph and its two collapsed views
# --------------------------------------------------------------------------- #
def build_trusted_subgraph(
    graph: nx.MultiDiGraph, cost_table: pd.DataFrame
) -> nx.MultiDiGraph:
    """The directed subgraph carrying an adopted terrain cost.

    Every node of the raw graph is kept, including the ones left with no
    trusted edge at all: a stop that snaps to such a node has to be reported as
    a model-coverage restriction, which needs the node to exist.

    The raw graph is not modified and nothing is written to disk.
    """
    trusted = cost_table[cost_table["terrain_cost_status"].isin(
        TRUSTED_STATUSES)]
    efforts = {
        (int(row.u), int(row.v), int(row.key)):
            float(row.terrain_effort_j_per_kg)
        for row in trusted.itertuples()
    }
    subgraph = nx.MultiDiGraph()
    subgraph.graph.update(graph.graph)
    subgraph.add_nodes_from((node, dict(data))
                            for node, data in graph.nodes(data=True))
    for u, v, key, data in graph.edges(keys=True, data=True):
        effort = efforts.get((int(u), int(v), int(key)))
        if effort is None:
            continue
        subgraph.add_edge(int(u), int(v), key=int(key),
                          **{ATTR_LENGTH: float(data[ATTR_LENGTH]),
                             ATTR_EFFORT: effort})
    return subgraph


def collapse_parallel_edges(
    subgraph: nx.MultiDiGraph, objective: str
) -> Tuple[nx.DiGraph, Dict[str, int]]:
    """One edge per ordered pair, chosen by this objective's own ordering.

    For the distance objective the survivor minimises
    ``(length, effort, key)``; for the effort objective it minimises
    ``(effort, length, key)``. The key is a last-resort tie-break used only
    when both metrics are identical, and it is there for reproducibility, not
    for meaning.

    Collapsing changes no answer: Dijkstra on a multigraph already takes the
    minimum over parallel edges. Doing it explicitly means the exact directed
    key of every hop is recorded rather than inferred afterwards.

    Self-loops are kept - they are part of the 6,294 trusted edges - and simply
    never enter a route, because a positive self-loop cannot improve any label.
    """
    primary, secondary = OBJECTIVE_ATTRIBUTES[objective]
    collapsed = nx.DiGraph()
    collapsed.add_nodes_from(subgraph.nodes())
    counters = {"parallel_pairs": 0, "dual_metric_ties": 0, "self_loops": 0}

    for u, v in {(u, v) for u, v, _k in subgraph.edges(keys=True)}:
        bundle = subgraph[u][v]
        if u == v:
            counters["self_loops"] += len(bundle)
        if len(bundle) > 1:
            counters["parallel_pairs"] += 1
        ordered = sorted(
            ((float(data[primary]), float(data[secondary]), int(key))
             for key, data in bundle.items()))
        chosen_primary, chosen_secondary, chosen_key = ordered[0]
        equivalent = sum(1 for p, s, _k in ordered
                         if p == chosen_primary and s == chosen_secondary)
        if equivalent > 1:
            counters["dual_metric_ties"] += 1
        data = subgraph[u][v][chosen_key]
        collapsed.add_edge(u, v, key=chosen_key,
                           **{ATTR_LENGTH: float(data[ATTR_LENGTH]),
                              ATTR_EFFORT: float(data[ATTR_EFFORT])},
                           parallel_candidates=len(bundle),
                           dual_metric_equivalent=bool(equivalent > 1))
    return collapsed, counters


def collapse_disagreements(
    distance_graph: nx.DiGraph, effort_graph: nx.DiGraph
) -> int:
    """Ordered pairs where the two objectives keep a different parallel key."""
    return sum(
        1 for u, v, data in distance_graph.edges(data=True)
        if effort_graph.has_edge(u, v)
        and int(effort_graph[u][v]["key"]) != int(data["key"])
    )


# --------------------------------------------------------------------------- #
# provenance and input guards
# --------------------------------------------------------------------------- #
def cost_dir(root: Path = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache" / COST_SUBDIR).resolve()


def load_cost_table(source_dir: Path) -> pd.DataFrame:
    path = Path(source_dir) / "edge_terrain_cost.csv"
    if not path.exists():
        raise ProvenanceMismatch(
            f"{path} is missing. Build the terrain-cost table first:\n"
            "    python scripts/build_terrain_cost.py")
    return pd.read_csv(path)


def load_cost_summary(source_dir: Path) -> Dict[str, Any]:
    path = Path(source_dir) / "terrain_cost_summary.json"
    if not path.exists():
        raise ProvenanceMismatch(
            f"{path} is missing, so the terrain costs cannot be tied to any "
            "graph. Build the terrain-cost table first:\n"
            "    python scripts/build_terrain_cost.py")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_cost_provenance(
    summary: Dict[str, Any], cost_table: pd.DataFrame,
    graph: nx.MultiDiGraph, graph_path: Path, graph_sha256: str,
) -> Dict[str, Any]:
    """Refuse to route on costs that were computed about a different graph.

    Checked, in order: the recorded graph hash equals the GraphML on disk; the
    cost table has exactly one row per raw (u, v, key), with no duplicates and
    no extras; every trusted row has a finite, strictly positive effort; the
    trusted edge count agrees with the recorded one; and each row's stored
    length agrees with the GraphML attribute to within the millimetre it was
    rounded to.

    Raises:
        ProvenanceMismatch: on any of them, with no bypass. Nothing upstream is
            regenerated or repaired.
    """
    recorded = (summary.get("graph") or {}).get("sha256")
    if recorded != graph_sha256:
        raise ProvenanceMismatch(
            f"The terrain costs were built against graph {recorded}, but "
            f"{graph_path} hashes to {graph_sha256}. The costs may refer to "
            "different edges. Re-run scripts/build_terrain_cost.py.")

    if len(cost_table) != graph.number_of_edges():
        raise ProvenanceMismatch(
            f"The cost table has {len(cost_table)} rows but the graph has "
            f"{graph.number_of_edges()} directed edges.")

    keys = list(zip(cost_table["u"], cost_table["v"], cost_table["key"]))
    typed = [(int(u), int(v), int(k)) for u, v, k in keys]
    if len(set(typed)) != len(typed):
        raise ProvenanceMismatch(
            f"The cost table contains {len(typed) - len(set(typed))} "
            "duplicated (u, v, key) row(s). One edge cannot carry two costs.")

    graph_edges = {(int(u), int(v), int(k))
                   for u, v, k in graph.edges(keys=True)}
    if set(typed) != graph_edges:
        missing = graph_edges - set(typed)
        extra = set(typed) - graph_edges
        raise ProvenanceMismatch(
            f"The cost table's edge set does not match the graph: "
            f"{len(missing)} graph edge(s) uncosted and {len(extra)} cost "
            f"row(s) absent from the graph, e.g. "
            f"{sorted(missing)[:1] or sorted(extra)[:1]}.")

    trusted = cost_table[cost_table["terrain_cost_status"].isin(
        TRUSTED_STATUSES)]
    effort = pd.to_numeric(trusted["terrain_effort_j_per_kg"], errors="coerce")
    bad = trusted[~np.isfinite(effort.to_numpy(dtype=float))
                  | (effort.to_numpy(dtype=float) <= 0)]
    if not bad.empty:
        first = bad.iloc[0]
        raise ProvenanceMismatch(
            f"{len(bad)} trusted edge(s) have a missing, non-finite or "
            f"non-positive terrain effort, e.g. ({first['u']}, {first['v']}, "
            f"{first['key']}). Every routable weight must be finite and "
            "strictly positive.")

    recorded_trusted = (summary.get("trusted") or {}).get("edges")
    if int(recorded_trusted or -1) != len(trusted):
        raise ProvenanceMismatch(
            f"The summary records {recorded_trusted} trusted edges but the "
            f"table has {len(trusted)}.")

    lengths = {(int(u), int(v), int(k)): float(data[ATTR_LENGTH])
               for u, v, k, data in graph.edges(keys=True, data=True)}
    worst_edge, worst_gap = None, 0.0
    for row in cost_table.itertuples():
        edge = (int(row.u), int(row.v), int(row.key))
        gap = abs(lengths[edge] - float(row.length_m))
        if gap > worst_gap:
            worst_edge, worst_gap = edge, gap
    if worst_gap > LENGTH_CROSSCHECK_TOLERANCE_M:
        raise ProvenanceMismatch(
            f"Edge {worst_edge} is {lengths[worst_edge]:.4f} m in the GraphML "
            f"but {worst_gap:.4f} m away in the cost table. Routing distance "
            "comes from the GraphML; the two must describe the same edges.")

    return {
        "graph_sha256": graph_sha256,
        "graph_sha256_matches_cost_table": True,
        "directed_edges": int(graph.number_of_edges()),
        "cost_rows": int(len(cost_table)),
        "edge_set_matches": True,
        "duplicate_rows": 0,
        "trusted_edges": int(len(trusted)),
        "trusted_effort_all_finite_positive": True,
        "max_length_crosscheck_gap_m": round(worst_gap, 6),
    }


# --------------------------------------------------------------------------- #
# the Week 4 raw baseline, re-verified rather than trusted
# --------------------------------------------------------------------------- #
def revalidate_week4_baseline(
    graph: nx.MultiDiGraph, routes: pd.DataFrame, origin_node: int,
    origin_component: Optional[str],
) -> Dict[str, Any]:
    """Reproduce the Week 4 raw-graph distance result and check every row.

    One single-source Dijkstra on the raw graph, through the existing Stage 5
    implementation, then per stop: the route status, the network distance, the
    node count and the edge count must all match what the historical CSV
    recorded. The two Jubilee Park rows must still be raw-topology component
    gaps.

    The historical CSV is never rewritten.

    Returns:
        Tuple of (report, exact raw distance per routed stop_id). The second
        value is the FULL-PRECISION distance this run reproduced. The
        historical CSV stores it rounded to two decimals, which is fine as an
        audit reference but would put a rounding residual of up to 5 mm into
        the restriction effect - and since the trusted graph is a subgraph of
        the raw one, that residual could turn a true zero restriction into a
        small negative number.

    Raises:
        BaselineMismatch: on any disagreement, with no bypass.
    """
    distances, paths = single_source_distances(graph, origin_node)
    exact: Dict[Any, float] = {}
    origin_label = resolve_origin_component(graph, origin_node,
                                            origin_component)
    checked = {"routed": 0, "component_gap": 0}
    for _index, stop in routes.iterrows():
        node = int(stop["nearest_node"])
        reached = node in distances
        status = classify_route(reached, origin_label, stop["component_id"])
        if status != stop["route_status"]:
            raise BaselineMismatch(
                f"Stop {stop['stop_id']}: the raw graph now gives status "
                f"{status!r} but Week 4 recorded {stop['route_status']!r}.")

        if status == STATUS_ROUTED:
            distance = float(distances[node])
            recorded = float(stop["network_distance_m"])
            if abs(distance - recorded) > BASELINE_TOLERANCE_M:
                raise BaselineMismatch(
                    f"Stop {stop['stop_id']}: the raw graph now gives "
                    f"{distance:.2f} m but Week 4 recorded {recorded:.2f} m.")
            nodes = paths[node]
            if len(nodes) != int(stop["route_node_count"]):
                raise BaselineMismatch(
                    f"Stop {stop['stop_id']}: {len(nodes)} nodes now but "
                    f"{int(stop['route_node_count'])} recorded.")
            if len(nodes) - 1 != int(stop["route_edge_count"]):
                raise BaselineMismatch(
                    f"Stop {stop['stop_id']}: {len(nodes) - 1} edges now but "
                    f"{int(stop['route_edge_count'])} recorded.")
            exact[stop["stop_id"]] = distance
            checked["routed"] += 1
        else:
            if pd.notna(stop["network_distance_m"]):
                raise BaselineMismatch(
                    f"Stop {stop['stop_id']} is unreachable on the raw graph "
                    f"but carries a recorded distance "
                    f"{stop['network_distance_m']!r}.")
            checked["component_gap"] += 1

    report = {
        "origin_node": int(origin_node),
        "origin_component": origin_label,
        "raw_reachable_nodes": int(len(distances)),
        "rows_verified": int(len(routes)),
        "routed_rows_verified": checked["routed"],
        "component_gap_rows_verified": checked["component_gap"],
        "checks": ["route_status", "network_distance_m", "route_node_count",
                   "route_edge_count"],
        "historical_csv_rewritten": False,
        "restriction_effect_baseline": (
            "the full-precision distance re-derived here, not the historical "
            "two-decimal CSV value"),
    }
    return report, exact


# --------------------------------------------------------------------------- #
# turning a node sequence into exact directed edges
# --------------------------------------------------------------------------- #
def route_hops(
    collapsed: nx.DiGraph, nodes: Sequence[int]
) -> List[Dict[str, Any]]:
    """The exact directed edges a node sequence uses on a collapsed graph."""
    hops: List[Dict[str, Any]] = []
    for index, (u, v) in enumerate(zip(nodes[:-1], nodes[1:])):
        if not collapsed.has_edge(u, v):
            raise TerrainRoutingError(
                f"The path uses a hop {u} -> {v} that is not an edge of the "
                "routing graph.")
        data = collapsed[u][v]
        hops.append({
            "hop_index": index,
            "u": int(u), "v": int(v), "key": int(data["key"]),
            "length_m": float(data[ATTR_LENGTH]),
            "terrain_effort_j_per_kg": float(data[ATTR_EFFORT]),
            "hop_resolution": (HOP_DUAL_METRIC_TIE
                               if data["dual_metric_equivalent"]
                               else HOP_UNIQUE),
        })
    return hops


def verify_route(
    hops: Sequence[Dict[str, Any]], nodes: Sequence[int], origin: int,
    destination: int, distance_m: float, effort_j_per_kg: float,
    graph: nx.MultiDiGraph, trusted_keys: set, label: str,
) -> None:
    """Every check a reported route must pass before it may be published.

    Raises:
        TerrainRoutingError: on any failure. There is no bypass; a route that
            cannot be reproduced from its own edge list is not a result.
    """
    if not nodes or nodes[0] != origin:
        raise TerrainRoutingError(
            f"{label}: the route starts at {nodes[0] if nodes else None}, not "
            f"at the origin {origin}.")
    if nodes[-1] != destination:
        raise TerrainRoutingError(
            f"{label}: the route ends at {nodes[-1]}, not at the snapped stop "
            f"node {destination}.")
    if len(hops) != len(nodes) - 1:
        raise TerrainRoutingError(
            f"{label}: {len(hops)} hops for {len(nodes)} nodes.")

    for index, hop in enumerate(hops):
        edge = (hop["u"], hop["v"], hop["key"])
        if edge not in trusted_keys:
            raise TerrainRoutingError(
                f"{label}: hop {index} uses {edge}, which has no trusted "
                "terrain cost.")
        if not graph.has_edge(*edge):
            raise TerrainRoutingError(
                f"{label}: hop {index} uses {edge}, which is not in the raw "
                "graph.")
        if hop["u"] != nodes[index] or hop["v"] != nodes[index + 1]:
            raise TerrainRoutingError(
                f"{label}: hop {index} is {hop['u']} -> {hop['v']} but the "
                f"node sequence goes {nodes[index]} -> {nodes[index + 1]}.")
        graph_length = float(graph.edges[edge][ATTR_LENGTH])
        if abs(graph_length - hop["length_m"]) > SUM_TOLERANCE:
            raise TerrainRoutingError(
                f"{label}: hop {index} records {hop['length_m']} m but the "
                f"GraphML says {graph_length} m.")

    summed_distance = sum(hop["length_m"] for hop in hops)
    if abs(summed_distance - distance_m) > SUM_TOLERANCE:
        raise TerrainRoutingError(
            f"{label}: the exact keys sum to {summed_distance:.9f} m but the "
            f"reported distance is {distance_m:.9f} m.")
    summed_effort = sum(hop["terrain_effort_j_per_kg"] for hop in hops)
    if abs(summed_effort - effort_j_per_kg) > SUM_TOLERANCE:
        raise TerrainRoutingError(
            f"{label}: the exact keys sum to {summed_effort:.9f} J/kg but the "
            f"reported effort is {effort_j_per_kg:.9f} J/kg.")


def verify_optimality(
    stop_id: Any, distance_route: Dict[str, Any], effort_route: Dict[str, Any],
) -> None:
    """Cross-checks between the two optima, including the lexicographic ties.

    Each route is optimal under its own objective, so neither can be beaten on
    its own primary metric. And because the secondary metric is the explicit
    tie-break, a route that ties on the primary metric cannot be worse on the
    secondary one than the other discovered optimal label.

    Raises:
        TerrainRoutingError: on any violation.
    """
    if effort_route["effort"] > distance_route["effort"] + SUM_TOLERANCE:
        raise TerrainRoutingError(
            f"Stop {stop_id}: the effort route costs "
            f"{effort_route['effort']:.6f} J/kg, more than the distance "
            f"route's {distance_route['effort']:.6f} J/kg.")
    if distance_route["distance"] > effort_route["distance"] + SUM_TOLERANCE:
        raise TerrainRoutingError(
            f"Stop {stop_id}: the distance route is "
            f"{distance_route['distance']:.6f} m, longer than the effort "
            f"route's {effort_route['distance']:.6f} m.")
    if abs(distance_route["distance"] - effort_route["distance"]) \
            <= SUM_TOLERANCE:
        if distance_route["effort"] > effort_route["effort"] + SUM_TOLERANCE:
            raise TerrainRoutingError(
                f"Stop {stop_id}: the two routes tie on distance, so the "
                "distance route's lexicographic tie-break should have taken "
                "the lower effort.")
    if abs(distance_route["effort"] - effort_route["effort"]) <= SUM_TOLERANCE:
        if effort_route["distance"] > distance_route["distance"] + SUM_TOLERANCE:
            raise TerrainRoutingError(
                f"Stop {stop_id}: the two routes tie on effort, so the effort "
                "route's lexicographic tie-break should have taken the shorter "
                "distance.")


# --------------------------------------------------------------------------- #
# the per-stop comparison
# --------------------------------------------------------------------------- #
COMPARISON_COLUMNS: Tuple[str, ...] = (
    "stop_id", "stop_name", "mode", "component_id", "nearest_node",
    "comparison_status", "status_reason", "status_note",
    "week4_raw_distance_m", "raw_revalidated_distance_m",
    "trusted_subgraph_distance_m",
    "restriction_effect_m", "restriction_effect_pct",
    "distance_route_distance_m", "distance_route_effort_j_per_kg",
    "distance_route_edges",
    "effort_route_distance_m", "effort_route_effort_j_per_kg",
    "effort_route_edges",
    "effort_saving_j_per_kg", "effort_saving_pct",
    "distance_detour_m", "distance_detour_pct",
    "edge_route_changed", "node_route_changed", "parallel_key_only_change",
    "shared_edge_count", "distance_only_edges", "effort_only_edges",
    "distance_route_mean_cost_j_per_kg_per_m",
    "effort_route_mean_cost_j_per_kg_per_m",
    "distance_route_dual_metric_tie_hops", "effort_route_dual_metric_tie_hops",
    "verified",
)


def solve_route(
    collapsed: nx.DiGraph, labels: Dict[int, Tuple[float, float]],
    predecessor: Dict[int, int], objective: str, origin: int, target: int,
) -> Dict[str, Any]:
    """One route, as node sequence, exact hops, distance and effort."""
    nodes = reconstruct_nodes(predecessor, origin, target)
    hops = route_hops(collapsed, nodes)
    first, second = labels[target]
    if objective == OBJECTIVE_DISTANCE:
        distance, effort = first, second
    else:
        effort, distance = first, second
    return {
        "objective": objective, "nodes": nodes, "hops": hops,
        "distance": float(distance), "effort": float(effort),
        "edges": [(h["u"], h["v"], h["key"]) for h in hops],
        "dual_metric_tie_hops": sum(
            1 for h in hops if h["hop_resolution"] == HOP_DUAL_METRIC_TIE),
    }


def verify_restriction_effect(
    stop_id: Any, restriction: float, trusted_distance: float,
    raw_revalidated_m: float, hops: Sequence[Dict[str, Any]],
) -> None:
    """The restriction effect must be a real, non-negative subgraph penalty.

    The trusted subgraph is a SUBGRAPH of the raw graph, so removing edges can
    only ever lengthen the shortest path. A negative restriction effect would
    mean either an arithmetic error or - the reason this is measured against
    the re-derived distance rather than the historical CSV - a rounding
    residual of up to 5 mm from the two-decimal stored value.

    The reported effect is also re-derived from the distance route's own summed
    hop lengths, so it cannot drift from the edges the route actually used.

    Raises:
        TerrainRoutingError: if the effect is negative beyond tolerance, or
            does not equal the trusted route length minus the exact raw
            distance.
    """
    if restriction < -SUM_TOLERANCE:
        raise TerrainRoutingError(
            f"Stop {stop_id}: the restriction effect is {restriction:.9f} m. "
            "The trusted graph is a subgraph of the raw graph, so removing "
            "edges cannot shorten the shortest path.")
    from_hops = sum(hop["length_m"] for hop in hops) - float(raw_revalidated_m)
    if abs(restriction - from_hops) > SUM_TOLERANCE:
        raise TerrainRoutingError(
            f"Stop {stop_id}: the restriction effect is {restriction:.9f} m "
            f"but the distance route's own hops give {from_hops:.9f} m "
            f"against the re-derived raw distance {raw_revalidated_m:.9f} m.")
    if abs(restriction - (float(trusted_distance)
                          - float(raw_revalidated_m))) > SUM_TOLERANCE:
        raise TerrainRoutingError(
            f"Stop {stop_id}: the restriction effect does not equal the "
            "trusted distance minus the re-derived raw distance.")


def compare_stop(
    stop: pd.Series, origin: int, graph: nx.MultiDiGraph, trusted_keys: set,
    subgraph: nx.MultiDiGraph, distance_graph: nx.DiGraph,
    effort_graph: nx.DiGraph, distance_labels: Dict[int, Tuple[float, float]],
    distance_predecessor: Dict[int, int],
    effort_labels: Dict[int, Tuple[float, float]],
    effort_predecessor: Dict[int, int],
    raw_revalidated_m: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One stop's comparison row, and its route-edge rows.

    ``raw_revalidated_m`` is the full-precision raw-graph distance this run
    re-derived for the stop. It is passed in explicitly rather than read from
    shared state, and it is what the restriction effect is measured against -
    the historical two-decimal value stays untouched in its own column.
    """
    node = int(stop["nearest_node"])
    row: Dict[str, Any] = {name: None for name in COMPARISON_COLUMNS}
    row.update({
        "stop_id": stop["stop_id"], "stop_name": stop.get("stop_name", ""),
        "mode": stop.get("mode", ""), "component_id": stop.get("component_id"),
        "nearest_node": node, "status_reason": "", "status_note": "",
        "week4_raw_distance_m": stop.get("network_distance_m"),
        "raw_revalidated_distance_m": round(float(raw_revalidated_m), 6)
        if raw_revalidated_m is not None else None,
        "verified": False,
    })

    if stop["route_status"] == STATUS_DIFFERENT_COMPONENT:
        row.update({"comparison_status": STATUS_RAW_GAP,
                    "status_reason": REASON_RAW_COMPONENT_GAP,
                    "status_note": JUBILEE_NOTE})
        return row, []

    if stop["route_status"] != STATUS_ROUTED:
        row.update({"comparison_status": STATUS_NO_RAW_BASELINE,
                    "status_reason": REASON_NOT_IN_BASELINE,
                    "status_note": (
                        f"Week 4 recorded status {stop['route_status']!r}, so "
                        "there is no raw distance baseline to compare "
                        "against.")})
        return row, []

    if node not in distance_labels or node not in effort_labels:
        no_trusted_edge = subgraph.degree(node) == 0
        row.update({
            "comparison_status": STATUS_NO_TRUSTED_PATH,
            "status_reason": (REASON_NO_TRUSTED_EDGE_AT_STOP if no_trusted_edge
                              else REASON_NO_TRUSTED_PATH),
            "status_note": (SUBGRAPH_RESTRICTION_NOTE if no_trusted_edge else (
                "No route reaches this stop using only edges with an adopted "
                "terrain cost. A trusted-cost SUBGRAPH restriction, i.e. model "
                "coverage - NOT a terrain effect and NOT a statement about the "
                "stop.")),
        })
        return row, []

    distance_route = solve_route(distance_graph, distance_labels,
                                 distance_predecessor, OBJECTIVE_DISTANCE,
                                 origin, node)
    effort_route = solve_route(effort_graph, effort_labels, effort_predecessor,
                               OBJECTIVE_EFFORT, origin, node)

    for route in (distance_route, effort_route):
        verify_route(route["hops"], route["nodes"], origin, node,
                     route["distance"], route["effort"], graph, trusted_keys,
                     f"stop {stop['stop_id']} {route['objective']} route")
    verify_optimality(stop["stop_id"], distance_route, effort_route)

    distance_edges = list(distance_route["edges"])
    effort_edges = list(effort_route["edges"])
    shared = len(set(distance_edges) & set(effort_edges))
    edge_changed = distance_edges != effort_edges
    node_changed = distance_route["nodes"] != effort_route["nodes"]

    if raw_revalidated_m is None:
        raise TerrainRoutingError(
            f"Stop {stop['stop_id']} is routed on the raw graph but no "
            "re-derived raw distance was supplied. The restriction effect "
            "must be measured against the full-precision value, never the "
            "two-decimal historical one.")
    restriction = distance_route["distance"] - float(raw_revalidated_m)
    verify_restriction_effect(stop["stop_id"], restriction,
                              distance_route["distance"],
                              float(raw_revalidated_m), distance_route["hops"])

    row.update({
        "comparison_status": STATUS_COMPARED,
        "trusted_subgraph_distance_m": round(distance_route["distance"], 3),
        "restriction_effect_m": round(restriction, 3),
        "restriction_effect_pct": round(
            100.0 * restriction / float(raw_revalidated_m), 4)
        if float(raw_revalidated_m) else None,
        "distance_route_distance_m": round(distance_route["distance"], 3),
        "distance_route_effort_j_per_kg": round(distance_route["effort"], 4),
        "distance_route_edges": len(distance_edges),
        "effort_route_distance_m": round(effort_route["distance"], 3),
        "effort_route_effort_j_per_kg": round(effort_route["effort"], 4),
        "effort_route_edges": len(effort_edges),
        "effort_saving_j_per_kg": round(
            distance_route["effort"] - effort_route["effort"], 4),
        "effort_saving_pct": round(
            100.0 * (distance_route["effort"] - effort_route["effort"])
            / distance_route["effort"], 4) if distance_route["effort"] else None,
        "distance_detour_m": round(
            effort_route["distance"] - distance_route["distance"], 3),
        "distance_detour_pct": round(
            100.0 * (effort_route["distance"] - distance_route["distance"])
            / distance_route["distance"], 4)
        if distance_route["distance"] else None,
        "edge_route_changed": bool(edge_changed),
        "node_route_changed": bool(node_changed),
        "parallel_key_only_change": bool(edge_changed and not node_changed),
        "shared_edge_count": shared,
        "distance_only_edges": len(set(distance_edges) - set(effort_edges)),
        "effort_only_edges": len(set(effort_edges) - set(distance_edges)),
        "distance_route_mean_cost_j_per_kg_per_m": round(
            distance_route["effort"] / distance_route["distance"], 5)
        if distance_route["distance"] else None,
        "effort_route_mean_cost_j_per_kg_per_m": round(
            effort_route["effort"] / effort_route["distance"], 5)
        if effort_route["distance"] else None,
        "distance_route_dual_metric_tie_hops":
            distance_route["dual_metric_tie_hops"],
        "effort_route_dual_metric_tie_hops":
            effort_route["dual_metric_tie_hops"],
        "verified": True,
    })

    edge_rows: List[Dict[str, Any]] = []
    for route in (distance_route, effort_route):
        for hop in route["hops"]:
            edge_rows.append({
                "stop_id": stop["stop_id"], "metric": route["objective"],
                **hop,
            })
    return row, edge_rows


def attach_cost_status(
    edge_rows: pd.DataFrame, cost_table: pd.DataFrame
) -> pd.DataFrame:
    """Carry each hop's terrain-cost status through to the audit table."""
    if edge_rows.empty:
        return edge_rows
    statuses = {
        (int(row.u), int(row.v), int(row.key)): row.terrain_cost_status
        for row in cost_table.itertuples()
    }
    edge_rows = edge_rows.copy()
    edge_rows["terrain_cost_status"] = [
        statuses[(int(u), int(v), int(k))]
        for u, v, k in zip(edge_rows["u"], edge_rows["v"], edge_rows["key"])
    ]
    return edge_rows[list(EDGE_COLUMNS)]


EDGE_COLUMNS: Tuple[str, ...] = (
    "stop_id", "metric", "hop_index", "u", "v", "key", "length_m",
    "terrain_effort_j_per_kg", "terrain_cost_status", "hop_resolution",
)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def distribution(values: pd.Series) -> Dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(finite)),
        "median": round(float(finite.median()), 3),
        "p90": round(float(finite.quantile(0.90)), 3),
        "max": round(float(finite.max()), 3),
    }


def build_summary(
    comparison: pd.DataFrame, edge_rows: pd.DataFrame, graph_sha256: str,
    cost_sha256: str, graph_path: Path, provenance: Dict[str, Any],
    baseline: Dict[str, Any], subgraph: nx.MultiDiGraph,
    counters: Dict[str, Dict[str, int]], disagreements: int,
    reachable: Dict[str, int],
) -> Dict[str, Any]:
    compared = comparison[comparison["comparison_status"] == STATUS_COMPARED]
    return {
        "stage": "Week 5C-2B distance route against terrain-effort route",
        "purpose": (
            "Both metrics on exactly the same trusted-cost directed subgraph. "
            "The historical Week 4 raw-graph distance is a separate baseline "
            "and is never mixed into the terrain comparison. Statuses describe "
            "this model's coverage, not whether anywhere can be walked."),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "graph": {"path": str(graph_path), "sha256": graph_sha256,
                  "topology": "raw", "corrections_applied": []},
        "edge_terrain_cost_sha256": cost_sha256,
        "cost_provenance": provenance,
        "week4_baseline_revalidation": baseline,
        "method": {
            "objective_definitions": {
                "distance_route": "lexicographic minimum of "
                                  "(total_distance_m, total_effort_j_per_kg)",
                "effort_route": "lexicographic minimum of "
                                "(total_effort_j_per_kg, total_distance_m)",
            },
            "secondary_metric_role": (
                "tie-break only - the two metrics are never combined, no "
                "weighting is applied and no metres-to-joules conversion is "
                "invented"),
            "distance_source": "raw GraphML edge attribute 'length'",
            "effort_source": "edge_terrain_cost.csv terrain_effort_j_per_kg",
            "trusted_statuses": list(TRUSTED_STATUSES),
            "parallel_collapse": {
                "distance_graph": "min (length_m, terrain_effort_j_per_kg, key)",
                "effort_graph": "min (terrain_effort_j_per_kg, length_m, key)",
                "key_role": "deterministic tie-break only when BOTH metrics "
                            "are identical",
            },
            "direction": "origin -> stop only, matching Week 4",
            "directional_note": (
                "Minetti cost is directional: the reverse leg can differ and "
                "is future analysis, not assumed symmetric (D-21)."),
        },
        "subgraph": {
            "nodes": int(subgraph.number_of_nodes()),
            "trusted_directed_edges": int(subgraph.number_of_edges()),
            "nodes_with_no_trusted_edge": int(sum(
                1 for node in subgraph.nodes if subgraph.degree(node) == 0)),
            "reachable_nodes_raw_graph": reachable["raw"],
            "reachable_nodes_trusted_subgraph": reachable["trusted"],
        },
        "parallel_edges": {
            "ordered_pairs_with_parallel_edges":
                counters[OBJECTIVE_DISTANCE]["parallel_pairs"],
            "exact_two_metric_equivalent_ties":
                counters[OBJECTIVE_DISTANCE]["dual_metric_ties"],
            "pairs_where_the_objectives_choose_different_keys": disagreements,
            "trusted_self_loop_edges":
                counters[OBJECTIVE_DISTANCE]["self_loops"],
            "note": "Structural counts about the graph and the cost table. "
                    "Not accessibility findings.",
        },
        "status_counts": comparison["comparison_status"].value_counts().to_dict(),
        "compared": {
            "stops": int(len(compared)),
            "routes_with_a_changed_edge_sequence": int(
                compared["edge_route_changed"].eq(True).sum()),
            "routes_with_a_changed_node_sequence": int(
                compared["node_route_changed"].eq(True).sum()),
            "routes_changed_only_by_a_parallel_key": int(
                compared["parallel_key_only_change"].eq(True).sum()),
            "effort_saving_j_per_kg": distribution(
                compared["effort_saving_j_per_kg"]),
            "effort_saving_pct": distribution(compared["effort_saving_pct"]),
            "distance_detour_m": distribution(compared["distance_detour_m"]),
            "distance_detour_pct": distribution(
                compared["distance_detour_pct"]),
        },
        "restriction_effect": {
            "note": (
                "trusted_subgraph_distance_m minus the FULL-PRECISION raw "
                "distance re-derived by this run (raw_revalidated_distance_m), "
                "not the two-decimal historical CSV value, which is kept as an "
                "audit column. It "
                "isolates model-coverage and topology restriction from terrain "
                "optimisation and is NEVER part of the effort comparison."),
            "metres": distribution(compared["restriction_effect_m"]),
            "percent": distribution(compared["restriction_effect_pct"]),
        },
        "route_edges_rows": int(len(edge_rows)),
        "not_done": [
            "no stop ranking", "no combined distance-effort score",
            "no cost for untrusted edges", "no stairs penalty",
            "OSM-FIX-001 not applied", "no reverse (stop -> origin) leg",
            "Week 4 outputs not rewritten", "raw GraphML not modified",
            "interactive demo unchanged",
        ],
    }


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 5C-2B distance vs terrain-effort route comparison.")
    parser.add_argument("--graph", default=audit.DEFAULT_GRAPH)
    parser.add_argument("--cost-dir", default=None)
    options = parser.parse_args(argv)

    graph_path = Path(options.graph)
    if not graph_path.is_absolute():
        graph_path = (CONFIG_ROOT / graph_path).resolve()
    source_dir = Path(options.cost_dir).resolve() if options.cost_dir \
        else cost_dir()
    cache = CONFIG_ROOT / "data" / "cache"

    print("Week 5C-2B - distance route against terrain-effort route")
    print("Both metrics on EXACTLY the same trusted-cost directed subgraph.")
    print("Distance route: lexicographic minimum of (distance, effort).")
    print("Effort route:   lexicographic minimum of (effort, distance).")
    print("The second metric is a tie-break only - the two are never combined.")
    print("The Week 4 raw-graph distance stays a separate baseline; the gap is")
    print("reported as restriction_effect and is not a terrain effect.")
    print("")

    try:
        cost_table = load_cost_table(source_dir)
        cost_summary = load_cost_summary(source_dir)
    except ProvenanceMismatch as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    graph_sha256 = audit.file_sha256(graph_path)
    cost_sha256 = audit.file_sha256(source_dir / "edge_terrain_cost.csv")
    graph = load_graph(graph_path)

    try:
        provenance = verify_cost_provenance(cost_summary, cost_table, graph,
                                            graph_path, graph_sha256)
    except ProvenanceMismatch as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Graph        {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} directed edges")
    print(f"             sha256 {graph_sha256}")
    print(f"Cost table   {len(cost_table)} rows, "
          f"{provenance['trusted_edges']} trusted")
    print(f"             sha256 {cost_sha256}")
    print("Provenance verified: the terrain costs were built against this "
          "graph.")

    origin_record = load_json(cache / "snapped_origin.json")
    origin = int(origin_record["snap"]["nearest_node"])
    routes = pd.read_csv(cache / "week4_distance_routes.csv")

    try:
        baseline, raw_revalidated = revalidate_week4_baseline(
            graph, routes, origin, origin_record["snap"].get("component_id"))
    except BaselineMismatch as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Week 4 baseline re-verified: {baseline['routed_rows_verified']} "
          f"routed and {baseline['component_gap_rows_verified']} "
          "component-gap rows reproduced exactly.")
    print("")

    subgraph = build_trusted_subgraph(graph, cost_table)
    trusted_keys = {(int(u), int(v), int(k))
                    for u, v, k in subgraph.edges(keys=True)}
    counters: Dict[str, Dict[str, int]] = {}
    graphs: Dict[str, nx.DiGraph] = {}
    for objective in OBJECTIVES:
        graphs[objective], counters[objective] = collapse_parallel_edges(
            subgraph, objective)
    disagreements = collapse_disagreements(graphs[OBJECTIVE_DISTANCE],
                                           graphs[OBJECTIVE_EFFORT])

    print(f"Trusted subgraph  {subgraph.number_of_nodes()} nodes, "
          f"{subgraph.number_of_edges()} directed edges")
    print(f"                  {counters[OBJECTIVE_DISTANCE]['parallel_pairs']} "
          "ordered pairs carry parallel trusted edges; the two objectives "
          f"choose different keys on {disagreements} of them")
    print(f"                  "
          f"{counters[OBJECTIVE_DISTANCE]['dual_metric_ties']} pairs tie on "
          "BOTH metrics; "
          f"{counters[OBJECTIVE_DISTANCE]['self_loops']} trusted self-loops "
          "are kept and never enter a route")

    distance_labels, distance_predecessor = lexicographic_dijkstra(
        graphs[OBJECTIVE_DISTANCE], origin, ATTR_LENGTH, ATTR_EFFORT)
    effort_labels, effort_predecessor = lexicographic_dijkstra(
        graphs[OBJECTIVE_EFFORT], origin, ATTR_EFFORT, ATTR_LENGTH)
    reachable = {"raw": baseline["raw_reachable_nodes"],
                 "trusted": int(len(distance_labels))}
    print(f"                  reachable from the origin: "
          f"{reachable['trusted']} nodes on the trusted subgraph, "
          f"{reachable['raw']} on the raw graph")
    print("")

    rows: List[Dict[str, Any]] = []
    edge_records: List[Dict[str, Any]] = []
    for _index, stop in routes.iterrows():
        row, hops = compare_stop(
            stop, origin, graph, trusted_keys, subgraph,
            graphs[OBJECTIVE_DISTANCE], graphs[OBJECTIVE_EFFORT],
            distance_labels, distance_predecessor,
            effort_labels, effort_predecessor,
            raw_revalidated.get(stop["stop_id"]))
        rows.append(row)
        edge_records.extend(hops)

    comparison = pd.DataFrame(rows)[list(COMPARISON_COLUMNS)]
    edge_rows = attach_cost_status(pd.DataFrame(edge_records), cost_table)

    graph_sha256_after = audit.file_sha256(graph_path)
    if graph_sha256_after != graph_sha256:
        raise TerrainRoutingError(
            f"{graph_path} changed during the comparison. It is read-only.")

    out_dir = audit.ensure_directory(audit.cache_dir(OUTPUT_SUBDIR))
    comparison.to_csv(out_dir / "route_comparison.csv", index=False)
    edge_rows.to_csv(out_dir / "route_edges.csv", index=False)
    summary = build_summary(comparison, edge_rows, graph_sha256_after,
                            cost_sha256, graph_path, provenance, baseline,
                            subgraph, counters, disagreements, reachable)
    with open(out_dir / "terrain_routing_summary.json", "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("=" * 78)
    print("Comparison status")
    print("=" * 78)
    for status in STATUS_ORDER:
        count = int((comparison["comparison_status"] == status).sum())
        print(f"  {status:<32}{count:>4}")

    compared = comparison[comparison["comparison_status"] == STATUS_COMPARED]
    if not compared.empty:
        print("")
        print("=" * 78)
        print("Where the effort route differs from the distance route")
        print("=" * 78)
        print(f"  edge sequence changed            "
              f"{int(compared['edge_route_changed'].eq(True).sum())} of "
              f"{len(compared)}")
        print(f"  node sequence changed            "
              f"{int(compared['node_route_changed'].eq(True).sum())}")
        print(f"  changed only by a parallel key   "
              f"{int(compared['parallel_key_only_change'].eq(True).sum())}")
        for name, column in (("effort saving (J/kg)", "effort_saving_j_per_kg"),
                             ("distance detour (m)", "distance_detour_m"),
                             ("restriction effect (m)",
                              "restriction_effect_m")):
            stats = distribution(compared[column])
            print(f"  {name:<32} median {stats['median']}   p90 "
                  f"{stats['p90']}   max {stats['max']}")
        print("")
        print("  restriction effect is the gap to the Week 4 RAW baseline: "
              "model coverage")
        print("  and topology, not terrain. It is not part of the effort "
              "comparison.")

    print("")
    print(f"Wrote data/cache/{OUTPUT_SUBDIR}/ (route_comparison.csv, "
          "route_edges.csv,")
    print("     terrain_routing_summary.json)")
    print(f"The Stage 3 graph is unchanged: sha256 {graph_sha256_after}")
    print("")
    print("Two independent optima and their trade-off. No stop is ranked, the "
          "two metrics")
    print("are never combined into a score, and no route is claimed for an "
          "edge without")
    print("an adopted terrain cost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
