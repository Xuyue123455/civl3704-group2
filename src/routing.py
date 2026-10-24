"""Stage 5 - distance-only routing smoke test.

Runs Dijkstra over edge ``length`` only, from the provisional A-02 test origin
to each physical candidate stop, on the RAW walking graph.

What this proves, and what it does not
--------------------------------------
It proves one thing: the GTFS candidate stops, the OSM walking network and the
test origin all live on one routable graph, and a path can be found between
them. That is a plumbing test.

It is NOT an accessibility model and NOT a ranking. Distance is the baseline
that a later week's terrain-aware cost will be compared against. Nothing here
decides which stop is easiest to reach, and no stop is ordered against another.

Deliberately absent: DEM, slope, Minetti, Tobler, walking effort, travel time,
accessibility scoring, combined totals, route geometry, and any topology
correction. OSM-FIX-001 stays proposed and unapplied.

Distances are graph distances only
----------------------------------
``network_distance_m`` is measured node to node along graph edges. It never
includes an off-network snap gap. Those gaps already exist elsewhere and are
deliberately not combined here: the origin's is in
``data/cache/snapped_origin.json`` and each stop's is the ``snap_distance_m``
column of the Stage 4 table. Whether and how to combine them is a later
decision that this stage must not pre-empt.

Unreachability is a result, not an error
----------------------------------------
Two candidate stops sit in a weakly connected component that the origin's
component cannot reach, because of a known OSM topology defect (I-04). They are
reported with a status explaining why, a blank distance, and the caveat carried
forward in provenance. No route is invented for them and nothing is repaired.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

from src.osm_network import component_labels

logger = logging.getLogger(__name__)

# Week 4 routes on edge length alone. Note that networkx defaults to hop count
# when weight is not given, which would silently answer a different question.
WEIGHT = "length"

STATUS_ROUTED = "routed"
STATUS_DIFFERENT_COMPONENT = "unreachable_different_component"
STATUS_SAME_COMPONENT = "unreachable_same_component"

# Columns Stage 5 adds. Stage 4's columns are preserved ahead of these,
# untouched and in their original order.
ROUTE_COLUMNS: List[str] = [
    "route_status",
    "network_distance_m",
    "route_node_count",
    "route_edge_count",
]


class RoutingError(RuntimeError):
    """Raised with an actionable message when routing cannot proceed."""


class ProvenanceMismatch(RoutingError):
    """Raised when the inputs did not come from the graph being routed on.

    There is deliberately no override. Mixing a snapping result with a
    different graph would produce node ids that mean something else, and the
    output would look entirely plausible. The fix is to regenerate Stage 4, not
    to bypass the check.
    """


# --------------------------------------------------------------------------- #
# provenance gate
# --------------------------------------------------------------------------- #
def file_sha256(path: Path) -> str:
    """Hash a file, so inputs can be tied to the graph they came from."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(
    graph: nx.MultiDiGraph,
    graph_path: Path,
    snapping_provenance: Dict[str, Any],
    snapped_origin: Dict[str, Any],
    snapped_stops: pd.DataFrame,
) -> Dict[str, Any]:
    """Refuse to route unless every input provably came from this graph.

    Checks, in order:

    1. the graph on disk hashes to what the Stage 4 snapping recorded;
    2. the snapped origin file records that same hash;
    3. the snapping declares raw topology with no corrections applied;
    4. the origin node exists in the graph;
    5. every stop's nearest_node exists in the graph.

    Args:
        graph: the loaded walking graph.
        graph_path: the file it was loaded from.
        snapping_provenance: parsed data/cache/snapping_provenance.json.
        snapped_origin: parsed data/cache/snapped_origin.json.
        snapped_stops: the Stage 4 snapped stop table.

    Returns:
        A dict summarising what was verified, for the routing provenance.

    Raises:
        ProvenanceMismatch: on any failure. There is no bypass.
    """
    actual = file_sha256(graph_path)
    recorded = (snapping_provenance.get("graph") or {}).get("sha256")
    if recorded != actual:
        raise ProvenanceMismatch(
            f"The graph at {graph_path} hashes to {actual}, but the Stage 4 "
            f"snapping was produced against {recorded}. The snapped node ids "
            "may refer to different nodes. Re-run scripts/snap_to_network.py "
            "against the current graph."
        )

    origin_recorded = (snapped_origin.get("graph") or {}).get("sha256")
    if origin_recorded != actual:
        raise ProvenanceMismatch(
            f"snapped_origin.json was produced against graph {origin_recorded}, "
            f"but the graph on disk is {actual}. The origin and the stop table "
            "must come from the same Stage 4 run. Re-run "
            "scripts/snap_to_network.py."
        )

    topology = snapping_provenance.get("topology")
    if topology != "raw":
        raise ProvenanceMismatch(
            f"The snapping declares topology {topology!r}, not 'raw'. Stage 5 "
            "routes on the raw graph only; it must know which topology its "
            "inputs came from."
        )

    corrections = snapping_provenance.get("corrections_applied")
    if corrections != []:
        raise ProvenanceMismatch(
            f"The snapping declares corrections_applied={corrections!r}. Stage 5 "
            "expects an uncorrected graph; OSM-FIX-001 remains proposed."
        )

    origin_node = snapped_origin["snap"]["nearest_node"]
    if origin_node not in graph.nodes:
        raise ProvenanceMismatch(
            f"The snapped origin node {origin_node} is not in the graph. The "
            "snapping is stale; re-run scripts/snap_to_network.py."
        )

    missing = [
        (stop_id, node)
        for stop_id, node in zip(snapped_stops["stop_id"],
                                 snapped_stops["nearest_node"].astype(int))
        if node not in graph.nodes
    ]
    if missing:
        raise ProvenanceMismatch(
            f"{len(missing)} snapped node(s) are not in the graph, e.g. "
            f"stop_id {missing[0][0]} -> node {missing[0][1]}. The snapping is "
            "stale; re-run scripts/snap_to_network.py."
        )

    return {
        "graph_sha256": actual,
        "graph_sha256_matches_snapping": True,
        "graph_sha256_matches_origin": True,
        "topology": topology,
        "corrections_applied": corrections,
        "origin_node_present": True,
        "stop_nodes_present": int(len(snapped_stops)),
    }


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def single_source_distances(
    graph: nx.MultiDiGraph, origin_node: int, weight: str = WEIGHT
) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    """One Dijkstra from the origin, reused for every stop.

    A single search rather than one per stop, not mainly for speed - the graph
    is small - but for consistency: all answers then come from one relaxation
    and cannot disagree with each other. Unreachability also falls out for
    free, as absence from the result.

    Args:
        graph: the walking graph.
        origin_node: the snapped origin node.
        weight: edge attribute to minimise. Week 4 uses ``length``.

    Returns:
        Tuple of (node -> distance, node -> path). Unreachable nodes are absent
        from both.

    Raises:
        RoutingError: if the origin is not in the graph.
    """
    if origin_node not in graph.nodes:
        raise RoutingError(f"Origin node {origin_node} is not in the graph.")
    distances, paths = nx.single_source_dijkstra(
        graph, origin_node, weight=weight
    )
    logger.info(
        "Dijkstra from %s reached %s of %s nodes",
        origin_node, len(distances), graph.number_of_nodes(),
    )
    return distances, paths


def shortest_path_by_distance(
    graph: nx.MultiDiGraph, origin_node: int, destination_node: int
) -> Tuple[Optional[float], Optional[List[int]]]:
    """Shortest walking path between two graph nodes, weighted by length.

    Args:
        graph: pedestrian graph from osm_network.
        origin_node: snapped origin node id.
        destination_node: snapped stop node id.

    Returns:
        Tuple of (distance in metres, list of node ids). Both are ``None``
        when no path exists - unreachable stops are reported, not hidden.
        Genuine unreachability is a result of the graph, not an error.

    Raises:
        RoutingError: if either node is not in the graph. A missing node is a
            different problem from an absent path, and is not silently
            reported as unreachable.
    """
    for node, role in ((origin_node, "origin"), (destination_node, "destination")):
        if node not in graph.nodes:
            raise RoutingError(f"The {role} node {node} is not in the graph.")
    try:
        distance, path = nx.single_source_dijkstra(
            graph, origin_node, target=destination_node, weight=WEIGHT
        )
    except nx.NetworkXNoPath:
        return None, None
    return float(distance), [int(n) for n in path]


def classify_route(
    reached: bool, origin_component: str, stop_component: str
) -> str:
    """Say why a stop was or was not reached.

    ``origin_component`` must be the ORIGIN's component, obtained from
    :func:`resolve_origin_component`. Passing the stop's own component here
    makes the comparison trivially equal and silently forces the
    same-component branch for every unreachable stop.

    The two unreachable cases are kept apart on purpose. A different weak
    component guarantees there is no path, and here that is a known and
    explained OSM defect. The same weak component failing is a different thing
    entirely - a directed dead end - and would be a new, unexplained finding
    that must not be hidden inside a generic "unreachable".
    """
    if reached:
        return STATUS_ROUTED
    if stop_component != origin_component:
        return STATUS_DIFFERENT_COMPONENT
    return STATUS_SAME_COMPONENT


def resolve_origin_component(
    graph: nx.MultiDiGraph,
    origin_node: int,
    origin_component: Optional[str] = None,
) -> str:
    """Determine which component the origin node is in.

    The origin's component comes from the Stage 4 record that already states
    it, or failing that from the graph itself. It is NEVER inferred from the
    stop table.

    That distinction caused a real bug. An earlier version looked for a stop
    whose ``nearest_node`` equalled the origin node and took its component. No
    stop snaps to the origin's node - there is no reason one should - so the
    lookup found nothing, and a fallback then substituted each stop's OWN
    component. The comparison was therefore always equal, and every unreachable
    stop was misclassified as ``unreachable_same_component``, including two
    that are demonstrably in a different component.

    Args:
        graph: the walking graph.
        origin_node: the snapped origin node.
        origin_component: the label recorded in snapped_origin.json, if known.

    Returns:
        The origin's component label.

    Raises:
        RoutingError: if the origin node is not in the graph, or if a supplied
            label contradicts the graph. A contradiction means the origin
            record is stale and must not be silently preferred.
    """
    if origin_node not in graph.nodes:
        raise RoutingError(f"Origin node {origin_node} is not in the graph.")
    labels, _sizes = component_labels(graph)
    derived = labels[origin_node]
    if origin_component is None:
        return derived
    if str(origin_component) != derived:
        raise RoutingError(
            f"snapped_origin.json says the origin node {origin_node} is in "
            f"component {origin_component!r}, but the graph puts it in "
            f"{derived!r}. The origin record is stale; re-run "
            "scripts/snap_to_network.py."
        )
    return derived


def route_to_all_stops(
    graph: nx.MultiDiGraph,
    origin_node: int,
    snapped_stops: pd.DataFrame,
    origin_component: Optional[str] = None,
) -> pd.DataFrame:
    """Route from the origin to every candidate stop.

    Stage 4's columns are preserved exactly, in their original order, and the
    Stage 5 columns are appended. Row order and row count are preserved: no
    stop is dropped, reordered or ranked.

    Args:
        graph: pedestrian graph from osm_network.
        origin_node: snapped origin node id.
        snapped_stops: output of snapping.snap_stops.
        origin_component: the origin's component label from
            snapped_origin.json. Optional; when omitted it is derived from the
            graph. It is never inferred from the stop table.

    Returns:
        One row per stop with network distance and node count. Stops that
        could not be reached keep a null distance and are flagged with the
        reason.

    Raises:
        RoutingError: if the stop table is empty, lacks required columns, or
            the origin's component cannot be determined.
    """
    if snapped_stops.empty:
        raise RoutingError(
            "The snapped stop table is empty. Run scripts/snap_to_network.py "
            "first."
        )
    for column in ("stop_id", "nearest_node", "component_id"):
        if column not in snapped_stops.columns:
            raise RoutingError(f"The snapped stop table has no {column} column.")

    distances, paths = single_source_distances(graph, origin_node)
    origin_label = resolve_origin_component(graph, origin_node, origin_component)

    statuses: List[str] = []
    network_distances: List[Optional[float]] = []
    node_counts: List[Optional[int]] = []
    edge_counts: List[Optional[int]] = []

    for node, component in zip(snapped_stops["nearest_node"].astype(int),
                               snapped_stops["component_id"]):
        reached = node in distances
        statuses.append(classify_route(reached, origin_label, component))
        if reached:
            path = paths[node]
            network_distances.append(round(float(distances[node]), 2))
            node_counts.append(len(path))
            edge_counts.append(len(path) - 1)
        else:
            # Blank, never 0 and never infinity. Zero would read as "already
            # there" and infinity would poison any later summary statistic.
            network_distances.append(None)
            node_counts.append(None)
            edge_counts.append(None)

    routed = snapped_stops.copy()
    routed["route_status"] = statuses
    routed["network_distance_m"] = network_distances
    # Nullable integers, so an unreachable row writes an empty cell while a
    # routed row writes "3" rather than "3.0". A plain float column would turn
    # every count into a decimal simply because some rows are blank.
    routed["route_node_count"] = pd.array(node_counts, dtype="Int64")
    routed["route_edge_count"] = pd.array(edge_counts, dtype="Int64")

    original = list(snapped_stops.columns)
    routed = routed[original + ROUTE_COLUMNS]

    logger.info(
        "Routed %s of %s stops",
        sum(1 for s in statuses if s == STATUS_ROUTED), len(statuses),
    )
    return routed


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def distance_summary(distances: Sequence[Optional[float]]) -> Dict[str, float]:
    """Summarise the routed distances. No ranking, no threshold.

    Both ``None`` and ``NaN`` are treated as absent. The distinction matters:
    a column built from a list containing ``None`` comes back out of pandas as
    ``NaN``, and ``NaN is not None``, so a naive identity filter would silently
    count unreachable stops as though they had a distance.
    """
    values = sorted(float(d) for d in distances if pd.notna(d))
    count = len(values)
    if not count:
        return {"count": 0}
    return {
        "count": count,
        "min_m": round(values[0], 2),
        "median_m": round(
            values[count // 2] if count % 2
            else (values[count // 2 - 1] + values[count // 2]) / 2, 2
        ),
        "mean_m": round(sum(values) / count, 2),
        "max_m": round(values[-1], 2),
    }


def routing_provenance(
    graph: nx.MultiDiGraph,
    graph_path: Path,
    verification: Dict[str, Any],
    snapping_provenance: Dict[str, Any],
    snapped_stops_path: Path,
    origin: Dict[str, Any],
    routed: pd.DataFrame,
    generated: str,
) -> Dict[str, Any]:
    """Record what was routed, on which graph, and with what caveats.

    ``known_data_quality_issues`` is carried forward verbatim from the Stage 4
    provenance rather than rewritten. Two independently worded copies of the
    same caveat would eventually drift apart, and the drifted one would be the
    one somebody quoted.
    """
    counts = routed["route_status"].value_counts().to_dict()
    return {
        "purpose": "Week 4 Stage 5 distance-only routing smoke test",
        "generated": generated,
        "_scope": (
            "A plumbing test proving the origin, the OSM walking network and "
            "the candidate stops share one routable graph. NOT an "
            "accessibility model, NOT a ranking, and no basis for any claim "
            "about which stop is easiest to reach."
        ),
        "topology": "raw",
        "corrections_applied": [],
        "routing_method": {
            "algorithm": "networkx.single_source_dijkstra, one search from the "
                         "origin reused for every stop",
            "weight": WEIGHT,
            "weight_units": "metres",
            "direction": "origin -> stop only. The graph is directed, so the "
                         "reverse direction is not assumed to be identical and "
                         "was not computed.",
            "distance_definition": (
                "Graph-network distance, node to node along edges. It does NOT "
                "include any off-network snap gap. The origin's gap is in "
                "snapped_origin.json and each stop's is snap_distance_m in the "
                "Stage 4 table; Stage 5 deliberately does not combine them."
            ),
            "not_included": [
                "DEM", "slope", "Minetti", "Tobler", "walking effort",
                "travel time", "accessibility ranking", "combined scoring",
                "route geometry", "topology correction",
            ],
        },
        "graph": {
            "path": str(graph_path).replace("\\", "/"),
            "sha256": verification["graph_sha256"],
            "cache_key": (snapping_provenance.get("graph") or {}).get("cache_key"),
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
        },
        "provenance_verification": verification,
        "inputs": {
            "snapped_stops_csv": str(snapped_stops_path).replace("\\", "/"),
            "snapped_stops_sha256": file_sha256(snapped_stops_path),
            "stops": int(len(routed)),
            "origin_node": origin["snap"]["nearest_node"],
            "origin_component": origin["snap"]["component_id"],
            "origin_component_source": "snapped_origin.json, cross-checked "
                                       "against the graph's own components",
            "origin_assumption": origin["origin"]["assumption"],
        },
        "route_status_counts": {str(k): int(v) for k, v in counts.items()},
        "network_distance_summary_m": distance_summary(
            routed["network_distance_m"].tolist()
        ),
        "known_data_quality_issues": snapping_provenance.get(
            "known_data_quality_issues", []
        ),
        "_unreachable_wording": (
            "Report unreachable stops as raw-graph routes unavailable due to "
            "the known I-04 topology artefact. Do NOT report them as a share "
            "of Glebe stops being unreachable, and do NOT present them as an "
            "accessibility finding."
        ),
    }


def save_routes(frame: pd.DataFrame, out_path: Path) -> None:
    """Write the routing result to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %s rows to %s", len(frame), out_path)


def load_json(path: Path) -> Dict[str, Any]:
    """Read a required JSON input, with an actionable error when it is absent."""
    path = Path(path)
    if not path.exists():
        raise RoutingError(
            f"{path} is missing. Run scripts/snap_to_network.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
