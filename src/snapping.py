"""Stage 4 - connect the test origin and the candidate stops to the walking graph.

GTFS stop coordinates almost never coincide with an OSM node, so both the
provisional test origin and every candidate stop must be attached to a node of
the pedestrian graph before any routing can happen. The snap distance is
reported, never assumed to be zero.

What Stage 4 does and does not do
---------------------------------
It records, faithfully, where each stop attaches to the RAW graph. It does not
route, does not decide which stop is nearest, does not judge reachability, and
does not repair anything.

In particular, two candidate stops (Jubilee Park Light Rail) attach to a small
component that is disconnected from the rest of the network in this OSM
snapshot. That is a known data-quality defect, recorded as I-04 with a proposed
correction OSM-FIX-001 which is NOT applied. Stage 4 keeps them exactly where
the raw graph puts them: it does not merge them, does not move them into the
main component, and does not drop them. Hiding the defect here would hide it
from every later stage.

The snap distance is a STRAIGHT LINE, not a walking distance. The distance from
a stop to its snapped node is an off-network gap that the walking graph does not
model - the walk from the physical stop to the nearest mapped footpath node.
Stage 5 must decide what to do with it; Stage 4's job is to make sure the number
survives.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from math import isfinite
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import osmnx as ox
import pandas as pd
from pyproj import Geod, Transformer

from src.osm_network import component_labels

logger = logging.getLogger(__name__)

# Columns Stage 4 adds. Stage 2's columns are preserved ahead of these,
# untouched and in their original order.
SNAP_COLUMNS: List[str] = [
    "nearest_node",
    "snap_distance_m",
    "node_lat",
    "node_lon",
    "component_id",
    "component_nodes",
    "shares_node_with",
]

GEOD = Geod(ellps="WGS84")

WGS84 = "EPSG:4326"

# GDA2020 / MGA Zone 56. The same metric CRS Stage 3 used for its snapping
# diagnostic, and the same one the routing buffer was measured in. Snapping on
# the projected graph rather than the unprojected one is a deliberate choice:
# measured against the real graph, both pick the identical nearest node for all
# 37 stops, but the projected distance agrees with an ellipsoidal geodesic
# measurement to about 6 mm while the unprojected spherical measurement differs
# by up to about 7 cm. Agreeing with Stage 3, and with the ellipsoid, is worth
# more than the small simplification of skipping the projection.
METRIC_CRS = "EPSG:7856"

# Recorded in provenance rather than enforced: a projected Euclidean
# measurement and an ellipsoidal geodesic measurement are different things and
# differ slightly. The difference is reported, not corrected.
DISTANCE_DEFINITION = (
    "Projected straight-line (Euclidean) distance in metres in EPSG:7856 "
    "(GDA2020 / MGA Zone 56), from the stop coordinate to the nearest graph "
    "node, as returned by osmnx.distance.nearest_nodes on the projected graph. "
    "NOT a walking distance."
)

TIE_BREAK_LIMITATION = (
    "Where two nodes are equidistant, the winner is decided by the neighbour "
    "search's internal ordering, which follows the order nodes were read from "
    "the GraphML file. This is reproducible for a given graph file but is not "
    "a principled rule. No tie detection is performed."
)


class SnappingError(RuntimeError):
    """Raised with an actionable message when snapping cannot be performed."""


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def project_for_snapping(
    graph: nx.MultiDiGraph, metric_crs: str = METRIC_CRS
) -> nx.MultiDiGraph:
    """Project the cached graph into the metric CRS used for snapping.

    Node ids are preserved by the projection, so a node found on the projected
    graph can be looked up in the original for its WGS84 coordinates. The
    cached graph on disk is not touched.
    """
    if graph.number_of_nodes() == 0:
        raise SnappingError("The walking graph has no nodes to snap to.")
    return ox.projection.project_graph(graph, to_crs=metric_crs)


def project_coordinates(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    metric_crs: str = METRIC_CRS,
) -> Tuple[List[float], List[float]]:
    """Project WGS84 coordinates into the metric CRS.

    Raises:
        SnappingError: if any coordinate projects to a non-finite value. That
            means the point lies outside the projection's valid domain, which
            in practice almost always means latitude and longitude were
            supplied the wrong way round. Returning infinity instead would let
            a swapped pair travel on as a plausible-looking result.
    """
    transform = Transformer.from_crs(WGS84, metric_crs, always_xy=True).transform
    xs: List[float] = []
    ys: List[float] = []
    for index, (latitude, longitude) in enumerate(zip(latitudes, longitudes)):
        x, y = transform(float(longitude), float(latitude))
        if not (isfinite(x) and isfinite(y)):
            raise SnappingError(
                f"Coordinate {index} (latitude {latitude}, longitude "
                f"{longitude}) does not project into {metric_crs}. Check that "
                "latitude and longitude have not been swapped."
            )
        xs.append(x)
        ys.append(y)
    return xs, ys


def snap_point(
    graph: nx.MultiDiGraph, latitude: float, longitude: float
) -> Tuple[int, float]:
    """Find the nearest graph node to one coordinate.

    Args:
        graph: pedestrian graph from osm_network, as cached (EPSG:4326).
        latitude: WGS84 latitude.
        longitude: WGS84 longitude.

    Returns:
        Tuple of (node id, snap distance in metres, measured in EPSG:7856).

    Raises:
        SnappingError: if the graph has no nodes, or the coordinate does not
            project into the metric CRS.

    Note:
        This projects the whole graph on every call. It is for single points;
        use :func:`snap_stops` for a table, which projects once and does one
        vectorised query.
    """
    nodes, distances = snap_many(graph, [latitude], [longitude])
    return nodes[0], distances[0]


def snap_many(
    graph: nx.MultiDiGraph,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    metric_crs: str = METRIC_CRS,
    projected: nx.MultiDiGraph = None,
) -> Tuple[List[int], List[float]]:
    """Snap many coordinates in one vectorised query, in the metric CRS.

    Args:
        graph: pedestrian graph as cached (EPSG:4326).
        latitudes: WGS84 latitudes.
        longitudes: WGS84 longitudes, in the same order.
        metric_crs: projected CRS the distance is measured in.
        projected: an already-projected graph, to avoid reprojecting.

    Returns:
        Tuple of (node ids, Euclidean distances in metres in ``metric_crs``).
    """
    if graph.number_of_nodes() == 0:
        raise SnappingError("The walking graph has no nodes to snap to.")
    if not len(latitudes):
        raise SnappingError("No coordinates to snap.")

    if projected is None:
        projected = project_for_snapping(graph, metric_crs)
    xs, ys = project_coordinates(latitudes, longitudes, metric_crs)
    nodes, distances = ox.distance.nearest_nodes(
        projected, xs, ys, return_dist=True
    )
    return [int(n) for n in nodes], [float(d) for d in distances]


def geodesic_distances(
    graph: nx.MultiDiGraph,
    nodes: Sequence[int],
    latitudes: Sequence[float],
    longitudes: Sequence[float],
) -> List[float]:
    """Independent QA measurement of the same distances, on the WGS84 ellipsoid.

    This exists to catch a whole class of silent errors - swapped latitude and
    longitude, a wrong CRS, a wrong node - that would leave plausible-looking
    numbers behind. It is a cross-check only: the reported distance stays the
    one OSMnx produced, and only the maximum disagreement is recorded, in
    provenance.
    """
    return [
        float(GEOD.inv(graph.nodes[node]["x"], graph.nodes[node]["y"],
                       float(longitude), float(latitude))[2])
        for node, latitude, longitude in zip(nodes, latitudes, longitudes)
    ]


# --------------------------------------------------------------------------- #
# annotation
# --------------------------------------------------------------------------- #
def annotate_components(
    frame: pd.DataFrame, graph: nx.MultiDiGraph
) -> pd.DataFrame:
    """Attach the Stage 3 component label to each snapped row.

    Labels come from ``osm_network.component_labels`` and are never recomputed
    here. Re-enumerating components locally would produce a different
    numbering, and the labels quoted in docs/ASSUMPTIONS.md (C0, C14) would
    stop meaning what they say.
    """
    labels, sizes = component_labels(graph)
    frame = frame.copy()
    frame["component_id"] = [labels.get(node, "?") for node in frame["nearest_node"]]
    frame["component_nodes"] = [
        sizes.get(label, 0) for label in frame["component_id"]
    ]
    return frame


def annotate_shared_nodes(frame: pd.DataFrame) -> pd.DataFrame:
    """Record which other stops snapped to the same node.

    Distinct physical stops can share a nearest node - two light rail platforms
    do. They stay as separate rows with their own coordinates, because they are
    separate boarding locations. This column makes the shared attachment
    visible, so that identical Stage 5 results for two stops are recognisable
    as a consequence of the topology rather than a bug.
    """
    frame = frame.copy()
    by_node: Dict[int, List[str]] = {}
    for stop_id, node in zip(frame["stop_id"], frame["nearest_node"]):
        by_node.setdefault(node, []).append(str(stop_id))
    frame["shares_node_with"] = [
        ";".join(sorted(set(by_node[node]) - {str(stop_id)}))
        for stop_id, node in zip(frame["stop_id"], frame["nearest_node"])
    ]
    return frame


# --------------------------------------------------------------------------- #
# stage 4 outputs
# --------------------------------------------------------------------------- #
def snap_stops(graph: nx.MultiDiGraph, stops: pd.DataFrame) -> pd.DataFrame:
    """Snap every candidate stop to the walking graph.

    Stage 2's columns are preserved exactly, in their original order, and the
    Stage 4 columns are appended. Row order and row count are preserved: no
    stop is dropped, merged, reordered or moved between components.

    Args:
        graph: pedestrian graph from osm_network, as cached (EPSG:4326).
            It is projected to EPSG:7856 internally for the snap.
        stops: candidate stops from gtfs_loader, with stop_id/stop_lat/stop_lon.

    Returns:
        The stop table with the Stage 4 columns added.

    Raises:
        SnappingError: if the graph or the stop table is empty, or a coordinate
            is not numeric.
    """
    if stops.empty:
        raise SnappingError(
            "The candidate stop table is empty. Run "
            "scripts/build_candidate_stops.py first."
        )
    for column in ("stop_id", "stop_lat", "stop_lon"):
        if column not in stops.columns:
            raise SnappingError(f"The stop table has no {column} column.")

    latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
    if latitudes.isna().any() or longitudes.isna().any():
        bad = stops.loc[latitudes.isna() | longitudes.isna(), "stop_id"].tolist()
        raise SnappingError(f"Non-numeric coordinates for stop_id {bad}.")

    projected = project_for_snapping(graph)
    nodes, distances = snap_many(
        graph, latitudes.tolist(), longitudes.tolist(), projected=projected
    )

    snapped = stops.copy()
    snapped["nearest_node"] = nodes
    snapped["snap_distance_m"] = [round(d, 2) for d in distances]
    snapped["node_lat"] = [graph.nodes[n]["y"] for n in nodes]
    snapped["node_lon"] = [graph.nodes[n]["x"] for n in nodes]
    snapped = annotate_components(snapped, graph)
    snapped = annotate_shared_nodes(snapped)

    original = [c for c in stops.columns]
    snapped = snapped[original + SNAP_COLUMNS]

    logger.info("Snapped %s stops to %s distinct nodes",
                len(snapped), snapped["nearest_node"].nunique())
    return snapped


def snap_origin(graph: nx.MultiDiGraph, origin: Dict[str, Any]) -> Dict[str, Any]:
    """Snap the provisional Week 4 test origin, keeping its warning attached.

    The warning travels in the output file, not only in the documentation,
    because the reader of the file is whoever writes Stage 5.

    Args:
        graph: pedestrian graph, as cached (EPSG:4326).
        origin: dict from config.get_test_origin with name/latitude/longitude.

    Returns:
        A dict with the labelled origin and its snap result. It records the
        origin's own component and nothing about any stop's reachability -
        that belongs to Stage 5.
    """
    node, distance = snap_point(graph, origin["latitude"], origin["longitude"])
    labels, sizes = component_labels(graph)
    label = labels.get(node, "?")
    return {
        "origin": {
            "role": "PROVISIONAL WEEK 4 TEST ORIGIN",
            "assumption": "A-02",
            "name": origin["name"],
            "latitude": origin["latitude"],
            "longitude": origin["longitude"],
            "_warning": (
                "One arbitrary point inside Glebe, used only to prove that the "
                "pipeline runs end to end. It is NOT a research choice, NOT a "
                "representative resident, and no conclusion may be drawn from "
                "it. See docs/ASSUMPTIONS.md (A-02)."
            ),
        },
        "snap": {
            "nearest_node": node,
            "snap_distance_m": round(distance, 2),
            "node_lat": graph.nodes[node]["y"],
            "node_lon": graph.nodes[node]["x"],
            "component_id": label,
            "component_nodes": sizes.get(label, 0),
            "_distance_definition": DISTANCE_DEFINITION,
        },
    }


def distance_summary(distances: Sequence[float]) -> Dict[str, float]:
    """Report the snapping-distance distribution. No threshold, no flagging."""
    values = sorted(float(d) for d in distances)
    count = len(values)
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


def file_sha256(path: Path) -> str:
    """Hash a file so downstream stages can prove which inputs were used."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapping_provenance(
    graph: nx.MultiDiGraph,
    graph_path: Path,
    graph_provenance: Dict[str, Any],
    stops_path: Path,
    snapped: pd.DataFrame,
    osmnx_distances: Sequence[float],
    geodesic_check: Sequence[float],
    generated: str,
) -> Dict[str, Any]:
    """Record exactly which graph and inputs produced this snapping result.

    Stage 5 can compare ``graph.sha256`` against the graph it loads and refuse
    to mix results from two different graphs. That check is the point of this
    file.
    """
    labels, sizes = component_labels(graph)
    differences = [
        abs(float(a) - float(b)) for a, b in zip(osmnx_distances, geodesic_check)
    ]
    return {
        "purpose": "Stage 4 snapping of the A-02 test origin and the candidate "
                   "stops onto the raw walking graph",
        "generated": generated,
        "topology": "raw",
        "corrections_applied": [],
        "corrections_file": "data/corrections/osm_topology_corrections.json",
        "corrections_available_but_not_applied": [
            "OSM-FIX-001 (status=proposed; must not be applied)"
        ],
        "graph": {
            "path": str(graph_path).replace("\\", "/"),
            "sha256": file_sha256(graph_path),
            "cache_key": graph_provenance.get("cache_key"),
            "retrieval_date": graph_provenance.get("retrieval_date"),
            "osmnx_version": graph_provenance.get("osmnx_version"),
            "network_type": graph_provenance.get("network_type"),
            "buffer_metres": graph_provenance.get("buffer_metres"),
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "total_components": len(sizes),
            "c0_nodes": sizes.get("C0"),
        },
        "inputs": {
            "stops_csv": str(stops_path).replace("\\", "/"),
            "stops_csv_sha256": file_sha256(stops_path),
            "stops_rows": int(len(snapped)),
            "origin": "config week4.test_origin (assumption A-02)",
        },
        "component_label_scheme": (
            "osm_network.component_labels - components sorted by size "
            "descending, C0 is always the largest. Reused, never recomputed."
        ),
        "snapping_method": {
            "library": f"osmnx {ox.__version__} distance.nearest_nodes",
            "graph_crs_cached": WGS84,
            "graph_crs_for_snapping": METRIC_CRS,
            "_crs_note": "The cached graph is projected to EPSG:7856 for the "
                         "nearest-node query, matching the Stage 3 diagnostic "
                         "baseline. The cached file itself is not modified.",
            "vectorised": True,
            "distance_definition": DISTANCE_DEFINITION,
            "tie_break_limitation": TIE_BREAK_LIMITATION,
        },
        "distance_quality_assurance": {
            "method": "pyproj.Geod(ellps='WGS84').inv - geodesic distance on "
                      "the WGS84 ellipsoid between the stop coordinate and the "
                      "node OSMnx selected. Independent of the OSMnx "
                      "measurement; used as a cross-check only.",
            "max_abs_difference_m": round(max(differences), 4) if differences else 0.0,
            "mean_abs_difference_m": (
                round(sum(differences) / len(differences), 4) if differences else 0.0
            ),
            "_note": "The official snap_distance_m is Euclidean straight-line "
                     "distance in EPSG:7856. The independent QA check measures "
                     "the same separation on the WGS84 ellipsoid via "
                     "pyproj.Geod. A difference of a few millimetres is "
                     "expected, because a projected Euclidean measurement and "
                     "an ellipsoidal geodesic measurement are not the same "
                     "thing. The projected OSMnx value is retained; nothing is "
                     "corrected here.",
        },
        "snap_distance_summary_m": distance_summary(osmnx_distances),
        "component_distribution": {
            label: int(count)
            for label, count in snapped["component_id"].value_counts().items()
        },
        "known_data_quality_issues": [
            {
                "issue": "I-04",
                "correction": "OSM-FIX-001",
                "correction_status": "proposed",
                "applied": False,
                "affected_stops": ["203779", "203780"],
                "stop_name": "Jubilee Park Light Rail",
                "component": "C14",
                "summary": "These stops attach to a 4-node footway fragment "
                           "that shares no node with the rest of the network "
                           "in this OSM snapshot.",
                "warning": "Any unreachability involving these stops is a "
                           "product of an OSM topology defect and MUST NOT be "
                           "reported as an accessibility finding. Transport "
                           "for NSW confirms that real pedestrian access to "
                           "Jubilee Park exists.",
            }
        ],
    }


def save_snapped_stops(frame: pd.DataFrame, out_path: Path) -> None:
    """Write the snapped stop table to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %s rows to %s", len(frame), out_path)
