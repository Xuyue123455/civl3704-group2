"""Stage 3 - OpenStreetMap pedestrian network for the Glebe study area.

Builds a routable walking graph with OSMnx and caches it to GraphML so OSM is
not re-downloaded on every run.

Two extents, deliberately not the same thing
--------------------------------------------
    candidate-stop extent   the official Glebe suburb polygon (A-06)
    routing-network extent  that polygon, buffered outwards (A-07)

A walking route may legitimately leave and re-enter Glebe, so a network clipped
to the suburb boundary would lengthen or break routes near the edge. The buffer
is created in a projected metric CRS, never by buffering WGS84 degrees: at
Glebe's latitude one degree of longitude is about 92.4 km while one degree of
latitude is about 111.0 km, so a degree buffer is roughly 17 per cent narrower
east-west than north-south. That is a direction-dependent error, not a rounding
one.

The buffer distance is 300 m, adopted on evidence rather than chosen: the
sensitivity run of 2026-08-29 found the candidate-stop network-distance matrix
already unchanged at 300 m when the extent was expanded to 600 m and again to
1000 m. It is a routing-network EXTENT and is NOT a walking catchment - no
catchment has been adopted (A-04). See docs/ASSUMPTIONS.md (A-07) and
scripts/buffer_sensitivity.py.

Scope
-----
At Week 4 the graph carries geometry, edge length and preserved OSM tags only.
No DEM, no gradient, no stairs penalty, no accessibility cost, no realtime -
those are later weeks. Tags are PRESERVED here so that later work is possible;
nothing in this module interprets them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"

# GDA2020 / MGA Zone 56 - the metric CRS already used for Glebe in the Stage 2
# parent-station inspection. Kept the same so every metric distance in this
# project is measured in one system.
DEFAULT_METRIC_CRS = "EPSG:7856"

STEPS_HIGHWAY = "steps"

# --------------------------------------------------------------------------- #
# Tags preserved for later accessibility work.
#
# OSMnx discards unlisted tags at PARSE time, so these must be registered
# before any download or they are simply not in the graph and recovering them
# means downloading again. Nothing here is interpreted at Stage 3.
# --------------------------------------------------------------------------- #
ACCESSIBILITY_TAGS_WAY: Tuple[str, ...] = (
    # pedestrian classification
    "footway", "path", "sidewalk", "foot",
    # crossings
    "crossing", "crossing:markings", "crossing:island", "traffic_signals",
    # steps
    "step_count", "handrail", "ramp", "ramp:wheelchair", "conveying",
    # walking surface
    "surface", "smoothness",
    # explicit accessibility tagging
    "wheelchair", "tactile_paving",
    # structure and environment
    "covered", "lit",
    # OSM's own incline tag. Sparse, inconsistent, often categorical
    # ("up", "steep") rather than a gradient. It is NOT a substitute for the
    # DEM work, which is out of scope at Stage 3.
    "incline",
)

ACCESSIBILITY_TAGS_NODE: Tuple[str, ...] = (
    "crossing", "kerb", "barrier", "tactile_paving", "wheelchair",
    "traffic_signals", "footway",
)

# Reported as a coverage percentage after the first real download, so that
# "which accessibility attributes actually exist in Glebe" is an observation
# rather than an assumption.
TAG_COVERAGE_REPORT: Tuple[str, ...] = (
    "highway", "footway", "surface", "smoothness", "width", "incline",
    "wheelchair", "tactile_paving", "crossing", "step_count", "handrail",
    "ramp", "access", "foot", "bridge", "tunnel", "lit",
)


class NetworkUnavailable(RuntimeError):
    """Raised with an actionable message when the walking graph cannot be used."""


# --------------------------------------------------------------------------- #
# extent
# --------------------------------------------------------------------------- #
def routing_polygon(
    polygon: BaseGeometry,
    buffer_metres: float,
    metric_crs: str = DEFAULT_METRIC_CRS,
    source_crs: str = WGS84,
) -> BaseGeometry:
    """Buffer the candidate-stop polygon outwards in a projected metric CRS.

    Project -> buffer in metres -> project back. Never buffers degrees.

    Args:
        polygon: the official candidate-stop polygon, in ``source_crs``.
        buffer_metres: outward buffer in metres. Zero returns the input extent.
        metric_crs: projected CRS the buffer is measured in.
        source_crs: CRS of ``polygon`` and of the returned geometry.

    Returns:
        The routing extent, in ``source_crs``.

    Raises:
        TypeError: if buffer_metres is not a real number.
        ValueError: if buffer_metres is negative, or the polygon is empty.
    """
    if isinstance(buffer_metres, bool) or not isinstance(buffer_metres, (int, float)):
        raise TypeError(
            f"buffer_metres must be a number in metres, got {buffer_metres!r}."
        )
    if buffer_metres < 0:
        raise ValueError(
            f"buffer_metres must not be negative, got {buffer_metres}. The "
            "routing extent may only grow outwards from the candidate extent."
        )
    if polygon.is_empty:
        raise ValueError("Cannot buffer an empty polygon.")

    to_metric = Transformer.from_crs(source_crs, metric_crs, always_xy=True).transform
    to_source = Transformer.from_crs(metric_crs, source_crs, always_xy=True).transform

    projected = shapely_transform(to_metric, polygon)
    buffered = projected if buffer_metres == 0 else projected.buffer(buffer_metres)
    return shapely_transform(to_source, buffered)


def bbox_polygon(bbox: Dict[str, float]) -> Polygon:
    """Turn a named bounding box into a rectangle polygon."""
    return Polygon([
        (bbox["west"], bbox["south"]),
        (bbox["east"], bbox["south"]),
        (bbox["east"], bbox["north"]),
        (bbox["west"], bbox["north"]),
    ])


def save_routing_extent(
    polygon: BaseGeometry, out_path: Path, properties: Optional[Dict[str, Any]] = None
) -> None:
    """Write the routing extent to GeoJSON so it can be inspected in QGIS."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": properties or {},
            "geometry": mapping(polygon),
        }],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def graph_cache_key(
    polygon: BaseGeometry,
    buffer_metres: float,
    network_type: str,
    simplify: bool,
    truncate_by_edge: bool,
    metric_crs: str = DEFAULT_METRIC_CRS,
    retain_all: bool = True,
) -> str:
    """Fingerprint every input that changes the graph.

    The candidate polygon is hashed rather than the buffered result, so the key
    does not drift with pyproj patch releases. Change the buffer, the filter or
    any construction flag and the key changes, which is what stops a cached
    graph from being silently reused for different settings.

    Returns:
        A hex sha256 digest.
    """
    payload = json.dumps(
        {
            "polygon_wkt_sha256": hashlib.sha256(
                polygon.wkt.encode("utf-8")
            ).hexdigest(),
            "buffer_metres": float(buffer_metres),
            "network_type": network_type,
            "simplify": bool(simplify),
            "truncate_by_edge": bool(truncate_by_edge),
            "retain_all": bool(retain_all),
            "metric_crs": metric_crs,
            "useful_tags_way": sorted(ACCESSIBILITY_TAGS_WAY),
            "useful_tags_node": sorted(ACCESSIBILITY_TAGS_NODE),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def configure_osm_tags() -> Dict[str, List[str]]:
    """Register the accessibility tags with OSMnx BEFORE any download.

    OSMnx drops unlisted tags while parsing the Overpass response, so calling
    this after a download is useless. The project tags are added to the OSMnx
    defaults rather than replacing them.

    Returns:
        The resulting way and node tag lists, for provenance.
    """
    ox.settings.useful_tags_way = sorted(
        set(ox.settings.useful_tags_way) | set(ACCESSIBILITY_TAGS_WAY)
    )
    ox.settings.useful_tags_node = sorted(
        set(ox.settings.useful_tags_node) | set(ACCESSIBILITY_TAGS_NODE)
    )
    return {
        "way": list(ox.settings.useful_tags_way),
        "node": list(ox.settings.useful_tags_node),
    }


def build_walk_graph_from_polygon(
    polygon: BaseGeometry,
    network_type: str = "walk",
    simplify: bool = True,
    truncate_by_edge: bool = True,
    retain_all: bool = True,
) -> nx.MultiDiGraph:
    """Download the pedestrian network covering a routing extent.

    This is the only function in the module that touches the network.

    ``network_type="walk"`` is the standard OSMnx preset. It is used rather
    than a custom highway filter because a custom filter is a modelling
    decision and Stage 3 has no evidence to base one on. In particular the
    preset RETAINS ``highway=steps``: this project exists to measure whether
    steps make a stop hard to reach, so filtering them out would delete the
    finding before it could be made.

    ``retain_all=True`` keeps disconnected components. Glebe is bounded by
    water and a motorway corridor, so isolated components are plausible and
    must be reported rather than silently discarded.

    ``truncate_by_edge=True`` keeps edges that cross the extent boundary, so
    the network is not cut mid-block.

    Args:
        polygon: routing extent in WGS84, from :func:`routing_polygon`.
        network_type: OSMnx preset name.
        simplify: collapse interstitial nodes.
        truncate_by_edge: keep boundary-crossing edges.
        retain_all: keep disconnected components.

    Returns:
        A routable MultiDiGraph in WGS84.
    """
    configure_osm_tags()
    logger.info(
        "Requesting OSM: network_type=%s simplify=%s truncate_by_edge=%s "
        "retain_all=%s", network_type, simplify, truncate_by_edge, retain_all,
    )
    return ox.graph_from_polygon(
        polygon,
        network_type=network_type,
        simplify=simplify,
        truncate_by_edge=truncate_by_edge,
        retain_all=retain_all,
    )


def build_walk_graph(
    bbox: Dict[str, float], network_type: str = "walk"
) -> nx.MultiDiGraph:
    """Download a pedestrian network for a bounding box.

    LEGACY. Retained so the Stage 1 structure tests keep passing. The Stage 3
    pipeline uses :func:`build_walk_graph_from_polygon` with the buffered
    official polygon; a rectangle is not the authoritative routing geography.

    Args:
        bbox: dict with keys north/south/east/west.
        network_type: OSMnx preset name.

    Returns:
        A routable MultiDiGraph in WGS84.
    """
    return build_walk_graph_from_polygon(bbox_polygon(bbox), network_type=network_type)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def drop_none_attributes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a copy with ``None``-valued attributes removed.

    Measured behaviour of OSMnx 2.1.1: ``save_graphml`` writes ``None`` as the
    four-character string ``"None"``, and ``load_graphml`` reads it back as
    that string. A tag whose value is the string "None" looks like a real
    value and is worse than an absent tag, so the key is dropped instead.

    This is the ONLY transformation applied before saving. List-valued tags -
    which simplification creates, e.g. ``highway=['footway', 'steps']`` - are
    left exactly as they are, because OSMnx 2.1.1 round-trips them losslessly
    (verified by tests/test_stage3_network.py). Nothing is flattened, joined
    or canonicalised, so no tag semantics are destroyed for the sake of
    serialisation convenience.

    Args:
        graph: the graph to clean.

    Returns:
        A copy. The input graph is not modified.
    """
    clean = graph.copy()
    for _node, data in clean.nodes(data=True):
        for key in [k for k, v in data.items() if v is None]:
            del data[key]
    for _u, _v, _k, data in clean.edges(keys=True, data=True):
        for key in [k for k, v in data.items() if v is None]:
            del data[key]
    return clean


def save_graph(graph: nx.MultiDiGraph, out_path: Path) -> None:
    """Save the graph to GraphML so later runs can reuse it.

    Args:
        graph: the walking graph.
        out_path: destination, normally data/cache/glebe_walk.graphml.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(drop_none_attributes(graph), out_path)
    logger.info(
        "Wrote %s nodes / %s edges to %s",
        f"{graph.number_of_nodes():,}", f"{graph.number_of_edges():,}", out_path,
    )


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a previously saved GraphML walking graph.

    OSMnx restores ``length``, ``osmid``, ``oneway``, ``reversed``, ``x`` and
    ``y`` to their proper types. The preserved accessibility tags come back as
    strings, or as lists of strings where simplification merged several source
    ways. That is deliberate: after simplification a tag such as ``step_count``
    can legitimately hold several values, so coercing it to a number here would
    fail. Numeric interpretation belongs to whichever later week actually
    models the attribute.

    Args:
        path: GraphML file written by :func:`save_graph`.

    Returns:
        The walking graph.

    Raises:
        NetworkUnavailable: if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise NetworkUnavailable(
            f"{path} is missing. Run scripts/build_walk_network.py first."
        )
    return ox.load_graphml(path)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def tag_values(value: Any) -> List[str]:
    """Flatten one edge attribute into a list of string values.

    Simplification can merge several source ways into one edge, giving a list
    where a single value would otherwise sit. Both forms are handled so that a
    merged ``['footway', 'steps']`` edge counts towards both classes.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def count_steps_edges(graph: nx.MultiDiGraph) -> int:
    """Count edges whose highway tag includes ``steps``."""
    return sum(
        1
        for _u, _v, _k, data in graph.edges(keys=True, data=True)
        if STEPS_HIGHWAY in tag_values(data.get("highway"))
    )


def describe_graph(graph: nx.MultiDiGraph) -> Dict[str, object]:
    """Return diagnostics: size, length statistics, extent, connectivity, tags.

    Edge length is checked for being finite and positive. It is deliberately
    NOT compared against the straight-line distance between the edge endpoints:
    after simplification an edge carries the real curved geometry, which is
    legitimately longer than its endpoint chord, so such a check would flag
    correct data as wrong.

    Args:
        graph: the walking graph.

    Returns:
        A dictionary of diagnostics. Nothing here fails the pipeline; the
        caller decides what warrants a warning.
    """
    lengths: List[float] = []
    missing_length = 0
    nonpositive_length = 0
    highway_counts: Dict[str, int] = {}

    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        raw = data.get("length")
        try:
            length = float(raw)
        except (TypeError, ValueError):
            missing_length += 1
        else:
            if length > 0 and length == length and length != float("inf"):
                lengths.append(length)
            else:
                nonpositive_length += 1
        for value in tag_values(data.get("highway")):
            highway_counts[value] = highway_counts.get(value, 0) + 1

    xs = [data["x"] for _n, data in graph.nodes(data=True) if "x" in data]
    ys = [data["y"] for _n, data in graph.nodes(data=True) if "y" in data]

    weak = list(nx.weakly_connected_components(graph)) if graph.number_of_nodes() else []
    largest = max((len(c) for c in weak), default=0)

    summary: Dict[str, object] = {
        "crs": graph.graph.get("crs"),
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "edges_with_valid_length": len(lengths),
        "edges_missing_length": missing_length,
        "edges_nonpositive_length": nonpositive_length,
        "total_length_km": round(sum(lengths) / 1000.0, 3) if lengths else 0.0,
        "n_weakly_connected_components": len(weak),
        "largest_component_nodes": largest,
        "largest_component_fraction": (
            round(largest / graph.number_of_nodes(), 4)
            if graph.number_of_nodes() else 0.0
        ),
        "steps_edges": count_steps_edges(graph),
        "highway_counts": dict(sorted(highway_counts.items())),
    }
    if lengths:
        summary["length_min_m"] = round(min(lengths), 2)
        summary["length_median_m"] = round(statistics.median(lengths), 2)
        summary["length_mean_m"] = round(statistics.fmean(lengths), 2)
        summary["length_max_m"] = round(max(lengths), 2)
    if xs and ys:
        summary["bbox"] = {
            "west": min(xs), "east": max(xs), "south": min(ys), "north": max(ys),
        }
    return summary


def tag_coverage(
    graph: nx.MultiDiGraph, tags: Sequence[str] = TAG_COVERAGE_REPORT
) -> Dict[str, Dict[str, float]]:
    """Report how many edges actually carry each preserved tag.

    Whether Glebe is tagged well enough for a given accessibility attribute is
    an observation, not an assumption. This turns it into a number.

    Args:
        graph: the walking graph.
        tags: tags to report on.

    Returns:
        Dict of tag -> {"edges": count, "percent": share of all edges}.
    """
    total = graph.number_of_edges()
    coverage: Dict[str, Dict[str, float]] = {}
    for tag in tags:
        present = sum(
            1
            for _u, _v, _k, data in graph.edges(keys=True, data=True)
            if tag_values(data.get(tag))
        )
        coverage[tag] = {
            "edges": present,
            "percent": round(100.0 * present / total, 2) if total else 0.0,
        }
    return coverage


def geodesic_area_km2(geometry: BaseGeometry) -> Optional[float]:
    """Geodesic area of a WGS84 geometry in square kilometres."""
    try:
        from pyproj import Geod
    except ImportError:      # pragma: no cover - pyproj is a hard dependency
        return None
    area, _perimeter = Geod(ellps="WGS84").geometry_area_perimeter(geometry)
    return abs(area) / 1_000_000.0


def graph_provenance(
    polygon: BaseGeometry,
    routing_extent: BaseGeometry,
    buffer_metres: float,
    network_type: str,
    simplify: bool,
    truncate_by_edge: bool,
    retain_all: bool,
    metric_crs: str,
    graph: nx.MultiDiGraph,
    retrieval_date: str,
    tag_settings: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Assemble the record of how this graph was produced."""
    minx, miny, maxx, maxy = routing_extent.bounds
    return {
        "purpose": "Stage 3 pedestrian routing network for Glebe",
        "status": "buffer_metres 300 m ADOPTED for the Week 4 routing network "
                  "after the sensitivity run of 2026-08-29 (A-07). Still "
                  "provisional as a research choice: to be re-tested once "
                  "Stage 5 produces real origin-to-stop routes and once a "
                  "walking catchment is settled (A-04).",
        "candidate_stop_extent": "data/boundaries/glebe_suburb.geojson "
                                 "(official NSW Glebe suburb polygon, A-06)",
        "buffer_metres": buffer_metres,
        "buffer_crs": metric_crs,
        "buffer_note": "Buffer applied in a projected metric CRS, not in "
                       "WGS84 degrees. It is NOT a walking catchment; no "
                       "catchment has been adopted (A-04).",
        "routing_extent_bbox_wgs84": {
            "west": minx, "east": maxx, "south": miny, "north": maxy,
        },
        "routing_extent_area_km2": geodesic_area_km2(routing_extent),
        "network_type": network_type,
        "simplify": simplify,
        "truncate_by_edge": truncate_by_edge,
        "retain_all": retain_all,
        "osmnx_version": ox.__version__,
        "networkx_version": nx.__version__,
        "useful_tags_way": tag_settings.get("way", []),
        "useful_tags_node": tag_settings.get("node", []),
        "retrieval_date": retrieval_date,
        "cache_key": graph_cache_key(
            polygon, buffer_metres, network_type, simplify,
            truncate_by_edge, metric_crs, retain_all,
        ),
        "graph": describe_graph(graph),
        "tag_coverage": tag_coverage(graph),
    }


# --------------------------------------------------------------------------- #
# connectivity diagnostics - read only, never modifies the graph
# --------------------------------------------------------------------------- #
EDGE_TAGS_REPORTED: Tuple[str, ...] = (
    "highway", "footway", "name", "surface", "access", "foot", "crossing",
    "bridge", "tunnel", "service", "incline", "step_count", "wheelchair",
    "width", "oneway", "osmid",
)


def component_labels(
    graph: nx.MultiDiGraph,
) -> Tuple[Dict[Any, str], Dict[str, int]]:
    """Label every node's weakly connected component, deterministically.

    ``nx.weakly_connected_components`` yields components in an order that
    depends on node insertion, which is stable for one file but meaningless as
    an identifier. Components are therefore sorted by size descending, ties
    broken by their smallest node id as text, and labelled ``C0``, ``C1``, ...
    ``C0`` is always the largest. Two runs over the same graph always agree,
    and the label can be quoted in a report.

    Args:
        graph: the walking graph. It is not modified.

    Returns:
        Tuple of (node -> label, label -> node count).
    """
    components = list(nx.weakly_connected_components(graph))
    ordered = sorted(
        components, key=lambda c: (-len(c), min(str(node) for node in c))
    )
    node_to_label: Dict[Any, str] = {}
    sizes: Dict[str, int] = {}
    for index, component in enumerate(ordered):
        label = f"C{index}"
        sizes[label] = len(component)
        for node in component:
            node_to_label[node] = label
    return node_to_label, sizes


def node_report(graph: nx.MultiDiGraph, node: Any) -> Dict[str, Any]:
    """Describe one node: position, degree and whatever tags OSM gave it."""
    data = graph.nodes[node]
    reserved = {"x", "y"}
    return {
        "node": node,
        "latitude": data.get("y"),
        "longitude": data.get("x"),
        "degree": graph.degree(node),
        "in_degree": graph.in_degree(node),
        "out_degree": graph.out_degree(node),
        "tags": {k: v for k, v in data.items() if k not in reserved},
    }


def incident_edges_report(
    graph: nx.MultiDiGraph, node: Any, tags: Sequence[str] = EDGE_TAGS_REPORTED
) -> List[Dict[str, Any]]:
    """List every edge touching a node, with the tags needed to read it.

    Both directions are reported, because a stop reachable one way but not the
    other is exactly the kind of topology that produces unreachable pairs.
    """
    rows: List[Dict[str, Any]] = []
    for source, target, key, data in graph.out_edges(node, keys=True, data=True):
        row = {"direction": "out", "from": source, "to": target, "key": key}
        row.update({tag: data.get(tag) for tag in tags})
        rows.append(row)
    for source, target, key, data in graph.in_edges(node, keys=True, data=True):
        row = {"direction": "in", "from": source, "to": target, "key": key}
        row.update({tag: data.get(tag) for tag in tags})
        rows.append(row)
    return rows


def nearest_node_in_set(
    graph: nx.MultiDiGraph,
    node: Any,
    candidates: Iterable[Any],
    metric_crs: str = DEFAULT_METRIC_CRS,
) -> Tuple[Optional[Any], Optional[float]]:
    """Nearest node from ``candidates``, by STRAIGHT-LINE distance in metres.

    This is diagnostic evidence about how far apart two components sit on the
    ground. It is emphatically NOT a re-snap: nothing is reassigned, no result
    is changed, and the distance is a straight line, not a walkable route -
    there is by definition no route, which is why this is being measured.

    Args:
        graph: the walking graph. It is not modified.
        node: the node to measure from.
        candidates: node ids to measure to.
        metric_crs: projected CRS used for the measurement.

    Returns:
        Tuple of (nearest node, distance in metres), or (None, None) if
        ``candidates`` is empty.
    """
    to_metric = Transformer.from_crs(WGS84, metric_crs, always_xy=True).transform
    origin_x, origin_y = to_metric(graph.nodes[node]["x"], graph.nodes[node]["y"])

    best_node = None
    best_distance = None
    for candidate in candidates:
        data = graph.nodes[candidate]
        x, y = to_metric(data["x"], data["y"])
        distance = ((x - origin_x) ** 2 + (y - origin_y) ** 2) ** 0.5
        if best_distance is None or distance < best_distance:
            best_node, best_distance = candidate, distance
    return best_node, best_distance


def neighbourhood_edges(
    graph: nx.MultiDiGraph, node: Any, hops: int = 2
) -> List[Dict[str, Any]]:
    """Edges within ``hops`` steps of a node, for manual interpretation.

    Enough context to tell "an isolated 3-node footway stub" from "a real path
    that simply has no crossing to the other side of the road".
    """
    undirected = graph.to_undirected(as_view=True)
    reachable = {node}
    frontier = {node}
    for _ in range(hops):
        nxt = set()
        for current in frontier:
            nxt |= set(undirected.neighbors(current))
        frontier = nxt - reachable
        reachable |= nxt
        if not frontier:
            break

    seen = set()
    rows: List[Dict[str, Any]] = []
    for source in reachable:
        for _s, target, key, data in graph.out_edges(source, keys=True, data=True):
            if target not in reachable:
                continue
            signature = tuple(sorted((str(source), str(target)))) + (str(key),)
            if signature in seen:
                continue
            seen.add(signature)
            row = {"from": source, "to": target}
            row.update({tag: data.get(tag) for tag in EDGE_TAGS_REPORTED})
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# snapping preview - DIAGNOSTIC ONLY
# --------------------------------------------------------------------------- #
def preview_stop_snapping(
    graph: nx.MultiDiGraph,
    stops,
    metric_crs: str = DEFAULT_METRIC_CRS,
):
    """Report how far the candidate stops sit from the nearest graph node.

    DIAGNOSTIC ONLY. This exists so Stage 3 can be accepted or rejected on
    evidence. It returns its result in memory and writes nothing: it must not
    add a node id to data/cache/glebe_stops.csv and must not produce the
    Stage 4 snapping output. Stage 4 owns snapping as a pipeline step, decides
    how ties and unreachable stops are handled, and this preview does not
    pre-empt any of that.

    The graph is projected to ``metric_crs`` first so distances are metres in
    the same system as the buffer, rather than degrees or an implicit
    great-circle fallback.

    Args:
        graph: the walking graph, in WGS84.
        stops: candidate stops with stop_id, stop_lat, stop_lon.
        metric_crs: projected CRS used for the distance measurement.

    Returns:
        Tuple of (summary dict, per-stop DataFrame). Neither is written to disk.
    """
    import pandas as pd

    if graph.number_of_nodes() == 0:
        raise NetworkUnavailable("The walking graph has no nodes to snap to.")

    projected = ox.projection.project_graph(graph, to_crs=metric_crs)
    to_metric = Transformer.from_crs(WGS84, metric_crs, always_xy=True).transform

    latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
    xs, ys = [], []
    for lat, lon in zip(latitudes, longitudes):
        x, y = to_metric(lon, lat)
        xs.append(x)
        ys.append(y)

    nodes, distances = ox.distance.nearest_nodes(projected, xs, ys, return_dist=True)

    component_of, _sizes = component_labels(graph)

    preview = pd.DataFrame({
        "stop_id": list(stops["stop_id"]),
        "stop_name": list(stops.get("stop_name", [""] * len(stops))),
        "nearest_node": list(nodes),
        "snap_distance_m": [round(float(d), 2) for d in distances],
        "component": [component_of.get(n, "?") for n in nodes],
    })

    components_used = sorted(set(preview["component"]))
    summary = {
        "stops": len(preview),
        "snap_min_m": round(float(preview["snap_distance_m"].min()), 2),
        "snap_median_m": round(float(preview["snap_distance_m"].median()), 2),
        "snap_mean_m": round(float(preview["snap_distance_m"].mean()), 2),
        "snap_max_m": round(float(preview["snap_distance_m"].max()), 2),
        "distinct_components_used": len(components_used),
        "all_stops_in_one_component": len(components_used) == 1,
        "components_used": components_used,
    }
    return summary, preview
