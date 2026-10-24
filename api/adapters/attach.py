"""Origin attachment for arbitrary taps (spatial correctness audit, STEP 7): a bounded nearest-reachable snap.

What the audit found
--------------------
The validated snapping (``WalkEngine.snap_origin`` -> Week 5 ``src.snapping.snap_point``) returns the single
nearest node of the raw pedestrian graph, with no radius and no connectivity check; the routing then runs on the
trusted graph (Step-free) or the Standard hybrid graph from that node.  For validated origins (O2 ...) the nearest
node is connected and everything is exact.  For an arbitrary phone tap, 13 % of random study-area points snap to a
node that cannot reach the target (often a node with no trusted edge at all, 4 m from the tap, while a connected
node sits 20 m away) and the product showed a stop with no routes.

What this module does
---------------------
It is a *graph attachment* step only - routing costs, objectives and every engine call are unchanged:

1. snap with the validated primitive (the nearest node);
2. if that node reaches the target on the profile's own routing graph (Standard: the hybrid graph, Step-free: the
   trusted graph; the distance view and the time view), keep it - the result is byte-identical to the desktop;
3. otherwise consider the ``CANDIDATES`` nearest raw-graph nodes within ``RADIUS_M`` (physical closeness first) and
   take the first one that reaches the target;
4. if none does, report a structured no-route reason (``origin_not_connected`` / ``no_reachable_snap_candidate`` /
   ``target_not_connected``) instead of a silent empty result.

When a later candidate is chosen, the origin record keeps the tap (``tap_latitude`` / ``tap_longitude``) and
carries the chosen node's own coordinates as ``input_latitude`` / ``input_longitude`` so every validated downstream
call that re-snaps (``tor.solve_origin`` inside ``phase2.run_phase2``) lands on the same node (0 m) - the WALK,
COMPARE and TRACK stages then agree on one origin node, as they do on the desktop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .engines import EngineHost

RADIUS_M = 60.0
CANDIDATES = 10
POLICY_NEAREST = "nearest_node"
POLICY_RESCUE = "nearest_reachable_candidate"
CODE_ORIGIN_NOT_CONNECTED = "origin_not_connected"
CODE_NO_REACHABLE_CANDIDATE = "no_reachable_snap_candidate"
CODE_TARGET_NOT_CONNECTED = "target_not_connected"
NO_ROUTE_MESSAGE = "No connected walking path from this location."


class NodeIndex:
    """Projected coordinates of every raw-graph node (EPSG:7856), for a bounded k-nearest query (diagnostic geometry only)."""

    def __init__(self, graph) -> None:
        import numpy as np
        from pyproj import Transformer
        self.nodes = np.fromiter((int(n) for n in graph.nodes), dtype="int64", count=graph.number_of_nodes())
        lons = np.fromiter((float(graph.nodes[int(n)]["x"]) for n in self.nodes), dtype="float64", count=len(self.nodes))
        lats = np.fromiter((float(graph.nodes[int(n)]["y"]) for n in self.nodes), dtype="float64", count=len(self.nodes))
        self.transform = Transformer.from_crs("EPSG:4326", "EPSG:7856", always_xy=True)
        xs, ys = self.transform.transform(lons, lats)
        self.xs, self.ys = np.asarray(xs), np.asarray(ys)

    def nearest(self, lat: float, lon: float, k: int, radius_m: float) -> List[Tuple[int, float]]:
        import numpy as np
        x, y = self.transform.transform(float(lon), float(lat))
        d = np.hypot(self.xs - x, self.ys - y)
        order = np.argsort(d, kind="stable")[: max(1, k)]
        return [(int(self.nodes[i]), float(d[i])) for i in order if d[i] <= radius_m]


def node_index(host: EngineHost) -> NodeIndex:
    index = getattr(host, "_node_index", None)
    if index is None:
        index = NodeIndex(host.engine.graph)
        setattr(host, "_node_index", index)
    return index


def profile_views(host: EngineHost, profile: str) -> Tuple[Any, Any, str]:
    """The routing graph views the profile actually searches: (distance view, time view, basis)."""
    standard = host.standard
    if profile == "standard" and standard is not None:
        return standard.collapsed["distance"], standard.collapsed_time, "standard_hybrid"
    return host.engine.context["collapsed"]["distance"], host.engine.context["collapsed_time"], "trusted"


def reaches(views: Tuple[Any, Any, str], node: int, target: int) -> bool:
    import networkx as nx
    distance, timed, _ = views
    node, target = int(node), int(target)
    return (node in distance and target in distance and nx.has_path(distance, node, target)
            and node in timed and target in timed and nx.has_path(timed, node, target))


def connected(views: Tuple[Any, Any, str], node: int) -> bool:
    distance, _, _ = views
    return int(node) in distance and distance.degree(int(node)) > 0


def attach_origin(host: EngineHost, lat: float, lon: float, stop_id: str, profile: str) -> Dict[str, Any]:
    """The origin record for the journey (``WalkEngine.snap_origin`` shape + attachment facts), or a no-route record."""
    engine = host.engine
    row = engine.stop_row(stop_id)
    if row is None:
        raise KeyError(f"unknown stop_id {stop_id!r}")
    target = int(row["nearest_node"])
    views = profile_views(host, profile)
    nearest = engine.snap_origin(lat, lon)                     # the validated primitive, unchanged
    nearest_node = int(nearest["nearest_node"])
    facts: Dict[str, Any] = {"policy": POLICY_NEAREST, "radius_m": RADIUS_M, "candidates_limit": CANDIDATES,
                             "nearest_node": nearest_node, "nearest_distance_m": round(float(nearest["snap_distance_m"]), 3),
                             "routing_basis": views[2], "target_node": target}
    if reaches(views, nearest_node, target):
        facts.update(nearest_reachable=True, chosen_rank=0, candidates_considered=1)
        return {**nearest, "attachment": facts}
    facts["nearest_reachable"] = False
    if not connected(views, target):
        facts.update(no_route=CODE_TARGET_NOT_CONNECTED, candidates_considered=0)
        return {**nearest, "attachment": facts}
    candidates = node_index(host).nearest(lat, lon, CANDIDATES, RADIUS_M)
    facts["candidates_considered"] = len(candidates)
    facts["candidates"] = [{"node": n, "distance_m": round(d, 2), "connected": connected(views, n), "reaches_target": reaches(views, n, target)}
                           for n, d in candidates]
    for rank, (node, distance) in enumerate(candidates):
        if node == nearest_node or not reaches(views, node, target):
            continue
        graph = engine.graph
        node_lat, node_lon = float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"])
        facts.update(policy=POLICY_RESCUE, chosen_rank=rank, chosen_distance_m=round(distance, 3))
        return {
            # downstream validated calls re-snap from input_latitude / input_longitude: the node's own coordinates
            # make them land on this node (0 m), so WALK / COMPARE / TRACK share one origin node
            "input_latitude": node_lat, "input_longitude": node_lon,
            "tap_latitude": float(lat), "tap_longitude": float(lon),
            "nearest_node": node, "snap_distance_m": float(distance),
            "node_lat": node_lat, "node_lon": node_lon,
            "role": nearest.get("role"), "snap_gap_note": nearest.get("snap_gap_note"),
            "attachment": facts,
        }
    any_connected = any(c["connected"] for c in facts["candidates"])
    facts["no_route"] = CODE_NO_REACHABLE_CANDIDATE if any_connected else CODE_ORIGIN_NOT_CONNECTED
    return {**nearest, "attachment": facts}


def no_route_block(origin: Dict[str, Any], fallback_reason: str) -> Optional[Dict[str, Any]]:
    """The structured reason the client shows (user wording) and keeps for Technical details (internal token)."""
    facts = origin.get("attachment") or {}
    code = facts.get("no_route")
    if code is None:
        return None
    detail = {
        CODE_TARGET_NOT_CONNECTED: "The stop's own graph node is not connected to the walking network under this profile.",
        CODE_NO_REACHABLE_CANDIDATE: f"None of the {facts.get('candidates_considered', 0)} graph nodes within {RADIUS_M:g} m of the tap "
                                     f"reaches the stop on the {facts.get('routing_basis')} graph.",
        CODE_ORIGIN_NOT_CONNECTED: f"No graph node within {RADIUS_M:g} m of the tap has a routable edge under this profile.",
    }[code]
    return {"code": code, "message": NO_ROUTE_MESSAGE, "detail": detail, "internal_reason": fallback_reason}


def stop_access(host: EngineHost, stop_id: str) -> Optional[Dict[str, Any]]:
    """The display-only connector from a stop's routing node to its GTFS coordinate (the marker).

    Routes end at the stop's nearest graph node (median 12.8 m from the GTFS point over the 37 stops, max 41 m).
    This connector lets the map join the line to the marker honestly; it is NOT a walking edge and carries no
    distance, time or effort - the metrics stay the engine's.
    """
    row = host.engine.stop_row(stop_id)
    if row is None:
        return None
    node = (float(row["node_lon"]), float(row["node_lat"]))
    stop = (float(row["stop_lon"]), float(row["stop_lat"]))
    return {"type": "LineString", "coordinates": [[node[0], node[1]], [stop[0], stop[1]]],
            "length_m": round(float(row["snap_distance_m"]), 2), "display_only": True,
            "meaning": "routing node -> GTFS stop coordinate; drawn to join the route to the marker; not part of distance, time or effort"}
