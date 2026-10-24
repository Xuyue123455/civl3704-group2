"""Week 9 Phase 2 - the THIRD routing objective: minimum modelled walking time.

WHAT THIS IS
------------
Week 5C-2B solves two single-source searches on the trusted-cost subgraph:

    distance route   lexicographic minimum of (length, effort)
    effort route     lexicographic minimum of (effort, length)

Week 9 attached a Tobler walking time ``walking_time_s`` to every trusted
edge (``scripts/build_terrain_cost.py``, same DEM cells, same weights as the
Minetti integral) and measured it ALONG those two routes. Phase 2 needs the
minimum modelled walking time itself, so this module adds:

    time route       lexicographic minimum of (walking_time_s, length)

on exactly the same trusted subgraph, with the SAME searcher
(``compare_terrain_routes.lexicographic_dijkstra``), the same parallel-edge
collapse rule and the same route verification discipline. The distance and
effort searches are not touched: they are still run by the comparison
script's own functions, and ``solve_origin`` below reproduces their
Week 7 / Week 9 answers exactly (tested).

Tie-break: ``length`` is the secondary metric, as a tie-break only. It mirrors
the existing pair ((length, effort) / (effort, length)); effort would have
been an equally deterministic choice, and distance was taken because it is
the quantity the project has reported from Week 4 onwards. Parallel edges
between the same ordered pair collapse to the survivor minimising
``(walking_time_s, length, key)`` - the integer key is a last-resort
reproducibility tie-break, never a meaning.

SEMANTICS
---------
The time route gives the MINIMUM MODELLED Tobler walking time on the trusted
graph. It is a model quantity: Tobler (1993) is a hiking function, not a
measurement of this pedestrian, and the Minetti/Tobler terms both come from
the same 5 m-baseline DEM gradients. It is never called a measured or
predicted real arrival time. From it:

    earliest_modelled_pedestrian_arrival = observed_at + time_route_walking_time_s

which the Phase 2 engine uses as the pedestrian side of catchability.

Effort and time remain SEPARATE objectives, reported side by side. There is
no alpha*effort + beta*time score here or anywhere else.
"""

from __future__ import annotations

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

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The adopted routing primitives, imported and used as-is. There is no second
# Dijkstra in this file.
cmp = load_script("compare_terrain_routes")

ATTR_LENGTH = cmp.ATTR_LENGTH                      # "length"
ATTR_EFFORT = cmp.ATTR_EFFORT                      # "terrain_effort_j_per_kg"
ATTR_TIME = "walking_time_s"                       # Week 9 cost-table column

OBJECTIVE_DISTANCE = cmp.OBJECTIVE_DISTANCE
OBJECTIVE_EFFORT = cmp.OBJECTIVE_EFFORT
OBJECTIVE_TIME = "time"
OBJECTIVES: Tuple[str, ...] = (OBJECTIVE_DISTANCE, OBJECTIVE_EFFORT, OBJECTIVE_TIME)

TIME_PRIMARY, TIME_SECONDARY = ATTR_TIME, ATTR_LENGTH

SUM_TOLERANCE = 1e-6
DISTANCE_DECIMALS = 3
EFFORT_DECIMALS = 4
TIME_DECIMALS = 3

STATUS_COMPARED = cmp.STATUS_COMPARED
STATUS_NO_TRUSTED_PATH = cmp.STATUS_NO_TRUSTED_PATH
STATUS_RAW_GAP = cmp.STATUS_RAW_GAP
STATUS_NO_TIMED_PATH = "no_timed_path"            # reachable in effort, not in time
REASON_NO_TIMED_PATH = "no_path_on_the_timed_subgraph"

CACHE_DIR = CONFIG_ROOT / "data" / "cache"
STOPS_CSV = CACHE_DIR / "glebe_stops_snapped.csv"


class TimeRoutingError(RuntimeError):
    """A time route failed a verification that every published route passes."""


# --------------------------------------------------------------------------- #
# the timed subgraph and its collapsed view
# --------------------------------------------------------------------------- #
def build_timed_subgraph(
    subgraph: nx.MultiDiGraph, cost_table: pd.DataFrame,
) -> Tuple[nx.MultiDiGraph, Dict[str, int]]:
    """The trusted subgraph with ``walking_time_s`` on every edge.

    Edges are taken from the trusted subgraph the comparison script built
    (``build_trusted_subgraph``), never from the raw graph, so the time route
    can only use edges the effort and distance routes may also use. An edge
    whose cost-table time is missing or not strictly positive is left out and
    counted: by the Week 9 cost invariants a trusted edge always carries a
    time, so the count is expected to be zero and is reported, not assumed.
    """
    times: Dict[Tuple[int, int, int], float] = {}
    if ATTR_TIME in cost_table.columns:
        for row in cost_table[["u", "v", "key", ATTR_TIME]].itertuples(index=False):
            value = row[3]
            if value is None or not np.isfinite(float(value)):
                continue
            times[(int(row.u), int(row.v), int(row.key))] = float(value)
    timed = nx.MultiDiGraph()
    timed.graph.update(subgraph.graph)
    timed.add_nodes_from((node, dict(data)) for node, data in subgraph.nodes(data=True))
    counters = {"trusted_edges": 0, "timed_edges": 0, "untimed_trusted_edges": 0,
                "non_positive_time_edges": 0}
    for u, v, key, data in subgraph.edges(keys=True, data=True):
        counters["trusted_edges"] += 1
        seconds = times.get((int(u), int(v), int(key)))
        if seconds is None:
            counters["untimed_trusted_edges"] += 1
            continue
        if seconds <= 0:
            counters["non_positive_time_edges"] += 1
            continue
        timed.add_edge(int(u), int(v), key=int(key),
                       **{ATTR_LENGTH: float(data[ATTR_LENGTH]),
                          ATTR_EFFORT: float(data[ATTR_EFFORT]),
                          ATTR_TIME: seconds})
        counters["timed_edges"] += 1
    return timed, counters


def collapse_for_time(timed: nx.MultiDiGraph) -> Tuple[nx.DiGraph, Dict[str, int]]:
    """One edge per ordered pair: the survivor minimising (time, length, key).

    The same construction as ``compare_terrain_routes.collapse_parallel_edges``
    with the time pair substituted. Collapsing changes no answer (Dijkstra on
    the multigraph already takes the minimum over parallel edges); it lets the
    exact directed key of every hop be recorded.
    """
    collapsed = nx.DiGraph()
    collapsed.add_nodes_from(timed.nodes())
    counters = {"parallel_pairs": 0, "dual_metric_ties": 0, "self_loops": 0}
    for u, v in {(u, v) for u, v, _k in timed.edges(keys=True)}:
        bundle = timed[u][v]
        if u == v:
            counters["self_loops"] += len(bundle)
        if len(bundle) > 1:
            counters["parallel_pairs"] += 1
        ordered = sorted(((float(d[ATTR_TIME]), float(d[ATTR_LENGTH]), int(k))
                          for k, d in bundle.items()))
        t0, l0, k0 = ordered[0]
        equivalent = sum(1 for t, l, _k in ordered if t == t0 and l == l0)
        if equivalent > 1:
            counters["dual_metric_ties"] += 1
        data = timed[u][v][k0]
        collapsed.add_edge(u, v, key=k0,
                           **{ATTR_LENGTH: float(data[ATTR_LENGTH]),
                              ATTR_EFFORT: float(data[ATTR_EFFORT]),
                              ATTR_TIME: float(data[ATTR_TIME])},
                           parallel_candidates=len(bundle),
                           dual_metric_equivalent=bool(equivalent > 1))
    return collapsed, counters


# --------------------------------------------------------------------------- #
# search, reconstruction, verification
# --------------------------------------------------------------------------- #
def time_search(collapsed_time: nx.DiGraph, origin_node: int
                ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, int]]:
    """Single-source lexicographic (walking_time_s, length) search."""
    return cmp.lexicographic_dijkstra(collapsed_time, origin_node,
                                      TIME_PRIMARY, TIME_SECONDARY)


def route_hops_timed(collapsed_time: nx.DiGraph, nodes: Sequence[int]
                     ) -> List[Dict[str, Any]]:
    """The exact directed edges a node sequence uses, with all three metrics."""
    hops: List[Dict[str, Any]] = []
    for index, (u, v) in enumerate(zip(nodes[:-1], nodes[1:])):
        if not collapsed_time.has_edge(u, v):
            raise TimeRoutingError(
                f"The path uses a hop {u} -> {v} that is not an edge of the "
                "timed routing graph.")
        data = collapsed_time[u][v]
        hops.append({
            "hop_index": index, "u": int(u), "v": int(v), "key": int(data["key"]),
            "length_m": float(data[ATTR_LENGTH]),
            "terrain_effort_j_per_kg": float(data[ATTR_EFFORT]),
            "walking_time_s": float(data[ATTR_TIME]),
            "hop_resolution": (cmp.HOP_DUAL_METRIC_TIE
                               if data["dual_metric_equivalent"] else cmp.HOP_UNIQUE),
        })
    return hops


def solve_time_route(
    collapsed_time: nx.DiGraph, labels: Dict[int, Tuple[float, float]],
    predecessor: Dict[int, int], origin: int, target: int,
) -> Dict[str, Any]:
    """One time route: node sequence, exact hops, and all three totals."""
    nodes = cmp.reconstruct_nodes(predecessor, origin, target)
    hops = route_hops_timed(collapsed_time, nodes)
    seconds, distance = labels[target]
    return {
        "objective": OBJECTIVE_TIME, "nodes": nodes, "hops": hops,
        "distance": float(distance),
        "effort": float(sum(h["terrain_effort_j_per_kg"] for h in hops)),
        "walking_time_s": float(seconds),
        "edges": [(h["u"], h["v"], h["key"]) for h in hops],
        "dual_metric_tie_hops": sum(
            1 for h in hops if h["hop_resolution"] == cmp.HOP_DUAL_METRIC_TIE),
    }


def verify_time_route(route: Dict[str, Any], origin: int, target: int,
                      timed_keys: set, label: str = "time route") -> None:
    """Every check a published time route must pass.

    Endpoints, contiguity, membership of every hop in the timed trusted
    subgraph, and that the reported time, distance and effort are the sums of
    the hops (so nothing was averaged, interpolated or scaled).
    """
    nodes = route["nodes"]
    if not nodes or nodes[0] != origin or nodes[-1] != target:
        raise TimeRoutingError(f"{label}: endpoints are not origin/target.")
    hops = route["hops"]
    if len(hops) != len(nodes) - 1:
        raise TimeRoutingError(f"{label}: hop count does not match the node sequence.")
    for hop, (u, v) in zip(hops, zip(nodes[:-1], nodes[1:])):
        if (hop["u"], hop["v"]) != (u, v):
            raise TimeRoutingError(f"{label}: hop {hop} is not contiguous.")
        if (hop["u"], hop["v"], hop["key"]) not in timed_keys:
            raise TimeRoutingError(
                f"{label}: hop {(hop['u'], hop['v'], hop['key'])} is not a "
                "timed trusted edge.")
    for field, column in (("walking_time_s", "walking_time_s"),
                          ("distance", "length_m"),
                          ("effort", "terrain_effort_j_per_kg")):
        total = sum(h[column] for h in hops)
        if abs(total - route[field]) > SUM_TOLERANCE * max(1.0, abs(total)):
            raise TimeRoutingError(
                f"{label}: {field} {route[field]} is not the hop sum {total}.")


def route_time_from_table(edges: Sequence[Tuple[int, int, int]],
                          time_lookup: Dict[Tuple[int, int, int], float]
                          ) -> Tuple[Optional[float], int]:
    """Sum of edge times along a route; (None, n_missing) when any edge lacks one."""
    total = 0.0
    missing = 0
    for edge in edges:
        value = time_lookup.get(tuple(int(x) for x in edge))
        if value is None:
            missing += 1
        else:
            total += value
    return (None if missing else total), missing


# --------------------------------------------------------------------------- #
# the routing context and the three-objective solve for one origin
# --------------------------------------------------------------------------- #
def build_context(cost_table: pd.DataFrame, graph_path: Optional[Path] = None,
                  graph: Optional[nx.MultiDiGraph] = None) -> Dict[str, Any]:
    """Everything the three searches need, built once per session.

    The distance and effort graphs are the comparison script's own
    (``build_trusted_subgraph`` + ``collapse_parallel_edges``); the timed
    graph is derived from that same trusted subgraph.
    """
    from src.osm_network import component_labels, load_graph

    path = Path(graph_path) if graph_path else (CONFIG_ROOT / cmp.audit.DEFAULT_GRAPH)
    if graph is None:
        graph = load_graph(path)
    subgraph = cmp.build_trusted_subgraph(graph, cost_table)
    collapsed = {obj: cmp.collapse_parallel_edges(subgraph, obj)[0]
                 for obj in cmp.OBJECTIVES}
    timed, timed_counters = build_timed_subgraph(subgraph, cost_table)
    collapsed_time, collapse_counters = collapse_for_time(timed)
    labels, _sizes = component_labels(graph)
    time_lookup = {(int(u), int(v), int(k)): float(d[ATTR_TIME])
                   for u, v, k, d in timed.edges(keys=True, data=True)}
    return {
        "graph": graph, "graph_path": path, "subgraph": subgraph,
        "collapsed": collapsed, "timed": timed, "collapsed_time": collapsed_time,
        "trusted_keys": {(int(u), int(v), int(k)) for u, v, k in subgraph.edges(keys=True)},
        "timed_keys": {(int(u), int(v), int(k)) for u, v, k in timed.edges(keys=True)},
        "time_lookup": time_lookup,
        "component_labels": labels,
        "counters": {**timed_counters, **{f"time_collapse_{k}": v
                                         for k, v in collapse_counters.items()}},
    }


def load_stops(path: Optional[Path] = None) -> pd.DataFrame:
    stops = pd.read_csv(path or STOPS_CSV)
    stops["stop_id"] = stops["stop_id"].astype(str)
    return stops


def search_all(context: Dict[str, Any], origin_node: int) -> Dict[str, Any]:
    """The three single-source searches from one origin node."""
    searches = {
        OBJECTIVE_DISTANCE: cmp.lexicographic_dijkstra(
            context["collapsed"][OBJECTIVE_DISTANCE], origin_node, ATTR_LENGTH, ATTR_EFFORT),
        OBJECTIVE_EFFORT: cmp.lexicographic_dijkstra(
            context["collapsed"][OBJECTIVE_EFFORT], origin_node, ATTR_EFFORT, ATTR_LENGTH),
        OBJECTIVE_TIME: time_search(context["collapsed_time"], origin_node),
    }
    return {"origin_node": int(origin_node), "searches": searches}


def solve_stop(context: Dict[str, Any], search: Dict[str, Any], stop_node: int,
               label: str = "") -> Dict[str, Any]:
    """All three routes to one stop node from a completed origin search."""
    origin = search["origin_node"]
    routes: Dict[str, Dict[str, Any]] = {}
    for obj in (OBJECTIVE_DISTANCE, OBJECTIVE_EFFORT):
        best, pred = search["searches"][obj]
        if stop_node not in best:
            return {"status": STATUS_NO_TRUSTED_PATH,
                    "status_reason": cmp.REASON_NO_TRUSTED_PATH, "routes": {}}
        routes[obj] = cmp.solve_route(context["collapsed"][obj], best, pred, obj,
                                      origin, stop_node)
        cmp.verify_route(routes[obj]["hops"], routes[obj]["nodes"], origin, stop_node,
                         routes[obj]["distance"], routes[obj]["effort"],
                         context["graph"], context["trusted_keys"], f"{label} {obj}")
    cmp.verify_optimality(label, routes[OBJECTIVE_DISTANCE], routes[OBJECTIVE_EFFORT])
    best, pred = search["searches"][OBJECTIVE_TIME]
    if stop_node not in best:
        return {"status": STATUS_NO_TIMED_PATH, "status_reason": REASON_NO_TIMED_PATH,
                "routes": routes}
    routes[OBJECTIVE_TIME] = solve_time_route(context["collapsed_time"], best, pred,
                                              origin, stop_node)
    verify_time_route(routes[OBJECTIVE_TIME], origin, stop_node,
                      context["timed_keys"], f"{label} time")
    # Tobler time ALONG the distance and effort routes: a sum of the same edge
    # times, from the same table, so the three are comparable.
    for obj in (OBJECTIVE_DISTANCE, OBJECTIVE_EFFORT):
        seconds, missing = route_time_from_table(routes[obj]["edges"], context["time_lookup"])
        routes[obj]["walking_time_s"] = seconds
        routes[obj]["edges_missing_time"] = missing
    # The time route can be no slower than either other route, by optimality.
    for obj in (OBJECTIVE_DISTANCE, OBJECTIVE_EFFORT):
        other = routes[obj]["walking_time_s"]
        if other is not None and routes[OBJECTIVE_TIME]["walking_time_s"] > other + SUM_TOLERANCE:
            raise TimeRoutingError(
                f"{label}: the time route ({routes[OBJECTIVE_TIME]['walking_time_s']} s) "
                f"is slower than the {obj} route ({other} s).")
    return {"status": STATUS_COMPARED, "status_reason": "", "routes": routes}


def route_record(prefix: str, route: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if route is None:
        return {f"{prefix}_distance_m": np.nan, f"{prefix}_effort_j_per_kg": np.nan,
                f"{prefix}_walking_time_s": np.nan, f"{prefix}_walking_time_min": np.nan,
                f"{prefix}_edges": np.nan}
    seconds = route.get("walking_time_s")
    return {
        f"{prefix}_distance_m": round(route["distance"], DISTANCE_DECIMALS),
        f"{prefix}_effort_j_per_kg": round(route["effort"], EFFORT_DECIMALS),
        f"{prefix}_walking_time_s": (round(seconds, TIME_DECIMALS)
                                     if seconds is not None else np.nan),
        f"{prefix}_walking_time_min": (round(seconds / 60.0, 3)
                                       if seconds is not None else np.nan),
        f"{prefix}_edges": len(route["edges"]),
    }


def solve_origin(context: Dict[str, Any], latitude: float, longitude: float,
                 stops: Optional[pd.DataFrame] = None
                 ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Three routes to every candidate stop from a coordinate.

    Returns ``(table, routes_by_stop, provenance)``. ``table`` has one row per
    candidate stop with the Week 7-style columns for the distance and effort
    routes (``distance_route_*``, ``effort_route_*`` - identical to
    ``rank_candidate_stops.routes_for_origin``) plus ``time_route_*``.
    ``routes_by_stop`` keeps the full hop lists for drawing and export.
    """
    from src.snapping import snap_point

    graph = context["graph"]
    stops = load_stops() if stops is None else stops
    origin_node, snap_m = snap_point(graph, latitude, longitude)
    origin_component = context["component_labels"].get(origin_node)
    search = search_all(context, origin_node)

    rows: List[Dict[str, Any]] = []
    routes_by_stop: Dict[str, Dict[str, Any]] = {}
    for stop in stops.itertuples(index=False):
        node = int(stop.nearest_node)
        stop_id = str(stop.stop_id)
        record: Dict[str, Any] = {
            "stop_id": stop_id, "stop_name": stop.stop_name,
            "mode": getattr(stop, "mode", ""),
            "component_id": getattr(stop, "component_id", ""),
            "nearest_node": node,
            "stop_lat": float(stop.stop_lat), "stop_lon": float(stop.stop_lon),
        }
        if origin_component is not None and \
                getattr(stop, "component_id", None) not in (None, "", origin_component):
            record.update(comparison_status=STATUS_RAW_GAP,
                          status_reason=cmp.REASON_RAW_COMPONENT_GAP)
            solved = {"routes": {}}
        else:
            solved = solve_stop(context, search, node, f"stop {stop_id}")
            record.update(comparison_status=solved["status"],
                          status_reason=solved["status_reason"])
        routes = solved["routes"]
        d = routes.get(OBJECTIVE_DISTANCE)
        e = routes.get(OBJECTIVE_EFFORT)
        t = routes.get(OBJECTIVE_TIME)
        if d is not None and e is not None:
            record.update({
                "distance_route_distance_m": round(d["distance"], DISTANCE_DECIMALS),
                "distance_route_effort_j_per_kg": round(d["effort"], EFFORT_DECIMALS),
                "effort_route_distance_m": round(e["distance"], DISTANCE_DECIMALS),
                "effort_route_effort_j_per_kg": round(e["effort"], EFFORT_DECIMALS),
                "effort_saving_pct": round(100.0 * (d["effort"] - e["effort"]) / d["effort"], 4)
                if d["effort"] else np.nan,
                "distance_detour_pct": round(100.0 * (e["distance"] - d["distance"]) / d["distance"], 4)
                if d["distance"] else np.nan,
            })
        record.update(route_record("distance_route", d))
        record.update(route_record("effort_route", e))
        record.update(route_record("time_route", t))
        if t is not None and e is not None and e.get("walking_time_s") is not None:
            record["time_saving_vs_effort_route_s"] = round(
                e["walking_time_s"] - t["walking_time_s"], TIME_DECIMALS)
            record["effort_penalty_of_time_route_pct"] = round(
                100.0 * (t["effort"] - e["effort"]) / e["effort"], 4) if e["effort"] else np.nan
            record["time_route_same_as_effort_route"] = bool(t["edges"] == e["edges"])
            record["time_route_same_as_distance_route"] = bool(
                d is not None and t["edges"] == d["edges"])
        rows.append(record)
        routes_by_stop[stop_id] = routes
    provenance = {
        "origin": {"latitude": float(latitude), "longitude": float(longitude),
                   "nearest_node": int(origin_node),
                   "snap_distance_m": round(float(snap_m), 3),
                   "component_id": origin_component},
        "graph": str(context["graph_path"]),
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "trusted_subgraph_edges": context["subgraph"].number_of_edges(),
        "timed_subgraph_edges": context["timed"].number_of_edges(),
        "counters": context["counters"],
        "objectives": {
            OBJECTIVE_DISTANCE: "lexicographic (length, effort)",
            OBJECTIVE_EFFORT: "lexicographic (effort, length)",
            OBJECTIVE_TIME: "lexicographic (walking_time_s, length)",
        },
    }
    return pd.DataFrame(rows), routes_by_stop, provenance


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--origin", nargs=2, type=float, metavar=("LAT", "LON"),
                        required=True)
    parser.add_argument("--cost-dir", default=None)
    parser.add_argument("--graph", default=None)
    args = parser.parse_args(argv)
    cost_dir = Path(args.cost_dir) if args.cost_dir else cmp.cost_dir()
    cost_table = cmp.load_cost_table(cost_dir)
    context = build_context(cost_table, Path(args.graph) if args.graph else None)
    table, _routes, provenance = solve_origin(context, args.origin[0], args.origin[1])
    print(f"origin snapped to node {provenance['origin']['nearest_node']} "
          f"({provenance['origin']['snap_distance_m']} m); timed edges "
          f"{provenance['timed_subgraph_edges']} of {provenance['trusted_subgraph_edges']}")
    shown = table[table.comparison_status == STATUS_COMPARED].sort_values(
        "time_route_walking_time_s")
    cols = ["stop_id", "stop_name", "distance_route_distance_m",
            "effort_route_effort_j_per_kg", "effort_route_walking_time_min",
            "time_route_walking_time_min", "time_route_effort_j_per_kg",
            "time_route_same_as_effort_route"]
    print(shown[cols].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
