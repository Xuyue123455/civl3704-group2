"""Week 12 STANDARD hybrid pedestrian routing (WALK Phase A, behind ``--hybrid-stairs``).

Reader's summary (cleanup pass 2)
---------------------------------
Responsibility  the Standard walking profile at runtime: the trusted graph PLUS the
                eligible modelled stair edges, searched with the same three objectives.
Inputs          the trusted ``WalkEngine`` context (6294 directed edges with production
                length / effort / time) and the offline stair cost cache
                ``data/cache/stair_model/stair_costs.csv`` + ``metadata.json`` (built once
                by ``scripts/build_stair_cost_cache.py``).
Outputs         ``StandardContext.solve()`` -> the same result shape as ``WalkEngine.solve``
                (three routes, each with its own distance / time / effort) plus stair
                usage per route and ``model_sensitive`` flags; ``as_phase2_context()`` /
                ``phase2_cost_table()`` so COMPARE and TRACK use the same basis.
Numbers         6294 trusted + 214 eligible (class A / P / B) = 6508 directed edges;
                class C (110) never added.
Depends on      ``walk_engine.py`` (the trusted context), ``compare_terrain_routes.
                lexicographic_dijkstra`` / ``time_optimal_route.time_search`` (unchanged).
Do not infer    a stair penalty or a new cost model here (stairs compete only through
                their modelled effort and time), that the DEM or Overpass data are read
                at runtime (never), or that Step-free means wheelchair accessible (it
                means mapped stairs are excluded).

A SECOND routing context beside the validated trusted one:

    WALKING PROFILE
        STANDARD  -> this module: trusted edges + eligible modelled stair edges
        STEP_FREE -> the existing trusted context (walk_engine / time_optimal_route), untouched

The Standard context is built from (a) every trusted directed edge with its PRODUCTION
length / effort / walking time (the values the trusted context already carries - nothing is
recomputed) and (b) the eligible (class A / P / B) stair-containing edges of the offline
stair cost cache ``data/cache/stair_model/stair_costs.csv`` (``scripts/build_stair_cost_cache.py``),
each with the same three attributes.  Class C stair edges are never added: no cost is
invented for them.  The trusted subgraph object is read, never modified.

Routing reuses the production machinery unchanged: ``compare_terrain_routes.
collapse_parallel_edges`` / ``lexicographic_dijkstra`` for the distance (length, effort) and
effort (effort, length) objectives and ``time_optimal_route.collapse_for_time`` /
``time_search`` for the time (walking_time_s, length) objective.  No combined score, no stair
penalty: stairs compete through their modelled effort and time only.  Every route reports
all three totals, summed over its own edges.

Sensitivity metadata: the feasibility study's low-cost/fast (effort x0.85, fast published
speeds) and high-cost/slow (effort x1.25, slow published speeds) scenarios are built as two
extra collapsed views; a route that uses stairs is ``model_sensitive`` when its edge sequence
differs under either scenario.  This is technical metadata only (Technical details), never
part of the route card.

The runtime never parses literature or Overpass data and never samples the DEM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from .walk_engine import OBJECTIVES, WalkEngine, phase2, tor
from .raw_shortest import MappedSteps

cmp = tor.cmp
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAIR_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "stair_model"
PROFILE_STANDARD = "standard"
PROFILE_STEP_FREE = "step_free"
PROFILES = (PROFILE_STANDARD, PROFILE_STEP_FREE)
ELIGIBLE_CLASSES = ("A", "P", "B")
UNAVAILABLE_MESSAGE = ("Stair cost cache not found - the Standard profile is off "
                       "(build it once with: python scripts/build_stair_cost_cache.py)")
ELIGIBILITY_TEXT = ("Standard walking may use mapped stair segments with sufficient model data. "
                    "Poorly resolved stair segments remain excluded.")
PROFILE_TEXT = {PROFILE_STANDARD: "Standard (mapped stairs with model data may be used)",
                PROFILE_STEP_FREE: "Step-free (mapped stairs excluded)"}
SCENARIOS = {"low_cost_fast": ("total_effort_low_j_per_kg", "total_time_low_s"),
             "high_cost_slow": ("total_effort_high_j_per_kg", "total_time_high_s")}
STATUS_NO_PATH = "no_standard_path"


class StandardContext:
    """Trusted edges + eligible modelled stair edges; the three collapsed views; per-origin searches."""

    def __init__(self, engine: WalkEngine, table: pd.DataFrame, metadata: Dict[str, Any],
                 mapped_steps: Optional[MappedSteps] = None) -> None:
        started = time.perf_counter()
        self.engine = engine
        self.metadata = metadata
        self.mapped_steps = mapped_steps
        eligible = table[table["eligible"].astype(bool)].copy()
        self.table = eligible
        self.excluded_classes = table.loc[~table["eligible"].astype(bool), "confidence_class"].value_counts().to_dict()
        graph = engine.graph
        trusted = engine.context["subgraph"]
        time_lookup = engine.context["time_lookup"]
        hybrid = nx.MultiDiGraph()
        hybrid.add_nodes_from(graph.nodes())
        for u, v, k, d in trusted.edges(keys=True, data=True):
            t = time_lookup.get((int(u), int(v), int(k)))
            if t is None:
                continue                                              # an untimed trusted edge (none today) cannot carry 3 metrics
            hybrid.add_edge(int(u), int(v), key=int(k), **{cmp.ATTR_LENGTH: float(d[cmp.ATTR_LENGTH]),
                                                            cmp.ATTR_EFFORT: float(d[cmp.ATTR_EFFORT]), tor.ATTR_TIME: float(t)},
                            stair=False)
        self.trusted_keys = {(int(u), int(v), int(k)) for u, v, k in trusted.edges(keys=True)}
        self.stair_keys: set = set()
        self.stair_rows: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.alt: Dict[str, nx.MultiDiGraph] = {name: nx.MultiDiGraph() for name in SCENARIOS}
        for g in self.alt.values():
            g.add_nodes_from(hybrid.nodes())
            g.add_edges_from(hybrid.edges(keys=True, data=True))
        for row in eligible.itertuples(index=False):
            u, v, k = int(row.u), int(row.v), int(row.key)
            if (u, v, k) in self.trusted_keys or not graph.has_edge(u, v, k):
                continue                                              # never override a trusted edge
            length = float(graph[u][v][k][cmp.ATTR_LENGTH])            # the graph's own length, counted once
            effort, seconds = float(row.total_effort_j_per_kg), float(row.total_time_s)
            if not (effort > 0 and seconds > 0):
                continue
            hybrid.add_edge(u, v, key=k, **{cmp.ATTR_LENGTH: length, cmp.ATTR_EFFORT: effort, tor.ATTR_TIME: seconds}, stair=True)
            for name, (ecol, tcol) in SCENARIOS.items():
                self.alt[name].add_edge(u, v, key=k, **{cmp.ATTR_LENGTH: length, cmp.ATTR_EFFORT: float(getattr(row, ecol)),
                                                        tor.ATTR_TIME: float(getattr(row, tcol))}, stair=True)
            self.stair_keys.add((u, v, k))
            self.stair_rows[(u, v, k)] = row._asdict()
        self.graph = hybrid
        self.collapsed = {obj: cmp.collapse_parallel_edges(hybrid, obj)[0] for obj in cmp.OBJECTIVES}
        self.collapsed_time, _ = tor.collapse_for_time(hybrid)
        self.alt_collapsed = {name: {"effort": cmp.collapse_parallel_edges(g, cmp.OBJECTIVE_EFFORT)[0],
                                     "time": tor.collapse_for_time(g)[0]} for name, g in self.alt.items()}
        self._searches: Dict[int, Dict[str, Any]] = {}
        self._alt_searches: Dict[int, Dict[str, Any]] = {}
        self.build_seconds = time.perf_counter() - started
        self.last_search_seconds: Optional[float] = None
        self.last_solve_seconds: Optional[float] = None

    # ----- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, engine: WalkEngine, cache_dir: Optional[Path] = None) -> Optional["StandardContext"]:
        directory = Path(cache_dir) if cache_dir else STAIR_CACHE_DIR
        csv, meta = directory / "stair_costs.csv", directory / "metadata.json"
        if not csv.exists() or not meta.exists():
            return None
        table = pd.read_csv(csv)
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        return cls(engine, table, metadata, MappedSteps.load())

    def summary(self) -> Dict[str, Any]:
        s = self.metadata.get("summary") or {}
        return {"trusted_edges": len(self.trusted_keys), "stair_edges_added": len(self.stair_keys),
                "hybrid_edges": self.graph.number_of_edges(), "eligible_classes": list(ELIGIBLE_CLASSES),
                "excluded_class_edges": self.excluded_classes, "model_version": self.metadata.get("model_version"),
                "eligible_segments": s.get("eligible_physical_segments"), "segments": s.get("physical_segments"),
                "eligible_stair_way_length_m": s.get("eligible_stair_way_length_m"), "total_stair_way_length_m": s.get("total_stair_way_length_m"),
                "effort_models": s.get("effort_models"), "build_seconds": round(self.build_seconds, 3)}

    # ----- the three searches (production machinery) --------------------------- #
    def search(self, origin_node: int) -> Dict[str, Any]:
        origin_node = int(origin_node)
        if origin_node not in self._searches:
            started = time.perf_counter()
            self._searches[origin_node] = {
                "origin_node": origin_node,
                "searches": {
                    tor.OBJECTIVE_DISTANCE: cmp.lexicographic_dijkstra(self.collapsed[tor.OBJECTIVE_DISTANCE], origin_node,
                                                                      cmp.ATTR_LENGTH, cmp.ATTR_EFFORT),
                    tor.OBJECTIVE_EFFORT: cmp.lexicographic_dijkstra(self.collapsed[tor.OBJECTIVE_EFFORT], origin_node,
                                                                    cmp.ATTR_EFFORT, cmp.ATTR_LENGTH),
                    tor.OBJECTIVE_TIME: tor.time_search(self.collapsed_time, origin_node),
                }}
            self.last_search_seconds = time.perf_counter() - started
        return self._searches[origin_node]

    def _alt_search(self, origin_node: int) -> Dict[str, Any]:
        origin_node = int(origin_node)
        if origin_node not in self._alt_searches:
            self._alt_searches[origin_node] = {
                name: {"effort": cmp.lexicographic_dijkstra(views["effort"], origin_node, cmp.ATTR_EFFORT, cmp.ATTR_LENGTH),
                       "time": tor.time_search(views["time"], origin_node)}
                for name, views in self.alt_collapsed.items()}
        return self._alt_searches[origin_node]

    def _route(self, collapsed: nx.DiGraph, labels, pred, origin: int, target: int, objective: str) -> Optional[Dict[str, Any]]:
        if target not in labels:
            return None
        nodes = cmp.reconstruct_nodes(pred, origin, target)
        edges = [(a, b, int(collapsed[a][b]["key"])) for a, b in zip(nodes[:-1], nodes[1:])]
        g = self.graph
        distance = sum(float(g[a][b][k][cmp.ATTR_LENGTH]) for a, b, k in edges)
        effort = sum(float(g[a][b][k][cmp.ATTR_EFFORT]) for a, b, k in edges)
        seconds = sum(float(g[a][b][k][tor.ATTR_TIME]) for a, b, k in edges)
        stairs = [e for e in edges if e in self.stair_keys]
        hops = [{"u": a, "v": b, "key": k, "length_m": float(g[a][b][k][cmp.ATTR_LENGTH]),
                 "terrain_effort_j_per_kg": float(g[a][b][k][cmp.ATTR_EFFORT]), "walking_time_s": float(g[a][b][k][tor.ATTR_TIME]),
                 "stair": (a, b, k) in self.stair_keys} for a, b, k in edges]
        return {"objective": objective, "nodes": nodes, "hops": hops, "edges": edges,
                "distance": distance, "effort": effort, "walking_time_s": seconds,
                "stair_edges": stairs, "stair_edge_count": len(stairs)}

    def solve(self, search: Dict[str, Any], stop_id: str) -> Dict[str, Any]:
        """All three Standard routes to one exact stop, in the WalkEngine.solve result shape (plus stair facts)."""
        started = time.perf_counter()
        row = self.engine.stop_row(stop_id)
        if row is None:
            raise KeyError(f"unknown stop_id {stop_id!r}")
        origin = int(search["origin_node"])
        target = int(row["nearest_node"])
        base = {"stop_id": str(stop_id), "stop_name": str(row["stop_name"]), "mode": str(row.get("mode", "")),
                "stop_node": target, "profile": PROFILE_STANDARD}
        routes: Dict[str, Any] = {}
        for obj in (tor.OBJECTIVE_DISTANCE, tor.OBJECTIVE_EFFORT):
            labels, pred = search["searches"][obj]
            route = self._route(self.collapsed[obj], labels, pred, origin, target, obj)
            if route is not None:
                routes[obj] = route
        labels, pred = search["searches"][tor.OBJECTIVE_TIME]
        route = self._route(self.collapsed_time, labels, pred, origin, target, tor.OBJECTIVE_TIME)
        if route is not None:
            routes[tor.OBJECTIVE_TIME] = route
        if len(routes) < 3:
            self.last_solve_seconds = time.perf_counter() - started
            return {**base, "available": False, "status": STATUS_NO_PATH, "reason": "no path on the Standard hybrid graph",
                    "routes": routes, "route_identity": None, "stair_usage": {}, "model_sensitive": {}}
        d, e, t = routes["distance"], routes["effort"], routes["time"]
        # optimality self-checks (the three objectives are independent searches on one graph)
        if e["effort"] > d["effort"] + 1e-6 or e["effort"] > t["effort"] + 1e-6:
            raise RuntimeError(f"Standard effort route is not minimal at stop {stop_id}")
        if t["walking_time_s"] > d["walking_time_s"] + 1e-6 or t["walking_time_s"] > e["walking_time_s"] + 1e-6:
            raise RuntimeError(f"Standard time route is not minimal at stop {stop_id}")
        if d["distance"] > e["distance"] + 1e-6 or d["distance"] > t["distance"] + 1e-6:
            raise RuntimeError(f"Standard distance route is not minimal at stop {stop_id}")
        usage = {obj: self._usage(routes[obj]) for obj in OBJECTIVES}
        sensitive = self._sensitivity(search, origin, target, routes) if any(routes[o]["stair_edge_count"] for o in OBJECTIVES) \
            else {obj: False for obj in OBJECTIVES}
        self.last_solve_seconds = time.perf_counter() - started
        return {
            **base, "available": True, "reason": "", "status": tor.STATUS_COMPARED, "routes": routes,
            "route_identity": phase2.route_identity(routes),
            "effort_saving_vs_shortest_pct": (100.0 * (d["effort"] - e["effort"]) / d["effort"]) if d["effort"] else None,
            "extra_distance_vs_shortest_pct": (100.0 * (e["distance"] - d["distance"]) / d["distance"]) if d["distance"] else None,
            "time_saving_vs_shortest_s": d["walking_time_s"] - t["walking_time_s"],
            "stair_usage": usage, "model_sensitive": sensitive,
            "uses_stairs": any(u["stair_edge_count"] for u in usage.values()),
        }

    def _usage(self, route: Dict[str, Any]) -> Dict[str, Any]:
        rows = [self.stair_rows[e] for e in route["stair_edges"]]
        names = sorted({str(r["name"]) for r in rows if r.get("name") not in (None, "") and str(r.get("name")) != "nan"})
        ways: Dict[Any, float] = {}
        for r in rows:
            n = r.get("step_count")
            if n is not None and n == n:
                ways[r["stair_way_id"]] = ways.get(r["stair_way_id"], 0.0) + float(n)
        return {"stair_edge_count": len(rows), "names": names,
                "classes": sorted({str(r["confidence_class"]) for r in rows}),
                "directions": sorted({str(r["direction"]) for r in rows}),
                "stair_length_m": round(sum(float(r["stair_length_m"]) for r in rows), 1),
                "step_count": (round(sum(ways.values())) if ways else None),
                "effort_models": sorted({str(r["effort_model"]) for r in rows})}

    def _sensitivity(self, search: Dict[str, Any], origin: int, target: int, routes: Dict[str, Any]) -> Dict[str, bool]:
        """True when the route's edge sequence changes under the published low / high stair-cost scenarios."""
        alt = self._alt_search(origin)
        out = {tor.OBJECTIVE_DISTANCE: False}                            # distance never depends on the stair model
        for obj, key in ((tor.OBJECTIVE_EFFORT, "effort"), (tor.OBJECTIVE_TIME, "time")):
            changed = False
            for name, views in self.alt_collapsed.items():
                labels, pred = alt[name][key]
                if target not in labels:
                    changed = True
                    continue
                nodes = cmp.reconstruct_nodes(pred, origin, target)
                view = views[key]
                edges = [(a, b, int(view[a][b]["key"])) for a, b in zip(nodes[:-1], nodes[1:])]
                if edges != routes[obj]["edges"]:
                    changed = True
            out[obj] = changed
        return out

    # ----- Phase B-1: the same context in the shape the validated Phase 2 engine reads ----- #
    def as_phase2_context(self) -> Dict[str, Any]:
        """The Standard hybrid graph as a ``time_optimal_route.build_context``-shaped dict.

        ``phase2_decision.run_phase2`` -> ``tor.solve_origin`` / ``search_all`` / ``solve_stop`` read a plain
        dict: ``graph`` (the raw graph, for snapping and the length check), ``subgraph`` (the routable edges
        with length + effort), ``collapsed`` (one DiGraph per objective), ``timed`` (the routable edges with
        walking time), ``collapsed_time``, ``trusted_keys`` / ``timed_keys`` (what ``verify_route`` /
        ``verify_time_route`` accept), ``time_lookup``, ``component_labels`` and ``counters``.  Here the
        routable set is the hybrid graph: every trusted edge with its production values plus the eligible
        stair edges, each carrying all three metrics - so no hop of any Standard Phase 2 route can lack an
        effort or a walking time.  The engine does not know which context it was given.  Built once, cached;
        the graph objects are the ones the WALK searches already use (nothing is rebuilt per Compare run).
        """
        if getattr(self, "_phase2_context", None) is None:
            trusted = self.engine.context
            hybrid_keys = set(self.trusted_keys) | set(self.stair_keys)
            self._phase2_context = {
                "graph": trusted["graph"], "graph_path": trusted["graph_path"],
                "subgraph": self.graph, "collapsed": self.collapsed,
                "timed": self.graph, "collapsed_time": self.collapsed_time,
                "trusted_keys": hybrid_keys, "timed_keys": hybrid_keys,
                "time_lookup": {(int(u), int(v), int(k)): float(d[tor.ATTR_TIME])
                                for u, v, k, d in self.graph.edges(keys=True, data=True)},
                "component_labels": trusted["component_labels"],
                "counters": {**trusted["counters"], "standard_hybrid_edges": self.graph.number_of_edges(),
                             "standard_stair_edges_added": len(self.stair_keys),
                             "standard_trusted_edges": len(self.trusted_keys)},
                "routing_basis": "standard_hybrid",
            }
        return self._phase2_context

    def phase2_cost_table(self) -> pd.DataFrame:
        """The production cost table with the 214 eligible stair rows carrying the hybrid Phase A costs.

        ``run_phase2`` -> ``phase1_table`` -> ``rank_candidate_stops.route_walking_times`` sums Tobler per
        edge from the COST TABLE (not from the context).  The production table lists the stair edges as
        ``special_structure`` rows with no time and no effort, so a hybrid route through them would get a
        missing Phase 1 time.  This view replaces exactly those rows (same ``u, v, key``, same columns, no
        column added) with the stair cost cache's whole-edge values: ordinary remainder + stair part, counted
        once, as validated in Phase A.  Every trusted row is byte-for-byte the production row; the production
        table object is never modified.  Built once, cached.
        """
        if getattr(self, "_phase2_cost_table", None) is None:
            base = self.engine.cost_table
            keys = list(zip(base["u"].astype(int), base["v"].astype(int), base["key"].astype(int)))
            position = {key: i for i, key in enumerate(keys)}
            kept = base.loc[[k not in self.stair_keys for k in keys]]
            template = {column: None for column in base.columns}
            rows = []
            for (u, v, k), row in self.stair_rows.items():
                record = dict(template)
                if (u, v, k) in position:
                    original = base.iloc[position[(u, v, k)]]
                    record.update({c: original[c] for c in ("edge_id", "name", "highway", "special_markers",
                                                             "week5c2a_classification", "component_id")
                                   if c in base.columns})
                length = float(self.graph[u][v][k][cmp.ATTR_LENGTH])
                seconds, effort = float(row["total_time_s"]), float(row["total_effort_j_per_kg"])
                record.update({"u": u, "v": v, "key": k, "length_m": length,
                               "terrain_cost_status": "standard_hybrid_stair_edge",
                               "cost_method": f"{self.metadata.get('model_version')} (ordinary remainder + stair part, once)",
                               "terrain_effort_j_per_kg": effort, "mean_cost_j_per_kg_per_m": effort / length if length else None,
                               tor.ATTR_TIME: seconds, "walking_time_min": seconds / 60.0,
                               "walking_time_method": f"{self.metadata.get('model_version')} stair time + Tobler remainder",
                               "mean_walking_speed_ms": length / seconds if seconds else None})
                rows.append(record)
            table = pd.concat([kept, pd.DataFrame(rows, columns=list(base.columns))], ignore_index=True)
            table["u"], table["v"], table["key"] = table["u"].astype(int), table["v"].astype(int), table["key"].astype(int)
            self._phase2_cost_table = table
        return self._phase2_cost_table

    def stair_segments(self, result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mapped-steps cache records (clipped geometry) for the stair edges used by ANY of the three routes."""
        if not result or not result.get("available") or self.mapped_steps is None:
            return []
        keys: List[Tuple[int, int, int]] = []
        for obj in OBJECTIVES:
            for e in result["routes"][obj]["stair_edges"]:
                if e not in keys:
                    keys.append(e)
        return [self.mapped_steps.by_edge[e] for e in keys if e in self.mapped_steps.by_edge]


# --------------------------------------------------------------------------- #
# wording (pure formatting)
# --------------------------------------------------------------------------- #
HUMAN = {"distance": "Shortest", "effort": "Gentlest", "time": "Fastest"}


def stair_usage_lines(result: Optional[Dict[str, Any]]) -> List[str]:
    if not result or not result.get("available") or not result.get("uses_stairs"):
        return []
    usage = result["stair_usage"]
    names: Dict[str, List[str]] = {}
    for obj in OBJECTIVES:
        u = usage[obj]
        if u["stair_edge_count"]:
            label = ", ".join(u["names"]) if u["names"] else "unnamed mapped steps"
            names.setdefault(label, []).append(HUMAN[obj])
    lines = []
    for label, objs in names.items():
        lines.append(f"Mapped steps on route: {label}")
        lines.append("used by " + ", ".join(objs))
    counts = {usage[o]["step_count"] for o in OBJECTIVES if usage[o]["stair_edge_count"] and usage[o]["step_count"]}
    if counts:
        lines.append(f"{max(counts)} mapped steps" if len(counts) == 1 else f"up to {max(counts)} mapped steps")
    return lines


def technical_lines(context: Optional["StandardContext"], result: Optional[Dict[str, Any]], profile: str) -> List[str]:
    lines = [f"WALKING PROFILE: {profile}  -  {PROFILE_TEXT.get(profile, profile)}",
             "STANDARD HYBRID (Week 12 Phase A, --hybrid-stairs; trusted context untouched)"]
    if context is None:
        lines.append("  " + UNAVAILABLE_MESSAGE)
        return lines
    s = context.summary()
    lines.append(f"  hybrid edges {s['hybrid_edges']} = trusted {s['trusted_edges']} + eligible stair edges {s['stair_edges_added']}  "
                 f"(classes {', '.join(s['eligible_classes'])}; excluded class C edges {s['excluded_class_edges'].get('C', 0)})")
    lines.append(f"  eligible stair segments {s['eligible_segments']} of {s['segments']}  "
                 f"({s['eligible_stair_way_length_m']} of {s['total_stair_way_length_m']} m of mapped stair run); "
                 f"effort models {s['effort_models']}; cache {s['model_version']}")
    lines.append("  " + ELIGIBILITY_TEXT)
    lines.append("  objectives: Shortest (length, effort)  Gentlest (effort, length)  Fastest (walking_time_s, length); "
                 "no combined score, no stair penalty")
    if result and result.get("available") and result.get("profile") == PROFILE_STANDARD:
        for obj in OBJECTIVES:
            r = result["routes"][obj]; u = result["stair_usage"][obj]
            sens = result["model_sensitive"].get(obj)
            lines.append(f"  {HUMAN[obj]:9s} {r['distance']:9.3f} m  {r['effort']:10.4f} J/kg  {r['walking_time_s']:9.3f} s  "
                         f"{len(r['edges']):4d} edges  stairs {u['stair_edge_count']}"
                         + (f" ({', '.join(u['names']) or 'unnamed'}; class {', '.join(u['classes'])}; {', '.join(u['directions'])}; "
                            f"{u['stair_length_m']} m stair run" + (f"; {u['step_count']} steps" if u['step_count'] else "") + ")"
                            if u["stair_edge_count"] else ""))
            if u["stair_edge_count"]:
                lines.append("    Stair-model sensitivity: " + ("route choice changed under the tested published cost/speed range"
                                                                 if sens else "stable"))
    if context.last_search_seconds is not None:
        lines.append(f"  timings: context build {context.build_seconds:.3f} s, last origin search {context.last_search_seconds:.4f} s, "
                     f"last solve {context.last_solve_seconds or 0:.4f} s")
    lines.append("  Step-free here means no stairs mapped as highway=steps in the current OSM network. "
                 "It does not establish wheelchair accessibility.")
    return lines
