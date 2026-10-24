"""Week 11 STEP-FREE Phase A: the raw-distance shortest route and its mapped stairs.

A SEPARATE, additive layer beside the validated engine.  It never touches the
trusted subgraph, the cost table, Minetti, Tobler, the three validated solvers,
Compare or Track.  It answers one question for the WALK view:

    on the project's RAW OSM walking graph (every edge, mapped stairs included),
    what is the geometrically shortest path from the origin node to the stop node,
    and which mapped stairs does it use?

* Objective: DISTANCE ONLY - the edge attribute ``length`` in metres.  No Minetti
  effort, no Tobler time, no stair penalty, no assumed stair speed: none of those
  models is defined for a stairway in this project, so the raw route carries
  neither an effort nor a walking time.
* Search: NetworkX ``single_source_dijkstra`` on a plain DiGraph view of the raw
  MultiDiGraph (parallel edges collapsed to the shortest, the chosen key kept so
  the hop can be drawn from the stored OSM geometry).  No bespoke path search.
* Stairs: ``data/cache/mapped_steps.json`` built once by
  ``scripts/build_mapped_steps_cache.py`` (OSM ``highway=steps``, including the
  simplified edges whose ``highway`` is a list containing ``steps``; the stair
  sub-segment and one deterministic marker per edge come from the raw Overpass
  cache).  Without that file the layer reports itself unavailable and the WALK
  view behaves exactly as before.

The trusted Shortest from the validated engine IS the step-free alternative: no
step-free solver exists here.  ``presentation()`` only decides which of the two
the map shows, from the viewer's ``avoid_steps`` preference.

Wording (product principle): "step-free" means no stairs mapped as
``highway=steps`` in the current OSM network.  It does not establish wheelchair
accessibility, and nothing here claims it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPED_STEPS_CACHE = PROJECT_ROOT / "data" / "cache" / "mapped_steps.json"
ATTR_LENGTH = "length"
SHORTER_TOLERANCE_M = 0.5          # the raw route must beat the trusted Shortest by more than this

# step_free_status values (AppState.step_free_status)
STATUS_NOT_APPLICABLE = "not_applicable"       # no stair shortcut: the view is the validated one
STATUS_MAPPED_STEPS_FOUND = "mapped_steps_found"   # the raw route over mapped stairs is displayed
STATUS_STEP_FREE = "step_free"                 # avoid_steps: the trusted (step-free) Shortest is displayed
STATUS_NO_STEP_FREE_ROUTE = "no_step_free_route"   # avoid_steps, but no trusted route exists

UNAVAILABLE_MESSAGE = ("Mapped-steps cache not found - stair shortcuts are off "
                       "(build it once with: python scripts/build_mapped_steps_cache.py)")
NO_STEP_FREE_ROUTE_TEXT = "No step-free route was found in the current mapped walking network."
STEP_FREE_CAVEAT = ("Step-free here means no stairs mapped as highway=steps in the current OSM network. "
                    "It does not establish wheelchair accessibility.")
STAIR_SOURCE = "OSM highway=steps"


# --------------------------------------------------------------------------- #
# the mapped-steps cache
# --------------------------------------------------------------------------- #
class MappedSteps:
    """``data/cache/mapped_steps.json`` as a lookup by directed edge (u, v, key)."""

    def __init__(self, payload: Dict[str, Any], path: Optional[Path] = None) -> None:
        self.path = path
        self.payload = payload
        self.by_edge: Dict[Tuple[int, int, int], Dict[str, Any]] = {
            (int(r["u"]), int(r["v"]), int(r["key"])): r for r in payload.get("edges", [])}
        self.graph_sha256: Optional[str] = (payload.get("graph") or {}).get("sha256")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["MappedSteps"]:
        path = Path(path) if path else MAPPED_STEPS_CACHE
        if not path.exists():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")), path)

    def __len__(self) -> int:
        return len(self.by_edge)

    def summary(self) -> Dict[str, Any]:
        return dict(self.payload.get("summary") or {})

    def on_route(self, edges: List[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
        """The cache records of the route's edges that are mapped stairs, in route order."""
        return [self.by_edge[e] for e in edges if e in self.by_edge]


def stair_summary(step_edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One stairway = one raw OSM way: step_count and name are per way, never summed per edge."""
    ways: Dict[Any, Dict[str, Any]] = {}
    for rec in step_edges:
        ids = rec.get("raw_step_way_ids") or rec.get("osmid") or [(rec["u"], rec["v"], rec["key"])]
        way = tuple(ids)
        entry = ways.setdefault(way, {"way_ids": list(ids), "name": rec.get("name"), "step_count": rec.get("step_count"),
                                      "handrail": rec.get("handrail"), "ramp": rec.get("ramp"),
                                      "ramp_wheelchair": rec.get("ramp_wheelchair"), "incline": rec.get("incline"),
                                      "stair_length_m": 0.0, "edges": 0, "localisation": rec.get("stair_localisation")})
        entry["stair_length_m"] += float(rec.get("stair_length_m") or 0.0)
        entry["edges"] += 1
    stairways = list(ways.values())
    for s in stairways:
        s["stair_length_m"] = round(s["stair_length_m"], 1)
    names = [s["name"] for s in stairways if s.get("name")]
    counts = [int(s["step_count"]) for s in stairways if str(s.get("step_count") or "").isdigit()]
    return {
        "stairways": stairways, "stair_edges": len(step_edges),
        "stair_length_m": round(sum(float(r.get("stair_length_m") or 0.0) for r in step_edges), 1),
        "names": names,
        "step_count_total": (sum(counts) if counts and len(counts) == len(stairways) else None),
        "step_count_known_for": len(counts),
    }


# --------------------------------------------------------------------------- #
# the raw-distance route
# --------------------------------------------------------------------------- #
class RawShortestLayer:
    """Distance-only shortest paths on the raw walking graph, one search per origin."""

    def __init__(self, graph: nx.MultiDiGraph, mapped_steps: Optional[MappedSteps]) -> None:
        started = time.perf_counter()
        self.graph = graph
        self.mapped_steps = mapped_steps
        self.view = self._collapse(graph)
        self._searches: Dict[int, Tuple[Dict[int, float], Dict[int, List[int]]]] = {}
        self.build_seconds = time.perf_counter() - started
        self.last_search_seconds: Optional[float] = None
        self.last_route_seconds: Optional[float] = None

    @property
    def available(self) -> bool:
        return self.mapped_steps is not None

    @staticmethod
    def _collapse(graph: nx.MultiDiGraph) -> nx.DiGraph:
        """Parallel edges -> the shortest one (its key kept); self-loops never help a shortest path."""
        view = nx.DiGraph()
        view.add_nodes_from(graph.nodes)
        for u, v, key, data in graph.edges(keys=True, data=True):
            if u == v:
                continue
            length = float(data[ATTR_LENGTH])
            if view.has_edge(u, v) and view[u][v][ATTR_LENGTH] <= length:
                continue
            view.add_edge(u, v, **{ATTR_LENGTH: length, "key": int(key)})
        return view

    def search(self, origin_node: int) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
        origin_node = int(origin_node)
        if origin_node not in self._searches:
            started = time.perf_counter()
            # positional (G, source, target, cutoff, edge attribute): the edge attribute is ATTR_LENGTH, metres
            self._searches[origin_node] = nx.single_source_dijkstra(self.view, origin_node, None, None, ATTR_LENGTH)
            self.last_search_seconds = time.perf_counter() - started
        return self._searches[origin_node]

    def route(self, origin_node: int, stop_node: int) -> Dict[str, Any]:
        """The raw shortest route origin -> stop with its mapped stairs (plain, JSON-able)."""
        started = time.perf_counter()
        base = {"objective": "raw_distance", "origin_node": int(origin_node), "stop_node": int(stop_node),
                "weight": ATTR_LENGTH, "effort_j_per_kg": None, "walking_time_s": None}
        if not self.available:
            return {**base, "available": False, "status": "mapped_steps_cache_missing", "distance_m": None,
                    "nodes": [], "edges": [], "step_edges": [], "has_mapped_steps": False, "stairs": stair_summary([])}
        distances, paths = self.search(origin_node)
        if int(stop_node) not in distances:
            self.last_route_seconds = time.perf_counter() - started
            return {**base, "available": False, "status": "no_raw_path", "distance_m": None,
                    "nodes": [], "edges": [], "step_edges": [], "has_mapped_steps": False, "stairs": stair_summary([])}
        nodes = [int(n) for n in paths[int(stop_node)]]
        edges = [(nodes[i], nodes[i + 1], int(self.view[nodes[i]][nodes[i + 1]]["key"])) for i in range(len(nodes) - 1)]
        step_edges = self.mapped_steps.on_route(edges)
        self.last_route_seconds = time.perf_counter() - started
        return {**base, "available": True, "status": "raw_route", "distance_m": float(distances[int(stop_node)]),
                "nodes": nodes, "edges": edges, "step_edges": step_edges,
                "has_mapped_steps": bool(step_edges), "stairs": stair_summary(step_edges)}


# --------------------------------------------------------------------------- #
# which Shortest the map shows
# --------------------------------------------------------------------------- #
def trusted_shortest(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not result or not result.get("available"):
        return None
    return (result.get("routes") or {}).get("distance")


def stair_shortcut_exists(raw: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]],
                          tolerance_m: float = SHORTER_TOLERANCE_M) -> bool:
    """The raw route is valid, uses mapped stairs, and is genuinely shorter than the trusted Shortest
    (or the trusted engine has no route at all)."""
    if not raw or not raw.get("available") or not raw.get("has_mapped_steps"):
        return False
    trusted = trusted_shortest(result)
    if trusted is None:
        return True
    return float(raw["distance_m"]) < float(trusted["distance"]) - tolerance_m


def presentation(raw: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]], avoid_steps: bool,
                 enabled: bool = True) -> Dict[str, Any]:
    """The displayed Shortest: ``shortest`` is 'raw' | 'trusted' | None, with the step_free_status."""
    trusted = trusted_shortest(result)
    if not enabled or not stair_shortcut_exists(raw, result):
        return {"status": STATUS_NOT_APPLICABLE, "shortest": "trusted" if trusted is not None else None,
                "stair_shortcut": False, "detour_m": None}
    detour = (float(trusted["distance"]) - float(raw["distance_m"])) if trusted is not None else None
    if not avoid_steps:
        return {"status": STATUS_MAPPED_STEPS_FOUND, "shortest": "raw", "stair_shortcut": True, "detour_m": detour}
    if trusted is None:
        return {"status": STATUS_NO_STEP_FREE_ROUTE, "shortest": None, "stair_shortcut": True, "detour_m": None}
    return {"status": STATUS_STEP_FREE, "shortest": "trusted", "stair_shortcut": True, "detour_m": detour}


# --------------------------------------------------------------------------- #
# wording (pure formatting)
# --------------------------------------------------------------------------- #
def fmt_m(metres: Optional[float]) -> str:
    return "-" if metres is None else f"{metres:,.0f} m"


def stair_lines(raw: Dict[str, Any]) -> List[str]:
    """'Mapped steps: Palmerston Avenue Steps' / '36 mapped steps' (only what the tags say)."""
    stairs = raw.get("stairs") or {}
    names = stairs.get("names") or []
    lines = ["Mapped steps: " + ", ".join(names) if names else "Mapped steps on route (unnamed)"]
    if stairs.get("step_count_total") is not None:
        lines.append(f"{stairs['step_count_total']} mapped steps")
    elif stairs.get("stair_length_m"):
        lines.append(f"{stairs['stair_length_m']:.0f} m of mapped stairs")
    return lines


def panel_shortest_lines(view: Dict[str, Any], raw: Optional[Dict[str, Any]],
                         result: Optional[Dict[str, Any]]) -> List[str]:
    """The lines under the Shortest row for the three states of the stair shortcut."""
    status = view["status"]
    if status == STATUS_MAPPED_STEPS_FOUND:
        return stair_lines(raw) + ["Click the stairs marker to avoid mapped steps."]
    if status == STATUS_STEP_FREE:
        return [f"+{view['detour_m']:,.0f} m compared with the stair route", "No mapped steps on this route"]
    if status == STATUS_NO_STEP_FREE_ROUTE:
        return [NO_STEP_FREE_ROUTE_TEXT]
    return []


def technical_lines(raw: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]], view: Dict[str, Any],
                    layer: Optional[RawShortestLayer]) -> List[str]:
    """The STEP INFORMATION block of Technical details."""
    lines = ["STEP INFORMATION (Week 11 step-free Phase A; raw distance-only layer, trusted routes unchanged)"]
    if layer is None or not layer.available:
        lines.append("  " + UNAVAILABLE_MESSAGE)
        return lines
    if raw is None:
        lines.append("  no raw route computed")
        return lines
    lines.append(f"  status {view['status']}   displayed Shortest: {view.get('shortest') or 'none'}")
    if raw.get("available"):
        lines.append(f"  raw pedestrian shortest: {raw['distance_m']:.1f} m  ({len(raw['edges'])} edges, weight = edge length; "
                     "no Minetti effort, no Tobler time)")
    else:
        lines.append(f"  raw pedestrian shortest: none ({raw.get('status')})")
    for s in (raw.get("stairs") or {}).get("stairways") or []:
        lines.append(f"  mapped stairs: {s.get('name') or 'unnamed'}  {s['stair_length_m']:.1f} m stair segment  "
                     f"{s['edges']} edge(s)" + (f"  {s['step_count']} steps" if s.get("step_count") else "")
                     + (f"  handrail: {s['handrail']}" if s.get("handrail") else "")
                     + (f"  ramp: {s['ramp']}" if s.get("ramp") else "")
                     + f"  (OSM way {', '.join(map(str, s['way_ids']))}; {s.get('localisation')})")
    trusted = trusted_shortest(result)
    lines.append(f"  step-free alternative (trusted Shortest): {trusted['distance']:.1f} m" if trusted is not None
                 else f"  step-free alternative: none ({(result or {}).get('status') or 'no trusted route'})")
    if view.get("detour_m") is not None:
        lines.append(f"  avoiding mapped steps adds {view['detour_m']:.1f} m")
    lines.append(f"  source: {STAIR_SOURCE}  cache {getattr(layer.mapped_steps, 'path', None) and layer.mapped_steps.path.name}"
                 f"  ({len(layer.mapped_steps)} directed stair edges)")
    if layer.last_search_seconds is not None:
        lines.append(f"  timings: raw graph view {layer.build_seconds:.3f} s, search {layer.last_search_seconds:.4f} s, "
                     f"route {layer.last_route_seconds or 0:.4f} s")
    lines.append("  Caveat: " + STEP_FREE_CAVEAT)
    return lines
