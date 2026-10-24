"""Week 12 Phase B-1: the walking profile's routing basis for COMPARE and TRACK.

At checkpoint b871080 the STANDARD profile existed in WALK only: ``CompareEngine.run`` always handed
the validated Phase 2 engine the trusted (step-free) context, so entering Compare silently changed the
routing basis (O2 -> 203723: 404 m via Palmerston Avenue Steps in WALK, 623 m step-free in Compare).

This module makes the basis follow the profile WITHOUT touching the engines:

    walking profile     Phase 2 routing basis                      who builds it
    STANDARD            StandardContext.as_phase2_context()        standard_hybrid (hybrid graph: trusted
                        + StandardContext.phase2_cost_table()      edges + eligible A/P/B stair edges, all
                                                                   three metrics on every edge)
    STEP_FREE           WalkEngine.context + WalkEngine.cost_table the validated trusted context, unchanged
    (no --hybrid-stairs) the same trusted context                   -> CompareEngine.run, byte-identical path

``run_scene`` is ``CompareEngine.run`` with the context and cost table of the profile substituted: the
same feed-mode selection, the same snapshot (fetched once, never on a timer), the same
``phase2_decision.run_phase2`` call with the same margin, schedule fallback, stops and trip policy.  Nothing
about trips, catchability, slack or ranking is reimplemented here: the engine computes them from whatever
routable graph it is given, so for STANDARD the pedestrian arrival is ``observed_at`` + the hybrid Fastest
walking time + margin, and every other quantity follows.

``verify_scene_routes`` is the B-1 consistency check (tests, Technical details): every route of every
compared stop is re-summed over its own exact edge keys on the basis graph and must equal the displayed
distance / effort / walking time, and every edge must be a routable edge of that basis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .standard_hybrid import PROFILE_STANDARD, PROFILE_STEP_FREE, StandardContext
from .walk_engine import WalkEngine, phase2, tor

cmp = tor.cmp
BASIS_STANDARD = "standard_hybrid"
BASIS_TRUSTED = "trusted_step_free"
BASIS_TEXT = {BASIS_STANDARD: "Standard hybrid (trusted edges + eligible modelled stair edges)",
              BASIS_TRUSTED: "Step-free trusted (mapped stairs excluded)"}
PROFILE_LABEL = {PROFILE_STANDARD: "Walking profile: Standard", PROFILE_STEP_FREE: "Walking profile: Step-free"}
PROFILE_WORDS = {PROFILE_STANDARD: "Standard", PROFILE_STEP_FREE: "Step-free"}
# Phase B-2 product decision (supersedes the Phase B audit's reduced-mobility reading): Step-free is a route
# constraint - mapped stairs excluded - and both profiles report the same MODELLED pedestrian reference values.
REFERENCE_NOTE = ("Walking time and terrain effort are modelled pedestrian reference estimates (Tobler / Minetti; "
                  "stair model on Standard) and may differ from an individual user's actual walking time or effort.")
EXPORT_LIMITATION = "External scene export has not yet been made walking-profile-aware."
SUM_TOLERANCE = 1e-6


class Phase2View:
    """One routing basis in the two shapes the Phase 2 engine reads: a context dict and a cost table."""

    def __init__(self, basis: str, profile: str, context: Dict[str, Any], cost_table, label: str) -> None:
        self.basis = basis
        self.profile = profile
        self.context = context
        self.cost_table = cost_table
        self.label = label

    @property
    def routable_keys(self) -> set:
        return self.context["trusted_keys"]

    @property
    def edges(self) -> int:
        return self.context["subgraph"].number_of_edges()

    def describe(self) -> str:
        return f"{self.label}: {self.edges} routable edges"


def standard_view(standard: StandardContext) -> Phase2View:
    return Phase2View(BASIS_STANDARD, PROFILE_STANDARD, standard.as_phase2_context(), standard.phase2_cost_table(),
                      BASIS_TEXT[BASIS_STANDARD])


def trusted_view(engine: WalkEngine) -> Phase2View:
    return Phase2View(BASIS_TRUSTED, PROFILE_STEP_FREE, engine.context, engine.cost_table, BASIS_TEXT[BASIS_TRUSTED])


def view_for_profile(standard: Optional[StandardContext], profile: str) -> Optional[Phase2View]:
    """The basis the profile routes on; None = no hybrid context (flag off) -> the unchanged CompareEngine path."""
    if standard is None or profile != PROFILE_STANDARD:
        return None
    return standard_view(standard)


def run_scene(compare: Any, view: Phase2View, origin: Dict[str, Any], service_key: str,
              trip_id: Optional[str] = None, origin_label: str = "") -> Dict[str, Any]:
    """``CompareEngine.run`` on the profile's basis (same feed handling, same engine call, other context)."""
    compare.set_feed_mode(compare.mode_of_service(service_key))
    realtime = compare.snapshot()
    return phase2.run_phase2(
        view.context, compare.tables, origin["input_latitude"], origin["input_longitude"],
        service_key, realtime, view.cost_table, margin_s=compare.margin_s,
        schedule_fallback=compare.schedule_fallback, stops=compare.walk.stops,
        origin_label=origin_label, trip_id=trip_id)


def verify_scene_routes(scene: Dict[str, Any], view: Phase2View) -> Dict[str, Any]:
    """Re-sum every compared stop's routes over their exact edge keys on the basis graph.

    Returns counts and the list of discrepancies (empty when consistent).  A discrepancy is a route whose
    displayed distance / effort / walking time is not the sum of its own edges on this basis, or a route
    using an edge the basis does not route on.
    """
    graph = view.context["subgraph"]
    time_lookup = view.context["time_lookup"]
    keys = view.routable_keys
    problems: List[str] = []
    routes_checked = 0
    stair_routes = 0
    for stop in scene.get("stops") or []:
        for obj, route in (stop.get("routes") or {}).items():
            if route is None:
                continue
            routes_checked += 1
            edges = [tuple(int(x) for x in e) for e in route["edge_keys"]]
            bad = [e for e in edges if e not in keys]
            if bad:
                problems.append(f"{stop['stop_id']} {obj}: {len(bad)} edge(s) outside the basis, e.g. {bad[0]}")
                continue
            distance = sum(float(graph[u][v][k][cmp.ATTR_LENGTH]) for u, v, k in edges)
            effort = sum(float(graph[u][v][k][cmp.ATTR_EFFORT]) for u, v, k in edges)
            seconds = sum(float(time_lookup[e]) for e in edges)
            for name, shown, summed in (("distance_m", route["distance_m"], distance),
                                        ("effort_j_per_kg", route["effort_j_per_kg"], effort),
                                        ("walking_time_s", route["walking_time_s"], seconds)):
                if shown is None or abs(float(shown) - summed) > max(1e-3, SUM_TOLERANCE * abs(summed)):
                    problems.append(f"{stop['stop_id']} {obj}: {name} shown {shown} != edge sum {summed:.6f}")
            if view.basis == BASIS_STANDARD and any(graph[u][v][k].get("stair") for u, v, k in edges):
                stair_routes += 1
    return {"basis": view.basis, "routes_checked": routes_checked, "stair_routes": stair_routes,
            "consistent": not problems, "problems": problems}


def stair_edges_in_scene(scene: Dict[str, Any], standard: Optional[StandardContext]) -> Dict[str, List[str]]:
    """stop_id -> the objectives whose route uses an eligible stair edge (STANDARD basis only)."""
    if standard is None:
        return {}
    out: Dict[str, List[str]] = {}
    for stop in scene.get("stops") or []:
        for obj, route in (stop.get("routes") or {}).items():
            if route and any(tuple(int(x) for x in e) in standard.stair_keys for e in route["edge_keys"]):
                out.setdefault(stop["stop_id"], []).append(obj)
    return out


def compact_stair_lines(result: Optional[Dict[str, Any]]) -> List[str]:
    """The WALK card's mapped-steps summary in one line per stairway (Phase B-2 overflow fix): the step counts,
    classes, directions and effort models stay in Technical details (``standard_hybrid.technical_lines``)."""
    if not result or not result.get("available") or not result.get("uses_stairs"):
        return []
    usage = result["stair_usage"]
    human = {"distance": "Shortest", "effort": "Gentlest", "time": "Fastest"}
    by_label: Dict[str, List[str]] = {}
    for obj in ("distance", "effort", "time"):
        u = usage[obj]
        if u["stair_edge_count"]:
            label = ", ".join(u["names"]) if u["names"] else "unnamed mapped steps"
            by_label.setdefault(label, []).append(human[obj])
    lines = []
    for label, objs in by_label.items():
        who = "all three routes" if len(objs) == 3 else ", ".join(objs)
        lines.append(f"Mapped steps on route: {label} \u2014 used by {who}")
    return lines


# --------------------------------------------------------------------------- #
# Route-specific metric rows for the Compare objective-winner block (presentation; nothing re-routed)
# --------------------------------------------------------------------------- #
OBJECTIVE_ORDER = (("effort", "Gentlest"), ("time", "Fastest"), ("distance", "Shortest"))
WINNER_HEADING = "Objective winners among feasible stops"
WINNER_NOTE = "Each row: that route's own distance, walking time and effort."


def route_triplet(route: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One route -> ITS OWN three totals (scene ``route_summary`` fields) and its exact path."""
    if not route:
        return None
    return {"distance_m": route["distance_m"], "walking_time_s": route["walking_time_s"],
            "effort_j_per_kg": route["effort_j_per_kg"], "edge_keys": [list(e) for e in route["edge_keys"]],
            "path": tuple(tuple(int(x) for x in e) for e in route["edge_keys"]), "edges": route.get("edges")}


def stop_route_triplets(entry: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    """{"distance": Shortest route's triplet, "effort": Gentlest's, "time": Fastest's} for one candidate stop."""
    return {obj: route_triplet((entry.get("routes") or {}).get(obj)) for obj in ("distance", "effort", "time")}


def objective_rows(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The objective-winner rows: one row per (winning stop, exact path), each with THAT route's own metrics.

    The winners are the engine's (``summary.lowest_terrain_effort_stop`` / ``lowest_modelled_walking_time_stop`` /
    ``shortest_walk_stop``), unchanged.  What changes against the earlier ``walking_choice_blocks`` is the bundle:
    the Gentlest row carries the Gentlest route's distance, time and effort (not the Shortest route's distance and
    the Fastest route's time), and two objectives share one row only when they chose the same stop AND the same
    edge sequence (the scene's route identity) - a same-stop different-path pair gets two rows.
    """
    summary = scene.get("summary") or {}
    winners = {"effort": summary.get("lowest_terrain_effort_stop"), "time": summary.get("lowest_modelled_walking_time_stop"),
               "distance": summary.get("shortest_walk_stop")}
    entries = {s["stop_id"]: s for s in scene.get("stops") or []}
    rows: List[Dict[str, Any]] = []
    for obj, name in OBJECTIVE_ORDER:
        stop_id = winners.get(obj)
        entry = entries.get(stop_id) if stop_id is not None else None
        triplet = route_triplet((entry or {}).get("routes", {}).get(obj)) if entry else None
        if entry is None or triplet is None:
            continue
        same = next((r for r in rows if r["stop_id"] == stop_id and r["path"] == triplet["path"]), None)
        if same is not None:
            same["names"].append(name)
            same["objectives"].append(obj)
            continue
        catch = entry.get("catchability") or {}
        rows.append({"stop_id": stop_id, "stop_name": entry.get("stop_name", ""), "names": [name], "objectives": [obj],
                     "distance_m": triplet["distance_m"], "walking_time_s": triplet["walking_time_s"],
                     "effort_j_per_kg": triplet["effort_j_per_kg"], "edge_keys": triplet["edge_keys"], "path": triplet["path"],
                     "slack_s": catch.get("slack_s"), "status": catch.get("status"),
                     "highlights": list(entry.get("highlights") or []),
                     "same_stop_other_path": any(r["stop_id"] == stop_id for r in rows)})
    return rows


def verify_objective_rows(scene: Dict[str, Any], view: Phase2View) -> Dict[str, Any]:
    """Every objective row's three numbers re-summed over ITS OWN edges on the basis graph."""
    graph, lookup, keys = view.context["subgraph"], view.context["time_lookup"], view.routable_keys
    problems: List[str] = []
    rows = objective_rows(scene)
    for row in rows:
        edges = list(row["path"])
        if any(e not in keys for e in edges):
            problems.append(f"{row['stop_id']} {row['names']}: edge outside the basis")
            continue
        sums = {"distance_m": sum(float(graph[u][v][k][cmp.ATTR_LENGTH]) for u, v, k in edges),
                "effort_j_per_kg": sum(float(graph[u][v][k][cmp.ATTR_EFFORT]) for u, v, k in edges),
                "walking_time_s": sum(float(lookup[e]) for e in edges)}
        for name, summed in sums.items():
            if abs(float(row[name]) - summed) > max(1e-3, SUM_TOLERANCE * abs(summed)):
                problems.append(f"{row['stop_id']} {row['names']}: {name} shown {row[name]} != own-path sum {summed:.6f}")
    return {"rows": len(rows), "consistent": not problems, "problems": problems}


def route_triplet_lines(entry: Optional[Dict[str, Any]]) -> List[str]:
    """Technical details: the selected stop's three routes, each with its own totals and path identity."""
    if not entry:
        return []
    lines = ["  " + WINNER_NOTE + "  (objective-winner rows; two objectives share a row only on the same stop AND the same path)",
             "  selected stop routes (one route -> one internally consistent triplet):"]
    human = {"distance": "Shortest", "effort": "Gentlest", "time": "Fastest"}
    for obj in ("distance", "effort", "time"):
        t = route_triplet((entry.get("routes") or {}).get(obj))
        if t is None:
            lines.append(f"    {human[obj]:9s} no route")
            continue
        lines.append(f"    {human[obj]:9s} {t['distance_m']:9.3f} m  {t['walking_time_s']:9.3f} s  {t['effort_j_per_kg']:10.4f} J/kg  "
                     f"{len(t['path'])} edges")
    identity = entry.get("route_identity") or {}
    if identity:
        lines.append(f"    route identity: {identity.get('groups')} (distinct paths {identity.get('distinct_paths')})")
    return lines


def technical_lines(view: Optional[Phase2View], scene: Optional[Dict[str, Any]],
                    standard: Optional[StandardContext], profile: str) -> List[str]:
    """The COMPARE / TRACK routing-basis block of Technical details (hybrid flag on only)."""
    basis = view.basis if view is not None else BASIS_TRUSTED
    lines = ["", "ROUTING BASIS (Week 12 Phase B-1: Compare / Track follow the walking profile)",
             f"  walking profile {profile}  ->  {BASIS_TEXT[basis]}"]
    if view is not None:
        lines.append(f"  routable edges {view.edges}  (trusted {view.context['counters'].get('standard_trusted_edges')} + "
                     f"eligible stair edges {view.context['counters'].get('standard_stair_edges_added')})")
    else:
        lines.append("  routable edges: the trusted subgraph (unchanged validated path)")
    if scene is not None and view is not None:
        check = verify_scene_routes(scene, view)
        lines.append(f"  route consistency: {check['routes_checked']} routes re-summed on the basis, "
                     f"{'consistent' if check['consistent'] else 'INCONSISTENT: ' + '; '.join(check['problems'][:3])}; "
                     f"routes using eligible stairs {check['stair_routes']}")
        using = stair_edges_in_scene(scene, standard)
        if using:
            shown = ", ".join(f"{s} ({'/'.join(objs)})" for s, objs in list(using.items())[:6])
            lines.append(f"  stops whose Standard route uses mapped stairs: {len(using)}  e.g. {shown}")
    lines.append("  " + REFERENCE_NOTE)
    lines.append("  " + EXPORT_LIMITATION)
    return lines
