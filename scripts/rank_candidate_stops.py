"""Week 9 - candidate-stop comparison table with a Phase 1 effort ranking.

WHAT THIS IS
------------
One table, one row per candidate stop, answering the Phase 1 question of the
proposal - "which nearby transit stop requires the lowest terrain-aware
walking effort?" - from ONE origin, with the walking distance and the Tobler
walking time of the same route reported beside the effort, never folded into
it.

Two ways to obtain the routes:

* ``--from-week7`` (default) reads the Week 5C-2B comparison outputs exactly
  as they were written (``route_comparison.csv`` and ``route_edges.csv``) and
  only ADDS the Week 9 walking time by joining each route's exact directed
  edges to the rebuilt cost table. No route is recomputed, so every distance
  and effort in the result is the Week 7 number.

* ``--origin LAT LON --label NAME`` snaps a new origin and solves the same two
  lexicographic searches on the same trusted subgraph, by importing the
  routing primitives from ``compare_terrain_routes.py`` READ-ONLY. This is the
  path the representative-validation origins will use. It writes only under
  ``results/candidate_ranking/<label>/`` and never touches ``data/cache``.

Run with the A-02 origin, the second path reproduces the first exactly; the
script checks that and says so.

THE PHASE 1 RANKING RULE, STATED ONCE
-------------------------------------
Eligible stops (``comparison_status == compared``) are ordered by the effort
of their effort-optimal route, ascending. Ties on effort are broken by that
route's distance, ascending, then by stop_id, ascending - a deterministic
tie-break only, never a weighting. Stops without a trusted path, and the two
platforms behind the I-04 topology artefact, keep their status and get NO
rank: they are listed, not scored.

A NEAR-TIE IS REPORTED, NOT DECIDED
-----------------------------------
A ranked stop whose effort is within ``--near-tie-pct`` per cent of the
lowest effort in its group is placed in that group, so a group's whole span is
bounded by the threshold; the per-row gap to the previous rank is reported
separately. Members are "provisionally near-tied for reporting". The default
of 5 per cent is a PROVISIONAL REPORTING THRESHOLD and nothing more: it is NOT
derived from a measurement uncertainty, the 1.38 per cent same-stop rerouting
effect is NOT an uncertainty estimate, and the street-level 5 m vs 10 m
gradient comparison is NOT a route-level ranking uncertainty. A defensible
tolerance needs representative-origin and route-level sensitivity evidence -
see docs/week9_representative_origin_validation.md.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No combined score (no alpha*effort + beta*distance, no alpha*effort +
beta*time), no "best" label, no catchment rule (the candidate set is the
existing study-area set, which the proposal has not yet settled), no realtime
in the ranking (catchability is a Phase 2 question and is kept to a separate,
optional, offline helper here), no change to any routing weight, no change to
compare_terrain_routes.py or src/routing.py, no Minetti change.
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

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tobler = load_script("tobler_walking_time")

CACHE_DIR = CONFIG_ROOT / "data" / "cache"
COST_DIR = CACHE_DIR / "terrain_cost"
ROUTING_DIR = CACHE_DIR / "terrain_routing"
RESULTS_ROOT = CONFIG_ROOT / "results" / "candidate_ranking"

STATUS_COMPARED = "compared"
OBJECTIVE_DISTANCE = "distance"
OBJECTIVE_EFFORT = "effort"

# Reporting flag for near ties. 2.0 is the MEASURED route-level envelope from
# the Week 9 gradient-baseline sensitivity (3 m / 5 m / 10 m, routes fixed,
# five origins): the largest 5 m effort gap that reversed order under another
# baseline was 1.74 per cent. It covers that ONE source of variation only and
# it flags; it does not decide. See docs/week9_representative_origin_validation.md.
DEFAULT_NEAR_TIE_PCT = 2.0

# Candidate walking catchment - OPTIONAL and OFF by default (A-04 is open).
# When given, it ANNOTATES rows (within_catchment, effort_rank_in_catchment);
# it never removes a row, so the full-set Phase 1 rank is always reported too.
# The cited defaults are Transport for NSW (2013), Integrated Public Transport
# Service Planning Guidelines - Sydney Metropolitan Area, Table 10 p. 20:
# "Walking catchment is generally within 400 metres (as the crow flies) of the
# stop for bus services" and "within 800 metres (as the crow flies) of the
# station" for train / light rail. The standard is straight-line, so that is
# the default measure; "network" applies the same metres to the computed
# shortest-route distance and is stricter. This is NOT the 300 m routing
# buffer, which governs how much graph is loaded.
CATCHMENT_TFNSW_2013_M = {"bus": 400.0, "light_rail": 800.0}
CATCHMENT_SOURCE = ("Transport for NSW (2013). Integrated Public Transport Service "
                    "Planning Guidelines - Sydney Metropolitan Area, Table 10, p. 20 "
                    "(400 m bus / 800 m train & light rail, as the crow flies)")

TABLE_COLUMNS: Tuple[str, ...] = (
    "effort_rank", "stop_id", "stop_name", "mode", "eligibility",
    "status_reason",
    # the effort-optimal route: the Phase 1 quantity and its companions
    "effort_j_per_kg", "distance_m", "tobler_time_s", "tobler_time_min",
    # the distance-optimal route, for the side-by-side
    "shortest_route_distance_m", "shortest_route_effort_j_per_kg",
    "shortest_route_tobler_time_s", "shortest_route_tobler_time_min",
    "effort_saving_pct", "distance_detour_pct",
    # near-tie reporting
    "effort_gap_to_previous_pct", "near_tie_with_previous", "near_tie_group",
    # walking-time coverage of the route (every edge timed, or not)
    "effort_route_edges", "effort_route_edges_missing_time",
    "shortest_route_edges", "shortest_route_edges_missing_time",
    # optional catchment annotation (empty unless --catchment is given)
    "straight_line_m", "within_catchment", "effort_rank_in_catchment",
)


class RankingError(RuntimeError):
    """Raised with an actionable message when the table cannot be formed."""


# --------------------------------------------------------------------------- #
# walking time along a route = sum of its edges' walking times
# --------------------------------------------------------------------------- #
def route_walking_times(
    route_edges: pd.DataFrame, cost_table: pd.DataFrame
) -> pd.DataFrame:
    """Sum the per-edge Tobler time along every recorded route.

    Each route is the exact directed edge sequence the comparison wrote
    (``u, v, key`` per hop). The time is looked up per edge from the cost
    table and summed; an edge with no walking time (an untrusted edge, which
    cannot appear on a trusted route but is guarded against anyway) makes the
    route's time missing and is counted, never skipped silently.

    Returns:
        A frame indexed by (stop_id, metric) with ``walking_time_s``,
        ``edges_with_time`` and ``edges_missing_time``.
    """
    required = {"stop_id", "metric", "u", "v", "key"}
    missing = required - set(route_edges.columns)
    if missing:
        raise RankingError(f"route_edges lacks {sorted(missing)}")
    if "walking_time_s" not in cost_table.columns:
        raise RankingError(
            "the cost table has no walking_time_s column: rebuild it with the "
            "Week 9 scripts/build_terrain_cost.py first.")

    times = {
        (int(r.u), int(r.v), int(r.key)): r.walking_time_s
        for r in cost_table[["u", "v", "key", "walking_time_s"]].itertuples()
    }
    rows: List[Dict[str, Any]] = []
    for (stop_id, metric), hops in route_edges.groupby(["stop_id", "metric"],
                                                        sort=True):
        per_edge = [times.get((int(h.u), int(h.v), int(h.key)))
                    for h in hops.itertuples()]
        present = [t for t in per_edge if t is not None and pd.notna(t)]
        missing_count = len(per_edge) - len(present)
        rows.append({
            "stop_id": stop_id, "metric": metric,
            "walking_time_s": float(sum(present)) if missing_count == 0
            else np.nan,
            "edges_with_time": len(present),
            "edges_missing_time": missing_count,
        })
    frame = pd.DataFrame(rows, columns=["stop_id", "metric", "walking_time_s",
                                        "edges_with_time",
                                        "edges_missing_time"])
    return frame.set_index(["stop_id", "metric"])


# --------------------------------------------------------------------------- #
# the Phase 1 ranking and the near-tie report
# --------------------------------------------------------------------------- #
def rank_by_effort(table: pd.DataFrame) -> pd.DataFrame:
    """Assign ``effort_rank`` to eligible rows; everything else stays unranked.

    Ordering is (effort asc, distance asc, stop_id asc). A stable sort on a
    fully specified key, so two runs give the same ranks.
    """
    out = table.copy()
    out["effort_rank"] = np.nan
    eligible = out["eligibility"].eq(STATUS_COMPARED) & out["effort_j_per_kg"].notna()
    ordered = out[eligible].sort_values(
        ["effort_j_per_kg", "distance_m", "stop_id"], kind="mergesort")
    out.loc[ordered.index, "effort_rank"] = np.arange(1, len(ordered) + 1)
    return out


def flag_near_ties(table: pd.DataFrame, near_tie_pct: float) -> pd.DataFrame:
    """Report near ties without changing any rank.

    Two things are written, and they are deliberately different:

    * ``effort_gap_to_previous_pct`` / ``near_tie_with_previous`` - the gap to
      the row ranked immediately above, and whether it is below the threshold.
      This is a per-row statement about neighbours.

    * ``near_tie_group`` - a group is anchored on its lowest-effort member and
      extends only while a row's effort is within ``near_tie_pct`` of THAT
      anchor. So a group's whole span is bounded by the threshold. Members are
      "provisionally near-tied for reporting" - the threshold is a reporting
      convention, not an uncertainty estimate, so the group says nothing about
      which member is really lower. Chaining consecutive gaps instead would
      make the relation transitive: eleven stops each 2 per cent apart would
      become one group spanning 20 per cent.
    """
    out = table.copy()
    out["effort_gap_to_previous_pct"] = np.nan
    out["near_tie_with_previous"] = False
    out["near_tie_group"] = pd.array([pd.NA] * len(out), dtype="Int64")
    ranked = out[out["effort_rank"].notna()].sort_values("effort_rank")
    group = 0
    anchor_effort: Optional[float] = None
    anchor_index = None
    anchor_grouped = False
    previous_effort: Optional[float] = None
    for index, row in ranked.iterrows():
        effort = float(row["effort_j_per_kg"])
        if previous_effort is not None and previous_effort > 0:
            gap = 100.0 * (effort - previous_effort) / previous_effort
            out.at[index, "effort_gap_to_previous_pct"] = round(gap, 4)
            out.at[index, "near_tie_with_previous"] = bool(gap < near_tie_pct)
        within_anchor = (anchor_effort is not None and anchor_effort > 0 and
                         100.0 * (effort - anchor_effort) / anchor_effort < near_tie_pct)
        if within_anchor:
            if not anchor_grouped:
                group += 1
                out.at[anchor_index, "near_tie_group"] = group
                anchor_grouped = True
            out.at[index, "near_tie_group"] = group
        else:
            anchor_effort, anchor_index, anchor_grouped = effort, index, False
        previous_effort = effort
    return out


# --------------------------------------------------------------------------- #
# assembling the table from comparison rows and route times
# --------------------------------------------------------------------------- #
def build_experiment_table(
    comparison: pd.DataFrame, route_times: pd.DataFrame,
    near_tie_pct: float = DEFAULT_NEAR_TIE_PCT,
) -> pd.DataFrame:
    """One row per candidate stop, from the comparison and the route times.

    The comparison's distance and effort values pass through untouched; the
    only new numbers are the walking times, the ranks and the near-tie flags.
    """
    def lookup(stop_id, metric, column):
        try:
            return route_times.loc[(stop_id, metric), column]
        except KeyError:
            return np.nan

    rows: List[Dict[str, Any]] = []
    for row in comparison.itertuples(index=False):
        stop_id = row.stop_id
        effort_t = lookup(stop_id, OBJECTIVE_EFFORT, "walking_time_s")
        short_t = lookup(stop_id, OBJECTIVE_DISTANCE, "walking_time_s")
        rows.append({
            "stop_id": stop_id,
            "stop_name": getattr(row, "stop_name", ""),
            "mode": getattr(row, "mode", ""),
            "eligibility": row.comparison_status,
            "status_reason": getattr(row, "status_reason", "") or "",
            "effort_j_per_kg": getattr(row, "effort_route_effort_j_per_kg", np.nan),
            "distance_m": getattr(row, "effort_route_distance_m", np.nan),
            "tobler_time_s": effort_t,
            "tobler_time_min": tobler.seconds_to_minutes(effort_t)
            if pd.notna(effort_t) else np.nan,
            "shortest_route_distance_m": getattr(
                row, "distance_route_distance_m", np.nan),
            "shortest_route_effort_j_per_kg": getattr(
                row, "distance_route_effort_j_per_kg", np.nan),
            "shortest_route_tobler_time_s": short_t,
            "shortest_route_tobler_time_min": tobler.seconds_to_minutes(short_t)
            if pd.notna(short_t) else np.nan,
            "effort_saving_pct": getattr(row, "effort_saving_pct", np.nan),
            "distance_detour_pct": getattr(row, "distance_detour_pct", np.nan),
            "effort_route_edges": lookup(stop_id, OBJECTIVE_EFFORT,
                                         "edges_with_time"),
            "effort_route_edges_missing_time": lookup(
                stop_id, OBJECTIVE_EFFORT, "edges_missing_time"),
            "shortest_route_edges": lookup(stop_id, OBJECTIVE_DISTANCE,
                                           "edges_with_time"),
            "shortest_route_edges_missing_time": lookup(
                stop_id, OBJECTIVE_DISTANCE, "edges_missing_time"),
        })
    table = pd.DataFrame(rows)
    table = rank_by_effort(table)
    table = flag_near_ties(table, near_tie_pct)
    for column in TABLE_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    table = table[list(TABLE_COLUMNS)]
    return table.sort_values(
        ["effort_rank", "eligibility", "stop_id"], na_position="last",
        kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Phase 2 helper: a catchable service, kept separate and optional
# --------------------------------------------------------------------------- #
CATCHABLE_VIA_COMPUTED_ROUTE = "catchable_via_computed_route"
NOT_PROVEN_CATCHABLE = "not_proven_catchable_with_current_routes"


def classify_arrivals(
    observed_at_epoch_s: float, walking_time_s: float,
    arrivals_epoch_s: Sequence[float], margin_s: float = 0.0,
) -> Dict[str, Any]:
    """Label each arrival against the walking time of a COMPUTED route.

    The project has a distance-optimal and an effort-optimal route, and
    Tobler times along them. It has NO time-optimal route, so the walking time
    passed in is an UPPER BOUND on the unknown minimum walking time. That
    asymmetry is the whole logic here:

    * ``earliest_proven_arrival = observed_at + walking_time + margin`` is a
      time the pedestrian is PROVEN able to reach the stop by, along a route
      that has actually been computed.
    * A service expected at or after that moment is
      ``catchable_via_computed_route``.
    * A service expected BEFORE that moment is ``not_proven_catchable_with_
      current_routes``. It is NOT labelled missed or uncatchable: a faster,
      uncomputed route may exist. Deciding that it is missed would require a
      time-optimal route, which is a separate future objective.

    The margin defaults to ZERO because the project has not justified a
    value; a caller who passes one must document it. Arrivals are compared as
    given; nothing is predicted. This function knows nothing about
    destinations - see docs/week9_ranking_design.md B.4.
    """
    if walking_time_s is None or not np.isfinite(float(walking_time_s)):
        return {"earliest_proven_arrival_epoch_s": None,
                "first_catchable_via_computed_route_epoch_s": None,
                "not_proven_catchable": None, "labels": [],
                "reason": "no walking time along a computed route"}
    proven = float(observed_at_epoch_s) + float(walking_time_s) + float(margin_s)
    ordered = sorted(float(t) for t in arrivals_epoch_s if np.isfinite(float(t)))
    labels = [(t, CATCHABLE_VIA_COMPUTED_ROUTE if t >= proven
               else NOT_PROVEN_CATCHABLE) for t in ordered]
    catchable = [t for t, lab in labels if lab == CATCHABLE_VIA_COMPUTED_ROUTE]
    return {
        "earliest_proven_arrival_epoch_s": proven,
        "first_catchable_via_computed_route_epoch_s":
            catchable[0] if catchable else None,
        "wait_after_proven_arrival_s": (catchable[0] - proven) if catchable else None,
        "not_proven_catchable": sum(1 for _t, lab in labels
                                    if lab == NOT_PROVEN_CATCHABLE),
        "labels": labels,
        "reason": "" if catchable else
                  "no arrival at or after the proven arrival time in the snapshot",
    }


def first_catchable_service(*args, **kwargs) -> Dict[str, Any]:
    """Kept as a name; see :func:`classify_arrivals` for the semantics."""
    return classify_arrivals(*args, **kwargs)


# --------------------------------------------------------------------------- #
# path A: from the Week 7 comparison outputs (no re-routing)
# --------------------------------------------------------------------------- #
def load_cost_table(cost_dir: Path = COST_DIR) -> pd.DataFrame:
    path = cost_dir / "edge_terrain_cost.csv"
    if not path.exists():
        raise RankingError(f"cost table not found: {path}")
    table = pd.read_csv(path)
    if "walking_time_s" not in table.columns:
        raise RankingError(
            f"{path} predates Week 9 (no walking_time_s). Rebuild it with "
            "python scripts/build_terrain_cost.py")
    return table


def from_week7(cost_table: pd.DataFrame, routing_dir: Path = ROUTING_DIR,
               near_tie_pct: float = DEFAULT_NEAR_TIE_PCT) -> pd.DataFrame:
    comparison = pd.read_csv(routing_dir / "route_comparison.csv")
    route_edges = pd.read_csv(routing_dir / "route_edges.csv")
    times = route_walking_times(route_edges, cost_table)
    return build_experiment_table(comparison, times, near_tie_pct)


# --------------------------------------------------------------------------- #
# path B: a new origin, through the same routing primitives (read-only import)
# --------------------------------------------------------------------------- #
def routes_for_origin(
    latitude: float, longitude: float, cost_table: pd.DataFrame,
    graph_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Solve both objectives from a new origin on the trusted subgraph.

    Everything that decides a route - the trusted subgraph, the parallel-edge
    collapse, the lexicographic search, the hop reconstruction - is the
    comparison script's own function, imported and called. Nothing is
    re-implemented and nothing in that script is modified.
    """
    from src.osm_network import component_labels, load_graph
    from src.snapping import snap_point

    cmp = load_script("compare_terrain_routes")
    path = Path(graph_path) if graph_path else (CONFIG_ROOT / cmp.audit.DEFAULT_GRAPH)
    graph = load_graph(path)
    origin_node, snap_m = snap_point(graph, latitude, longitude)
    labels, _sizes = component_labels(graph)
    origin_component = labels.get(origin_node)

    stops = pd.read_csv(CACHE_DIR / "glebe_stops_snapped.csv")
    subgraph = cmp.build_trusted_subgraph(graph, cost_table)
    collapsed = {obj: cmp.collapse_parallel_edges(subgraph, obj)[0]
                 for obj in cmp.OBJECTIVES}
    searches = {
        cmp.OBJECTIVE_DISTANCE: cmp.lexicographic_dijkstra(
            collapsed[cmp.OBJECTIVE_DISTANCE], origin_node,
            cmp.ATTR_LENGTH, cmp.ATTR_EFFORT),
        cmp.OBJECTIVE_EFFORT: cmp.lexicographic_dijkstra(
            collapsed[cmp.OBJECTIVE_EFFORT], origin_node,
            cmp.ATTR_EFFORT, cmp.ATTR_LENGTH),
    }

    comparison_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    for stop in stops.itertuples(index=False):
        node = int(stop.nearest_node)
        record: Dict[str, Any] = {
            "stop_id": stop.stop_id, "stop_name": stop.stop_name,
            "mode": getattr(stop, "mode", ""),
            "component_id": getattr(stop, "component_id", ""),
            "nearest_node": node,
        }
        if origin_component is not None and \
                getattr(stop, "component_id", None) not in (None, "", origin_component):
            record.update(comparison_status=cmp.STATUS_RAW_GAP,
                          status_reason=cmp.REASON_RAW_COMPONENT_GAP)
            comparison_rows.append(record)
            continue
        solved = {}
        for obj in cmp.OBJECTIVES:
            best, pred = searches[obj]
            if node not in best:
                solved = None
                break
            solved[obj] = cmp.solve_route(collapsed[obj], best, pred, obj,
                                          origin_node, node)
        if not solved:
            record.update(comparison_status=cmp.STATUS_NO_TRUSTED_PATH,
                          status_reason=cmp.REASON_NO_TRUSTED_PATH)
            comparison_rows.append(record)
            continue
        d, e = solved[cmp.OBJECTIVE_DISTANCE], solved[cmp.OBJECTIVE_EFFORT]
        record.update({
            "comparison_status": cmp.STATUS_COMPARED, "status_reason": "",
            "distance_route_distance_m": round(d["distance"], 3),
            "distance_route_effort_j_per_kg": round(d["effort"], 4),
            "effort_route_distance_m": round(e["distance"], 3),
            "effort_route_effort_j_per_kg": round(e["effort"], 4),
            "effort_saving_pct": round(
                100.0 * (d["effort"] - e["effort"]) / d["effort"], 4)
            if d["effort"] else np.nan,
            "distance_detour_pct": round(
                100.0 * (e["distance"] - d["distance"]) / d["distance"], 4)
            if d["distance"] else np.nan,
        })
        comparison_rows.append(record)
        for obj, route in ((cmp.OBJECTIVE_DISTANCE, d), (cmp.OBJECTIVE_EFFORT, e)):
            for hop in route["hops"]:
                edge_rows.append({"stop_id": stop.stop_id, "metric": obj,
                                  **hop})
    provenance = {
        "origin": {"latitude": latitude, "longitude": longitude,
                   "nearest_node": int(origin_node),
                   "snap_distance_m": round(float(snap_m), 3),
                   "component_id": origin_component},
        "graph": str(path), "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "trusted_subgraph_edges": subgraph.number_of_edges(),
    }
    return (pd.DataFrame(comparison_rows), pd.DataFrame(edge_rows), provenance)


# --------------------------------------------------------------------------- #
# optional candidate catchment: annotate, never drop
# --------------------------------------------------------------------------- #
def annotate_catchment(
    table: pd.DataFrame, origin_lat: float, origin_lon: float,
    limits_m: Dict[str, float], measure: str = "straight_line",
) -> pd.DataFrame:
    """Add straight-line distance, a within-catchment flag and a rank among the
    stops inside the catchment. The full-set effort_rank is left untouched."""
    from pyproj import Geod
    stops = pd.read_csv(CACHE_DIR / "glebe_stops_snapped.csv")[["stop_id", "stop_lat", "stop_lon"]]
    out = table.merge(stops, on="stop_id", how="left")
    geod = Geod(ellps="WGS84")
    out["straight_line_m"] = [
        round(geod.inv(origin_lon, origin_lat, lon, lat)[2], 1)
        if pd.notna(lat) and pd.notna(lon) else np.nan
        for lat, lon in zip(out.stop_lat, out.stop_lon)]
    distance = out["straight_line_m"] if measure == "straight_line" \
        else out["shortest_route_distance_m"]
    limit = out["mode"].map(limits_m)
    out["within_catchment"] = (distance <= limit) & distance.notna() & limit.notna()
    out["effort_rank_in_catchment"] = np.nan
    inside = out[out.within_catchment & out.effort_rank.notna()].sort_values("effort_rank")
    out.loc[inside.index, "effort_rank_in_catchment"] = np.arange(1, len(inside) + 1)
    return out.drop(columns=["stop_lat", "stop_lon"])


# --------------------------------------------------------------------------- #
# summary and writing
# --------------------------------------------------------------------------- #
def summarise(table: pd.DataFrame, near_tie_pct: float) -> Dict[str, Any]:
    ranked = table[table["effort_rank"].notna()].sort_values("effort_rank")
    unavailable = table[table["effort_rank"].isna()]
    groups = []
    for gid, part in ranked[ranked["near_tie_group"].notna()].groupby(
            "near_tie_group"):
        groups.append({
            "group": int(gid),
            "ranks": [int(r) for r in part["effort_rank"]],
            "stops": [f"{int(s)} {n}" for s, n in zip(part["stop_id"],
                                                     part["stop_name"])],
            "effort_span_pct": round(
                100.0 * (part["effort_j_per_kg"].max()
                         - part["effort_j_per_kg"].min())
                / part["effort_j_per_kg"].min(), 3),
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranking_rule": "effort_j_per_kg asc, then distance_m asc, then stop_id asc",
        "near_tie_threshold_pct": near_tie_pct,
        "near_tie_threshold_status": ("reporting flag only; 2.0 = measured gradient-baseline "
                                      "envelope (largest reversed gap 1.74 %, 3/5/10 m, routes "
                                      "fixed, 5 origins); covers that one source of variation"),
        "candidates": int(len(table)),
        "ranked": int(len(ranked)),
        "unavailable": {
            status: int(n) for status, n in
            unavailable["eligibility"].value_counts().items()
        },
        "top_five": [
            {"rank": int(r.effort_rank), "stop_id": int(r.stop_id),
             "stop_name": r.stop_name,
             "effort_j_per_kg": float(r.effort_j_per_kg),
             "distance_m": float(r.distance_m),
             "tobler_time_min": round(float(r.tobler_time_min), 3)
             if pd.notna(r.tobler_time_min) else None}
            for r in ranked.head(5).itertuples()
        ],
        "near_tie_groups": groups,
        "routes_with_untimed_edges": int(
            (ranked["effort_route_edges_missing_time"].fillna(0) > 0).sum()),
        "tobler_time_min_range_over_ranked": [
            round(float(ranked["tobler_time_min"].min()), 3),
            round(float(ranked["tobler_time_min"].max()), 3)]
        if not ranked.empty else None,
    }


def write_outputs(table: pd.DataFrame, summary: Dict[str, Any],
                  out_dir: Path, provenance: Optional[Dict[str, Any]] = None,
                  comparison: Optional[pd.DataFrame] = None,
                  route_edges: Optional[pd.DataFrame] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "candidate_stop_ranking.csv", index=False)
    if comparison is not None:
        comparison.to_csv(out_dir / "route_comparison.csv", index=False)
    if route_edges is not None:
        route_edges.to_csv(out_dir / "route_edges.csv", index=False)
    payload = dict(summary)
    if provenance:
        payload["provenance"] = provenance
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2),
                                          encoding="utf-8")


def print_summary(table: pd.DataFrame, summary: Dict[str, Any]) -> None:
    print("")
    print(f"Candidates {summary['candidates']}  ranked {summary['ranked']}  "
          f"unavailable {summary['unavailable']}")
    print(f"Ranking rule: {summary['ranking_rule']}")
    print(f"Near-tie threshold: {summary['near_tie_threshold_pct']} % "
          f"({summary['near_tie_threshold_status']})")
    print("")
    show = table[table["effort_rank"].notna()][
        ["effort_rank", "stop_id", "stop_name", "effort_j_per_kg",
         "distance_m", "tobler_time_min", "effort_gap_to_previous_pct",
         "near_tie_group"]]
    with pd.option_context("display.width", 140, "display.max_rows", 60):
        print(show.to_string(index=False))
    if summary["near_tie_groups"]:
        print("")
        for g in summary["near_tie_groups"]:
            print(f"near-tie group {g['group']}: ranks {g['ranks']} span "
                  f"{g['effort_span_pct']} % - {'; '.join(g['stops'])}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--origin", nargs=2, type=float, metavar=("LAT", "LON"),
                        help="snap this origin and solve both routes afresh "
                             "(omit to use the Week 7 comparison outputs)")
    parser.add_argument("--label", default=None,
                        help="results subfolder name (required with --origin)")
    parser.add_argument("--near-tie-pct", type=float, default=DEFAULT_NEAR_TIE_PCT)
    parser.add_argument("--cost-dir", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--catchment", choices=["tfnsw2013"], default=None,
                        help="annotate rows with a candidate walking catchment "
                             "(bus 400 m / light rail 800 m, TfNSW 2013); off by default")
    parser.add_argument("--catchment-measure", choices=["straight_line", "network"],
                        default="straight_line",
                        help="straight_line = as the standard states; network = same "
                             "metres applied to the shortest computed route (stricter)")
    parser.add_argument("--check-reproduces-week7", action="store_true",
                        help="with --origin, assert the distances and efforts "
                             "equal the Week 7 comparison exactly")
    args = parser.parse_args(argv)

    cost_table = load_cost_table(Path(args.cost_dir) if args.cost_dir else COST_DIR)

    if args.origin is None:
        label = args.label or "week7_origin_A02"
        table = from_week7(cost_table, near_tie_pct=args.near_tie_pct)
        provenance = {"source": "data/cache/terrain_routing (Week 7 outputs, "
                                "routes not recomputed)",
                      "origin": json.loads(
                          (CACHE_DIR / "snapped_origin.json").read_text(
                              encoding="utf-8"))["origin"]}
        comparison = route_edges = None
    else:
        if not args.label:
            parser.error("--label is required with --origin")
        label = args.label
        lat, lon = args.origin
        comparison, route_edges, provenance = routes_for_origin(
            lat, lon, cost_table, Path(args.graph) if args.graph else None)
        times = route_walking_times(route_edges, cost_table)
        table = build_experiment_table(comparison, times, args.near_tie_pct)
        if args.check_reproduces_week7:
            week7 = pd.read_csv(ROUTING_DIR / "route_comparison.csv")
            merged = week7.merge(comparison, on="stop_id", suffixes=("_w7", "_new"))
            for col in ("distance_route_distance_m", "effort_route_distance_m",
                        "distance_route_effort_j_per_kg",
                        "effort_route_effort_j_per_kg"):
                a = merged[f"{col}_w7"].astype(float)
                b = merged[f"{col}_new"].astype(float)
                ok = ((a - b).abs() < 1e-3) | (a.isna() & b.isna())
                if not ok.all():
                    bad = merged.loc[~ok, "stop_id"].tolist()
                    raise RankingError(
                        f"{col} differs from the Week 7 comparison for "
                        f"stops {bad[:5]} - routing is NOT reproduced.")
            print("Reproduction check: every distance and effort equals the "
                  "Week 7 comparison to within 1 mm / 1e-3 J/kg.")

    if args.catchment:
        o_lat, o_lon = (args.origin if args.origin else
                        (provenance["origin"]["latitude"], provenance["origin"]["longitude"]))
        table = annotate_catchment(table, o_lat, o_lon, CATCHMENT_TFNSW_2013_M,
                                   args.catchment_measure)
    summary = summarise(table, args.near_tie_pct)
    if args.catchment:
        inside = table[table.within_catchment.fillna(False)]
        summary["catchment"] = {
            "rule": args.catchment, "measure": args.catchment_measure,
            "limits_m": CATCHMENT_TFNSW_2013_M, "source": CATCHMENT_SOURCE,
            "candidates_within": int(len(inside)),
            "note": "annotation only; effort_rank is still over the full compared set",
        }
    out_dir = RESULTS_ROOT / label
    write_outputs(table, summary, out_dir, provenance, comparison, route_edges)
    print_summary(table, summary)
    print("")
    print(f"written: {out_dir / 'candidate_stop_ranking.csv'}")
    print(f"         {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
