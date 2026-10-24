"""Week 5C-2A.1 - can an edge-scale gradient stand in for the 5 m method?

Why this exists
---------------
The real Week 5C-2A audit found that missing DEM is not the obstacle to a
terrain-aware Glebe network: DEM coverage is complete on all 6,832 directed
edges. The obstacle is graph resolution. 772 directed edges are physically too
short to support two eligible 5 m centred-gradient estimates. They are only
1.04 per cent of network length, but 31 of the 35 audited Week 4 distance
routes touch at least one of them - and across all 35 routes only 12 unique
short edges are responsible.

That is a model-feasibility problem, not a finding about the street.

The question, and what this does NOT do
---------------------------------------
    Can an evidence-based edge-scale gradient give a defensible terrain
    estimate for ordinary edges too short for the adopted 5 m method?

This script only gathers the evidence. It adopts nothing, writes no routing
weight, changes no Week 5C-2A classification, and leaves the ordinary method
exactly as it is: raw ~1 m DEM -> 5 m centred gradient -> Minetti Cw ->
J/kg integration. Normal edges keep that method regardless of what is found
here.

The two candidates, both diagnostic
-----------------------------------
A. ENDPOINT: (last valid elevation - first valid elevation) divided by the
   distance between those two samples, measured along the actual sampled
   geometry. Simple and physically interpretable; on a 2-5 m edge it rests on
   two DEM cells.
B. REGRESSION: the least-squares slope of elevation on distance over every
   valid raw sample of the edge. Uses all the evidence, but can flatten real
   local structure.

Neither is smoothed, interpolated, filled, clamped, extended into neighbouring
edges or penalised. Where an edge has only two valid samples the two methods
are arithmetically identical, and that is reported rather than hidden.

The validation that carries the weight
--------------------------------------
The 772 real short edges have no independent reference slope, so judging the
candidates on them would be circular. Instead the already-valid ORDINARY edges
provide the test population: short windows are cut from their raw profiles and
each candidate is compared against the eligible 5 m estimate covering the same
place on the same edge.

That 5 m estimate is NOT ground truth. It is the project's adopted baseline,
and this measures how far an edge-scale approximation departs from it.

Windows are cut by sample index, so a nominal 5-interval window can span
materially less than 5 m when it touches a profile's short final interval. A
window is kept in its requested length stratum only when its ACTUAL metric span
is within half the parent's typical sample spacing of that length; otherwise it
is dropped rather than mislabelled, and it is never stretched to fit.

No acceptable-error threshold is set here. The error distribution is reported
and the decision comes afterwards. The headline weighted figures are weighted
AVERAGES of the per-length stratum metrics, not pooled weighted medians or
percentiles - a distinction the summary carries explicitly, because averaging
stratum medians does not produce the median of a reweighted population.

Usage
-----
    python scripts/validate_short_edge_gradient.py
    python scripts/validate_short_edge_gradient.py --validation-edges 0
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
from src.osm_network import load_graph  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Everything upstream comes through the Week 5C-2A audit module, so Week 5A
# sampling, the Week 5B gradient, the Week 5C-1 Minetti polynomial and the
# baseline-eligibility rule all have exactly one implementation.
audit = load_script("audit_terrain_readiness")
week5a = audit.week5a
week5b = audit.week5b
week5c1 = audit.week5c1


AUDIT_SUBDIR = audit.OUTPUT_SUBDIR                    # terrain_readiness
OUTPUT_SUBDIR = f"{AUDIT_SUBDIR}/short_edge_validation"

TARGET_CLASS = audit.CLASS_TOO_SHORT

# Window lengths, in metres, cut from the validation profiles. The four fixed
# bins the research question names, plus every whole-metre span present in the
# REAL short-edge population so the summary can be weighted by it.
FIXED_WINDOW_LENGTHS_M: Tuple[int, ...] = (2, 3, 4, 5)

# A parent profile must be long enough to hold a 5 m window and still have
# eligible 5 m estimates on either side of it.
MIN_PARENT_LENGTH_M = 20.0

# How many windows to cut per parent edge per length. Spread evenly along the
# edge rather than clustered at its start.
DEFAULT_WINDOWS_PER_LENGTH = 6

# How many ordinary edges to draw for the validation population. 0 means all
# eligible ones.
DEFAULT_VALIDATION_EDGES = 800

# Fixed so the experiment is reproducible. Nothing about the draw depends on
# routes, stops or gradients.
VALIDATION_SEED = 20260904

# A window's reference is the eligible 5 m estimate whose centre is nearest to
# the window's centre, and only when it is within half a sample interval - the
# same quantisation tolerance the Week 5C-2A audit uses for baselines.
REFERENCE_MATCH_FACTOR = 0.5

# Sign agreement on a near-level surface is dominated by noise, so it is
# reported twice: over everything, and over references steeper than this.
SIGN_AGREEMENT_MIN_GRADIENT_PCT = 1.0

METHODS: Tuple[str, ...] = ("endpoint", "regression")

LEVEL_COST = week5c1.LEVEL_COST_J_PER_KG_PER_M


class ShortEdgeError(RuntimeError):
    """Raised with an actionable message when the diagnostic cannot be trusted."""


# --------------------------------------------------------------------------- #
# the two candidate estimators
# --------------------------------------------------------------------------- #
def valid_samples(profile: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Distances and elevations of the samples that actually have a value.

    Missing samples are DROPPED, never replaced. Nothing is interpolated,
    smoothed or carried forward, and the caller's frame is not modified.
    """
    numeric = audit.numeric_elevations(profile)
    distances = numeric["distance_m"].to_numpy(dtype=float)
    elevations = numeric["elevation_m_ahd"].to_numpy(dtype=float)
    keep = np.isfinite(elevations) & np.isfinite(distances)
    return distances[keep], elevations[keep]


def endpoint_gradient(profile: pd.DataFrame) -> Dict[str, Any]:
    """Candidate A: rise between the first and last valid samples over the run.

    The run is the distance between those two samples along the sampled
    geometry - not the OSM ``length`` attribute, which is computed differently
    and would put a rise measured on one line over a run measured on another.

    Returns:
        Dict with the gradient, the run and rise it used, the sample count,
        and None for the gradient when fewer than two samples have a value.
    """
    distances, elevations = valid_samples(profile)
    result: Dict[str, Any] = {
        "endpoint_gradient": None,
        "endpoint_run_m": None,
        "endpoint_rise_m": None,
        "endpoint_valid_samples": int(distances.size),
    }
    if distances.size < 2:
        return result
    run = float(distances[-1] - distances[0])
    if run <= 0:
        return result
    rise = float(elevations[-1] - elevations[0])
    result.update({
        "endpoint_gradient": rise / run,
        "endpoint_run_m": round(run, 4),
        "endpoint_rise_m": round(rise, 4),
    })
    return result


def regression_gradient(profile: pd.DataFrame) -> Dict[str, Any]:
    """Candidate B: least-squares slope of elevation on distance.

    Fitted on every valid raw sample of this edge and nothing else. With
    exactly two valid samples the fit passes through both and the slope is
    arithmetically identical to the endpoint gradient; that is reported through
    ``regression_equals_endpoint`` rather than presented as a second opinion.

    Returns:
        Dict with the slope, the intercept, the sample count, R-squared and the
        residual standard deviation - the last two so a fit that hides real
        structure is visible as a poor fit rather than as a confident number.
    """
    distances, elevations = valid_samples(profile)
    result: Dict[str, Any] = {
        "regression_gradient": None,
        "regression_intercept_m": None,
        "regression_valid_samples": int(distances.size),
        "regression_r_squared": None,
        "regression_residual_std_m": None,
        "regression_equals_endpoint": None,
    }
    if distances.size < 2:
        return result
    spread = distances - distances.mean()
    denominator = float(np.sum(spread * spread))
    if denominator <= 0:
        return result
    slope = float(np.sum(spread * (elevations - elevations.mean())) / denominator)
    intercept = float(elevations.mean() - slope * distances.mean())
    fitted = intercept + slope * distances
    residuals = elevations - fitted
    total = float(np.sum((elevations - elevations.mean()) ** 2))
    result.update({
        "regression_gradient": slope,
        "regression_intercept_m": round(intercept, 4),
        "regression_r_squared": (
            round(1.0 - float(np.sum(residuals ** 2)) / total, 6)
            if total > 0 else None),
        "regression_residual_std_m": round(float(residuals.std(ddof=0)), 5),
        "regression_equals_endpoint": bool(distances.size == 2),
    })
    return result


def candidate_gradients(profile: pd.DataFrame) -> Dict[str, Any]:
    """Both candidates for one profile. Takes a profile and nothing else.

    Deliberately unaware of stop ids, route usage and how often an edge is
    walked. How common an edge is cannot be allowed to change the terrain
    estimated on it.
    """
    combined = dict(endpoint_gradient(profile))
    combined.update(regression_gradient(profile))
    return combined


def minetti_diagnostics(gradient: Optional[float], length_m: Optional[float],
                        prefix: str) -> Dict[str, Any]:
    """HYPOTHETICAL Minetti values for one candidate gradient.

    Every key is prefixed ``hypothetical_``. These are here so the magnitude of
    the decision is visible - nothing more. They are not a cost, they are not
    adopted, and nothing writes them into a graph.
    """
    keys = {
        f"{prefix}_in_minetti_domain": None,
        f"hypothetical_{prefix}_cw_j_per_kg_per_m": None,
        f"hypothetical_{prefix}_effort_j_per_kg": None,
    }
    if gradient is None or not np.isfinite(gradient):
        return keys
    inside = week5c1.in_domain(gradient)
    cost = week5c1.minetti_cost(gradient)
    keys[f"{prefix}_in_minetti_domain"] = bool(inside)
    keys[f"hypothetical_{prefix}_cw_j_per_kg_per_m"] = round(cost, 5)
    if length_m is not None and np.isfinite(length_m):
        keys[f"hypothetical_{prefix}_effort_j_per_kg"] = round(
            cost * float(length_m), 4)
    return keys


# --------------------------------------------------------------------------- #
# scope guard
# --------------------------------------------------------------------------- #
def assert_only_short_ordinary(table: pd.DataFrame) -> None:
    """Refuse to run on anything but edges whose only problem is their length.

    A short stair is still a stair. This diagnostic exists for edges that would
    be ordinary if the graph gave them a few more metres, and for no others.

    Raises:
        ShortEdgeError: if any row is not classified
            ``too_short_for_5m_gradient``, or carries a special marker.
    """
    wrong = table[table["classification"] != TARGET_CLASS]
    if not wrong.empty:
        found = sorted(set(wrong["classification"]))
        raise ShortEdgeError(
            f"{len(wrong)} edge(s) in the short-edge population are not "
            f"classified {TARGET_CLASS!r} but {found}. This diagnostic must "
            "never be used to give a special structure an ordinary gradient."
        )
    marked = table[table["special_markers"].apply(
        lambda value: bool(audit.marker_list(value)))]
    if not marked.empty:
        first = marked.iloc[0]
        raise ShortEdgeError(
            f"{len(marked)} short edge(s) carry a special marker, e.g. "
            f"({first['u']}, {first['v']}, {first['key']}) with "
            f"{first['special_markers']!r}. Special structures are out of "
            "scope for the short-edge candidates."
        )


# --------------------------------------------------------------------------- #
# reading the Week 5C-2A outputs
# --------------------------------------------------------------------------- #
def audit_dir(root: Path = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache" / AUDIT_SUBDIR).resolve()


def load_audit_table(source_dir: Path = None) -> pd.DataFrame:
    source_dir = Path(source_dir) if source_dir else audit_dir()
    path = source_dir / "edge_readiness_audit.csv"
    if not path.exists():
        raise ShortEdgeError(
            f"{path} is missing. Run the Week 5C-2A audit first:\n"
            "    python scripts/audit_terrain_readiness.py"
        )
    return pd.read_csv(path)


def load_route_table(source_dir: Path = None) -> pd.DataFrame:
    source_dir = Path(source_dir) if source_dir else audit_dir()
    path = source_dir / "stop_route_readiness.csv"
    if not path.exists():
        raise ShortEdgeError(
            f"{path} is missing. Run the Week 5C-2A audit first:\n"
            "    python scripts/audit_terrain_readiness.py"
        )
    return pd.read_csv(path)


def parse_class_field(value: Any) -> Dict[str, float]:
    """Parse a ``class=value;class=value`` cell from the route table."""
    parsed: Dict[str, float] = {}
    for item in audit.marker_list(value):
        name, _sep, number = item.partition("=")
        try:
            parsed[name] = float(number)
        except ValueError:
            continue
    return parsed


def parse_unresolved_details(value: Any) -> List[Tuple[str, str]]:
    """Parse ``u-v-key:class(len m)`` entries into (edge id, class) pairs.

    Ambiguous-hop entries carry no edge id and are skipped here; the audit
    already counted them separately.
    """
    pairs: List[Tuple[str, str]] = []
    for item in audit.marker_list(value):
        identifier, _sep, rest = item.partition(":")
        classification = rest.split("(")[0].split("[")[0]
        if ("->" in identifier or not classification
                or classification == audit.HOP_AMBIGUOUS):
            continue
        pairs.append((identifier, classification))
    return pairs


def route_usage_counts(routes: pd.DataFrame) -> Dict[str, int]:
    """How many audited routes each short edge appears on.

    A REPORTING column only. It decides which edges are worth showing in the
    focused table; it can never reach an estimator.
    """
    counts: Dict[str, int] = {}
    for _index, row in routes.iterrows():
        if row.get("route_audit_status") != audit.ROUTE_AUDITED:
            continue
        seen = {identifier for identifier, classification
                in parse_unresolved_details(row.get("unresolved_edge_details"))
                if classification == TARGET_CLASS}
        for identifier in seen:
            counts[identifier] = counts.get(identifier, 0) + 1
    return counts


def edge_identifier(u: Any, v: Any, key: Any) -> str:
    return f"{int(u)}-{int(v)}-{int(key)}"


# --------------------------------------------------------------------------- #
# the real short-edge population
# --------------------------------------------------------------------------- #
def geometry_coordinates(graph, u: int, v: int, key: int) -> List[Tuple[float, float]]:
    line, _stored = week5a.edge_geometry(graph, u, v, key)
    return [(round(x, 9), round(y, 9)) for x, y in line.coords]


def find_geometric_reverse(
    graph, u: int, v: int, key: int
) -> Tuple[Optional[int], str]:
    """The edge that is genuinely the reverse of this one, if there is one.

    Week 5C-2A established that ``(v, u, key)`` is NOT the counterpart of
    ``(u, v, key)`` - MultiDiGraph keys are assigned per ordered pair. So every
    edge from v to u is examined and one qualifies only when its coordinate
    sequence is exactly the reverse of this edge's. Where several qualify the
    pairing is ambiguous and none is used.

    Returns:
        Tuple of (key of the reverse edge or None, a reason).
    """
    if not graph.has_edge(v, u):
        return None, "no_edge_in_the_other_direction"
    forward = geometry_coordinates(graph, u, v, key)
    matches = [
        candidate for candidate in graph[v][u]
        if geometry_coordinates(graph, v, u, candidate) == forward[::-1]
    ]
    if not matches:
        return None, "no_candidate_has_the_reversed_geometry"
    if len(matches) > 1:
        return None, "several_candidates_share_the_reversed_geometry"
    return int(matches[0]), "verified_reverse_geometry"


def describe_short_edge(
    graph, coverage, row: pd.Series, usage: Dict[str, int],
    with_reverse: bool = True,
) -> Dict[str, Any]:
    """One row of the real short-edge population, with both candidates."""
    u, v, key = int(row["u"]), int(row["v"]), int(row["key"])
    profile = week5a.build_profile(graph, coverage, u, v, key,
                                   week5a.SAMPLE_SPACING_M)
    distances, _elevations = valid_samples(profile)
    sampled_length = float(profile["distance_m"].max())

    record: Dict[str, Any] = {
        "u": u, "v": v, "key": key,
        "edge_id": edge_identifier(u, v, key),
        "name": row.get("name", ""),
        "highway": row.get("highway", ""),
        "surface": row.get("surface", ""),
        "special_markers": row.get("special_markers", ""),
        "length_m": row.get("length_m"),
        "sampled_length_m": round(sampled_length, 3),
        "samples": int(len(profile)),
        "valid_samples": int(distances.size),
        "dem_status": row.get("dem_status"),
        "has_stored_geometry": row.get("has_stored_geometry"),
        "estimates_from_5m_method": row.get("estimates"),
        "eligible_5m_estimates": row.get("eligible_estimates"),
        "component_id": row.get("component_id"),
        "routes_using_this_edge": int(
            usage.get(edge_identifier(u, v, key), 0)),
    }
    record.update(candidate_gradients(profile))
    record["candidate_difference"] = (
        abs(record["endpoint_gradient"] - record["regression_gradient"])
        if record["endpoint_gradient"] is not None
        and record["regression_gradient"] is not None else None)
    record.update(minetti_diagnostics(record["endpoint_gradient"],
                                      sampled_length, "endpoint"))
    record.update(minetti_diagnostics(record["regression_gradient"],
                                      sampled_length, "regression"))

    record.update({
        "reverse_key": None, "reverse_pairing": "not_checked",
        "reverse_endpoint_gradient": None, "reverse_regression_gradient": None,
        "endpoint_reverse_sum": None, "regression_reverse_sum": None,
    })
    if with_reverse:
        reverse_key, reason = find_geometric_reverse(graph, u, v, key)
        record["reverse_pairing"] = reason
        if reverse_key is not None:
            reverse_profile = week5a.build_profile(
                graph, coverage, v, u, reverse_key, week5a.SAMPLE_SPACING_M)
            reverse = candidate_gradients(reverse_profile)
            record["reverse_key"] = reverse_key
            record["reverse_endpoint_gradient"] = reverse["endpoint_gradient"]
            record["reverse_regression_gradient"] = \
                reverse["regression_gradient"]
            for method in METHODS:
                forward_value = record[f"{method}_gradient"]
                reverse_value = reverse[f"{method}_gradient"]
                if forward_value is not None and reverse_value is not None:
                    # Perfect antisymmetry sums to zero. The residual is the
                    # measurement, so it is reported rather than judged.
                    record[f"{method}_reverse_sum"] = round(
                        forward_value + reverse_value, 8)
    return record


# --------------------------------------------------------------------------- #
# the pseudo-short validation experiment
# --------------------------------------------------------------------------- #
def window_start_indices(
    distances: np.ndarray, span_m: int, windows: int
) -> List[int]:
    """Evenly spread start indices for windows of a given whole-metre span.

    Spread along the parent rather than taken from its start, so the windows
    are not all cut from the same end of every street.
    """
    last_start = int(len(distances) - 1 - span_m)
    if last_start < 0:
        return []
    if windows <= 1 or last_start == 0:
        return [0]
    step = max(1, last_start // (windows - 1))
    starts = list(range(0, last_start + 1, step))[:windows]
    return starts


def reference_for_centre(
    references: pd.DataFrame, centre_m: float, tolerance_m: float
) -> Optional[pd.Series]:
    """The eligible 5 m estimate covering the same place on the same edge.

    Nearest centre wins, and only within half a sample interval. Matching a
    window to a 5 m estimate from somewhere else on the street would compare
    two different pieces of ground and call the difference method error.
    """
    if references.empty:
        return None
    centres = references["centre_distance_m"].to_numpy(dtype=float)
    index = int(np.argmin(np.abs(centres - centre_m)))
    if abs(float(centres[index]) - centre_m) > tolerance_m:
        return None
    return references.iloc[index]


def window_span_tolerance(profile: pd.DataFrame) -> float:
    """How far a window's real span may sit from its nominal length.

    Half of the parent profile's own typical sample spacing - the same
    discrete-sampling tolerance D-18 adopted for the 5 m baseline, applied to
    the same problem one level down.
    """
    return audit.baseline_tolerance(
        week5b.sample_spacing(audit.numeric_elevations(profile)))


def pseudo_short_rows(
    profile: pd.DataFrame, references: pd.DataFrame, edge_id: str,
    spans: Sequence[int], windows_per_length: int, tolerance_m: float,
) -> List[Dict[str, Any]]:
    """Cut short windows from one valid ordinary profile and score them.

    A window is cut by SAMPLE INDEX, so ``start + span`` is a nominal length in
    sample intervals, not a measured distance. Week 5A steps by one metre and
    then appends the far end of the line, so the last interval of a profile is
    short and a window touching it spans materially less than its nominal
    length.

    Because the whole validation is stratified by short-segment length, a
    window is kept in its requested stratum only when its ACTUAL span is within
    half a sample interval of that length. Anything further out is dropped -
    never relabelled, and never stretched by inventing an endpoint.
    """
    distances = profile["distance_m"].to_numpy(dtype=float)
    span_tolerance = window_span_tolerance(profile)
    rows: List[Dict[str, Any]] = []
    for span in spans:
        for start in window_start_indices(distances, span, windows_per_length):
            end = start + span
            actual_span = float(distances[end] - distances[start])
            span_error = actual_span - float(span)
            if abs(span_error) > span_tolerance:
                continue
            window = profile.iloc[start:end + 1]
            centre = float((distances[start] + distances[end]) / 2.0)
            reference = reference_for_centre(references, centre, tolerance_m)
            if reference is None:
                continue
            reference_gradient = float(reference["gradient"])
            if not np.isfinite(reference_gradient):
                continue

            row: Dict[str, Any] = {
                "edge_id": edge_id,
                "window_start_m": round(float(distances[start]), 3),
                "window_end_m": round(float(distances[end]), 3),
                "window_centre_m": round(centre, 3),
                "requested_window_length_m": int(span),
                "actual_window_length_m": round(actual_span, 4),
                "window_length_error_m": round(span_error, 4),
                "window_span_tolerance_m": round(span_tolerance, 4),
                "window_samples": int(len(window)),
                "reference_centre_m": round(
                    float(reference["centre_distance_m"]), 3),
                "reference_offset_m": round(
                    abs(float(reference["centre_distance_m"]) - centre), 4),
                "reference_gradient": reference_gradient,
                "reference_gradient_pct": round(reference_gradient * 100.0, 4),
                "reference_cw_j_per_kg_per_m": round(
                    week5c1.minetti_cost(reference_gradient), 6),
                "reference_in_minetti_domain": bool(
                    week5c1.in_domain(reference_gradient)),
            }
            row.update(candidate_gradients(window))
            for method in METHODS:
                value = row[f"{method}_gradient"]
                if value is None or not np.isfinite(value):
                    continue
                cost = week5c1.minetti_cost(value)
                row[f"{method}_gradient_pct"] = round(value * 100.0, 4)
                row[f"{method}_signed_error"] = value - reference_gradient
                row[f"{method}_abs_error"] = abs(value - reference_gradient)
                row[f"{method}_signed_error_pct"] = round(
                    (value - reference_gradient) * 100.0, 4)
                row[f"{method}_abs_error_pct"] = round(
                    abs(value - reference_gradient) * 100.0, 4)
                row[f"{method}_sign_agrees"] = bool(
                    np.sign(value) == np.sign(reference_gradient))
                row[f"{method}_in_minetti_domain"] = bool(
                    week5c1.in_domain(value))
                row[f"{method}_cw_j_per_kg_per_m"] = round(cost, 6)
                row[f"{method}_cw_difference"] = (
                    cost - row["reference_cw_j_per_kg_per_m"])
                row[f"{method}_cw_relative_difference"] = (
                    (cost - row["reference_cw_j_per_kg_per_m"])
                    / row["reference_cw_j_per_kg_per_m"]
                    if row["reference_cw_j_per_kg_per_m"] else None)
                # Terrain-to-level ratio, the quantity a route comparison
                # would actually feel: Cw divided by level walking.
                row[f"{method}_ratio_error"] = (
                    cost - row["reference_cw_j_per_kg_per_m"]) / LEVEL_COST
            rows.append(row)
    return rows


def method_metrics(part: pd.DataFrame, method: str) -> Dict[str, Any]:
    """Every metric the research question asks for, for one method and slice."""
    errors = part[f"{method}_abs_error"].dropna()
    signed = part[f"{method}_signed_error"].dropna()
    metrics: Dict[str, Any] = {
        "method": method,
        "windows": int(len(part)),
        "scored_windows": int(len(errors)),
    }
    if errors.empty:
        return metrics
    steep = part[part["reference_gradient_pct"].abs()
                 >= SIGN_AGREEMENT_MIN_GRADIENT_PCT]
    false_violation = part[(part["reference_in_minetti_domain"])
                           & (part[f"{method}_in_minetti_domain"].eq(False))]
    missed_violation = part[(~part["reference_in_minetti_domain"])
                            & (part[f"{method}_in_minetti_domain"] == True)]  # noqa: E712
    metrics.update({
        "median_signed_error_pct": round(float(signed.median() * 100), 4),
        "mean_signed_error_pct": round(float(signed.mean() * 100), 4),
        "median_abs_error_pct": round(float(errors.median() * 100), 4),
        "mean_abs_error_pct": round(float(errors.mean() * 100), 4),
        "p90_abs_error_pct": round(float(errors.quantile(0.90) * 100), 4),
        "p95_abs_error_pct": round(float(errors.quantile(0.95) * 100), 4),
        "max_abs_error_pct": round(float(errors.max() * 100), 4),
        "sign_agreement": round(
            float(part[f"{method}_sign_agrees"].mean()), 4),
        "sign_agreement_above_1pct_reference": round(
            float(steep[f"{method}_sign_agrees"].mean()), 4)
        if not steep.empty else None,
        "false_domain_violations": int(len(false_violation)),
        "missed_domain_violations": int(len(missed_violation)),
        "reference_windows_outside_domain": int(
            (~part["reference_in_minetti_domain"]).sum()),
        "median_abs_cw_difference": round(
            float(part[f"{method}_cw_difference"].abs().median()), 5),
        "p95_abs_cw_difference": round(
            float(part[f"{method}_cw_difference"].abs().quantile(0.95)), 5),
        "median_abs_ratio_error": round(
            float(part[f"{method}_ratio_error"].abs().median()), 5),
        "p95_abs_ratio_error": round(
            float(part[f"{method}_ratio_error"].abs().quantile(0.95)), 5),
    })
    return metrics


WEIGHTED_STRATUM_LABEL = "real-distribution weighted average of stratum metrics"

WEIGHTED_METRIC_NOTE = (
    "A weighted AVERAGE of each length stratum's metric, using how common that "
    "length is among the real short edges. It is NOT a pooled weighted median "
    "and NOT a pooled weighted p95: the weighted mean of stratum medians is a "
    "different quantity from the median of a reweighted population. No "
    "pseudo-observation was invented to compute a weighted quantile."
)

# Below this share of the real short-edge population, the weighted row cannot
# be described as representing that population.
REPRESENTATIVE_WEIGHT_FLOOR = 0.99

WEIGHTED_METRICS: Tuple[str, ...] = (
    "median_abs_error_pct", "p95_abs_error_pct", "sign_agreement",
    "median_abs_ratio_error",
)


def represented_population_weight(
    validation: pd.DataFrame, population_weights: Dict[int, float],
) -> float:
    """How much of the real short-edge population the strata actually cover.

    A length present in the real population but absent from the validation
    contributes nothing to the weighted row, so its weight is missing mass. It
    is reported rather than quietly renormalised away, because a future dataset
    could put real short edges at lengths this experiment never cut.
    """
    present = {int(span) for span
               in validation["requested_window_length_m"].unique()}
    return float(sum(weight for span, weight in population_weights.items()
                     if int(span) in present))


def summarise_pseudo_short(
    validation: pd.DataFrame, population_weights: Dict[int, float],
) -> pd.DataFrame:
    """Metrics per method, stratified by requested window length.

    Three kinds of row per method: one per length stratum, one over all
    windows, and one re-weighting the strata by how common each length is among
    the real short edges.

    That last row is deliberately named and columned differently. Averaging
    stratum medians with population weights does NOT give the median of a
    reweighted population, and presenting it in the ``median_abs_error_pct``
    column would invite exactly that reading. Its values live in
    ``weighted_avg_of_stratum_*`` columns, carry
    :data:`WEIGHTED_METRIC_NOTE`, and state how much of the real population the
    strata actually represent.
    """
    strata = sorted(validation["requested_window_length_m"].unique())
    represented = represented_population_weight(validation, population_weights)
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        by_stratum: Dict[int, Dict[str, Any]] = {}
        for span in strata:
            part = validation[
                validation["requested_window_length_m"] == span]
            entry = method_metrics(part, method)
            entry["stratum"] = f"{int(span)} m windows"
            entry["requested_window_length_m"] = int(span)
            entry["real_population_weight"] = round(
                population_weights.get(int(span), 0.0), 4)
            entry["median_actual_window_length_m"] = round(
                float(part["actual_window_length_m"].median()), 4) \
                if not part.empty else None
            by_stratum[int(span)] = entry
            rows.append(entry)

        overall = method_metrics(validation, method)
        overall["stratum"] = "all windows (unweighted)"
        overall["requested_window_length_m"] = None
        overall["real_population_weight"] = None
        rows.append(overall)

        weighted: Dict[str, Any] = {
            "method": method,
            "stratum": WEIGHTED_STRATUM_LABEL,
            "requested_window_length_m": None,
            "windows": int(len(validation)),
            "represented_population_weight": round(represented, 4),
            "represents_full_short_edge_population": bool(
                represented >= REPRESENTATIVE_WEIGHT_FLOOR),
            "metric_note": WEIGHTED_METRIC_NOTE,
        }
        for metric in WEIGHTED_METRICS:
            total = 0.0
            weight_used = 0.0
            for span in strata:
                weight = population_weights.get(int(span), 0.0)
                value = by_stratum[int(span)].get(metric)
                if weight and value is not None:
                    total += weight * value
                    weight_used += weight
            weighted[f"weighted_avg_of_stratum_{metric}"] = (
                round(total / weight_used, 4) if weight_used else None)
        weighted["weight_actually_used"] = round(
            sum(population_weights.get(int(span), 0.0) for span in strata
                if by_stratum[int(span)].get("scored_windows")), 4)
        rows.append(weighted)
    return pd.DataFrame(rows)


def population_length_weights(population: pd.DataFrame) -> Dict[int, float]:
    """How common each whole-metre length is among the real short edges."""
    spans = population["sampled_length_m"].round().astype(int)
    counts = spans.value_counts().to_dict()
    total = sum(counts.values())
    return {int(span): count / total for span, count in counts.items()} \
        if total else {}


# --------------------------------------------------------------------------- #
# route-scale context
# --------------------------------------------------------------------------- #
def route_exposure(routes: pd.DataFrame) -> pd.DataFrame:
    """How much of each audited route rests on a short edge.

    Read straight out of the Week 5C-2A route audit. No route is re-computed
    and no ranking is produced.
    """
    rows: List[Dict[str, Any]] = []
    for _index, row in routes.iterrows():
        if row.get("route_audit_status") != audit.ROUTE_AUDITED:
            continue
        counts = parse_class_field(row.get("class_counts"))
        lengths = parse_class_field(row.get("class_lengths_m"))
        short_length = lengths.get(TARGET_CLASS, 0.0)
        short_count = int(counts.get(TARGET_CLASS, 0))
        if short_count == 0:
            continue
        total = float(row["network_distance_m"])
        specials = sorted(name for name in counts
                          if name in audit.SPECIAL_CLASSES)
        short_edges = sorted({
            identifier for identifier, classification
            in parse_unresolved_details(row.get("unresolved_edge_details"))
            if classification == TARGET_CLASS})
        rows.append({
            "stop_id": row["stop_id"],
            "stop_name": row.get("stop_name", ""),
            "network_distance_m": round(total, 2),
            "route_edges": row.get("audited_edges"),
            "short_edges": short_count,
            "short_edge_length_m": round(short_length, 2),
            "short_edge_share_of_route_pct": round(
                100.0 * short_length / total, 3) if total else None,
            "contains_special_structure": bool(specials),
            "special_structures": ";".join(specials),
            "short_edge_ids": ";".join(short_edges),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_error_vs_length(validation: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spans = sorted(validation["requested_window_length_m"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, method, colour in zip(axes, METHODS, ("#2171b5", "#d95f02")):
        data = [validation.loc[validation["requested_window_length_m"] == span,
                               f"{method}_abs_error_pct"].dropna()
                for span in spans]
        # The keyword that names boxplot categories was renamed between
        # Matplotlib versions (labels -> tick_labels), so neither is used.
        # Boxes sit at the default 1-based positions and the ticks are set
        # afterwards, which behaves the same on every version.
        axis.boxplot(data, showfliers=False)
        axis.set_xticks(range(1, len(spans) + 1))
        axis.set_xticklabels([f"{int(span)} m" for span in spans])
        axis.set_title(f"{method} gradient", fontsize=10, color=colour)
        axis.set_xlabel("pseudo-short window length")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    axes[0].set_ylabel("absolute gradient error vs the 5 m reference\n"
                       "(percentage points of grade)")
    figure.suptitle("How far an edge-scale gradient departs from the adopted "
                    "5 m estimate", fontsize=11)
    figure.text(0.5, 0.005,
                "Diagnostic. The 5 m estimate is the project baseline, not "
                "ground truth. Outliers hidden; see the summary for p95 and "
                "maximum.",
                ha="center", fontsize=7.5, color="#555555")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_method_comparison(validation: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spans = sorted(validation["requested_window_length_m"].unique())
    figure, axis = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    positions = np.arange(len(spans))
    for offset, method, colour in zip((-width / 2, width / 2), METHODS,
                                      ("#2171b5", "#d95f02")):
        medians, p95 = [], []
        for span in spans:
            part = validation[validation["requested_window_length_m"] == span]
            errors = part[f"{method}_abs_error_pct"].dropna()
            medians.append(float(errors.median()) if not errors.empty else 0.0)
            p95.append(float(errors.quantile(0.95)) if not errors.empty else 0.0)
        axis.bar(positions + offset, medians, width, color=colour,
                 label=f"{method} (median)")
        axis.plot(positions + offset, p95, "k_", markersize=18,
                  label=f"{method} (p95)" if offset > 0 else None)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{int(s)} m" for s in spans])
    axis.set_xlabel("pseudo-short window length")
    axis.set_ylabel("absolute gradient error\n(percentage points of grade)")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    axis.set_title("Endpoint against linear trend, by window length",
                   fontsize=11)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_candidate_vs_reference(validation: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    limit = float(np.nanmax(np.abs(
        validation["reference_gradient_pct"].to_numpy(dtype=float)))) * 1.1
    for axis, method, colour in zip(axes, METHODS, ("#2171b5", "#d95f02")):
        axis.axline((0, 0), slope=1, color="#999999", linewidth=0.9,
                    linestyle="--", label="exact agreement")
        axis.scatter(validation["reference_gradient_pct"],
                     validation[f"{method}_gradient_pct"],
                     s=6, alpha=0.25, color=colour, edgecolors="none")
        axis.set_xlabel("adopted 5 m reference gradient (%)")
        axis.set_title(f"{method} gradient", fontsize=10)
        axis.grid(True, linewidth=0.4, alpha=0.4)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("edge-scale candidate gradient (%)")
    for boundary in (-45.0, 45.0):
        for axis in axes:
            axis.axhline(boundary, color="#e31a1c", linewidth=0.7, alpha=0.6)
            axis.axvline(boundary, color="#e31a1c", linewidth=0.7, alpha=0.6)
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    figure.suptitle("Edge-scale candidate against the adopted 5 m estimate",
                    fontsize=11)
    figure.text(0.5, 0.005,
                "Red lines mark the Minetti studied domain. Points crossing "
                "one axis but not the other are domain disagreements.",
                ha="center", fontsize=7.5, color="#555555")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
def select_validation_edges(
    table: pd.DataFrame, how_many: int, min_length_m: float = MIN_PARENT_LENGTH_M,
) -> pd.DataFrame:
    """Ordinary edges long enough to cut short windows from.

    Drawn at random with a fixed seed, so the experiment is reproducible and
    the draw cannot have been chosen to flatter either method. Nothing about
    the draw looks at routes, stops or gradients.
    """
    eligible = table[
        (table["classification"] == audit.CLASS_ORDINARY)
        & (table["minetti_status"] == audit.MINETTI_VALID)
        & (table["dem_status"] == audit.DEM_COMPLETE)
        & (table["sampled_length_m"] >= min_length_m)
    ].sort_values(["u", "v", "key"]).reset_index(drop=True)
    if how_many and how_many < len(eligible):
        rng = np.random.default_rng(VALIDATION_SEED)
        chosen = rng.choice(len(eligible), size=how_many, replace=False)
        eligible = eligible.iloc[np.sort(chosen)].reset_index(drop=True)
    return eligible


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 5C-2A.1 short-edge gradient validation (diagnostic).")
    parser.add_argument("--dem", nargs="+", default=[audit.DEFAULT_DEM_DIR])
    parser.add_argument("--graph", default=audit.DEFAULT_GRAPH)
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--validation-edges", type=int,
                        default=DEFAULT_VALIDATION_EDGES,
                        help="ordinary edges to cut pseudo-short windows from "
                             "(0 = every eligible edge)")
    parser.add_argument("--windows-per-length", type=int,
                        default=DEFAULT_WINDOWS_PER_LENGTH)
    parser.add_argument("--skip-reverse-check", action="store_true")
    options = parser.parse_args(argv)

    graph_path = Path(options.graph)
    if not graph_path.is_absolute():
        graph_path = (CONFIG_ROOT / graph_path).resolve()
    source_dir = Path(options.audit_dir).resolve() if options.audit_dir \
        else audit_dir()

    print("Week 5C-2A.1 - short-edge terrain-cost policy VALIDATION")
    print("No policy is adopted here. The ordinary method is unchanged: raw "
          "1 m DEM ->")
    print("5 m centred gradient -> Minetti Cw -> J/kg. This measures how far "
          "an")
    print("edge-scale estimate would depart from that baseline on edges too "
          "short")
    print("to support it. Short edges are a graph-resolution problem, not an")
    print("accessibility finding.")
    print("")

    try:
        table = load_audit_table(source_dir)
        routes = load_route_table(source_dir)
    except ShortEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    short = table[table["classification"] == TARGET_CLASS].copy()
    assert_only_short_ordinary(short)
    if short.empty:
        print("No edge is classified too_short_for_5m_gradient. Nothing to do.")
        return 0

    dem_paths = week5a.discover_tiles(options.dem)
    if not dem_paths:
        print(f"ERROR: no DEM tiles found under {options.dem}.", file=sys.stderr)
        return 1

    graph_hash_before = audit.file_sha256(graph_path)
    graph = load_graph(graph_path)
    try:
        coverage = week5a.DemCoverage(dem_paths)
    except week5a.DemError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    usage = route_usage_counts(routes)
    print(f"Short-edge population   {len(short)} directed edges, "
          f"{short['length_m'].sum():.2f} m "
          f"({100 * short['length_m'].sum() / table['length_m'].sum():.3f} % "
          "of graph length)")
    print(f"Route-used short edges  {len(usage)} unique edges across the "
          "audited routes")
    print("")

    try:
        print("Measuring both candidates on every real short edge...")
        population = pd.DataFrame([
            describe_short_edge(graph, coverage, row, usage,
                                with_reverse=not options.skip_reverse_check)
            for _index, row in short.iterrows()
        ])

        parents = select_validation_edges(table, options.validation_edges)
        spans = sorted(set(FIXED_WINDOW_LENGTHS_M)
                       | {int(round(v)) for v
                          in population["sampled_length_m"].dropna()
                          if 1 <= round(v) <= 5})
        print(f"Cutting pseudo-short windows from {len(parents)} valid "
              f"ordinary edges, spans {spans} m...")

        tolerance = REFERENCE_MATCH_FACTOR * week5a.SAMPLE_SPACING_M
        rows: List[Dict[str, Any]] = []
        for count, (_index, parent) in enumerate(parents.iterrows(), start=1):
            u, v, key = int(parent["u"]), int(parent["v"]), int(parent["key"])
            profile = week5a.build_profile(graph, coverage, u, v, key,
                                           week5a.SAMPLE_SPACING_M)
            gradient = audit.gradient_for_profile(profile,
                                                  audit.GRADIENT_BASELINE_M)
            references = gradient["gradients"]
            if references is None:
                continue
            rows.extend(pseudo_short_rows(
                profile, references, edge_identifier(u, v, key), spans,
                options.windows_per_length, tolerance))
            if count % 100 == 0:
                print(f"  {count}/{len(parents)} validation edges, "
                      f"{len(rows)} windows")
        validation = pd.DataFrame(rows)
    finally:
        coverage.close()

    graph_hash_after = audit.file_sha256(graph_path)
    if graph_hash_after != graph_hash_before:
        raise ShortEdgeError(
            f"{graph_path} changed during the diagnostic. It is read-only.")

    if validation.empty:
        print("ERROR: no pseudo-short window could be matched to a 5 m "
              "reference.", file=sys.stderr)
        return 1

    weights = population_length_weights(population)
    summary_table = summarise_pseudo_short(validation, weights)
    exposure = route_exposure(routes)
    route_used = population[population["routes_using_this_edge"] > 0] \
        .sort_values(["routes_using_this_edge", "edge_id"], ascending=[False, True])

    out_dir = audit.ensure_directory(audit.cache_dir(OUTPUT_SUBDIR))
    population.to_csv(out_dir / "short_edge_population.csv", index=False)
    route_used.to_csv(out_dir / "route_used_short_edges.csv", index=False)
    validation.to_csv(out_dir / "pseudo_short_validation.csv", index=False)
    summary_table.to_csv(out_dir / "pseudo_short_summary.csv", index=False)
    exposure.to_csv(out_dir / "route_short_edge_exposure.csv", index=False)
    plot_error_vs_length(validation, out_dir / "error_vs_window_length.png")
    plot_method_comparison(validation, out_dir / "endpoint_vs_regression.png")
    plot_candidate_vs_reference(validation, out_dir / "candidate_vs_reference.png")

    reverse_checked = population[
        population["reverse_pairing"] == "verified_reverse_geometry"]
    summary: Dict[str, Any] = {
        "diagnostic": "Week 5C-2A.1 short-edge gradient validation",
        "status": "VALIDATION ONLY - no short-edge method has been adopted",
        "purpose": (
            "Measures how far an edge-scale gradient departs from the adopted "
            "5 m centred gradient. Short edges are a graph-resolution and "
            "model-feasibility problem."
            " This is not an accessibility finding,"
            " and no value here is a routing weight."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "graph_sha256": graph_hash_after,
        "audit_source": str(source_dir),
        "adopted_method_unchanged": (
            "raw ~1 m DEM -> 5 m centred gradient -> Minetti Cw -> J/kg"),
        "short_edge_population": {
            "edges": int(len(population)),
            "total_length_m": round(float(short["length_m"].sum()), 2),
            "share_of_graph_length_pct": round(
                100 * float(short["length_m"].sum())
                / float(table["length_m"].sum()), 3),
            "length_quantiles_m": {
                str(q): round(float(population["sampled_length_m"].quantile(q)), 2)
                for q in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)},
            "edges_with_stored_geometry": int(
                population["has_stored_geometry"].eq(True).sum()),
            "edges_with_two_samples_only": int(
                (population["valid_samples"] < 3).sum()),
            "regression_identical_to_endpoint": int(
                population["regression_equals_endpoint"].eq(True).sum()),
            "highway_counts": population["highway"].value_counts().to_dict(),
            "unique_edges_used_by_audited_routes": int(len(route_used)),
        },
        "candidates_on_real_short_edges": {
            method: {
                "estimated": int(population[f"{method}_gradient"].notna().sum()),
                "median_abs_gradient_pct": round(float(
                    population[f"{method}_gradient"].abs().median() * 100), 4),
                "max_abs_gradient_pct": round(float(
                    population[f"{method}_gradient"].abs().max() * 100), 4),
                "outside_minetti_domain": int(
                    population[f"{method}_in_minetti_domain"].eq(False).sum()),
            } for method in METHODS
        },
        "pseudo_short_experiment": {
            "validation_edges": int(len(parents)),
            "windows": int(len(validation)),
            "window_lengths_m": spans,
            "windows_per_length_per_edge": options.windows_per_length,
            "reference": (
                "the eligible 5 m centred-gradient estimate nearest the "
                "window centre, within half a sample interval; the project "
                "baseline, NOT ground truth"),
            "reference_match_tolerance_m": tolerance,
            "selection_seed": VALIDATION_SEED,
            "window_length_rule": (
                "a window is cut by sample index and kept in its requested "
                "stratum only when its ACTUAL metric span is within half the "
                "parent's typical sample spacing of that length; otherwise it "
                "is dropped, never relabelled and never stretched"),
            "real_population_length_weights": {
                str(k): round(v, 4) for k, v in sorted(weights.items())},
            "represented_population_weight": round(
                represented_population_weight(validation, weights), 4),
            "unrepresented_lengths_m": sorted(
                int(span) for span in weights
                if int(span) not in set(
                    validation["requested_window_length_m"].unique())),
            "represents_full_short_edge_population": bool(
                represented_population_weight(validation, weights)
                >= REPRESENTATIVE_WEIGHT_FLOOR),
            "weighted_metric_note": WEIGHTED_METRIC_NOTE,
            "structural_note_at_the_baseline_length": (
                "At a window length equal to the gradient baseline (5 m) the "
                "endpoint candidate uses exactly the two samples the 5 m "
                "centred reference is built from, so its error in that "
                "stratum is identically zero by construction. The 5 m stratum "
                "is therefore NOT an independent test of the endpoint method. "
                "The regression candidate uses every sample in the window and "
                "is not affected."),
        },
        "reverse_consistency": {
            "short_edges_with_a_verified_reverse": int(len(reverse_checked)),
            "pairing_outcomes":
                population["reverse_pairing"].value_counts().to_dict(),
            **{
                f"{method}_median_abs_reverse_sum": (
                    round(float(reverse_checked[f"{method}_reverse_sum"]
                                .abs().median()), 8)
                    if not reverse_checked.empty else None)
                for method in METHODS
            },
        },
        "route_exposure": {
            "audited_routes": int(
                (routes["route_audit_status"] == audit.ROUTE_AUDITED).sum()),
            "routes_touching_a_short_edge": int(len(exposure)),
            "max_short_edge_share_of_a_route_pct": round(float(
                exposure["short_edge_share_of_route_pct"].max()), 3)
            if not exposure.empty else None,
            "median_short_edge_share_of_a_route_pct": round(float(
                exposure["short_edge_share_of_route_pct"].median()), 3)
            if not exposure.empty else None,
        },
        "not_done": [
            "no short-edge method adopted",
            "no routing weight written",
            "no effort Dijkstra",
            "no stop ranking",
            "no Week 5C-2A classification changed",
            "no special structure given an ordinary gradient",
        ],
    }
    with open(out_dir / "short_edge_validation_summary.json", "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("")
    print("=" * 78)
    print("Pseudo-short validation against the adopted 5 m estimate")
    print("=" * 78)
    print(f"{'method':<12}{'stratum':<14}{'n':>8}{'median':>10}{'p90':>9}"
          f"{'p95':>9}{'max':>9}{'sign':>8}{'actual m':>10}")
    for row in summary_table.itertuples():
        if pd.isna(getattr(row, "requested_window_length_m", None)):
            continue
        print(f"{row.method:<12}{str(row.stratum):<14}{row.windows:>8}"
              f"{row.median_abs_error_pct:>10.3f}{row.p90_abs_error_pct:>9.3f}"
              f"{row.p95_abs_error_pct:>9.3f}{row.max_abs_error_pct:>9.3f}"
              f"{row.sign_agreement:>8.3f}"
              f"{row.median_actual_window_length_m:>10.3f}")
    print("  errors in percentage points of grade, against the adopted 5 m "
          "estimate")
    print("  'actual m' is the median REAL span of the windows in that "
          "stratum; a window")
    print("  whose real span was further than half a sample interval from its "
          "nominal")
    print("  length was dropped, never relabelled.")

    print("")
    print("Weighted by the real short-edge length distribution:")
    for row in summary_table[
            summary_table["stratum"] == WEIGHTED_STRATUM_LABEL].itertuples():
        print(f"  {row.method:<12}"
              f"median {row.weighted_avg_of_stratum_median_abs_error_pct:>7.3f}"
              f"   p95 {row.weighted_avg_of_stratum_p95_abs_error_pct:>7.3f}"
              f"   sign {row.weighted_avg_of_stratum_sign_agreement:>6.3f}")
    represented = float(
        summary_table["represented_population_weight"].dropna().max())
    print("  These are weighted AVERAGES of the per-length stratum metrics - "
          "NOT pooled")
    print("  weighted medians or percentiles. The two are different "
          "quantities.")
    print(f"  Strata represent {represented:.1%} of the real short-edge "
          "population.")
    if represented < REPRESENTATIVE_WEIGHT_FLOOR:
        print("  WARNING: that is materially below the whole population, so "
              "the weighted")
        print("  values do NOT describe every real short edge. The unmatched "
              "lengths are")
        print("  reported in short_edge_validation_summary.json.")

    print("")
    print(f"Wrote data/cache/{OUTPUT_SUBDIR}/")
    print(f"The Stage 3 graph is unchanged: sha256 {graph_hash_after}")
    print("")
    print("EVIDENCE ONLY. No error threshold has been set, neither candidate "
          "has been")
    print("selected, and the hypothetical Cw and J/kg columns are diagnostic "
          "magnitudes,")
    print("not costs. The short-edge policy remains OPEN (see D-19).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
