"""Week 5C-2A - network-wide terrain-cost READINESS audit.

What this answers
-----------------
For every directed edge of the existing raw Glebe pedestrian graph: is this
edge ready to be given an ordinary DEM + 5 m gradient + Minetti walking cost,
or does it need handling this project has not designed yet?

That is all it answers. It is a MODEL-READINESS audit, not an accessibility
finding. "special", "unresolved" and "requires_special_handling" say something
about OUR model, never about a person's ability to walk somewhere. An edge is
marked unresolved when the ordinary model cannot honestly be applied to it -
which is a statement about the evidence available, not about the street.

The chain being audited, reused rather than reimplemented
---------------------------------------------------------
    Week 5A   scripts/inspect_dem_profiles.py       1 m LiDAR -> raw profile
    Week 5B   scripts/analyse_gradient_sensitivity  profile -> 5 m gradient
    Week 5C-1 scripts/validate_walking_cost.py      gradient -> Minetti J/kg

This script writes no new sampling, gradient or cost mathematics. It adds a
classification ladder, a loop, and a route audit.

Directed edges are audited INDEPENDENTLY
----------------------------------------
A MultiDiGraph key is assigned per ordered node pair, so ``(v, u, k)`` is not
the reverse of ``(u, v, k)``. Measured on this graph, pairing by key puts edges
up to 174.9 m apart into the same "pair". Every directed edge is therefore
sampled on its own. Correctness beats halving the DEM work.

The classification ladder
-------------------------
``classification`` is mutually exclusive and is decided in this order:

    steps -> bridge -> tunnel -> elevator -> conveying -> corridor
      -> access_restricted
      -> DEM_missing_or_unusable
      -> degenerate_sampling_geometry
      -> too_short_for_5m_gradient
      -> Minetti_out_of_domain
      -> ordinary

``special_markers`` is NOT exclusive and keeps every marker that matched, so
the priority rule cannot hide how common bridges, tunnels or steps really are.
``covered=yes`` is a marker only and never a classification.

Only genuine 5 m gradients are used
-----------------------------------
Week 5B reports the baseline each estimate actually achieved, and the terminal
estimate of a profile is routinely short because the final sample interval is
short - on a constant 10 per cent ramp it came out at 12.5 per cent, measured
over 4 m. An estimate is used only when its actual baseline is within half a
sample interval of 5 m, the natural tolerance for hitting a baseline with
discrete samples. Rejected estimates take no part in the gradient range, the
domain test or the effort, and the length they represented becomes uncovered
endpoint length. Nothing is invented in their place.

Conservative on purpose
-----------------------
* Any missing, NoData or out-of-coverage DEM sample on an otherwise ordinary
  edge blocks a trusted cost. Nothing is interpolated or filled.
* Any 5 m gradient estimate outside the Minetti domain [-0.45, +0.45] blocks a
  trusted cost. Nothing is clamped and no extrapolated value is trusted.
* ``ordinary_effort_j_per_kg`` is populated if and only if the classification
  is ``ordinary``. That invariant is asserted over the whole table before
  anything is written.

Special structures are still profiled where the DEM allows it, because the
elevation evidence is interesting, but a profile on a stairway or a bridge deck
can never turn into an ordinary walking cost.

Usage
-----
    python scripts/audit_terrain_readiness.py
    python scripts/audit_terrain_readiness.py --dem data/dem
    python scripts/audit_terrain_readiness.py --sample-edges 200   # timing only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from src.osm_network import component_labels, load_graph  # noqa: E402
from src.routing import (  # noqa: E402
    STATUS_ROUTED,
    load_json,
    resolve_origin_component,
    single_source_distances,
    verify_inputs,
)


# --------------------------------------------------------------------------- #
# the Week 5A/5B/5C-1 chain, loaded by path because scripts/ is not a package
# --------------------------------------------------------------------------- #
def load_script(name: str):
    """Import one of this project's scripts as a module."""
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


week5a = load_script("inspect_dem_profiles")
week5b = load_script("analyse_gradient_sensitivity")
week5c1 = load_script("validate_walking_cost")


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
OUTPUT_SUBDIR = "terrain_readiness"
DEFAULT_DEM_DIR = "data/dem"
DEFAULT_GRAPH = "data/cache/glebe_walk.graphml"

SAMPLE_SPACING_M = week5a.SAMPLE_SPACING_M          # 1.0, Week 5A
GRADIENT_BASELINE_M = week5c1.ADOPTED_BASELINE_M    # 5.0, Week 5B
MINETTI_DOMAIN = week5c1.MINETTI_DOMAIN             # (-0.45, +0.45)

# Two estimates is not a style choice: Week 5C-1's representative-length
# integration needs at least two support points to have a length to integrate
# over. The short-edge rule is therefore the method's own feasibility limit,
# not an arbitrary number of metres.
MIN_GRADIENT_ESTIMATES = 2

# An estimate counts as a 5 m gradient only if the baseline it actually
# achieved is within half a sample interval of 5 m. With discrete ~1 m samples
# the baseline can only be hit to the nearest sample, so half an interval is
# the quantisation tolerance the sampling itself implies - not a chosen
# percentage. The terminal estimate of a profile routinely falls outside it,
# because the final sample interval is short.
BASELINE_TOLERANCE_FACTOR = 0.5

# Two parallel edges count as tied when their lengths are indistinguishable in
# floating point, not merely close.
LENGTH_TIE_TOLERANCE_M = 1e-9

# The Stage 5 distances are stored rounded to 2 dp, so the reconstruction is
# checked to the same tolerance Stage 5 itself used.
DISTANCE_TOLERANCE_M = 0.1

CLASS_STEPS = "steps"
CLASS_BRIDGE = "bridge"
CLASS_TUNNEL = "tunnel"
CLASS_ELEVATOR = "elevator"
CLASS_CONVEYING = "conveying"
CLASS_CORRIDOR = "corridor"
CLASS_ACCESS_RESTRICTED = "access_restricted"
CLASS_DEM_MISSING = "DEM_missing_or_unusable"
CLASS_DEGENERATE = "degenerate_sampling_geometry"
CLASS_TOO_SHORT = "too_short_for_5m_gradient"
CLASS_OUT_OF_DOMAIN = "Minetti_out_of_domain"
CLASS_ORDINARY = "ordinary"

# Marker -> classification, in priority order. Steps outrank bridge by an
# explicit project decision: the Glebe Tram Stop footbridge is both, and the
# step treatment is the more fundamental departure from graded walking.
# covered=yes is deliberately NOT here; it is a marker only.
MARKER_PRIORITY: Tuple[Tuple[str, str], ...] = (
    ("highway=steps", CLASS_STEPS),
    ("step_count", CLASS_STEPS),
    ("bridge", CLASS_BRIDGE),
    ("tunnel", CLASS_TUNNEL),
    ("highway=elevator", CLASS_ELEVATOR),
    ("conveying", CLASS_CONVEYING),
    ("highway=corridor", CLASS_CORRIDOR),
    ("access=no", CLASS_ACCESS_RESTRICTED),
)

MARKER_ONLY: Tuple[str, ...] = ("covered=yes",)

SPECIAL_CLASSES: Tuple[str, ...] = tuple(
    dict.fromkeys(cls for _marker, cls in MARKER_PRIORITY)
)

UNRESOLVED_CLASSES: Tuple[str, ...] = SPECIAL_CLASSES + (
    CLASS_DEM_MISSING, CLASS_DEGENERATE, CLASS_TOO_SHORT, CLASS_OUT_OF_DOMAIN,
)

CLASS_ORDER: Tuple[str, ...] = UNRESOLVED_CLASSES + (CLASS_ORDINARY,)

DEM_COMPLETE = "complete"
DEM_PARTIAL = "partial"
DEM_NONE = "none"
DEM_NOT_SAMPLED = "not_sampled"

GRADIENT_COMPUTED = "computed"
GRADIENT_TOO_SHORT = "too_short"
GRADIENT_DEM_GAPS = "dem_gaps"
GRADIENT_DEGENERATE = "degenerate_sampling_geometry"
GRADIENT_NOT_ATTEMPTED = "not_attempted"

MINETTI_VALID = "valid"
MINETTI_FLAGGED = "flagged_extrapolated"
MINETTI_SPECIAL = "not_applicable_special_structure"
MINETTI_NOT_COMPUTED = "not_computed"

HOP_UNIQUE = "unique"
HOP_TIED_EQUIVALENT = "tied_but_equivalent"
HOP_AMBIGUOUS = "parallel_min_ambiguous"

ROUTE_AUDITED = "audited"
ROUTE_NO_PATH = "no_path_available_to_audit"

JUBILEE_NOTE = (
    "No path exists in the current RAW OSM topology, so there is no route to "
    "audit. This is I-04, a known upstream data artefact; OSM-FIX-001 remains "
    "proposed and is NOT applied. This is NOT a route with zero special edges."
)


class ReadinessError(RuntimeError):
    """Raised with an actionable message when the audit cannot be trusted."""


# --------------------------------------------------------------------------- #
# tag handling
# --------------------------------------------------------------------------- #
def tag_set(data: Dict[str, Any], key: str) -> Set[str]:
    """The values of one OSM tag, lower-cased, as a set.

    OSMnx stores a merged way's tag either as a string or as a LIST of strings
    (``['footway', 'steps']``). A plain equality test misses the list form
    entirely, which is exactly how a stairway would slip into the ordinary
    class, so every tag read goes through here.
    """
    value = data.get(key)
    if value is None:
        return set()
    items: Sequence[Any]
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = str(value).split(";")
    return {str(item).strip().lower() for item in items if str(item).strip()}


def tag_text(data: Dict[str, Any], key: str) -> str:
    """The raw tag value as one string, for the audit table."""
    value = data.get(key)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def has_positive_tag(data: Dict[str, Any], key: str) -> bool:
    """Whether a yes/no style tag is present and says something other than no."""
    values = tag_set(data, key)
    return bool(values) and values != {"no"}


def detect_markers(data: Dict[str, Any]) -> List[str]:
    """Every special-structure marker on this edge, in a stable order.

    Not mutually exclusive on purpose. The Glebe Tram Stop footbridge carries
    both ``highway=steps`` and ``bridge``, and the aggregate report must be
    able to say so even though the classification can only pick one.
    """
    markers: List[str] = []
    highway = tag_set(data, "highway")
    if "steps" in highway:
        markers.append("highway=steps")
    step_count = data.get("step_count")
    if step_count is not None and str(step_count).strip() != "":
        markers.append("step_count")
    if has_positive_tag(data, "bridge"):
        markers.append("bridge")
    if has_positive_tag(data, "tunnel"):
        markers.append("tunnel")
    if "elevator" in highway:
        markers.append("highway=elevator")
    if has_positive_tag(data, "conveying"):
        markers.append("conveying")
    if "corridor" in highway:
        markers.append("highway=corridor")
    if "no" in tag_set(data, "access"):
        markers.append("access=no")
    if "yes" in tag_set(data, "covered"):
        markers.append("covered=yes")
    return markers


def marker_list(value: Any) -> List[str]:
    """The markers recorded in one audit cell, as a list.

    Written as an explicit parser rather than a pandas fill so that nothing in
    this file ever substitutes a value for a missing one - not even an empty
    string for a blank marker cell.
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return [item for item in str(value).split(";") if item]


def has_marker(value: Any, marker: str) -> bool:
    return marker in marker_list(value)


def special_classification(markers: Sequence[str]) -> Optional[str]:
    """The single special class for a set of markers, or None if ordinary.

    ``covered=yes`` never reaches this function's answer: it is absent from
    MARKER_PRIORITY, so an edge whose only marker is ``covered=yes`` stays
    eligible for the ordinary model with the marker recorded against it.
    """
    present = set(markers)
    for marker, classification in MARKER_PRIORITY:
        if marker in present:
            return classification
    return None


def ordinary_guard_tags(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalised tags for the Week 5C-1 guard.

    Week 5C-1's :func:`special_structure_reason` splits on semicolons, so a
    Python list reaches it as the literal text ``"['footway', 'steps']"`` and
    matches nothing. Feeding it semicolon-joined values lets that independent
    guard actually fire, which is the point of keeping it: it is a second
    opinion on the same question, not decoration.
    """
    tags: Dict[str, Any] = {}
    for key in ("highway", "bridge", "tunnel"):
        values = tag_set(data, key)
        if values:
            tags[key] = ";".join(sorted(values))
    step_count = data.get("step_count")
    if step_count is not None and str(step_count).strip() != "":
        tags["step_count"] = step_count
    return tags


# --------------------------------------------------------------------------- #
# hashing and provenance
# --------------------------------------------------------------------------- #
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_dir(subdir: str, root: Path = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache" / subdir).resolve()


def ensure_directory(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# --------------------------------------------------------------------------- #
# auditing one directed edge
# --------------------------------------------------------------------------- #
def blank_gradient_result() -> Dict[str, Any]:
    return {
        "estimates": 0,
        "eligible_estimates": 0,
        "rejected_baseline_estimates": 0,
        "finite_estimates": 0,
        "gradient_status": GRADIENT_NOT_ATTEMPTED,
        "min_gradient_pct": None,
        "max_gradient_pct": None,
        "sample_spacing_m": None,
        "baseline_tolerance_m": None,
        "min_actual_baseline_m": None,
        "max_actual_baseline_m": None,
        "min_actual_baseline_used_m": None,
        "max_actual_baseline_used_m": None,
        "duplicate_sample_distances": False,
        "out_of_domain_estimates": 0,
        "gradients": None,
    }


def numeric_elevations(profile: pd.DataFrame) -> pd.DataFrame:
    """A copy of the profile whose elevation column is numeric.

    Week 5A writes a missing elevation as ``None``. When EVERY sample is
    missing, pandas keeps the column as objects and arithmetic on it fails
    with a type error rather than producing NaN. Coercing to numeric turns
    each ``None`` into NaN and nothing else: no value is filled, replaced or
    guessed, and the caller's frame is not touched.
    """
    coerced = profile.copy()
    coerced["elevation_m_ahd"] = pd.to_numeric(
        profile["elevation_m_ahd"], errors="coerce")
    return coerced


def has_duplicate_sample_distances(profile: pd.DataFrame) -> bool:
    """Whether two samples were recorded at the same distance along the edge.

    Week 5A rounds the sample distance to millimetres and always appends the
    far end of the line, so an edge whose projected length lands within half a
    millimetre of a whole number of samples records its final sample at the
    same distance as the previous one. Reported as a diagnostic; the estimate
    that uses it is removed by the baseline-eligibility rule below.
    """
    distances = profile["distance_m"].to_numpy(dtype=float)
    return bool(np.any(np.diff(distances) <= 0))


def baseline_tolerance(sample_spacing_m: float) -> float:
    """How far an actual baseline may sit from the target and still count.

    Half of one sample interval. With discrete samples about a metre apart, a
    5 m baseline can only ever be hit to the nearest sample, so half an
    interval is the natural quantisation tolerance around the request. It is
    derived from the data's own spacing rather than being a chosen percentage.
    """
    return BASELINE_TOLERANCE_FACTOR * float(sample_spacing_m)


def eligible_baseline_mask(
    gradients: pd.DataFrame, baseline_m: float, tolerance_m: float
) -> np.ndarray:
    """Which estimates were actually measured over the requested baseline.

    THIS IS THE VALIDITY FIX. Week 5B honestly reports the baseline each
    estimate achieved, and the last one is routinely short because the final
    sample interval is short: on a constant 10 per cent ramp the terminal
    estimate came out at 12.5 per cent, measured over 4 m rather than 5 m.

    Left in, a spurious terminal value would inflate ``max_gradient_pct`` and
    could push an otherwise valid edge into ``Minetti_out_of_domain``. It is
    therefore excluded from every judgement - gradient range, domain test and
    effort - while staying visible in the diagnostic counts.

    Nothing is invented in its place: the length it represented becomes
    uncovered endpoint length, exactly like the metres beyond the first and
    last estimate.
    """
    achieved = gradients["actual_baseline_m"].to_numpy(dtype=float)
    return np.abs(achieved - float(baseline_m)) <= tolerance_m


def is_contiguous(mask: np.ndarray) -> bool:
    """Whether a boolean mask selects one unbroken run.

    A gap in the middle would let the integration span across rejected
    estimates and quietly cover ground no accepted estimate measured, which is
    the filling this audit refuses to do. In practice only terminal estimates
    are ever rejected; anything else is reported instead of integrated over.
    """
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return False
    return bool(indices[-1] - indices[0] + 1 == indices.size)


def gradient_for_profile(
    profile: pd.DataFrame, baseline_m: float = GRADIENT_BASELINE_M
) -> Dict[str, Any]:
    """Run the adopted Week 5B method on one profile and describe the outcome.

    Nothing here decides a classification; it reports what the existing method
    could and could not do, and the ladder in :func:`audit_edge` reads that.

    Only estimates whose ACTUAL baseline is within half a sample interval of
    the adopted 5 m baseline are used for anything. The rest are counted and
    reported, never used and never replaced.
    """
    result = blank_gradient_result()
    result["duplicate_sample_distances"] = has_duplicate_sample_distances(profile)
    numeric = numeric_elevations(profile)
    try:
        gradients = week5b.centred_gradient(numeric, baseline_m)
        spacing = week5b.sample_spacing(numeric)
    except week5b.SensitivityError:
        # The profile is shorter than the baseline needs. Not an error: an
        # answer. No endpoint gradient is invented to fill the gap.
        result["gradient_status"] = GRADIENT_TOO_SHORT
        return result

    tolerance = baseline_tolerance(spacing)
    achieved = gradients["actual_baseline_m"].to_numpy(dtype=float)
    mask = eligible_baseline_mask(gradients, baseline_m, tolerance)

    result["estimates"] = int(len(gradients))
    result["eligible_estimates"] = int(mask.sum())
    result["rejected_baseline_estimates"] = int(len(gradients) - mask.sum())
    result["sample_spacing_m"] = round(float(spacing), 4)
    result["baseline_tolerance_m"] = round(float(tolerance), 4)
    result["min_actual_baseline_m"] = round(float(achieved.min()), 3)
    result["max_actual_baseline_m"] = round(float(achieved.max()), 3)
    if mask.any():
        result["min_actual_baseline_used_m"] = round(
            float(achieved[mask].min()), 3)
        result["max_actual_baseline_used_m"] = round(
            float(achieved[mask].max()), 3)

    # Shortness is decided on the ELIGIBLE estimates: an edge with only
    # short-baseline estimates has no 5 m gradient, whatever Week 5B produced.
    if result["eligible_estimates"] < MIN_GRADIENT_ESTIMATES:
        result["gradient_status"] = GRADIENT_TOO_SHORT
        return result

    if not is_contiguous(mask):
        result["gradient_status"] = GRADIENT_DEGENERATE
        return result

    eligible = gradients[mask].reset_index(drop=True)
    centres = eligible["centre_distance_m"].to_numpy(dtype=float)
    if np.any(np.diff(centres) <= 0):
        result["gradient_status"] = GRADIENT_DEGENERATE
        return result

    slopes = eligible["gradient"].to_numpy(dtype=float)
    finite = np.isfinite(slopes)
    result["finite_estimates"] = int(finite.sum())
    result["gradients"] = eligible

    if not finite.any():
        result["gradient_status"] = GRADIENT_DEM_GAPS
        return result

    result["min_gradient_pct"] = round(float(np.nanmin(slopes) * 100.0), 3)
    result["max_gradient_pct"] = round(float(np.nanmax(slopes) * 100.0), 3)
    inside = week5c1.in_domain_array(slopes[finite])
    result["out_of_domain_estimates"] = int((~inside).sum())
    result["gradient_status"] = (
        GRADIENT_COMPUTED if finite.all() else GRADIENT_DEM_GAPS
    )
    return result


def audit_edge(
    graph,
    coverage,
    u: int,
    v: int,
    key: int,
    component: str,
    spacing_m: float = SAMPLE_SPACING_M,
    baseline_m: float = GRADIENT_BASELINE_M,
) -> Dict[str, Any]:
    """Classify one directed edge and, only if ordinary, cost it.

    Returns:
        One audit row. ``ordinary_effort_j_per_kg`` is present only when the
        classification is ``ordinary``; every other row carries None there.
    """
    data = graph.edges[u, v, key]
    markers = detect_markers(data)
    special = special_classification(markers)

    row: Dict[str, Any] = {
        "u": int(u), "v": int(v), "key": int(key),
        "osmid": tag_text(data, "osmid"),
        "length_m": round(float(data.get("length", float("nan"))), 3),
        "highway": tag_text(data, "highway"),
        "name": tag_text(data, "name"),
        "surface": tag_text(data, "surface"),
        "incline_osm": tag_text(data, "incline"),
        "bridge": tag_text(data, "bridge"),
        "tunnel": tag_text(data, "tunnel"),
        "step_count": tag_text(data, "step_count"),
        "access": tag_text(data, "access"),
        "covered": tag_text(data, "covered"),
        "conveying": tag_text(data, "conveying"),
        "special_markers": ";".join(markers),
        "component_id": component,
    }

    # ----- sample the DEM. Special structures are profiled too: the evidence
    # is worth having, and it can never become an ordinary cost. -----------
    profile: Optional[pd.DataFrame] = None
    dem_status = DEM_NOT_SAMPLED
    samples = valid_samples = 0
    sampled_length = None
    tiles: List[str] = []
    has_geometry = None
    try:
        profile = week5a.build_profile(graph, coverage, u, v, key, spacing_m)
    except week5a.DemError as error:
        row["sampling_error"] = str(error)
    else:
        has_geometry = bool(profile.attrs.get("has_stored_geometry"))
        samples = int(len(profile))
        elevations = profile["elevation_m_ahd"]
        valid_samples = int(elevations.notna().sum())
        sampled_length = round(float(profile["distance_m"].max()), 3)
        tiles = sorted({t for t in profile["dem_tile"].dropna().unique()})
        if valid_samples == 0:
            dem_status = DEM_NONE
        elif valid_samples == samples:
            dem_status = DEM_COMPLETE
        else:
            dem_status = DEM_PARTIAL

    row.update({
        "has_stored_geometry": has_geometry,
        "sampled_length_m": sampled_length,
        "samples": samples,
        "valid_samples": valid_samples,
        "nodata_or_outside_samples": samples - valid_samples,
        "dem_status": dem_status,
        "dem_tiles_touched": ";".join(tiles),
    })

    gradient = (
        gradient_for_profile(profile, baseline_m) if profile is not None
        else blank_gradient_result()
    )
    row.update({
        "estimates": gradient["estimates"],
        "eligible_estimates": gradient["eligible_estimates"],
        "rejected_baseline_estimates": gradient["rejected_baseline_estimates"],
        "finite_estimates": gradient["finite_estimates"],
        "gradient_status": gradient["gradient_status"],
        "min_gradient_pct": gradient["min_gradient_pct"],
        "max_gradient_pct": gradient["max_gradient_pct"],
        "sample_spacing_m": gradient["sample_spacing_m"],
        "baseline_tolerance_m": gradient["baseline_tolerance_m"],
        "min_actual_baseline_m": gradient["min_actual_baseline_m"],
        "max_actual_baseline_m": gradient["max_actual_baseline_m"],
        "min_actual_baseline_used_m": gradient["min_actual_baseline_used_m"],
        "max_actual_baseline_used_m": gradient["max_actual_baseline_used_m"],
        "duplicate_sample_distances": gradient["duplicate_sample_distances"],
        "out_of_domain_estimates": gradient["out_of_domain_estimates"],
    })

    # ----- the classification ladder -------------------------------------- #
    if special is not None:
        classification = special
    elif dem_status != DEM_COMPLETE:
        classification = CLASS_DEM_MISSING
    elif gradient["gradient_status"] == GRADIENT_DEGENERATE:
        classification = CLASS_DEGENERATE
    elif gradient["gradient_status"] == GRADIENT_TOO_SHORT:
        classification = CLASS_TOO_SHORT
    elif gradient["gradient_status"] == GRADIENT_DEM_GAPS:
        # Belt and braces: a complete DEM should not produce a missing
        # gradient. If it somehow does, the edge stays unresolved.
        classification = CLASS_DEM_MISSING
    elif gradient["out_of_domain_estimates"] > 0:
        classification = CLASS_OUT_OF_DOMAIN
    else:
        classification = CLASS_ORDINARY

    row["classification"] = classification
    row["requires_special_handling"] = classification in UNRESOLVED_CLASSES

    # ----- cost, and ONLY for an ordinary edge ---------------------------- #
    row.update({
        "minetti_status": MINETTI_NOT_COMPUTED,
        "gradient_covered_length_m": None,
        "uncovered_endpoint_length_m": None,
        "out_of_domain_length_m": None,
        "ordinary_effort_j_per_kg": None,
        "terrain_to_level_ratio": None,
    })

    if classification in SPECIAL_CLASSES:
        row["minetti_status"] = MINETTI_SPECIAL
        return row

    if classification == CLASS_OUT_OF_DOMAIN:
        row["minetti_status"] = MINETTI_FLAGGED
        table = week5c1.build_effort_table(gradient["gradients"])
        outside = ~table["in_minetti_domain"].to_numpy()
        row["out_of_domain_length_m"] = round(
            float(table.loc[outside, "integration_weight_m"].sum()), 3)
        row["gradient_covered_length_m"] = round(
            float(table["integration_weight_m"].sum()), 3)
        if sampled_length is not None:
            row["uncovered_endpoint_length_m"] = round(
                sampled_length - row["gradient_covered_length_m"], 3)
        return row

    if classification != CLASS_ORDINARY:
        return row

    # An independent second opinion from Week 5C-1, on normalised tags, before
    # a single joule is computed. If the two guards ever disagree this raises
    # rather than quietly preferring the permissive one.
    week5c1.assert_ordinary(f"edge ({u}, {v}, {key})", ordinary_guard_tags(data))

    table = week5c1.build_effort_table(gradient["gradients"])
    covered = float(table["integration_weight_m"].sum())
    effort = float(table["incremental_effort_j_per_kg"].sum())
    level = week5c1.LEVEL_COST_J_PER_KG_PER_M * covered

    row["minetti_status"] = MINETTI_VALID
    row["gradient_covered_length_m"] = round(covered, 3)
    row["uncovered_endpoint_length_m"] = (
        round(sampled_length - covered, 3) if sampled_length is not None
        else None
    )
    row["out_of_domain_length_m"] = 0.0
    row["ordinary_effort_j_per_kg"] = round(effort, 4)
    row["terrain_to_level_ratio"] = round(effort / level, 4) if level else None
    return row


AUDIT_COLUMNS: Tuple[str, ...] = (
    "u", "v", "key", "osmid", "length_m", "sampled_length_m",
    "highway", "name", "surface", "incline_osm",
    "bridge", "tunnel", "step_count", "access", "covered", "conveying",
    "special_markers", "classification", "requires_special_handling",
    "has_stored_geometry", "samples", "valid_samples",
    "nodata_or_outside_samples", "dem_status", "dem_tiles_touched",
    "estimates", "eligible_estimates", "rejected_baseline_estimates",
    "finite_estimates", "gradient_status",
    "min_gradient_pct", "max_gradient_pct",
    "sample_spacing_m", "baseline_tolerance_m",
    "min_actual_baseline_m", "max_actual_baseline_m",
    "min_actual_baseline_used_m", "max_actual_baseline_used_m",
    "duplicate_sample_distances",
    "out_of_domain_estimates", "out_of_domain_length_m", "minetti_status",
    "gradient_covered_length_m", "uncovered_endpoint_length_m",
    "ordinary_effort_j_per_kg", "terrain_to_level_ratio", "component_id",
)


def audit_graph(
    graph,
    coverage,
    spacing_m: float = SAMPLE_SPACING_M,
    baseline_m: float = GRADIENT_BASELINE_M,
    limit: Optional[int] = None,
    progress_every: int = 500,
    quiet: bool = False,
) -> pd.DataFrame:
    """Audit every directed edge, in a deterministic order."""
    labels, _sizes = component_labels(graph)
    edges = sorted(graph.edges(keys=True))
    if limit is not None:
        edges = edges[:limit]

    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for index, (u, v, key) in enumerate(edges, start=1):
        rows.append(audit_edge(graph, coverage, u, v, key,
                               labels.get(u, ""), spacing_m, baseline_m))
        if not quiet and progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"  {index}/{len(edges)} directed edges "
                  f"({elapsed:.1f} s, {index / max(elapsed, 1e-9):.1f} edges/s)")

    table = pd.DataFrame(rows)
    for column in AUDIT_COLUMNS:
        if column not in table.columns:
            table[column] = None
    extra = [c for c in table.columns if c not in AUDIT_COLUMNS]
    table = table[list(AUDIT_COLUMNS) + extra]
    table.attrs["elapsed_s"] = time.perf_counter() - started
    return table


def assert_no_special_effort(table: pd.DataFrame) -> None:
    """The invariant this whole file exists to protect.

    Raises:
        ReadinessError: if any edge that is not classified ``ordinary`` carries
            an ordinary walking cost.
    """
    if table.empty:
        return
    offending = table[
        (table["classification"] != CLASS_ORDINARY)
        & table["ordinary_effort_j_per_kg"].notna()
    ]
    if not offending.empty:
        first = offending.iloc[0]
        raise ReadinessError(
            f"{len(offending)} edge(s) that are not classified 'ordinary' "
            f"carry an ordinary Minetti cost, e.g. "
            f"({first['u']}, {first['v']}, {first['key']}) classified "
            f"{first['classification']!r}. Special and unresolved edges must "
            "never receive the ordinary-road model."
        )
    trusted = table[table["classification"] == CLASS_ORDINARY]
    missing = trusted[trusted["ordinary_effort_j_per_kg"].isna()]
    if not missing.empty:
        first = missing.iloc[0]
        raise ReadinessError(
            f"{len(missing)} edge(s) classified 'ordinary' have no cost, e.g. "
            f"({first['u']}, {first['v']}, {first['key']}). An ordinary "
            "classification means the model was applicable; a blank cost "
            "means it was not, and the two must not disagree."
        )


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #
def class_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and lengths, as a share of edges AND of length."""
    total_edges = int(len(table))
    total_length = float(table["length_m"].sum())
    rows: List[Dict[str, Any]] = []
    for classification in CLASS_ORDER:
        part = table[table["classification"] == classification]
        length = float(part["length_m"].sum())
        rows.append({
            "classification": classification,
            "requires_special_handling": classification in UNRESOLVED_CLASSES,
            "edges": int(len(part)),
            "total_length_m": round(length, 2),
            "share_of_edges_pct": round(100.0 * len(part) / total_edges, 3)
            if total_edges else None,
            "share_of_length_pct": round(100.0 * length / total_length, 3)
            if total_length else None,
        })
    return pd.DataFrame(rows)


def marker_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Raw marker prevalence, independent of which class won the priority rule.

    Without this, a footbridge with steps would be invisible in the bridge
    count simply because ``steps`` outranks ``bridge``.
    """
    total_edges = int(len(table))
    total_length = float(table["length_m"].sum())
    known = [marker for marker, _cls in MARKER_PRIORITY] + list(MARKER_ONLY)
    rows: List[Dict[str, Any]] = []
    for marker in known:
        mask = table["special_markers"].apply(
            lambda text, m=marker: has_marker(text, m))
        part = table[mask]
        length = float(part["length_m"].sum())
        classification = dict(MARKER_PRIORITY).get(marker)
        rows.append({
            "marker": marker,
            "is_a_classification": classification is not None,
            "maps_to_classification": classification or "",
            "edges_with_marker": int(len(part)),
            "total_length_m": round(length, 2),
            "share_of_edges_pct": round(100.0 * len(part) / total_edges, 3)
            if total_edges else None,
            "share_of_length_pct": round(100.0 * length / total_length, 3)
            if total_length else None,
            "edges_where_this_marker_won": int(
                (part["classification"] == classification).sum())
            if classification else 0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# route reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_hop(
    graph, a: int, b: int, classification_of: Dict[Tuple[int, int, int], str]
) -> Dict[str, Any]:
    """Work out which edge a Dijkstra hop used, and admit it when we cannot.

    Stage 5 recorded node paths only, so the edge key of each hop is not
    preserved. Where several parallel edges tie for the minimum length, the key
    Dijkstra happened to relax is simply not knowable from what was saved.

    Choosing the smallest key would look tidy and would be a fabrication, so:

    * one minimum -> ``unique``;
    * several minima that agree on readiness -> ``tied_but_equivalent``, safe
      to summarise because either answer gives the same audit outcome;
    * several minima that disagree -> ``parallel_min_ambiguous``, and the stop
      route that contains it cannot be called clean.
    """
    if not graph.has_edge(a, b):
        raise ReadinessError(
            f"The reconstructed path uses a hop {a} -> {b} that is not an edge "
            "of the graph. The path and the graph disagree."
        )
    lengths = {int(k): float(d["length"]) for k, d in graph[a][b].items()}
    minimum = min(lengths.values())
    ties = sorted(k for k, length in lengths.items()
                  if abs(length - minimum) <= LENGTH_TIE_TOLERANCE_M)
    classes = sorted({classification_of[(a, b, k)] for k in ties})

    if len(ties) == 1:
        status = HOP_UNIQUE
        edge: Optional[Tuple[int, int, int]] = (a, b, ties[0])
    elif len(classes) == 1:
        status = HOP_TIED_EQUIVALENT
        edge = (a, b, ties[0])
    else:
        status = HOP_AMBIGUOUS
        edge = None

    return {
        "u": a, "v": b, "length_m": minimum, "status": status,
        "edge": edge, "candidate_keys": ties, "candidate_classes": classes,
    }


def audit_path(
    graph, path: Sequence[int],
    classification_of: Dict[Tuple[int, int, int], str],
) -> Dict[str, Any]:
    """Audit the edges of one reconstructed shortest path."""
    hops = [reconstruct_hop(graph, a, b, classification_of)
            for a, b in zip(path[:-1], path[1:])]
    counts: Dict[str, int] = {}
    lengths: Dict[str, float] = {}
    for hop in hops:
        if hop["status"] == HOP_AMBIGUOUS:
            continue
        classification = classification_of[hop["edge"]]
        counts[classification] = counts.get(classification, 0) + 1
        lengths[classification] = lengths.get(classification, 0.0) + hop["length_m"]

    ambiguous = [h for h in hops if h["status"] == HOP_AMBIGUOUS]
    tied = [h for h in hops if h["status"] == HOP_TIED_EQUIVALENT]
    unresolved = {c: n for c, n in counts.items() if c in UNRESOLVED_CLASSES}
    return {
        "hops": hops,
        "total_length_m": float(sum(h["length_m"] for h in hops)),
        "class_counts": counts,
        "class_lengths": lengths,
        "ambiguous_hops": ambiguous,
        "tied_but_equivalent_hops": tied,
        "unresolved_counts": unresolved,
        "clean_ordinary_only": not unresolved and not ambiguous,
    }


def verify_reconstruction(
    stop_id: Any, audit: Dict[str, Any], path: Sequence[int],
    recorded_distance: float, recorded_nodes: int, recorded_edges: int,
) -> None:
    """Refuse to report on a path that does not match what Stage 5 recorded.

    Raises:
        ReadinessError: on any of the three mismatches. There is no bypass; a
            path that does not reproduce Stage 5 is a different path, and an
            audit of it would describe a route no one ever computed.
    """
    if abs(audit["total_length_m"] - recorded_distance) > DISTANCE_TOLERANCE_M:
        raise ReadinessError(
            f"Stop {stop_id}: the reconstructed path is "
            f"{audit['total_length_m']:.2f} m but Stage 5 recorded "
            f"{recorded_distance:.2f} m. Re-run scripts/route_distances.py; "
            "the routing output and the graph have diverged."
        )
    if len(path) != int(recorded_nodes):
        raise ReadinessError(
            f"Stop {stop_id}: the reconstructed path has {len(path)} nodes but "
            f"Stage 5 recorded {int(recorded_nodes)}."
        )
    if len(audit["hops"]) != int(recorded_edges):
        raise ReadinessError(
            f"Stop {stop_id}: the reconstructed path has {len(audit['hops'])} "
            f"edges but Stage 5 recorded {int(recorded_edges)}."
        )


def audit_stop_routes(
    graph, routes: pd.DataFrame, paths: Dict[int, List[int]],
    classification_of: Dict[Tuple[int, int, int], str],
) -> pd.DataFrame:
    """Audit the Week 4 distance route to every candidate stop.

    Stops with no route keep every audit field blank and say why. A blank is
    not a zero: "no path to audit" and "a path with no special edges" are
    opposite findings and must never be written the same way.
    """
    rows: List[Dict[str, Any]] = []
    for _index, stop in routes.iterrows():
        row: Dict[str, Any] = {
            "stop_id": stop["stop_id"],
            "stop_name": stop.get("stop_name", ""),
            "mode": stop.get("mode", ""),
            "component_id": stop.get("component_id", ""),
            "route_status": stop["route_status"],
            "network_distance_m": stop.get("network_distance_m"),
            "route_node_count": stop.get("route_node_count"),
            "route_edge_count": stop.get("route_edge_count"),
        }
        if stop["route_status"] != STATUS_ROUTED:
            row.update({
                "route_audit_status": ROUTE_NO_PATH,
                "route_audit_note": JUBILEE_NOTE,
                "clean_ordinary_only": pd.NA,
                "audited_edges": pd.NA,
                "ordinary_edges": pd.NA,
                "unresolved_edges": pd.NA,
                "unresolved_classes": "",
                "class_counts": "",
                "class_lengths_m": "",
                "ambiguous_hops": pd.NA,
                "tied_but_equivalent_hops": pd.NA,
                "unresolved_edge_details": "",
            })
            rows.append(row)
            continue

        node = int(stop["nearest_node"])
        path = paths[node]
        audit = audit_path(graph, path, classification_of)
        verify_reconstruction(
            stop["stop_id"], audit, path,
            float(stop["network_distance_m"]),
            int(stop["route_node_count"]), int(stop["route_edge_count"]),
        )

        details = []
        for hop in audit["hops"]:
            if hop["status"] == HOP_AMBIGUOUS:
                details.append(
                    f"{hop['u']}->{hop['v']}:{HOP_AMBIGUOUS}"
                    f"[{','.join(hop['candidate_classes'])}]")
            elif classification_of[hop["edge"]] in UNRESOLVED_CLASSES:
                details.append(
                    f"{hop['edge'][0]}-{hop['edge'][1]}-{hop['edge'][2]}:"
                    f"{classification_of[hop['edge']]}"
                    f"({hop['length_m']:.1f}m)")

        row.update({
            "route_audit_status": ROUTE_AUDITED,
            "route_audit_note": "",
            "clean_ordinary_only": bool(audit["clean_ordinary_only"]),
            "audited_edges": len(audit["hops"]),
            "ordinary_edges": audit["class_counts"].get(CLASS_ORDINARY, 0),
            "unresolved_edges": sum(audit["unresolved_counts"].values()),
            "unresolved_classes": ";".join(sorted(audit["unresolved_counts"])),
            "class_counts": ";".join(
                f"{c}={n}" for c, n in sorted(audit["class_counts"].items())),
            "class_lengths_m": ";".join(
                f"{c}={length:.1f}"
                for c, length in sorted(audit["class_lengths"].items())),
            "ambiguous_hops": len(audit["ambiguous_hops"]),
            "tied_but_equivalent_hops": len(audit["tied_but_equivalent_hops"]),
            "unresolved_edge_details": ";".join(details),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def build_summary(
    table: pd.DataFrame, stop_routes: Optional[pd.DataFrame],
    graph_path: Path, graph_hash: str, dem_paths: Sequence[Path],
    dem_hashes: Dict[str, str], elapsed_s: float,
) -> Dict[str, Any]:
    counts = table["classification"].value_counts().to_dict()
    markers: Dict[str, int] = {}
    for marker in [m for m, _c in MARKER_PRIORITY] + list(MARKER_ONLY):
        markers[marker] = int(table["special_markers"].apply(
            lambda t, m=marker: has_marker(t, m)).sum())

    summary: Dict[str, Any] = {
        "audit": "Week 5C-2A terrain-cost readiness",
        "purpose": (
            "Model readiness only. 'special' and 'unresolved' describe what "
            "this project's ordinary walking-cost model can be applied to. "
            "They are NOT accessibility findings and say nothing about "
            "whether a person can walk somewhere."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "elapsed_seconds": round(elapsed_s, 1),
        "graph": {
            "path": str(graph_path),
            "sha256": graph_hash,
            "topology": "raw",
            "corrections_applied": [],
            "directed_edges": int(len(table)),
            "total_length_m": round(float(table["length_m"].sum()), 2),
        },
        "dem": {
            "tiles": [p.name for p in dem_paths],
            "sha256": dem_hashes,
            "sample_spacing_m": SAMPLE_SPACING_M,
        },
        "method": {
            "gradient_baseline_m": GRADIENT_BASELINE_M,
            "gradient_method": "Week 5B centred difference on raw 1 m samples",
            "minetti_domain": list(MINETTI_DOMAIN),
            "units": "J/kg (no body mass applied)",
            "min_gradient_estimates_required": MIN_GRADIENT_ESTIMATES,
            "baseline_tolerance_rule": (
                "abs(actual_baseline_m - 5.0) <= 0.5 * typical sample spacing; "
                "estimates outside it are counted but never used"
            ),
            "baseline_tolerance_factor": BASELINE_TOLERANCE_FACTOR,
            "directed_edges_audited_independently": True,
            "reverse_edges_paired_by_key": False,
        },
        "class_counts": {c: int(counts.get(c, 0)) for c in CLASS_ORDER},
        "special_marker_counts": markers,
        "ordinary_cost_assigned": int(counts.get(CLASS_ORDINARY, 0)),
        "dem_missing_or_unusable": int(counts.get(CLASS_DEM_MISSING, 0)),
        "too_short_for_5m_gradient": int(counts.get(CLASS_TOO_SHORT, 0)),
        "minetti_out_of_domain": int(counts.get(CLASS_OUT_OF_DOMAIN, 0)),
        "degenerate_sampling_geometry": int(counts.get(CLASS_DEGENERATE, 0)),
        # pandas sums skip missing values, so no substitute is needed here.
        "gradient_estimates": {
            "produced": int(table["estimates"].sum()),
            "eligible_baseline": int(table["eligible_estimates"].sum()),
            "rejected_short_or_irregular_baseline": int(
                table["rejected_baseline_estimates"].sum()),
            "edges_with_a_rejected_estimate": int(
                (table["rejected_baseline_estimates"] > 0).sum()),
            "edges_with_duplicate_sample_distances": int(
                table["duplicate_sample_distances"].eq(True).sum()),
        },
        "dem_status_counts": table["dem_status"].value_counts().to_dict(),
        "gradient_status_counts":
            table["gradient_status"].value_counts().to_dict(),
        "minetti_status_counts":
            table["minetti_status"].value_counts().to_dict(),
    }
    if stop_routes is not None:
        audited = stop_routes[stop_routes["route_audit_status"] == ROUTE_AUDITED]
        summary["stop_routes"] = {
            "stops": int(len(stop_routes)),
            "audited": int(len(audited)),
            "no_path_available_to_audit": int(len(stop_routes) - len(audited)),
            "clean_ordinary_only": int(
                (audited["clean_ordinary_only"] == True).sum()),  # noqa: E712
            "with_unresolved_edges": int(
                (audited["unresolved_edges"] > 0).sum()),
            "with_parallel_min_ambiguity": int(
                (audited["ambiguous_hops"] > 0).sum()),
        }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 5C-2A terrain-cost readiness audit (diagnostic only)."
    )
    parser.add_argument("--dem", nargs="+", default=[DEFAULT_DEM_DIR],
                        help=f"DEM tiles or a directory (default "
                             f"{DEFAULT_DEM_DIR})")
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--baseline-m", type=float, default=GRADIENT_BASELINE_M)
    parser.add_argument("--sample-edges", type=int, default=None,
                        help="audit only the first N edges and write NOTHING; "
                             "use this to measure timing before the full run")
    parser.add_argument("--skip-route-audit", action="store_true",
                        help="edge audit only; the stop-route audit is skipped")
    options = parser.parse_args(argv)

    graph_path = Path(options.graph)
    if not graph_path.is_absolute():
        graph_path = (CONFIG_ROOT / graph_path).resolve()
    if not graph_path.exists():
        print(f"ERROR: {graph_path} is missing. Run "
              "scripts/build_walk_network.py first.", file=sys.stderr)
        return 1

    dem_paths = week5a.discover_tiles(options.dem)
    if not dem_paths:
        print(f"ERROR: no DEM tiles found under {options.dem}.", file=sys.stderr)
        return 1

    print("Week 5C-2A - terrain-cost readiness audit")
    print("A MODEL-READINESS audit. 'special' and 'unresolved' describe what "
          "the")
    print("ordinary walking-cost model can be applied to. They are NOT "
          "accessibility")
    print("findings and no conclusion about walkability may be drawn from "
          "them.")
    print("")

    graph_hash_before = file_sha256(graph_path)
    graph = load_graph(graph_path)
    print(f"Graph      {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} directed edges")
    print(f"           sha256 {graph_hash_before}")
    print(f"DEM        {len(dem_paths)} tile(s), {SAMPLE_SPACING_M:g} m "
          f"sampling")
    print(f"Method     {options.baseline_m:g} m centred gradient (Week 5B), "
          f"Minetti domain {MINETTI_DOMAIN}")
    print("")

    try:
        coverage = week5a.DemCoverage(dem_paths)
    except week5a.DemError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    try:
        if options.sample_edges:
            print(f"TIMING PROBE: auditing the first {options.sample_edges} "
                  "edges. Nothing will be written.")
            table = audit_graph(graph, coverage, SAMPLE_SPACING_M,
                                options.baseline_m, limit=options.sample_edges,
                                progress_every=100)
            elapsed = table.attrs["elapsed_s"]
            rate = len(table) / max(elapsed, 1e-9)
            total = graph.number_of_edges()
            print("")
            print(f"  {len(table)} edges in {elapsed:.1f} s "
                  f"({rate:.1f} edges/s)")
            print(f"  extrapolated full run: {total / rate / 60:.1f} min for "
                  f"{total} edges")
            print("  (an extrapolation from the first edges only; edge lengths "
                  "vary)")
            return 0

        print("Auditing every directed edge independently "
              "(reverse edges are NOT paired by key)...")
        table = audit_graph(graph, coverage, SAMPLE_SPACING_M,
                            options.baseline_m)
        assert_no_special_effort(table)
    finally:
        coverage.close()

    graph_hash_after = file_sha256(graph_path)
    if graph_hash_after != graph_hash_before:
        raise ReadinessError(
            f"{graph_path} changed during the audit "
            f"({graph_hash_before} -> {graph_hash_after}). The Stage 3 graph "
            "is read-only."
        )

    classification_of = {
        (int(r.u), int(r.v), int(r.key)): r.classification
        for r in table.itertuples()
    }

    stop_routes: Optional[pd.DataFrame] = None
    if not options.skip_route_audit:
        cache = CONFIG_ROOT / "data" / "cache"
        try:
            snapping = load_json(cache / "snapping_provenance.json")
            origin = load_json(cache / "snapped_origin.json")
            routes = pd.read_csv(cache / "week4_distance_routes.csv")
            verify_inputs(graph, graph_path, snapping, origin, routes)
        except Exception as error:  # noqa: BLE001 - reported, never bypassed
            print(f"ERROR: the Week 4 routing inputs could not be verified: "
                  f"{error}", file=sys.stderr)
            return 1

        origin_node = int(origin["snap"]["nearest_node"])
        resolve_origin_component(graph, origin_node,
                                 origin["snap"].get("component_id"))
        _distances, paths = single_source_distances(graph, origin_node)
        stop_routes = audit_stop_routes(graph, routes, paths, classification_of)

    out_dir = ensure_directory(cache_dir(OUTPUT_SUBDIR))
    table.to_csv(out_dir / "edge_readiness_audit.csv", index=False)
    classes = class_summary(table)
    classes.to_csv(out_dir / "class_summary.csv", index=False)
    markers = marker_summary(table)
    markers.to_csv(out_dir / "special_marker_summary.csv", index=False)
    if stop_routes is not None:
        stop_routes.to_csv(out_dir / "stop_route_readiness.csv", index=False)

    dem_hashes = {p.name: file_sha256(p) for p in dem_paths}
    summary = build_summary(table, stop_routes, graph_path, graph_hash_after,
                            dem_paths, dem_hashes, table.attrs["elapsed_s"])
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    # ----- console report -------------------------------------------------- #
    print("")
    print("=" * 78)
    print("Edge classification")
    print("=" * 78)
    print(f"{'class':<32}{'edges':>8}{'% edges':>10}{'length m':>12}"
          f"{'% length':>10}")
    for row in classes.itertuples():
        print(f"{row.classification:<32}{row.edges:>8}"
              f"{row.share_of_edges_pct:>10.2f}{row.total_length_m:>12.0f}"
              f"{row.share_of_length_pct:>10.2f}")

    print("")
    print("=" * 78)
    print("Gradient estimates")
    print("=" * 78)
    print(f"  produced by the {options.baseline_m:g} m method   "
          f"{int(table['estimates'].sum())}")
    print(f"  eligible (baseline within tolerance)  "
          f"{int(table['eligible_estimates'].sum())}")
    print(f"  rejected, short or irregular baseline "
          f"{int(table['rejected_baseline_estimates'].sum())}"
          f"  on {int((table['rejected_baseline_estimates'] > 0).sum())} edges")
    print("  Rejected estimates take no part in the gradient range, the domain")
    print("  test or the effort. The length each represented is left uncovered,")
    print("  not filled.")

    print("")
    print("=" * 78)
    print("Special markers, counted independently of the priority rule")
    print("=" * 78)
    for row in markers.itertuples():
        print(f"  {row.marker:<20} on {row.edges_with_marker:>5} edges"
              f"  ({row.share_of_edges_pct:.2f} % of edges)"
              + (f"  -> class {row.maps_to_classification}"
                 if row.is_a_classification else "  -> marker only"))

    if stop_routes is not None:
        audited = stop_routes[stop_routes["route_audit_status"] == ROUTE_AUDITED]
        clean = int((audited["clean_ordinary_only"] == True).sum())  # noqa: E712
        print("")
        print("=" * 78)
        print("Week 4 distance routes to the candidate stops")
        print("=" * 78)
        print(f"  stops in the table                 {len(stop_routes)}")
        print(f"  routes reconstructed and audited   {len(audited)}")
        print(f"  no path available to audit         "
              f"{len(stop_routes) - len(audited)}  (I-04; not 'zero special "
              "edges')")
        print(f"  clean ordinary-only distance route {clean}")
        print(f"  routes touching unresolved edges   "
              f"{int((audited['unresolved_edges'] > 0).sum())}")
        print(f"  routes with parallel ambiguity     "
              f"{int((audited['ambiguous_hops'] > 0).sum())}")

    print("")
    print(f"Wrote data/cache/{OUTPUT_SUBDIR}/ "
          "(edge_readiness_audit.csv, class_summary.csv, "
          "special_marker_summary.csv,")
    print("     stop_route_readiness.csv, audit_summary.json)")
    print(f"The Stage 3 graph is unchanged: sha256 {graph_hash_after}")
    print("")
    print("Readiness only. No routing weight was produced, no stop was ranked,")
    print("and no accessibility conclusion may be drawn from these classes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
