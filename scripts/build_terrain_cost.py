"""Week 5C-2A.2 - the final ordinary terrain-cost attribution.

What this produces
------------------
One terrain effort in J/kg for every directed edge of the raw Glebe pedestrian
graph that the project can honestly cost, plus an explicit reason for every
edge it cannot. This is the table a walking-effort route search would later
read. It is not a route search, and nothing here is written into the graph.

Week 9 addition - walking TIME on the same cells
------------------------------------------------
Every trusted edge now also carries a Tobler walking time, integrated over the
IDENTICAL represented-length cells as the Minetti effort (see
``scripts/tobler_walking_time.py``). Time and effort share one eligibility
gate and one trusted population: an edge with no trusted effort has no time,
and the two are never combined into a score. No Minetti value, no gradient, no
weight and no status rule changed; the time columns are purely additive.

Two changes from the Week 5C-2A audit, both decided on real evidence
--------------------------------------------------------------------
1. ORDINARY EDGES ARE NOW COSTED OVER THEIR WHOLE LENGTH.

   The Week 5C-1 validation integrated only between the first and last gradient
   centres, which leaves about 5 m of every edge uncovered. That was right for
   a validation - it never claimed ground no estimate measured - but summed
   over the network it loses a substantial amount of real walking, so it cannot
   be the final routing weight.

   The final rule gives every eligible 5 m estimate the stretch of edge nearest
   to it. For centres c0..cn on an edge of sampled length L the cell boundaries
   are

       b0 = 0
       bi = (c[i-1] + c[i]) / 2      for i = 1..n
       b[n+1] = L

   and estimate i is weighted by ``b[i+1] - b[i]``. The weights are positive,
   they do not overlap, they leave no gap, and they sum to L exactly. No new
   gradient is created, nothing is smoothed, no elevation is invented and the
   endpoint cells are carried by the local estimates nearest to them.

   Because the end cells extend past the estimates that carry them, each end is
   judged against the OUTERMOST estimate on that side, using the baseline that
   estimate actually achieved:

       first_limit = first_actual_baseline_m / 2 + 0.5 * profile spacing
       last_limit  = last_actual_baseline_m  / 2 + 0.5 * profile spacing

       c0     <= first_limit
       L - cn <= last_limit

   D-18 accepts an estimate whose achieved baseline is within half a sample
   interval of 5 m, so an accepted estimate can span less than 5 m and must not
   be credited with more reach than it has: an accepted 4.5 m estimate licenses
   2.25 + 0.5 = 2.75 m, not 3.00 m.

   The profile spacing is the typical sample spacing MEASURED on that edge -
   the value the reused Week 5B method already reports - not the nominal
   spacing the sampler was asked for. The half-sample term is NOT
   measurement. It is a small, explicit endpoint extrapolation, permitted
   only because the samples are discrete at about a metre. The claim is therefore the weaker one: the outer local gradient may
   represent the endpoint cell only when the endpoint lies within its measured
   half-baseline plus at most half one typical sample interval of
   sampling-quantisation extrapolation. Both terms come from the method and the
   data; neither is a percentage of anything. An edge that fails the guard is
   marked unresolved rather than given a cost over ground neither observed nor
   reasonably extrapolated to.

2. SHORT ORDINARY EDGES USE A WHOLE-EDGE ENDPOINT GRADIENT.

   Adopted on the Week 5C-2A.1 evidence (see D-19). One endpoint gradient
   represents the whole short edge:

       effort = Cw(endpoint gradient) * sampled_length_m

   The multi-centre integration above is NOT applied to it - there is only one
   estimate and it already spans the edge. An edge shorter than 1.5 m stays
   unresolved: 1.5 m is the lower edge of the empirically evaluated 2 m-and-up
   strata in this project's validation, not a universal walking or DEM
   threshold.

History is preserved
--------------------
The Week 5C-2A classification is copied through unchanged.
``too_short_for_5m_gradient`` still records, correctly, that the standard 5 m
method failed on that edge; the new ``terrain_cost_status`` records what was
done about it. Nothing under ``data/cache/terrain_readiness/`` is rewritten.

Still excluded, with no ordinary Minetti cost and no penalty of any kind:
steps, bridge, tunnel, elevator, conveying, corridor, access_restricted.

Usage
-----
    python scripts/build_terrain_cost.py
    python scripts/build_terrain_cost.py --dem data/dem
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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


# The whole upstream chain arrives through one import, so sampling, the 5 m
# gradient, baseline eligibility, the endpoint estimator and the Minetti
# polynomial each keep exactly one implementation.
short_edge = load_script("validate_short_edge_gradient")
audit = short_edge.audit
week5a = audit.week5a
week5b = audit.week5b
week5c1 = audit.week5c1

# Week 9: walking TIME over the same cells. Loaded the same way so the Tobler
# relation also has exactly one implementation. It consumes the (g_i, w_i)
# pairs this script already forms; it never re-samples or re-estimates.
tobler = load_script("tobler_walking_time")


OUTPUT_SUBDIR = "terrain_cost"
AUDIT_SUBDIR = audit.OUTPUT_SUBDIR

GRADIENT_BASELINE_M = audit.GRADIENT_BASELINE_M      # 5.0
SAMPLE_SPACING_M = week5a.SAMPLE_SPACING_M           # 1.0
MINETTI_DOMAIN = week5c1.MINETTI_DOMAIN              # (-0.45, +0.45)
LEVEL_COST = week5c1.LEVEL_COST_J_PER_KG_PER_M       # 2.5

# PROJECT-SPECIFIC. The lower edge of the 2 m-and-longer short-edge strata that
# Week 5C-2A.1 actually evaluated. It is NOT a universal walking threshold and
# NOT a DEM resolution limit; it is the shortest length at which this project
# has evidence for the endpoint fallback.
MIN_SHORT_EDGE_LENGTH_M = 1.5

# Weights must sum to the sampled length to well within a millimetre.
WEIGHT_SUM_TOLERANCE_M = 1e-6

STATUS_ORDINARY_5M = "ordinary_5m"
STATUS_SHORT_FALLBACK = "ordinary_short_endpoint_fallback"
STATUS_SHORT_LOW_CONFIDENCE = "unresolved_short_low_confidence"
STATUS_DOMAIN = "unresolved_minetti_domain"
STATUS_SPECIAL = "special_structure"
STATUS_DEM = "unresolved_dem"
STATUS_ENDPOINT_COVERAGE = "unresolved_endpoint_coverage"
STATUS_NO_GRADIENT = "unresolved_no_usable_gradient"

TRUSTED_STATUSES: Tuple[str, ...] = (STATUS_ORDINARY_5M, STATUS_SHORT_FALLBACK)

STATUS_ORDER: Tuple[str, ...] = (
    STATUS_ORDINARY_5M, STATUS_SHORT_FALLBACK, STATUS_SHORT_LOW_CONFIDENCE,
    STATUS_DOMAIN, STATUS_ENDPOINT_COVERAGE, STATUS_NO_GRADIENT, STATUS_DEM,
    STATUS_SPECIAL,
)

# The historical labels whose reproducibility is guarded per edge. Labels
# that were already DEM-related are not here: for them an incomplete DEM
# is the expected reading, not a change of evidential state.
GUARDED_CLASSIFICATIONS: Tuple[str, ...] = (
    audit.CLASS_ORDINARY, audit.CLASS_OUT_OF_DOMAIN,
    audit.CLASS_TOO_SHORT,
)

METHOD_FULL_LENGTH = "full_length_cell_integration_over_5m_gradients"
METHOD_ENDPOINT = "whole_edge_endpoint_gradient"
METHOD_NONE = "none"

# Week 9 walking-time methods. They name the SAME cells as the effort methods,
# because the time integration runs over the identical weights array.
TIME_METHOD_FULL_LENGTH = "tobler_over_full_length_cells"
TIME_METHOD_ENDPOINT = "tobler_over_whole_edge_endpoint_gradient"


# A final baseline is not a parameter. D-20 is a 5 m construction, and the
# console text, the summary JSON, the endpoint-support limits and the gradients
# themselves must all refer to the same number.
BASELINE_LOCK_TOLERANCE = 1e-9


class TerrainCostError(RuntimeError):
    """Raised with an actionable message when a cost cannot be trusted."""


class ProvenanceMismatch(TerrainCostError):
    """Raised when the historical audit does not belong to these inputs."""


def assert_adopted_baseline(baseline_m: float) -> float:
    """Refuse any baseline but the adopted 5 m one.

    The parameter survives only so tests can prove this refusal happens. There
    is no command-line option for it: a final terrain-cost table built on a
    different baseline would contradict D-20, and would silently disagree with
    the constants the console output and the summary JSON report.

    Raises:
        TerrainCostError: if the baseline is not the adopted one.
    """
    if abs(float(baseline_m) - GRADIENT_BASELINE_M) > BASELINE_LOCK_TOLERANCE:
        raise TerrainCostError(
            f"The final terrain-cost construction is defined on the adopted "
            f"{GRADIENT_BASELINE_M:g} m gradient baseline (D-20), but "
            f"{float(baseline_m)!r} was requested. A table built on another "
            "baseline would disagree with its own reported method."
        )
    return GRADIENT_BASELINE_M


# --------------------------------------------------------------------------- #
# provenance: does the historical audit belong to these inputs?
# --------------------------------------------------------------------------- #
def load_audit_summary(source_dir: Path) -> Dict[str, Any]:
    path = Path(source_dir) / "audit_summary.json"
    if not path.exists():
        raise ProvenanceMismatch(
            f"{path} is missing, so the historical classifications cannot be "
            "tied to any graph or DEM. Run the Week 5C-2A audit first:\n"
            "    python scripts/audit_terrain_readiness.py"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_audit_provenance(
    summary: Dict[str, Any], audit_table: pd.DataFrame, graph,
    graph_path: Path, graph_sha256: str, dem_paths: Sequence[Path],
    sample_spacing_m: float = SAMPLE_SPACING_M,
    baseline_m: float = GRADIENT_BASELINE_M,
) -> Dict[str, Any]:
    """Refuse to reuse a classification that was made about different inputs.

    This script does not re-derive the Week 5C-2A classification; it obeys it.
    An audit produced against another graph or another DEM would attach the
    right-looking words to the wrong edges, and nothing downstream would
    notice. So every link in that chain is checked before a single cost is
    computed, and any break is fatal.

    Checked, in order:

    1. the recorded graph hash equals the graph on disk now;
    2. the recorded directed-edge count equals this graph's;
    3. the (u, v, key) set in the audit table equals this graph's edge set
       exactly - not merely the same size;
    4. the recorded DEM tile names and hashes equal the tiles being used now;
    5. the recorded gradient baseline is the adopted 5 m;
    6. the recorded sample spacing is the adopted spacing.

    Raises:
        ProvenanceMismatch: on any of them. There is no bypass, and nothing is
            regenerated or repaired - the historical audit is evidence, not a
            cache to be refreshed.
    """
    recorded_graph = summary.get("graph") or {}
    recorded_hash = recorded_graph.get("sha256")
    if recorded_hash != graph_sha256:
        raise ProvenanceMismatch(
            f"The Week 5C-2A audit was produced against graph {recorded_hash}, "
            f"but {graph_path} hashes to {graph_sha256}. Its classifications "
            "may refer to different edges. Re-run "
            "scripts/audit_terrain_readiness.py against the current graph."
        )

    recorded_edges = recorded_graph.get("directed_edges")
    if int(recorded_edges or -1) != graph.number_of_edges():
        raise ProvenanceMismatch(
            f"The audit records {recorded_edges} directed edges but the graph "
            f"has {graph.number_of_edges()}."
        )

    if len(audit_table) != graph.number_of_edges():
        raise ProvenanceMismatch(
            f"The audit table has {len(audit_table)} rows but the graph has "
            f"{graph.number_of_edges()} directed edges.")

    keys = list(zip(audit_table["u"], audit_table["v"], audit_table["key"]))
    if len(set(keys)) != len(keys):
        # A set comparison alone cannot see this: duplicates vanish into the
        # set and a duplicated row could coexist with a complete match.
        seen, duplicated = set(), []
        for item in keys:
            if item in seen:
                duplicated.append(item)
            seen.add(item)
        raise ProvenanceMismatch(
            f"The audit table contains {len(duplicated)} duplicated "
            f"(u, v, key) row(s), e.g. {duplicated[0]}. One edge cannot carry "
            "two classifications.")

    graph_edges = {(int(u), int(v), int(k))
                   for u, v, k in graph.edges(keys=True)}
    table_edges = {(int(u), int(v), int(k)) for u, v, k
                   in zip(audit_table["u"], audit_table["v"],
                          audit_table["key"])}
    if table_edges != graph_edges:
        missing = graph_edges - table_edges
        extra = table_edges - graph_edges
        raise ProvenanceMismatch(
            f"The audit table's edge set does not match the graph: "
            f"{len(missing)} graph edge(s) absent from the audit and "
            f"{len(extra)} audit row(s) absent from the graph, e.g. "
            f"{sorted(missing)[:1] or sorted(extra)[:1]}."
        )

    recorded_dem = summary.get("dem") or {}
    recorded_tiles = list(recorded_dem.get("tiles") or [])
    current_tiles = [p.name for p in dem_paths]
    if sorted(recorded_tiles) != sorted(current_tiles):
        raise ProvenanceMismatch(
            f"The audit used DEM tiles {sorted(recorded_tiles)} but "
            f"{sorted(current_tiles)} are being used now."
        )
    recorded_hashes = dict(recorded_dem.get("sha256") or {})
    for path in dem_paths:
        current = audit.file_sha256(path)
        if recorded_hashes.get(path.name) != current:
            raise ProvenanceMismatch(
                f"DEM tile {path.name} hashes to {current}, but the audit "
                f"recorded {recorded_hashes.get(path.name)}. The elevations "
                "behind the historical classifications are not these ones."
            )

    recorded_spacing = recorded_dem.get("sample_spacing_m")
    if recorded_spacing is None or abs(
            float(recorded_spacing) - float(sample_spacing_m)) > 1e-9:
        raise ProvenanceMismatch(
            f"The audit sampled the DEM every {recorded_spacing} m but this "
            f"construction uses {sample_spacing_m} m."
        )

    recorded_baseline = (summary.get("method") or {}).get(
        "gradient_baseline_m")
    if recorded_baseline is None or abs(
            float(recorded_baseline) - float(baseline_m)) > 1e-9:
        raise ProvenanceMismatch(
            f"The audit used a {recorded_baseline} m gradient baseline but "
            f"this construction is defined on {baseline_m} m."
        )

    return {
        "graph_sha256": graph_sha256,
        "graph_sha256_matches_audit": True,
        "directed_edges": int(graph.number_of_edges()),
        "edge_set_matches_audit": True,
        "dem_tiles": current_tiles,
        "dem_sha256_match": True,
        "sample_spacing_m": float(sample_spacing_m),
        "gradient_baseline_m": float(baseline_m),
    }


def assert_classification_reproduces(
    classification: str, record: Dict[str, Any], gradient: Dict[str, Any],
    dem_complete: bool,
) -> None:
    """Check the historical classification still describes what the DEM says.

    A CONSISTENCY CHECK, never a reclassification. If an edge's evidence no
    longer supports the label the audit gave it, this fails loudly rather than
    quietly costing the edge under a meaning it no longer has.

        ordinary                  -> complete DEM, at least two eligible 5 m
                                     gradients, and no Minetti-domain violation
        Minetti_out_of_domain     -> complete DEM, at least two eligible 5 m
                                     gradients, and at least one violation
        too_short_for_5m_gradient -> complete DEM, and the 5 m method still
                                     fails the Week 5C-2A shortness rule
                                     (fewer than two eligible estimates)

    Complete DEM is required by all three because each of those labels was
    produced under complete coverage. A newly incomplete DEM means the
    evidential state has changed, so the label cannot be reproduced - and
    quietly turning it into ``unresolved_dem`` would be exactly the silent
    change of meaning this guard exists to prevent. Historical labels that were
    already DEM-related keep the ordinary unresolved-DEM behaviour; they are
    not passed here.

    Raises:
        TerrainCostError: naming the edge and both readings.
    """
    if classification not in GUARDED_CLASSIFICATIONS:
        return

    where = (f"edge ({record['u']}, {record['v']}, {record['key']}) "
             f"classified {classification!r} by Week 5C-2A")
    eligible = int(gradient.get("eligible_estimates") or 0)
    enough = eligible >= audit.MIN_GRADIENT_ESTIMATES
    violations = int(gradient.get("out_of_domain_estimates") or 0)

    if not dem_complete:
        raise TerrainCostError(
            f"{where} no longer has complete DEM coverage. That label was made "
            "under complete coverage, so it cannot be reproduced now and must "
            "not be silently turned into an unresolved-DEM edge.")

    if classification == audit.CLASS_ORDINARY:
        if not enough:
            raise TerrainCostError(
                f"{where} no longer yields {audit.MIN_GRADIENT_ESTIMATES} "
                f"eligible 5 m gradients (got {eligible}, status "
                f"{gradient.get('gradient_status')!r}).")
        if violations > 0:
            raise TerrainCostError(
                f"{where} now reproduces {violations} Minetti-domain "
                "violation(s). The upstream classification cannot be "
                "reproduced, so its meaning must not be changed silently.")
        return

    if classification == audit.CLASS_OUT_OF_DOMAIN:
        # Passing needs BOTH conditions. Requiring only "violations == 0 while
        # there are enough gradients" would let an edge through whenever the
        # 5 m method had stopped producing usable gradients at all, which is
        # not a reproduction of the label either.
        if not enough:
            raise TerrainCostError(
                f"{where} no longer yields {audit.MIN_GRADIENT_ESTIMATES} "
                f"eligible 5 m gradients (got {eligible}, status "
                f"{gradient.get('gradient_status')!r}), so the domain "
                "violation it was classified for cannot be reproduced.")
        if violations < 1:
            raise TerrainCostError(
                f"{where} now reproduces no Minetti-domain violation at all.")
        return

    if classification == audit.CLASS_TOO_SHORT:
        if enough:
            raise TerrainCostError(
                f"{where} now yields {eligible} eligible 5 m gradients, so it "
                "no longer meets the Week 5C-2A shortness rule. The endpoint "
                "fallback must not be applied to an edge the standard method "
                "can handle.")
        return


def profile_sample_spacing(
    gradient: Dict[str, Any], record: Dict[str, Any]
) -> float:
    """The typical sample spacing MEASURED on this profile.

    D-18 and D-20 both describe the endpoint allowance as half the profile's
    own typical sample spacing, so the guard must use that rather than the
    nominal spacing the sampler was asked for. The reused Week 5B method
    already reports it.

    There is deliberately no fallback to 1.0 m: on an edge that claims usable
    eligible gradients, a missing or nonsensical measured spacing means the
    tolerance cannot be derived from the data at all, and substituting the
    nominal value would make the guard claim a provenance it does not have.

    Raises:
        TerrainCostError: if the measured spacing is absent or not positive.
    """
    spacing = gradient.get("sample_spacing_m")
    if spacing is None or not np.isfinite(float(spacing)) or float(spacing) <= 0:
        raise TerrainCostError(
            f"edge ({record['u']}, {record['v']}, {record['key']}) has usable "
            f"eligible 5 m gradients but no measured sample spacing "
            f"({spacing!r}). The endpoint allowance is half the profile's own "
            "typical spacing and must not fall back to the nominal value."
        )
    return float(spacing)


# --------------------------------------------------------------------------- #
# full-length cell integration
# --------------------------------------------------------------------------- #
def cell_boundaries(
    centres: Sequence[float], sampled_length_m: float
) -> np.ndarray:
    """Boundaries of the stretch of edge each gradient estimate represents.

    Midway between neighbouring centres, with the edge's own ends closing the
    first and last cells. This is what makes the cost cover the whole edge
    instead of only the span between the outermost estimates.
    """
    values = np.asarray(centres, dtype=float)
    length = float(sampled_length_m)
    if values.size == 0:
        raise TerrainCostError("No gradient estimate to integrate.")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise TerrainCostError(
            "Gradient centres must be strictly ascending to be integrated.")
    if length <= 0:
        raise TerrainCostError(
            f"The sampled edge length must be positive, got {length}.")
    if values[0] <= 0 or values[-1] >= length:
        # A centre off the end of the edge would still yield positive cells
        # that sum to the length, because the boundary formula clips them - so
        # this has to be checked directly. An estimate that is not on this edge
        # cannot be given a stretch of it to carry.
        raise TerrainCostError(
            f"Gradient centres must lie strictly inside the edge (0, "
            f"{length:.4f}) m, but they run from {values[0]:.4f} to "
            f"{values[-1]:.4f} m."
        )
    interior = (values[:-1] + values[1:]) / 2.0 if values.size > 1 \
        else np.array([], dtype=float)
    return np.concatenate(([0.0], interior, [float(sampled_length_m)]))


def full_length_weights(
    centres: Sequence[float], sampled_length_m: float
) -> np.ndarray:
    """The length of edge each gradient estimate is responsible for.

    Returns:
        One weight per estimate, all positive, summing exactly to
        ``sampled_length_m``. Neighbouring cells share a boundary, so there is
        no overlap and no gap.

    Raises:
        TerrainCostError: if a weight would be non-positive, or the weights do
            not sum to the sampled length. Either means the estimates do not
            sit sensibly on this edge, and a cost built from them would be
            describing the wrong ground.
    """
    boundaries = cell_boundaries(centres, sampled_length_m)
    weights = np.diff(boundaries)
    if np.any(weights <= 0):
        raise TerrainCostError(
            "A represented cell has zero or negative length. The gradient "
            "centres do not lie inside this edge in ascending order."
        )
    total = float(weights.sum())
    if abs(total - float(sampled_length_m)) > WEIGHT_SUM_TOLERANCE_M:
        raise TerrainCostError(
            f"The represented cells sum to {total:.9f} m but the edge is "
            f"{float(sampled_length_m):.9f} m. Full-length integration must "
            "cover the edge exactly once."
        )
    return weights


def endpoint_support_limit(
    actual_baseline_m: float,
    sample_spacing_m: float = SAMPLE_SPACING_M,
) -> float:
    """How far an outer estimate's own end cell may reach beyond what it measured.

    An estimate centred at ``c`` with a MEASURED baseline of ``b`` spans
    ``c +/- b/2``: that is ground it actually observed. The extra half sample
    interval is not measurement. It is a small, explicit endpoint
    extrapolation, permitted only because the samples are discrete at about a
    metre and a centre can therefore only be placed to the nearest sample.

    So the claim this limit licenses is deliberately weaker than "the outer
    estimate measured all the way to the edge's end". It is: the outer local
    gradient may represent the endpoint cell only when the endpoint lies within
    its measured half-baseline plus at most half one typical sample interval of
    sampling-quantisation extrapolation.

    Both terms come from the method and the data. Neither is a percentage of
    anything.

    Args:
        actual_baseline_m: the baseline that estimate ACHIEVED, as reported by
            the Week 5B method - not the 5 m that was requested. D-18 accepts
            an estimate whose achieved baseline is within half a sample
            interval of 5 m, so an accepted estimate can genuinely span less
            than 5 m and must not be credited with more reach than it has.
        sample_spacing_m: the profile's typical sample spacing.
    """
    return float(actual_baseline_m) / 2.0 + audit.baseline_tolerance(
        sample_spacing_m)


def endpoint_cells_are_supported(
    gradients: pd.DataFrame, sampled_length_m: float,
    sample_spacing_m: float = SAMPLE_SPACING_M,
) -> Dict[str, Any]:
    """Whether each end cell is within reach of the estimate that carries it.

    The end cells extend beyond the estimates that carry them. Each end is
    therefore judged against the OUTERMOST estimate on that side, using the
    baseline that estimate actually achieved - so an accepted 4.5 m estimate
    licenses 2.25 + 0.5 = 2.75 m of reach, not the 3.00 m a full 5 m estimate
    would.

    An edge whose end lies beyond that is left unresolved rather than given a
    cost over ground nothing observed and nothing may reasonably extrapolate to.
    """
    centres = gradients["centre_distance_m"].to_numpy(dtype=float)
    baselines = gradients["actual_baseline_m"].to_numpy(dtype=float)
    first_baseline = float(baselines[0])
    last_baseline = float(baselines[-1])
    first_limit = endpoint_support_limit(first_baseline, sample_spacing_m)
    last_limit = endpoint_support_limit(last_baseline, sample_spacing_m)
    first_gap = float(centres[0])
    last_gap = float(sampled_length_m) - float(centres[-1])
    return {
        "first_actual_baseline_m": round(first_baseline, 4),
        "last_actual_baseline_m": round(last_baseline, 4),
        "first_endpoint_support_limit_m": round(first_limit, 4),
        "last_endpoint_support_limit_m": round(last_limit, 4),
        "first_estimate_gap_m": round(first_gap, 4),
        "last_estimate_gap_m": round(last_gap, 4),
        "supported": bool(first_gap <= first_limit + WEIGHT_SUM_TOLERANCE_M
                          and last_gap <= last_limit + WEIGHT_SUM_TOLERANCE_M),
    }


def integrate_full_length(
    gradients: pd.DataFrame, sampled_length_m: float
) -> Dict[str, Any]:
    """Terrain effort over the whole edge, from the existing 5 m estimates.

    No estimate is added, removed, smoothed or shifted: only the weight each
    one carries changes, from the validation's between-centres span to the
    stretch of edge nearest to it.
    """
    centres = gradients["centre_distance_m"].to_numpy(dtype=float)
    slopes = gradients["gradient"].to_numpy(dtype=float)
    weights = full_length_weights(centres, sampled_length_m)
    costs = week5c1.minetti_cost_array(slopes)
    increments = costs * weights
    # Week 9: the walking time is integrated over the SAME weights. One cells
    # array, two integrations - so time and effort cannot cover the edge
    # differently, and the Minetti sum above is untouched.
    timing = tobler.walking_time_over_cells(slopes, weights)
    return {
        "weights_m": weights,
        "estimates": int(len(centres)),
        "covered_length_m": float(weights.sum()),
        "effort_j_per_kg": float(np.sum(increments)),
        "min_weight_m": float(weights.min()),
        "max_weight_m": float(weights.max()),
        "walking_time_s": float(timing["walking_time_s"]),
        "min_speed_ms": float(timing["min_speed_ms"]),
        "max_speed_ms": float(timing["max_speed_ms"]),
        "mean_speed_ms": timing["mean_speed_ms"],
    }


# --------------------------------------------------------------------------- #
# the short-edge endpoint fallback
# --------------------------------------------------------------------------- #
def short_edge_effort(
    profile: pd.DataFrame, sampled_length_m: float
) -> Dict[str, Any]:
    """The adopted fallback: one endpoint gradient over the whole short edge.

    The estimator is Week 5C-2A.1's, unchanged, so the thing adopted is the
    thing that was validated. The multi-centre integration is deliberately not
    used here: there is one estimate and it already spans the edge.
    """
    estimate = short_edge.endpoint_gradient(profile)
    gradient = estimate["endpoint_gradient"]
    result: Dict[str, Any] = {
        "endpoint_gradient": gradient,
        "endpoint_run_m": estimate["endpoint_run_m"],
        "endpoint_valid_samples": estimate["endpoint_valid_samples"],
        "in_minetti_domain": None,
        "effort_j_per_kg": None,
        "walking_time_s": None,
        "speed_ms": None,
    }
    if gradient is None or not np.isfinite(gradient):
        return result
    result["in_minetti_domain"] = bool(week5c1.in_domain(gradient))
    if result["in_minetti_domain"]:
        result["effort_j_per_kg"] = float(
            week5c1.minetti_cost(gradient) * float(sampled_length_m))
        # Week 9: same gate, same single estimate, same length. The time is
        # only formed where the effort is, so the trusted population is one.
        timing = tobler.walking_time_whole_edge(gradient, sampled_length_m)
        result["walking_time_s"] = float(timing["walking_time_s"])
        result["speed_ms"] = float(timing["mean_speed_ms"])
    return result


# --------------------------------------------------------------------------- #
# costing one edge
# --------------------------------------------------------------------------- #
COST_COLUMNS: Tuple[str, ...] = (
    "u", "v", "key", "edge_id", "name", "highway", "special_markers",
    "week5c2a_classification", "terrain_cost_status", "cost_method",
    "length_m", "sampled_length_m", "samples", "valid_samples", "dem_status",
    "gradient_estimates", "eligible_estimates",
    "integration_covered_length_m", "integration_coverage_pct",
    "min_cell_length_m", "max_cell_length_m",
    "measured_sample_spacing_m",
    "first_actual_baseline_m", "last_actual_baseline_m",
    "first_endpoint_support_limit_m", "last_endpoint_support_limit_m",
    "first_estimate_gap_m", "last_estimate_gap_m",
    "endpoint_fallback_gradient", "endpoint_fallback_gradient_pct",
    "min_gradient_pct", "max_gradient_pct",
    "minetti_domain_status", "out_of_domain_estimates",
    "terrain_effort_j_per_kg", "mean_cost_j_per_kg_per_m",
    "terrain_to_level_ratio",
    # Week 9 walking time. Present only where terrain_cost_status is trusted;
    # None everywhere else, by the same rule as the effort columns.
    "walking_time_s", "walking_time_min", "walking_time_method",
    "mean_walking_speed_ms", "min_local_speed_ms", "max_local_speed_ms",
    "time_to_level_ratio",
    "unresolved_reason", "component_id",
)

DOMAIN_OK = "inside"
DOMAIN_VIOLATED = "outside"
DOMAIN_NOT_TESTED = "not_tested"


def cost_edge(
    graph, coverage, row: pd.Series,
    baseline_m: float = GRADIENT_BASELINE_M,
    spacing_m: float = SAMPLE_SPACING_M,
) -> Dict[str, Any]:
    """Decide one edge's terrain-cost status and, when trusted, its cost.

    The Week 5C-2A classification is carried through untouched and decides the
    branch. Everything numerical is re-derived from the DEM here rather than
    read back out of the audit table, so the cost cannot drift away from the
    evidence it claims to rest on.
    """
    baseline_m = assert_adopted_baseline(baseline_m)
    u, v, key = int(row["u"]), int(row["v"]), int(row["key"])
    classification = str(row["classification"])
    record: Dict[str, Any] = {
        "u": u, "v": v, "key": key,
        "edge_id": short_edge.edge_identifier(u, v, key),
        "name": row.get("name", ""),
        "highway": row.get("highway", ""),
        "special_markers": row.get("special_markers", ""),
        "week5c2a_classification": classification,
        "length_m": row.get("length_m"),
        "component_id": row.get("component_id"),
        "terrain_cost_status": None,
        "cost_method": METHOD_NONE,
        "sampled_length_m": None, "samples": None, "valid_samples": None,
        "dem_status": row.get("dem_status"),
        "gradient_estimates": None, "eligible_estimates": None,
        "integration_covered_length_m": None,
        "integration_coverage_pct": None,
        "min_cell_length_m": None, "max_cell_length_m": None,
        "measured_sample_spacing_m": None,
        "first_actual_baseline_m": None, "last_actual_baseline_m": None,
        "first_endpoint_support_limit_m": None,
        "last_endpoint_support_limit_m": None,
        "first_estimate_gap_m": None, "last_estimate_gap_m": None,
        "endpoint_fallback_gradient": None,
        "endpoint_fallback_gradient_pct": None,
        "min_gradient_pct": None, "max_gradient_pct": None,
        "minetti_domain_status": DOMAIN_NOT_TESTED,
        "out_of_domain_estimates": None,
        "terrain_effort_j_per_kg": None,
        "mean_cost_j_per_kg_per_m": None,
        "terrain_to_level_ratio": None,
        "walking_time_s": None, "walking_time_min": None,
        "walking_time_method": METHOD_NONE,
        "mean_walking_speed_ms": None,
        "min_local_speed_ms": None, "max_local_speed_ms": None,
        "time_to_level_ratio": None,
        "unresolved_reason": "",
    }

    # ----- special structures leave immediately, uncosted and unpenalised -- #
    if classification in audit.SPECIAL_CLASSES:
        record["terrain_cost_status"] = STATUS_SPECIAL
        record["unresolved_reason"] = (
            f"{classification}: outside the ordinary walking-cost model. No "
            "ordinary Minetti cost and no penalty is applied.")
        return record

    profile = week5a.build_profile(graph, coverage, u, v, key, spacing_m)
    sampled_length = float(profile["distance_m"].max())
    distances, _elevations = short_edge.valid_samples(profile)
    record.update({
        "sampled_length_m": round(sampled_length, 3),
        "samples": int(len(profile)),
        "valid_samples": int(distances.size),
    })

    dem_complete = int(distances.size) == int(len(profile))

    # The 5 m method is run for EVERY non-special edge, including the short
    # ones, so the historical classification can be checked against what the
    # DEM says today before anything is costed under it - and BEFORE an
    # incomplete DEM could quietly become an unresolved-DEM edge.
    gradient = audit.gradient_for_profile(profile, baseline_m)
    assert_classification_reproduces(classification, record, gradient,
                                     dem_complete)
    record["gradient_estimates"] = gradient["estimates"]
    record["eligible_estimates"] = gradient["eligible_estimates"]

    if not dem_complete:
        # Only reachable for a label that was already DEM-related; a guarded
        # label with an incomplete DEM raised above.
        record["terrain_cost_status"] = STATUS_DEM
        record["unresolved_reason"] = (
            f"{int(len(profile)) - int(distances.size)} of "
            f"{int(len(profile))} DEM samples are missing. Nothing is "
            "interpolated, so no trusted cost is assigned.")
        return record

    # ----- the adopted short-edge fallback -------------------------------- #
    if classification == audit.CLASS_TOO_SHORT:
        if sampled_length < MIN_SHORT_EDGE_LENGTH_M:
            record["terrain_cost_status"] = STATUS_SHORT_LOW_CONFIDENCE
            record["unresolved_reason"] = (
                f"sampled length {sampled_length:.2f} m is below the "
                f"{MIN_SHORT_EDGE_LENGTH_M:g} m lower bound of the strata "
                "Week 5C-2A.1 evaluated. This is a project-specific evidence "
                "boundary, not a walking or DEM threshold.")
            return record

        fallback = short_edge_effort(profile, sampled_length)
        record["endpoint_fallback_gradient"] = fallback["endpoint_gradient"]
        if fallback["endpoint_gradient"] is not None:
            record["endpoint_fallback_gradient_pct"] = round(
                fallback["endpoint_gradient"] * 100.0, 4)
            record["min_gradient_pct"] = record["endpoint_fallback_gradient_pct"]
            record["max_gradient_pct"] = record["endpoint_fallback_gradient_pct"]
        if fallback["endpoint_gradient"] is None:
            record["terrain_cost_status"] = STATUS_SHORT_LOW_CONFIDENCE
            record["unresolved_reason"] = (
                "fewer than two valid DEM samples, so no endpoint gradient "
                "exists for this edge.")
            return record

        record["minetti_domain_status"] = (
            DOMAIN_OK if fallback["in_minetti_domain"] else DOMAIN_VIOLATED)
        record["out_of_domain_estimates"] = int(
            not fallback["in_minetti_domain"])
        if not fallback["in_minetti_domain"]:
            record["terrain_cost_status"] = STATUS_DOMAIN
            record["unresolved_reason"] = (
                f"endpoint gradient {fallback['endpoint_gradient'] * 100:+.2f}"
                f" % lies outside the studied Minetti domain {MINETTI_DOMAIN}."
                " It is neither clamped nor extrapolated.")
            return record

        effort = float(fallback["effort_j_per_kg"])
        record.update({
            "terrain_cost_status": STATUS_SHORT_FALLBACK,
            "cost_method": METHOD_ENDPOINT,
            "integration_covered_length_m": round(sampled_length, 3),
            "integration_coverage_pct": 100.0,
            "min_cell_length_m": round(sampled_length, 3),
            "max_cell_length_m": round(sampled_length, 3),
            "terrain_effort_j_per_kg": round(effort, 4),
            "mean_cost_j_per_kg_per_m": round(effort / sampled_length, 5)
            if sampled_length else None,
            "terrain_to_level_ratio": round(
                effort / (LEVEL_COST * sampled_length), 4)
            if sampled_length else None,
        })
        record.update(_time_fields(
            fallback["walking_time_s"], sampled_length,
            fallback["speed_ms"], fallback["speed_ms"], fallback["speed_ms"],
            TIME_METHOD_ENDPOINT))
        return record

    # ----- everything else goes through the 5 m method --------------------- #
    eligible = gradient["gradients"]
    if eligible is None or len(eligible) < audit.MIN_GRADIENT_ESTIMATES:
        record["terrain_cost_status"] = STATUS_NO_GRADIENT
        record["unresolved_reason"] = (
            f"the 5 m method produced no usable gradient "
            f"({gradient['gradient_status']}), and this edge is not in the "
            "short-edge fallback population.")
        return record

    slopes = eligible["gradient"].to_numpy(dtype=float)
    record["min_gradient_pct"] = round(float(np.min(slopes) * 100.0), 3)
    record["max_gradient_pct"] = round(float(np.max(slopes) * 100.0), 3)
    inside = week5c1.in_domain_array(slopes)
    record["out_of_domain_estimates"] = int((~inside).sum())
    record["minetti_domain_status"] = (
        DOMAIN_OK if inside.all() else DOMAIN_VIOLATED)

    measured_spacing = profile_sample_spacing(gradient, record)
    record["measured_sample_spacing_m"] = round(measured_spacing, 4)
    support = endpoint_cells_are_supported(eligible, sampled_length,
                                           measured_spacing)
    record.update({key: support[key] for key in (
        "first_actual_baseline_m", "last_actual_baseline_m",
        "first_endpoint_support_limit_m", "last_endpoint_support_limit_m",
        "first_estimate_gap_m", "last_estimate_gap_m")})

    if not inside.all():
        record["terrain_cost_status"] = STATUS_DOMAIN
        record["unresolved_reason"] = (
            f"{int((~inside).sum())} of {len(slopes)} eligible 5 m gradients "
            f"lie outside the studied Minetti domain {MINETTI_DOMAIN}. Nothing "
            "is clamped and no extrapolated value is trusted.")
        return record

    if not support["supported"]:
        record["terrain_cost_status"] = STATUS_ENDPOINT_COVERAGE
        record["unresolved_reason"] = (
            f"the outermost estimates sit "
            f"{support['first_estimate_gap_m']} m and "
            f"{support['last_estimate_gap_m']} m from the edge's ends, "
            f"beyond the {support['first_endpoint_support_limit_m']} m and "
            f"{support['last_endpoint_support_limit_m']} m their measured "
            f"baselines of {support['first_actual_baseline_m']} m and "
            f"{support['last_actual_baseline_m']} m reach, allowing half a "
            "sample interval of sampling-quantisation extrapolation. The end "
            "cells would cover ground neither observed nor reasonably "
            "extrapolated to.")
        return record

    try:
        integration = integrate_full_length(eligible, sampled_length)
    except TerrainCostError as error:
        record["terrain_cost_status"] = STATUS_NO_GRADIENT
        record["unresolved_reason"] = str(error)
        return record

    effort = integration["effort_j_per_kg"]
    record.update({
        "terrain_cost_status": STATUS_ORDINARY_5M,
        "cost_method": METHOD_FULL_LENGTH,
        "integration_covered_length_m": round(
            integration["covered_length_m"], 3),
        "integration_coverage_pct": round(
            100.0 * integration["covered_length_m"] / sampled_length, 4)
        if sampled_length else None,
        "min_cell_length_m": round(integration["min_weight_m"], 4),
        "max_cell_length_m": round(integration["max_weight_m"], 4),
        "terrain_effort_j_per_kg": round(effort, 4),
        "mean_cost_j_per_kg_per_m": round(effort / sampled_length, 5)
        if sampled_length else None,
        "terrain_to_level_ratio": round(
            effort / (LEVEL_COST * sampled_length), 4) if sampled_length
        else None,
    })
    record.update(_time_fields(
        integration["walking_time_s"], sampled_length,
        integration["mean_speed_ms"], integration["min_speed_ms"],
        integration["max_speed_ms"], TIME_METHOD_FULL_LENGTH))
    return record


def _time_fields(
    walking_time_s: float, sampled_length_m: float,
    mean_speed_ms: Optional[float], min_speed_ms: float, max_speed_ms: float,
    method: str,
) -> Dict[str, Any]:
    """The Week 9 walking-time columns for one trusted edge.

    ``time_to_level_ratio`` is this edge's time over the time the same length
    would take on the level at Tobler's level speed - the time analogue of
    ``terrain_to_level_ratio``.
    """
    level_time = (float(sampled_length_m) / tobler.LEVEL_SPEED_MS
                  if sampled_length_m else None)
    seconds = round(float(walking_time_s), 4)
    return {
        "walking_time_s": seconds,
        "walking_time_min": round(seconds / 60.0, 5),
        "walking_time_method": method,
        "mean_walking_speed_ms": round(float(mean_speed_ms), 5)
        if mean_speed_ms is not None else None,
        "min_local_speed_ms": round(float(min_speed_ms), 5),
        "max_local_speed_ms": round(float(max_speed_ms), 5),
        "time_to_level_ratio": round(float(walking_time_s) / level_time, 4)
        if level_time else None,
    }


# --------------------------------------------------------------------------- #
# invariants over the finished table
# --------------------------------------------------------------------------- #
def assert_cost_invariants(table: pd.DataFrame) -> None:
    """Refuse to write a table that contradicts the adopted policy.

    Raises:
        TerrainCostError: on any of the four ways this could go wrong - a cost
            on an untrusted edge, a trusted edge with no cost, a special
            structure carrying an ordinary cost, or an ordinary edge whose
            cells do not cover it exactly once.
    """
    if table.empty:
        return
    costed = table["terrain_effort_j_per_kg"].notna()
    trusted = table["terrain_cost_status"].isin(TRUSTED_STATUSES)

    stray = table[costed & ~trusted]
    if not stray.empty:
        first = stray.iloc[0]
        raise TerrainCostError(
            f"{len(stray)} edge(s) carry a terrain cost without a trusted "
            f"status, e.g. ({first['u']}, {first['v']}, {first['key']}) with "
            f"status {first['terrain_cost_status']!r}.")

    missing = table[trusted & ~costed]
    if not missing.empty:
        first = missing.iloc[0]
        raise TerrainCostError(
            f"{len(missing)} edge(s) have a trusted status but no cost, e.g. "
            f"({first['u']}, {first['v']}, {first['key']}).")

    special = table[table["week5c2a_classification"].isin(
        audit.SPECIAL_CLASSES) & costed]
    if not special.empty:
        first = special.iloc[0]
        raise TerrainCostError(
            f"{len(special)} special structure(s) carry an ordinary terrain "
            f"cost, e.g. ({first['u']}, {first['v']}, {first['key']}) "
            f"classified {first['week5c2a_classification']!r}.")

    # Week 9: walking time follows the effort exactly - present on every
    # trusted edge, absent on every other, and never zero or negative.
    if "walking_time_s" in table.columns:
        timed = table["walking_time_s"].notna()
        if (timed != costed).any():
            first = table[timed != costed].iloc[0]
            raise TerrainCostError(
                f"{int((timed != costed).sum())} edge(s) have a walking time "
                f"without an effort or vice versa, e.g. ({first['u']}, "
                f"{first['v']}, {first['key']}). Time and effort must share "
                "one trusted population.")
        bad = table[timed & ~(table["walking_time_s"] > 0)]
        if not bad.empty:
            first = bad.iloc[0]
            raise TerrainCostError(
                f"{len(bad)} edge(s) carry a non-positive walking time, e.g. "
                f"({first['u']}, {first['v']}, {first['key']}).")

    ordinary = table[table["terrain_cost_status"] == STATUS_ORDINARY_5M]
    gap = (ordinary["integration_covered_length_m"]
           - ordinary["sampled_length_m"]).abs()
    if not gap.empty and float(gap.max()) > 1e-3:
        worst = ordinary.loc[gap.idxmax()]
        raise TerrainCostError(
            f"Full-length integration does not cover edge "
            f"({worst['u']}, {worst['v']}, {worst['key']}) exactly: "
            f"{worst['integration_covered_length_m']} m of "
            f"{worst['sampled_length_m']} m.")


def build_cost_table(
    graph, coverage, audit_table: pd.DataFrame,
    baseline_m: float = GRADIENT_BASELINE_M,
    spacing_m: float = SAMPLE_SPACING_M,
    limit: Optional[int] = None, progress_every: int = 500,
    quiet: bool = False,
) -> pd.DataFrame:
    """Cost every directed edge in the audit table, in a deterministic order."""
    baseline_m = assert_adopted_baseline(baseline_m)
    ordered = audit_table.sort_values(["u", "v", "key"]).reset_index(drop=True)
    if limit is not None:
        ordered = ordered.iloc[:limit]
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for index, (_position, row) in enumerate(ordered.iterrows(), start=1):
        rows.append(cost_edge(graph, coverage, row, baseline_m, spacing_m))
        if not quiet and progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"  {index}/{len(ordered)} directed edges ({elapsed:.1f} s)")
    table = pd.DataFrame(rows)
    for column in COST_COLUMNS:
        if column not in table.columns:
            table[column] = None
    extra = [c for c in table.columns if c not in COST_COLUMNS]
    table = table[list(COST_COLUMNS) + extra]
    table.attrs["elapsed_s"] = time.perf_counter() - started
    return table


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def status_summary(table: pd.DataFrame) -> pd.DataFrame:
    total_edges = int(len(table))
    total_length = float(table["length_m"].sum())
    rows: List[Dict[str, Any]] = []
    for status in STATUS_ORDER:
        part = table[table["terrain_cost_status"] == status]
        length = float(part["length_m"].sum())
        rows.append({
            "terrain_cost_status": status,
            "trusted": status in TRUSTED_STATUSES,
            "edges": int(len(part)),
            "total_length_m": round(length, 2),
            "share_of_edges_pct": round(100.0 * len(part) / total_edges, 3)
            if total_edges else None,
            "share_of_length_pct": round(100.0 * length / total_length, 3)
            if total_length else None,
            "total_effort_j_per_kg": round(
                float(part["terrain_effort_j_per_kg"].sum()), 2),
            "total_walking_time_s": round(
                float(part["walking_time_s"].sum()), 2)
            if "walking_time_s" in part.columns else None,
        })
    return pd.DataFrame(rows)


def build_summary(
    table: pd.DataFrame, graph_path: Path, graph_hash: str,
    dem_paths: Sequence[Path], elapsed_s: float,
) -> Dict[str, Any]:
    trusted = table[table["terrain_cost_status"].isin(TRUSTED_STATUSES)]
    total_length = float(table["length_m"].sum())
    return {
        "stage": "Week 5C-2A.2 final ordinary terrain-cost attribution",
        "purpose": (
            "One terrain effort in J/kg per directed edge where the project "
            "can honestly assign one, and an explicit reason everywhere else. "
            "Diagnostic table only: nothing is written into the graph, no "
            "route is computed and no stop is ranked. Statuses describe this "
            "model's reach, not whether anywhere can be walked."),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "elapsed_seconds": round(elapsed_s, 1),
        "graph": {"path": str(graph_path), "sha256": graph_hash,
                  "topology": "raw", "corrections_applied": [],
                  "directed_edges": int(len(table)),
                  "total_length_m": round(total_length, 2)},
        "dem_tiles": [p.name for p in dem_paths],
        "method": {
            "ordinary": (
                "full-length cell integration over the eligible 5 m centred "
                "gradients; cell boundaries midway between neighbouring "
                "centres, closed by the edge's own ends (D-20)"),
            "short_edge_fallback": (
                "whole-edge endpoint gradient over the actual sampled length, "
                "for edges >= "
                f"{MIN_SHORT_EDGE_LENGTH_M:g} m (D-19)"),
            "gradient_baseline_m": GRADIENT_BASELINE_M,
            "sample_spacing_m": SAMPLE_SPACING_M,
            "minetti_domain": list(MINETTI_DOMAIN),
            "units": "J/kg (no body mass applied)",
            "short_edge_minimum_length_m": MIN_SHORT_EDGE_LENGTH_M,
            "short_edge_minimum_length_note": (
                "PROJECT-SPECIFIC: the lower edge of the 2 m-and-longer "
                "strata Week 5C-2A.1 evaluated. Not a universal walking "
                "threshold and not a DEM resolution limit."),
            "endpoint_support_limit_rule": (
                "each end is judged against the OUTERMOST estimate on that "
                "side: its own achieved baseline / 2, plus half the typical "
                "sample spacing MEASURED on that profile by the reused "
                "Week 5B method. The second term is not measurement - it is "
                "a small, explicit endpoint extrapolation permitted because "
                "the samples are discrete at about a metre. The outer local "
                "gradient may represent the endpoint cell only when the "
                "endpoint lies within its measured half-baseline plus at "
                "most half one typical sample interval."),
            "endpoint_support_limit_for_a_full_5m_estimate_m": round(
                endpoint_support_limit(GRADIENT_BASELINE_M), 4),
            "special_structures_excluded": list(audit.SPECIAL_CLASSES),
            "stairs_penalty": "none - no penalty of any kind is applied",
        },
        "status_counts": table["terrain_cost_status"].value_counts().to_dict(),
        "week5c2a_classification_counts":
            table["week5c2a_classification"].value_counts().to_dict(),
        "trusted": {
            "edges": int(len(trusted)),
            "share_of_edges_pct": round(100.0 * len(trusted) / len(table), 3)
            if len(table) else None,
            "length_m": round(float(trusted["length_m"].sum()), 2),
            "share_of_length_pct": round(
                100.0 * float(trusted["length_m"].sum()) / total_length, 3)
            if total_length else None,
            "total_effort_j_per_kg": round(
                float(trusted["terrain_effort_j_per_kg"].sum()), 2),
            "by_method": trusted["cost_method"].value_counts().to_dict(),
        },
        "coverage_gain_over_validation_integration": (
            "The Week 5C-1 validation covered only the span between the first "
            "and last gradient centres. Full-length integration covers the "
            "whole sampled edge, so a routing weight no longer omits about "
            "5 m of every ordinary edge."),
        "not_done": [
            "no effort Dijkstra", "no stop ranking",
            "no change to the interactive demo", "no stairs penalty",
            "OSM-FIX-001 not applied", "raw GraphML not modified",
            "Week 5A/5B/5C-1/5C-2A outputs not rewritten",
        ],
    }


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 5C-2A.2 final ordinary terrain-cost attribution.")
    parser.add_argument("--dem", nargs="+", default=[audit.DEFAULT_DEM_DIR])
    parser.add_argument("--graph", default=audit.DEFAULT_GRAPH)
    parser.add_argument("--audit-dir", default=None)
    # No --baseline-m: D-20 is defined on the adopted 5 m baseline and
    # offering to change it here could only produce a table that
    # contradicts its own reported method.
    options = parser.parse_args(argv)

    graph_path = Path(options.graph)
    if not graph_path.is_absolute():
        graph_path = (CONFIG_ROOT / graph_path).resolve()
    source_dir = Path(options.audit_dir).resolve() if options.audit_dir \
        else short_edge.audit_dir()

    print("Week 5C-2A.2 - final ordinary terrain-cost attribution")
    print("Ordinary edges: full-length cell integration over the eligible 5 m "
          "gradients.")
    print(f"Short ordinary edges >= {MIN_SHORT_EDGE_LENGTH_M:g} m: whole-edge "
          "endpoint gradient.")
    print("Special structures carry no ordinary cost and no penalty. The "
          "Week 5C-2A")
    print("classification is preserved unchanged; a separate status records "
          "what was")
    print("done about it. Statuses describe this model's reach, not "
          "walkability.")
    print("")

    try:
        audit_table = short_edge.load_audit_table(source_dir)
        audit_summary = load_audit_summary(source_dir)
    except (short_edge.ShortEdgeError, TerrainCostError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    dem_paths = week5a.discover_tiles(options.dem)
    if not dem_paths:
        print(f"ERROR: no DEM tiles found under {options.dem}.",
              file=sys.stderr)
        return 1

    graph_hash_before = audit.file_sha256(graph_path)
    graph = load_graph(graph_path)
    print(f"Graph      {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} directed edges")
    print(f"           sha256 {graph_hash_before}")
    print(f"Audit      {len(audit_table)} rows from "
          f"{source_dir.name}/edge_readiness_audit.csv")

    try:
        provenance = verify_audit_provenance(
            audit_summary, audit_table, graph, graph_path, graph_hash_before,
            dem_paths, SAMPLE_SPACING_M, GRADIENT_BASELINE_M)
    except ProvenanceMismatch as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Provenance verified: the historical classifications were made "
          "against this")
    print("           graph, these DEM tiles, this sampling and this "
          "baseline.")
    quantisation = audit.baseline_tolerance(SAMPLE_SPACING_M)
    full_limit = endpoint_support_limit(GRADIENT_BASELINE_M)
    print("Endpoint support     each end against its OWN outer estimate: that "
          "estimate's")
    print(f"                     achieved baseline / 2 + {quantisation:.2f} m "
          "of sampling-quantisation")
    print(f"                     extrapolation ({full_limit:.2f} m for a full "
          f"{GRADIENT_BASELINE_M:g} m estimate, less for a shorter one)")
    print("")

    try:
        coverage = week5a.DemCoverage(dem_paths)
    except week5a.DemError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    try:
        print("Costing every directed edge...")
        table = build_cost_table(graph, coverage, audit_table,
                                 GRADIENT_BASELINE_M, SAMPLE_SPACING_M)
        assert_cost_invariants(table)
    finally:
        coverage.close()

    graph_hash_after = audit.file_sha256(graph_path)
    if graph_hash_after != graph_hash_before:
        raise TerrainCostError(
            f"{graph_path} changed during construction. It is read-only.")

    out_dir = audit.ensure_directory(audit.cache_dir(OUTPUT_SUBDIR))
    table.to_csv(out_dir / "edge_terrain_cost.csv", index=False)
    statuses = status_summary(table)
    statuses.to_csv(out_dir / "terrain_cost_status_summary.csv", index=False)
    summary = build_summary(table, graph_path, graph_hash_after, dem_paths,
                            table.attrs["elapsed_s"])
    summary["audit_provenance"] = provenance
    with open(out_dir / "terrain_cost_summary.json", "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("")
    print("=" * 78)
    print("Terrain-cost status")
    print("=" * 78)
    print(f"{'status':<38}{'edges':>7}{'% edges':>9}{'length m':>11}"
          f"{'% length':>10}")
    for row in statuses.itertuples():
        print(f"{row.terrain_cost_status:<38}{row.edges:>7}"
              f"{row.share_of_edges_pct:>9.2f}{row.total_length_m:>11.0f}"
              f"{row.share_of_length_pct:>10.2f}")
    trusted = summary["trusted"]
    print("")
    print(f"  trusted terrain cost on {trusted['edges']} edges "
          f"({trusted['share_of_length_pct']} % of network length)")
    print(f"  total terrain effort     "
          f"{trusted['total_effort_j_per_kg']} J/kg")
    print(f"  by method                {trusted['by_method']}")

    print("")
    print(f"Wrote data/cache/{OUTPUT_SUBDIR}/ (edge_terrain_cost.csv, "
          "terrain_cost_status_summary.csv,")
    print("     terrain_cost_summary.json)")
    print(f"The Stage 3 graph is unchanged: sha256 {graph_hash_after}")
    print("")
    print("A cost table, not a route search. No Dijkstra, no ranking, no "
          "stairs penalty,")
    print("no change to any earlier output. Statuses describe what this model "
          "can be")
    print("applied to; they are not accessibility findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
