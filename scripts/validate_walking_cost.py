"""Week 5C-1 - validate Minetti walking cost on ordinary profiles.

A NUMERICAL VALIDATION, not a routing weight. It takes the 5 m gradients Week
5B already produced for the three ordinary walking edges, converts them to
metabolic walking cost, and checks that the arithmetic and the integration are
right before any of it is allowed near a routing graph.

Deliberately NOT here
---------------------
The Glebe routing graph is untouched. No Dijkstra weight, no stop ranking, no
change to the interactive demo, and no edge attribute is written anywhere. This
script reads two directories and writes its own diagnostics.

The cost model
--------------
Minetti et al., the walking branch:

    Cw(i) = 280.5 i^5 - 58.7 i^4 - 76.8 i^3 + 51.9 i^2 + 19.6 i + 2.5

with i the decimal gradient (+5 % is +0.05) and Cw in J per kg per metre
travelled. The empirical study covered roughly -0.45 <= i <= +0.45; outside
that the polynomial is an extrapolation and is flagged as one. Nothing is
clamped and nothing is quietly extrapolated.

Everything stays in J/kg. Body mass is never requested and never applied: it is
a constant multiplier, so it cannot change the ORDER of two routes, and asking
for it would imply a personalisation this project has not designed.

The integration rule, which is where this is easy to get wrong
--------------------------------------------------------------
A 5 m gradient baseline does NOT mean each estimate represents 5 m of walking.
Week 5B produced one estimate per ~1 m of route; consecutive 5 m windows
overlap by 4 m. Multiplying every estimate by 5 m would count nearly every
metre of the street five times over.

Each estimate is therefore given the length of route it actually represents -
the span between the midpoints to its neighbours:

    ds_j = (c_{j+1} - c_{j-1}) / 2        interior
    ds_0 = (c_1 - c_0) / 2                first
    ds_n = (c_n - c_{n-1}) / 2            last

    effort = sum_j Cw(i_j) * ds_j

Those weights sum exactly to c_last - c_first, the covered length, so no metre
is counted twice and none is skipped.

Endpoint coverage
-----------------
A centred 5 m gradient produces no estimate in the first or last 2.5 m of a
profile. Those metres are reported as uncovered and left alone. Inventing an
endpoint gradient would put fabricated values exactly where there is no
evidence; how to handle them will be decided before anything runs network-wide.

Usage
-----
    python scripts/validate_walking_cost.py
    python scripts/validate_walking_cost.py --self-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402

WEEK5A_SUBDIR = "dem_diagnostics"
WEEK5B_SUBDIR = "gradient_sensitivity"
OUTPUT_SUBDIR = "walking_cost_validation"

# The baseline adopted for ordinary roads in Week 5B.
ADOPTED_BASELINE_M = 5.0

ORDINARY_KEYS: Tuple[str, ...] = ("westmoreland", "greek", "footway")

# Minetti et al. walking polynomial coefficients, highest power first.
MINETTI_COEFFICIENTS: Tuple[float, ...] = (
    280.5, -58.7, -76.8, 51.9, 19.6, 2.5,
)

# The gradient range the empirical study covered. Outside it the polynomial is
# an extrapolation, and this script says so rather than pretending otherwise.
MINETTI_DOMAIN: Tuple[float, float] = (-0.45, 0.45)

# OSM tags that mean "this is not an ordinary walking edge". A stairway, a
# bridge deck or a tunnel is not what Minetti measured on a treadmill, and the
# DEM does not describe the surface people actually walk on there either.
SPECIAL_STRUCTURE_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("highway", "steps", "highway=steps - a stairway is not a graded walking "
                         "surface and Minetti's walking branch does not "
                         "describe it"),
    ("bridge", "yes", "bridge=yes - the DEM samples the ground beneath the "
                      "deck, not the surface walked on"),
    ("tunnel", "yes", "tunnel=yes - the DEM samples the surface above, not "
                      "the passage walked through"),
)


class WalkingCostError(RuntimeError):
    """Raised with an actionable message when the validation cannot run."""


# --------------------------------------------------------------------------- #
# the cost model
# --------------------------------------------------------------------------- #
def minetti_cost(gradient: float, strict: bool = False) -> float:
    """Metabolic walking cost at one gradient, in J per kg per metre.

    Args:
        gradient: decimal gradient. +5 per cent is +0.05, not 5.
        strict: when True, refuse to evaluate outside the studied domain
            instead of returning an extrapolated value. Use this wherever an
            extrapolation must be impossible rather than merely visible.

    Returns:
        Cw in J kg^-1 m^-1. Always positive over the studied domain: walking
        downhill costs less than walking on the level, never nothing.

    Raises:
        WalkingCostError: if strict and the gradient is outside
            :data:`MINETTI_DOMAIN`.
    """
    value = float(gradient)
    if strict and not in_domain(value):
        low, high = MINETTI_DOMAIN
        raise WalkingCostError(
            f"gradient {value:+.4f} is outside the studied Minetti domain "
            f"[{low}, {high}]. Evaluating here would be an extrapolation."
        )
    cost = 0.0
    for coefficient in MINETTI_COEFFICIENTS:
        cost = cost * value + coefficient
    return cost


def minetti_cost_array(gradients: Sequence[float]) -> np.ndarray:
    """Vectorised :func:`minetti_cost`, without the strict check.

    Out-of-domain values are computed and flagged separately by
    :func:`in_domain_array`, so a caller always has both the number and the
    warning rather than a silently clamped value.
    """
    values = np.asarray(gradients, dtype=float)
    cost = np.zeros_like(values)
    for coefficient in MINETTI_COEFFICIENTS:
        cost = cost * values + coefficient
    return cost


def in_domain(gradient: float) -> bool:
    """Whether a gradient lies inside the range Minetti's study covered."""
    low, high = MINETTI_DOMAIN
    return bool(low <= float(gradient) <= high)


def in_domain_array(gradients: Sequence[float]) -> np.ndarray:
    low, high = MINETTI_DOMAIN
    values = np.asarray(gradients, dtype=float)
    return (values >= low) & (values <= high)


LEVEL_COST_J_PER_KG_PER_M = minetti_cost(0.0)


# --------------------------------------------------------------------------- #
# integration along distance
# --------------------------------------------------------------------------- #
def integration_weights(centres: Sequence[float]) -> np.ndarray:
    """The length of route each gradient estimate represents.

    THIS IS THE POINT OF THE WHOLE FILE. The estimates come from overlapping
    5 m windows spaced about 1 m apart, so multiplying each by its 5 m baseline
    would count nearly every metre of street five times. Each estimate instead
    gets the span between the midpoints to its neighbours, and the weights sum
    exactly to the covered length.

    Args:
        centres: the centre distance of each gradient estimate, ascending.

    Returns:
        One weight per estimate, in metres, summing to
        ``centres[-1] - centres[0]``.

    Raises:
        WalkingCostError: if there are fewer than two estimates, or they are
            not in ascending order.
    """
    values = np.asarray(centres, dtype=float)
    if values.size < 2:
        raise WalkingCostError(
            "At least two gradient estimates are needed to integrate."
        )
    if np.any(np.diff(values) <= 0):
        raise WalkingCostError(
            "Gradient centres must be strictly ascending to be integrated."
        )
    weights = np.empty_like(values)
    weights[1:-1] = (values[2:] - values[:-2]) / 2.0
    weights[0] = (values[1] - values[0]) / 2.0
    weights[-1] = (values[-1] - values[-2]) / 2.0
    return weights


def integrate_effort(
    centres: Sequence[float], costs: Sequence[float]
) -> Dict[str, Any]:
    """Integrate cost along distance using the representative-length weights.

    Returns:
        Dict with the per-estimate weights, incremental and cumulative effort
        in J/kg, the total, and the covered length.
    """
    weights = integration_weights(centres)
    cost_values = np.asarray(costs, dtype=float)
    incremental = cost_values * weights
    return {
        "weights_m": weights,
        "incremental_j_per_kg": incremental,
        "cumulative_j_per_kg": np.nancumsum(incremental),
        "total_j_per_kg": float(np.nansum(incremental)),
        "covered_length_m": float(weights.sum()),
    }


# --------------------------------------------------------------------------- #
# ordinary-edge guard
# --------------------------------------------------------------------------- #
def special_structure_reason(tags: Dict[str, Any]) -> Optional[str]:
    """Why this edge is not an ordinary walking edge, or None if it is.

    Checked against the real OSM tags recorded by Week 5A rather than against a
    hardcoded list of keys, so a profile cannot be treated as ordinary just
    because someone put its name in the wrong constant.
    """
    for tag, marker, reason in SPECIAL_STRUCTURE_RULES:
        raw = tags.get(tag)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        values = {part.strip().lower() for part in str(raw).split(";")}
        if marker in values:
            return reason
    step_count = tags.get("step_count")
    if step_count not in (None, "") and not (
        isinstance(step_count, float) and np.isnan(step_count)
    ):
        return f"step_count={step_count} - a stepped edge, not a graded surface"
    return None


def assert_ordinary(key: str, tags: Dict[str, Any]) -> None:
    """Refuse to give a special structure an ordinary walking cost.

    Raises:
        WalkingCostError: naming the tag that disqualifies it.
    """
    reason = special_structure_reason(tags)
    if reason is not None:
        raise WalkingCostError(
            f"{key} is not an ordinary walking edge: {reason}. Minetti "
            "ordinary-road treatment must not be applied to it; special "
            "structures are handled separately."
        )


# --------------------------------------------------------------------------- #
# reading Week 5A and Week 5B output
# --------------------------------------------------------------------------- #
def cache_dir(subdir: str, root: Path = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache" / subdir).resolve()


def load_gradients(
    key: str, baseline_m: float = ADOPTED_BASELINE_M, source_dir: Path = None
) -> pd.DataFrame:
    """Read one Week 5B gradient series exactly as it was written."""
    source_dir = Path(source_dir) if source_dir else cache_dir(WEEK5B_SUBDIR)
    path = source_dir / f"gradient_{key}_{baseline_m:g}m.csv"
    if not path.exists():
        raise WalkingCostError(
            f"{path} is missing. Run the Week 5B sensitivity analysis first:\n"
            "    python scripts/analyse_gradient_sensitivity.py"
        )
    frame = pd.read_csv(path)
    for column in ("centre_distance_m", "gradient", "gradient_pct"):
        if column not in frame.columns:
            raise WalkingCostError(f"{path} has no {column} column.")
    return frame


def load_profile_extent(key: str, source_dir: Path = None) -> float:
    """The full sampled length of a Week 5A profile, for coverage reporting."""
    source_dir = Path(source_dir) if source_dir else cache_dir(WEEK5A_SUBDIR)
    path = source_dir / f"profile_{key}.csv"
    if not path.exists():
        raise WalkingCostError(
            f"{path} is missing. Run the Week 5A profiles first:\n"
            "    python scripts/inspect_dem_profiles.py --dem <tiles>"
        )
    return float(pd.read_csv(path)["distance_m"].max())


def load_tags(key: str, source_dir: Path = None) -> Dict[str, Any]:
    """The OSM tags Week 5A recorded for one example."""
    source_dir = Path(source_dir) if source_dir else cache_dir(WEEK5A_SUBDIR)
    path = source_dir / "dem_profile_diagnostics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    match = frame[frame["key"].astype(str) == key]
    return {} if match.empty else match.iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# the per-road table
# --------------------------------------------------------------------------- #
def build_effort_table(gradients: pd.DataFrame) -> pd.DataFrame:
    """Cost and integrated effort for one road's 5 m gradient series."""
    centres = gradients["centre_distance_m"].to_numpy(dtype=float)
    slopes = gradients["gradient"].to_numpy(dtype=float)
    costs = minetti_cost_array(slopes)
    inside = in_domain_array(slopes)
    integration = integrate_effort(centres, costs)

    return pd.DataFrame({
        "distance_m": centres,
        "gradient": slopes,
        "gradient_pct": gradients["gradient_pct"].to_numpy(dtype=float),
        "minetti_cw_j_per_kg_per_m": costs,
        "integration_weight_m": integration["weights_m"],
        "incremental_effort_j_per_kg": integration["incremental_j_per_kg"],
        "cumulative_effort_j_per_kg": integration["cumulative_j_per_kg"],
        "in_minetti_domain": inside,
    })


def summarise(
    table: pd.DataFrame, total_length_m: float, baseline_m: float,
) -> Dict[str, Any]:
    """Coverage, effort and domain diagnostics for one road."""
    covered = float(table["integration_weight_m"].sum())
    total_effort = float(table["incremental_effort_j_per_kg"].sum())
    level_equivalent = LEVEL_COST_J_PER_KG_PER_M * covered
    outside = ~table["in_minetti_domain"].to_numpy()

    summary: Dict[str, Any] = {
        "baseline_m": baseline_m,
        "estimates": int(len(table)),
        "total_profile_length_m": round(total_length_m, 2),
        "gradient_covered_length_m": round(covered, 2),
        "uncovered_endpoint_length_m": round(total_length_m - covered, 2),
        "covered_share": round(covered / total_length_m, 4)
        if total_length_m else None,
        "total_effort_j_per_kg": round(total_effort, 2),
        "level_equivalent_effort_j_per_kg": round(level_equivalent, 2),
        "terrain_to_level_ratio": round(total_effort / level_equivalent, 4)
        if level_equivalent else None,
        "level_cost_j_per_kg_per_m": LEVEL_COST_J_PER_KG_PER_M,
        "min_gradient_pct": round(float(table["gradient_pct"].min()), 3),
        "max_gradient_pct": round(float(table["gradient_pct"].max()), 3),
        "min_cw_j_per_kg_per_m": round(
            float(table["minetti_cw_j_per_kg_per_m"].min()), 4),
        "max_cw_j_per_kg_per_m": round(
            float(table["minetti_cw_j_per_kg_per_m"].max()), 4),
        "out_of_domain_estimates": int(outside.sum()),
        "minetti_domain": list(MINETTI_DOMAIN),
    }
    if outside.any():
        summary["out_of_domain_distance_m"] = round(
            float(table.loc[outside, "integration_weight_m"].sum()), 2)
        summary["out_of_domain_effort_j_per_kg"] = round(
            float(table.loc[outside, "incremental_effort_j_per_kg"].sum()), 2)
    return summary


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_effort(table: pd.DataFrame, label: str, out_path: Path) -> None:
    """Three diagnostic panels sharing a distance axis. Not presentation art."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].axhline(0.0, color="#999999", linewidth=0.8)
    axes[0].plot(table["distance_m"], table["gradient_pct"],
                 color="#2171b5", linewidth=1.3)
    axes[0].set_ylabel("gradient (%)")
    axes[0].grid(True, linewidth=0.4, alpha=0.4)

    axes[1].axhline(LEVEL_COST_J_PER_KG_PER_M, color="#999999",
                    linewidth=0.9, linestyle="--",
                    label=f"level walking = {LEVEL_COST_J_PER_KG_PER_M} J/kg/m")
    axes[1].plot(table["distance_m"], table["minetti_cw_j_per_kg_per_m"],
                 color="#d95f02", linewidth=1.3)
    outside = ~table["in_minetti_domain"]
    if outside.any():
        axes[1].scatter(table.loc[outside, "distance_m"],
                        table.loc[outside, "minetti_cw_j_per_kg_per_m"],
                        s=18, color="#e31a1c", zorder=4,
                        label="outside the studied domain (extrapolated)")
    axes[1].set_ylabel("Minetti Cw\n(J/kg/m)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, linewidth=0.4, alpha=0.4)

    axes[2].plot(table["distance_m"], table["cumulative_effort_j_per_kg"],
                 color="#1b9e77", linewidth=1.6, label="terrain-aware")
    level_line = (LEVEL_COST_J_PER_KG_PER_M
                  * table["integration_weight_m"].cumsum())
    axes[2].plot(table["distance_m"], level_line, color="#999999",
                 linewidth=1.1, linestyle="--", label="level-walking equivalent")
    axes[2].set_ylabel("cumulative effort\n(J/kg)")
    axes[2].set_xlabel("distance along the OSM geometry (m)")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, linewidth=0.4, alpha=0.4)

    figure.suptitle(f"{label} - Minetti walking cost on the 5 m gradient",
                    fontsize=11)
    figure.text(
        0.5, 0.005,
        "Diagnostic. Effort is integrated over the length each estimate "
        "represents, not over the 5 m baseline. All values J/kg; no body mass "
        "is applied.",
        ha="center", fontsize=7.5, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
def self_check() -> int:
    """Print the qualitative behaviour of the cost curve. No data needed."""
    print("Minetti walking cost, Cw(i) in J/kg/m")
    print(f"  studied domain           {MINETTI_DOMAIN}")
    print(f"  level walking Cw(0)      {LEVEL_COST_J_PER_KG_PER_M}")
    print("")
    print("   gradient      Cw      vs level")
    for gradient in (-0.45, -0.30, -0.20, -0.15, -0.10, -0.05, 0.0,
                     0.05, 0.10, 0.20, 0.30, 0.45):
        cost = minetti_cost(gradient)
        marker = "cheaper" if cost < LEVEL_COST_J_PER_KG_PER_M else "dearer"
        print(f"  {gradient:+7.2f}   {cost:8.4f}   {marker}")
    print("")
    print("  uphill and downhill are NOT mirror images: "
          f"Cw(+0.10)={minetti_cost(0.10):.4f} but "
          f"Cw(-0.10)={minetti_cost(-0.10):.4f}")
    print("  mild downhill is cheaper than level; steep downhill rises again, "
          "because braking costs energy too.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Minetti walking cost on the ordinary profiles."
    )
    parser.add_argument("--baseline-m", type=float, default=ADOPTED_BASELINE_M,
                        help=f"gradient baseline to use (default "
                             f"{ADOPTED_BASELINE_M:g} m, as adopted in Week 5B)")
    parser.add_argument("--self-check", action="store_true",
                        help="print the cost curve's behaviour and exit")
    parser.add_argument("--gradients-dir", default=None)
    parser.add_argument("--profiles-dir", default=None)
    options = parser.parse_args(argv)

    if options.self_check:
        return self_check()

    gradients_dir = Path(options.gradients_dir).resolve() \
        if options.gradients_dir else cache_dir(WEEK5B_SUBDIR)
    profiles_dir = Path(options.profiles_dir).resolve() \
        if options.profiles_dir else cache_dir(WEEK5A_SUBDIR)
    out_dir = cache_dir(OUTPUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Week 5C-1 - Minetti walking cost validation")
    print("Reads the Week 5B gradients. No DEM re-sampling, no gradient "
          "recomputation,")
    print("no routing weight, no change to the Glebe graph.")
    print("")
    print(f"Cost model      Minetti walking polynomial, domain "
          f"{MINETTI_DOMAIN}")
    print(f"Level walking   {LEVEL_COST_J_PER_KG_PER_M} J/kg/m")
    print(f"Baseline        {options.baseline_m:g} m (Week 5B)")
    print("Units           J/kg throughout. Body mass is never requested: it "
          "is a constant")
    print("                multiplier and cannot change the order of two "
          "routes.")
    print("")

    summaries: List[Dict[str, Any]] = []
    for key in ORDINARY_KEYS:
        try:
            tags = load_tags(key, profiles_dir)
            assert_ordinary(key, tags)
            gradients = load_gradients(key, options.baseline_m, gradients_dir)
            total_length = load_profile_extent(key, profiles_dir)
        except WalkingCostError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

        label = str(tags.get("label") or tags.get("name") or key)
        table = build_effort_table(gradients)
        summary = summarise(table, total_length, options.baseline_m)
        summary["key"] = key
        summary["label"] = label
        summary["highway"] = tags.get("highway", "")
        summaries.append(summary)

        table.to_csv(out_dir / f"walking_cost_{key}.csv", index=False)
        plot_effort(table, label, out_dir / f"walking_cost_{key}.png")

        print("=" * 78)
        print(f"{label}  [{tags.get('highway', '')}]")
        print("=" * 78)
        print(f"  gradient estimates            {summary['estimates']}")
        print(f"  total profile length          "
              f"{summary['total_profile_length_m']} m")
        print(f"  gradient-covered length       "
              f"{summary['gradient_covered_length_m']} m")
        print(f"  uncovered endpoint length     "
              f"{summary['uncovered_endpoint_length_m']} m  "
              f"(first and last {options.baseline_m / 2:g} m; no gradient "
              "invented there)")
        print(f"  gradient range                "
              f"{summary['min_gradient_pct']:+.2f} % to "
              f"{summary['max_gradient_pct']:+.2f} %")
        print(f"  Cw range                      "
              f"{summary['min_cw_j_per_kg_per_m']} to "
              f"{summary['max_cw_j_per_kg_per_m']} J/kg/m")
        print("")
        print(f"  effort over covered length    "
              f"{summary['total_effort_j_per_kg']} J/kg")
        print(f"  level-walking equivalent      "
              f"{summary['level_equivalent_effort_j_per_kg']} J/kg  "
              f"({LEVEL_COST_J_PER_KG_PER_M} x "
              f"{summary['gradient_covered_length_m']} m)")
        print(f"  terrain / level ratio         "
              f"{summary['terrain_to_level_ratio']}")
        print("")
        print(f"  out-of-domain estimates       "
              f"{summary['out_of_domain_estimates']}")
        if summary["out_of_domain_estimates"]:
            print(f"     representing               "
                  f"{summary['out_of_domain_distance_m']} m and "
                  f"{summary['out_of_domain_effort_j_per_kg']} J/kg")
            print("     WARNING: those values are EXTRAPOLATED beyond the "
                  "studied domain.")
            print("     They are included in the total and flagged, not "
                  "clamped and not dropped.")
        print("")

    if summaries:
        summary_path = out_dir / "walking_cost_validation_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        print(f"Wrote {out_dir.relative_to(CONFIG_ROOT)}/ "
              f"({len(summaries)} roads, one CSV and one PNG each, plus "
              f"{summary_path.name})")

    print("")
    print("Integration check: effort was summed over the length each estimate")
    print("represents, so the weights add up to the covered length. Multiplying")
    print(f"each estimate by the {options.baseline_m:g} m baseline instead would "
          "have counted almost")
    print("every metre of street five times.")
    print("")
    print("This is validation only. Nothing here is a routing weight yet, "
          "endpoint")
    print("handling is still undecided, and no Glebe result is interpreted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
