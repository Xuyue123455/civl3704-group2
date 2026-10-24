"""Week 5B - how does the estimated gradient change with the baseline length?

A SENSITIVITY DIAGNOSTIC, not a model and not a decision. It takes the raw 1 m
DEM samples that Week 5A already produced and asks one question: over what
spatial baseline does a local gradient estimate stop being noise and start
being the street?

It answers nothing. It produces the evidence and stops.

Deliberately NOT here
---------------------
No Minetti, no Tobler, no walking effort or cost, no stop ranking, no
terrain-aware routing, and no change to the Week 4 pipeline. No baseline is
adopted: the script prints every metric side by side and leaves the choice to a
human, because no single metric should decide it.

It also does not re-sample the DEM. Week 5A already wrote the raw samples to
data/cache/dem_diagnostics/profile_<key>.csv and those files are read as they
are. The raw elevations are never modified, smoothed, filled or resampled.

Ordinary roads only
-------------------
Only Westmoreland Street, Greek Street and the unnamed 350 m footway are used.
The Ernest Pederson Steps and the Glebe Tram Stop footbridge behave nothing
like an ordinary walking edge, so including them would let a stairway or a
bridge deck decide how ordinary streets are measured. They remain Week 5A
evidence about special structures and are handled separately later.

The gradient method
-------------------
For a target baseline L and samples spaced about 1 m apart, let
n = max(1, round(L / spacing)). For each i:

    gradient      = (z[i + n] - z[i]) / (d[i + n] - d[i])
    attributed to   (d[i] + d[i + n]) / 2

That is a centred estimate: it belongs to the midpoint of the pair, not to
either end. It uses only raw samples, so nothing is interpolated. The ACTUAL
baseline achieved is d[i + n] - d[i], which is reported rather than assumed,
because the last sample interval of a profile is usually shorter than 1 m.

Endpoint handling: none. There is no padding, reflection or one-sided
difference at the ends. A baseline of L simply produces no estimate in the
first and last L/2 of the profile, and the series is that much shorter. Making
up edge values would put invented gradients exactly where the evidence is
thinnest.

Missing data: if either endpoint of a pair is NoData, that gradient is missing.
It is never filled.

Usage
-----
    python scripts/analyse_gradient_sensitivity.py
    python scripts/analyse_gradient_sensitivity.py --baselines 1 3 5 10 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402

WEEK5A_SUBDIR = "dem_diagnostics"
OUTPUT_SUBDIR = "gradient_sensitivity"

# The baselines under test, in metres.
DEFAULT_BASELINES_M: Tuple[float, ...] = (1.0, 3.0, 5.0, 10.0)

# Ordinary walking edges. These, and only these, inform the choice of baseline
# for ordinary roads.
ORDINARY_KEYS: Tuple[str, ...] = ("westmoreland", "greek", "footway")

# Excluded from the ordinary selection, on the evidence of the Week 5A
# profiles. Kept here by name so their exclusion is explicit and auditable
# rather than silent.
SPECIAL_STRUCTURES: Dict[str, str] = {
    "pederson_steps":
        "66 concrete steps over 50 m. A stairway is not an ordinary walking "
        "edge, and letting it influence the baseline would size the method to "
        "the least representative feature in Glebe.",
    "tram_bridge":
        "A bridge deck over ground the DEM samples underneath. The profile "
        "describes the terrain below the structure, not the surface anyone "
        "walks on.",
}

# Reference grades used only to tabulate how often large local gradients occur.
# They are reading aids, deliberately several of them, so that no single one
# becomes the deciding criterion. Nothing is filtered or rejected by them.
REFERENCE_GRADES_PCT: Tuple[float, ...] = (5.0, 8.0, 10.0, 15.0, 20.0)

# The OSM tag on Greek Street. Recorded for comparison. NOT ground truth -
# see compare_with_osm_incline.
GREEK_STREET_OSM_INCLINE_PCT = -1.5


class SensitivityError(RuntimeError):
    """Raised with an actionable message when the analysis cannot run."""


# --------------------------------------------------------------------------- #
# reading Week 5A output
# --------------------------------------------------------------------------- #
def week5a_dir(root: Path = None) -> Path:
    root = root or CONFIG_ROOT
    return (Path(root) / "data" / "cache" / WEEK5A_SUBDIR).resolve()


def load_profile(key: str, source_dir: Path = None) -> pd.DataFrame:
    """Read one raw Week 5A profile exactly as it was written.

    Raises:
        SensitivityError: if the profile has not been generated.
    """
    source_dir = source_dir or week5a_dir()
    path = Path(source_dir) / f"profile_{key}.csv"
    if not path.exists():
        raise SensitivityError(
            f"{path} is missing. Run the Week 5A profiles first:\n"
            "    python scripts/inspect_dem_profiles.py --dem <tiles>"
        )
    frame = pd.read_csv(path)
    for column in ("distance_m", "elevation_m_ahd"):
        if column not in frame.columns:
            raise SensitivityError(f"{path} has no {column} column.")
    return frame


def sample_spacing(frame: pd.DataFrame) -> float:
    """The typical spacing between samples, taken from the data itself.

    The median is used rather than the mean because the final interval of a
    profile is usually shorter than the rest.
    """
    steps = frame["distance_m"].diff().dropna()
    if steps.empty:
        raise SensitivityError("The profile has fewer than two samples.")
    return float(steps.median())


# --------------------------------------------------------------------------- #
# the gradient method
# --------------------------------------------------------------------------- #
def centred_gradient(
    frame: pd.DataFrame, baseline_m: float
) -> pd.DataFrame:
    """Centred local gradient over an approximately ``baseline_m`` baseline.

    See the module docstring for the definition and for why there is no
    endpoint padding. The raw elevations in ``frame`` are not modified.

    Returns:
        A frame with one row per estimate: the midpoint distance, the actual
        baseline achieved, the rise, the gradient as a decimal slope, and the
        same gradient as a percent grade.
    """
    if baseline_m <= 0:
        raise SensitivityError(f"baseline must be positive, got {baseline_m}")

    spacing = sample_spacing(frame)
    n = max(1, int(round(baseline_m / spacing)))
    distances = frame["distance_m"].to_numpy()
    elevations = frame["elevation_m_ahd"].to_numpy()
    if len(distances) <= n:
        raise SensitivityError(
            f"A {baseline_m:g} m baseline needs more than {n} sample "
            f"intervals, but this profile has only {len(distances)} samples."
        )

    lower_d, upper_d = distances[:-n], distances[n:]
    lower_z, upper_z = elevations[:-n], elevations[n:]
    run = upper_d - lower_d
    rise = upper_z - lower_z
    gradient = rise / run

    result = pd.DataFrame({
        "centre_distance_m": (lower_d + upper_d) / 2.0,
        "actual_baseline_m": run,
        "rise_m": rise,
        "gradient": gradient,             # decimal slope, rise over run
        "gradient_pct": gradient * 100.0,  # percent grade
    })
    result.attrs["target_baseline_m"] = float(baseline_m)
    result.attrs["samples_spanned"] = n
    result.attrs["sample_spacing_m"] = spacing
    return result


def overall_gradient(frame: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Start-to-end average gradient, for context only.

    Uses the first and last samples that actually have an elevation, so a
    NoData patch at either end does not silently shorten the run.
    """
    valid = frame.dropna(subset=["elevation_m_ahd"])
    if len(valid) < 2:
        return None
    run = float(valid["distance_m"].iloc[-1] - valid["distance_m"].iloc[0])
    rise = float(valid["elevation_m_ahd"].iloc[-1]
                 - valid["elevation_m_ahd"].iloc[0])
    if run == 0:
        return None
    return {
        "run_m": round(run, 2),
        "rise_m": round(rise, 3),
        "gradient": rise / run,
        "gradient_pct": round(100.0 * rise / run, 3),
    }


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def gradient_diagnostics(
    gradients: pd.DataFrame,
    reference_grades_pct: Sequence[float] = REFERENCE_GRADES_PCT,
) -> Dict[str, Any]:
    """Summarise one baseline's gradient series. Nothing is judged.

    Two independent readings of "unusually large" are reported, on purpose:

    * counts against a ladder of fixed reference grades, so no single grade is
      the criterion; and
    * a self-referential count, |g| beyond median + 3 x MAD of the same
      series, which needs no external number at all.

    Two noise indicators are reported alongside, because they are what
    actually separate a noisy baseline from a smooth one:

    * the median absolute change between consecutive estimates - a jittery
      series moves a lot from one estimate to the next; and
    * how often the gradient changes sign per 100 m - a real street does not
      alternate between uphill and downhill every metre.
    """
    values = gradients["gradient_pct"].dropna()
    absolute = values.abs()
    total = int(len(values))

    diagnostics: Dict[str, Any] = {
        "target_baseline_m": gradients.attrs.get("target_baseline_m"),
        "actual_baseline_m_median": round(
            float(gradients["actual_baseline_m"].median()), 3),
        "samples_spanned": gradients.attrs.get("samples_spanned"),
        "estimates": total,
        "missing_estimates": int(gradients["gradient_pct"].isna().sum()),
    }
    if not total:
        return diagnostics

    mad = float((absolute - absolute.median()).abs().median())
    outlier_level = float(absolute.median() + 3.0 * mad)

    diagnostics.update({
        "median_abs_gradient_pct": round(float(absolute.median()), 3),
        "p95_abs_gradient_pct": round(float(absolute.quantile(0.95)), 3),
        "max_abs_gradient_pct": round(float(absolute.max()), 3),
        "mad_abs_gradient_pct": round(mad, 3),
        "self_referential_outlier_level_pct": round(outlier_level, 3),
        "self_referential_outliers": int((absolute > outlier_level).sum()),
        "self_referential_outlier_share": round(
            float((absolute > outlier_level).mean()), 4),
        "median_abs_change_between_estimates_pct": round(
            float(values.diff().abs().median()), 3),
    })

    signs = values[values != 0]
    sign_changes = int((signs.apply(lambda v: 1 if v > 0 else -1)
                        .diff().abs() > 0).sum())
    span = float(gradients["centre_distance_m"].max()
                 - gradients["centre_distance_m"].min())
    diagnostics["sign_changes"] = sign_changes
    diagnostics["sign_changes_per_100m"] = (
        round(100.0 * sign_changes / span, 2) if span > 0 else None
    )

    for grade in reference_grades_pct:
        diagnostics[f"over_{grade:g}pct_count"] = int((absolute > grade).sum())
        diagnostics[f"over_{grade:g}pct_share"] = round(
            float((absolute > grade).mean()), 4)
    return diagnostics


def compare_with_osm_incline(
    dem_overall_pct: Optional[float],
    osm_incline_pct: float = GREEK_STREET_OSM_INCLINE_PCT,
) -> Dict[str, Any]:
    """Put the DEM trend beside the OSM incline tag, without ranking them.

    The OSM tag is context, NOT ground truth, and a discrepancy is not by
    itself evidence that the DEM is wrong. Four reasons the two can legitimately
    differ:

    1. Direction. OSM ``incline`` is relative to the way's own node order,
       which need not match the direction this edge was sampled in. A sign
       disagreement may be a convention difference and nothing more.
    2. Extent. OSMnx simplification merges ways, so the sampled edge may cover
       more or less ground than the way the tag was placed on.
    3. Provenance. An ``incline`` value is usually a mapper's estimate or a
       rounded figure, not a survey.
    4. Quantity. A start-to-end average over the whole edge is not the same
       thing as the "typical" gradient a mapper had in mind.

    Returns:
        Both values, their signed and absolute differences, and whether the
        signs agree - as evidence, with no verdict attached.
    """
    comparison: Dict[str, Any] = {
        "osm_incline_pct": osm_incline_pct,
        "osm_incline_is_ground_truth": False,
        "dem_overall_gradient_pct": dem_overall_pct,
    }
    if dem_overall_pct is None:
        comparison["note"] = "no DEM trend available for this profile"
        return comparison
    comparison.update({
        "signed_difference_pct": round(dem_overall_pct - osm_incline_pct, 3),
        "absolute_difference_pct": round(
            abs(abs(dem_overall_pct) - abs(osm_incline_pct)), 3),
        "signs_agree": (dem_overall_pct < 0) == (osm_incline_pct < 0),
    })
    return comparison


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_baseline_comparison(
    series: Dict[float, pd.DataFrame], label: str, out_path: Path,
) -> None:
    """Overlay every baseline's gradient series for one road."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {1.0: "#c6dbef", 3.0: "#6baed6", 5.0: "#2171b5", 10.0: "#08306b"}
    widths = {1.0: 0.8, 3.0: 1.1, 5.0: 1.5, 10.0: 2.1}

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.axhline(0.0, color="#999999", linewidth=0.8, zorder=1)
    for baseline in sorted(series):
        frame = series[baseline]
        axis.plot(frame["centre_distance_m"], frame["gradient_pct"],
                  label=f"{baseline:g} m baseline",
                  color=colours.get(baseline, "#08306b"),
                  linewidth=widths.get(baseline, 1.4),
                  alpha=0.95, zorder=2)
    axis.set_xlabel("distance along the OSM geometry (m)")
    axis.set_ylabel("gradient (%)")
    axis.set_title(
        f"{label} - local gradient by baseline length", fontsize=11, pad=10)
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.legend(loc="best", fontsize=9, frameon=True, framealpha=0.9)
    figure.text(
        0.5, 0.01,
        "Centred differences on the raw 1 m DEM samples. No smoothing, no gap "
        "filling, no baseline adopted.",
        ha="center", fontsize=7.5, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare local-gradient estimates across spatial baselines."
    )
    parser.add_argument("--baselines", type=float, nargs="+",
                        default=list(DEFAULT_BASELINES_M),
                        help="baselines in metres (default: 1 3 5 10)")
    parser.add_argument("--profiles-dir", default=None,
                        help="where the Week 5A profiles are "
                             "(default: data/cache/dem_diagnostics)")
    options = parser.parse_args(argv)

    source_dir = Path(options.profiles_dir).resolve() if options.profiles_dir \
        else week5a_dir()
    out_dir = (CONFIG_ROOT / "data" / "cache" / OUTPUT_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from importlib import util as importlib_util
        spec = importlib_util.spec_from_file_location(
            "inspect_dem_profiles", PROJECT_ROOT / "scripts" / "inspect_dem_profiles.py")
        week5a = importlib_util.module_from_spec(spec)
        sys.modules[spec.name] = week5a
        spec.loader.exec_module(week5a)
        labels = {e.key: e.label for e in week5a.EXAMPLES}
    except Exception:                    # pragma: no cover - labels are cosmetic
        labels = {}

    print("Week 5B - gradient baseline sensitivity")
    print("Reads the raw Week 5A samples. Nothing is re-sampled or smoothed.")
    print("")
    print("Ordinary walking edges used for this comparison:")
    for key in ORDINARY_KEYS:
        print(f"  {key:<16} {labels.get(key, key)}")
    print("")
    print("Excluded, and NOT allowed to influence the ordinary baseline:")
    for key, reason in SPECIAL_STRUCTURES.items():
        print(f"  {key:<16} {labels.get(key, key)}")
        print(f"                   {reason}")
    print("")

    all_diagnostics: List[Dict[str, Any]] = []
    for key in ORDINARY_KEYS:
        try:
            profile = load_profile(key, source_dir)
        except SensitivityError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

        label = labels.get(key, key)
        overall = overall_gradient(profile)
        spacing = sample_spacing(profile)

        print("=" * 78)
        print(f"{label}")
        print("=" * 78)
        print(f"  raw samples              {len(profile)}")
        print(f"  median sample spacing    {spacing:.3f} m")
        if overall:
            print(f"  start-to-end run         {overall['run_m']} m")
            print(f"  start-to-end rise        {overall['rise_m']} m")
            print(f"  overall average gradient {overall['gradient_pct']:+.3f} % "
                  f"({overall['gradient']:+.5f} decimal)   [context only]")
        print("")

        series: Dict[float, pd.DataFrame] = {}
        for baseline in options.baselines:
            try:
                gradients = centred_gradient(profile, baseline)
            except SensitivityError as error:
                print(f"  {baseline:g} m baseline skipped: {error}")
                continue
            series[float(baseline)] = gradients
            gradients.to_csv(
                out_dir / f"gradient_{key}_{baseline:g}m.csv", index=False)

            diagnostics = gradient_diagnostics(gradients)
            diagnostics["key"] = key
            diagnostics["label"] = label
            if overall:
                diagnostics["overall_gradient_pct"] = overall["gradient_pct"]
            all_diagnostics.append(diagnostics)

            print(f"  --- {baseline:g} m baseline "
                  f"(spans {diagnostics['samples_spanned']} sample intervals, "
                  f"actual {diagnostics['actual_baseline_m_median']:g} m) ---")
            for name in ("estimates", "median_abs_gradient_pct",
                         "p95_abs_gradient_pct", "max_abs_gradient_pct",
                         "median_abs_change_between_estimates_pct",
                         "sign_changes_per_100m",
                         "self_referential_outlier_level_pct",
                         "self_referential_outliers",
                         "self_referential_outlier_share"):
                if diagnostics.get(name) is not None:
                    print(f"      {name:<44} {diagnostics[name]}")
            counts = "  ".join(
                f">{g:g}%: {diagnostics[f'over_{g:g}pct_count']}"
                f" ({100*diagnostics[f'over_{g:g}pct_share']:.1f}%)"
                for g in REFERENCE_GRADES_PCT
            )
            print(f"      large local gradients                        {counts}")
            print("")

        if series:
            plot_baseline_comparison(
                series, label, out_dir / f"gradient_baselines_{key}.png")

        if key == "greek":
            comparison = compare_with_osm_incline(
                overall["gradient_pct"] if overall else None)
            print("  --- Greek Street: DEM trend beside the OSM incline tag ---")
            for name, value in comparison.items():
                print(f"      {name:<44} {value}")
            print("      The OSM tag is CONTEXT, NOT GROUND TRUTH. The two can")
            print("      legitimately differ: incline is relative to the way's")
            print("      own node order, OSMnx simplification may merge ways so")
            print("      the sampled extent differs, the tag is usually a")
            print("      mapper's rounded estimate rather than a survey, and a")
            print("      start-to-end average is not the 'typical' gradient a")
            print("      mapper had in mind. A discrepancy is not by itself")
            print("      evidence that the DEM is wrong.")
            print("")
            pd.DataFrame([comparison]).to_csv(
                out_dir / "greek_street_osm_incline_comparison.csv", index=False)

    if all_diagnostics:
        summary = out_dir / "gradient_sensitivity_diagnostics.csv"
        pd.DataFrame(all_diagnostics).to_csv(summary, index=False)
        print(f"Wrote {out_dir.relative_to(CONFIG_ROOT)}/ "
              f"({len(all_diagnostics)} baseline/road combinations, per-baseline "
              f"CSVs, one comparison PNG per road, plus {summary.name})")

    print("")
    print("NO BASELINE HAS BEEN CHOSEN.")
    print("Read the plots and the table together. The question is whether 1 m "
          "is dominated by")
    print("noise, whether 3-5 m is stable while still local, and where 10 m "
          "starts erasing real")
    print("variation. No single metric settles that, and this script does not "
          "try to.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
