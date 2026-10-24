"""Tobler walking time over the adopted represented-length cells (Week 9).

WHAT THIS IS
------------
Tobler's hiking function, applied to the gradient estimates and represented
lengths that Week 5C-2A.2 (D-20) already produces, to give a walking TIME for
an edge alongside its Minetti walking EFFORT.

    v(g) = 6 * exp(-3.5 * |g + 0.05|)        [km/h]

with ``g`` the dimensionless gradient (rise/run; +10 per cent is +0.10, not 10)
and ``v`` a walking speed in kilometres per hour. The function peaks at exactly
6 km/h on a 5 per cent DOWNHILL (g = -0.05) and falls away on both sides, so a
steep descent is slow as well as a steep climb. On the level it gives

    v(0) = 6 * exp(-0.175) = 5.0367 km/h = 1.3991 m/s

Reference: Tobler, W. R. (1993). Three presentations on geographical analysis
and modeling (Technical Report 93-1). NCGIA, University of California, Santa
Barbara. The project already cites this reference; nothing here is re-derived
or re-fitted.

WHY A SEPARATE METRIC
---------------------
Time and effort are different questions and the project keeps them apart.
Minetti answers "how much metabolic work is this walk?"; Tobler answers "how
long does this walk take?". They are NOT combined into a single score, there is
no alpha*effort + beta*time anywhere, and adding this module changes no Minetti
value and no routing decision.

HOW IT REUSES THE EXISTING PIPELINE
-----------------------------------
It does not re-sample the DEM and it does not re-estimate gradients. The caller
hands over the SAME two arrays the Minetti integration consumes - the eligible
local gradients g_i and their non-overlapping represented lengths w_i, which
cover the edge exactly once - and this module turns each pair into a local time

    t_i = w_i / v(g_i)        [s]      with v converted to m/s

and sums them. One weights array, two integrations. Because the weights are the
adopted D-20 cells, full-edge coverage and the no-double-counting property are
inherited rather than re-asserted here.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
No DEM sampling, no gradient estimation, no smoothing, no route search, no stop
ranking, no catchability, no combined score, no personalisation (no age, mass,
fitness, mobility aid or individual speed), and no eligibility policy of its
own: an edge that the terrain-cost build does not trust gets no walking time
from this module either.

A NOTE ON WHAT TOBLER IS AND IS NOT
-----------------------------------
Tobler's function is an off-road hiking relation fitted to empirical travel
rates, not a measurement of urban footpath walking and not a validated model of
any particular person. Applying it to Glebe footpaths is a modelling assumption
of exactly the same kind as applying Minetti: it is adopted, visible and
documented, not validated here. It contains no waiting, crossing, signal,
kerb-ramp, surface or crowding term.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- #
# the published relation
# --------------------------------------------------------------------------- #
# Tobler (1993). Highest speed, the exponential decay rate, and the gradient at
# which the peak sits. These are the published constants, not project choices.
TOBLER_PEAK_SPEED_KMH = 6.0
TOBLER_DECAY = 3.5
TOBLER_PEAK_GRADIENT = -0.05

#: Exact kilometres-per-hour to metres-per-second conversion (1000 / 3600).
KMH_TO_MS = 1.0 / 3.6

# A guard, not a model parameter. A speed at or below this is treated as no
# usable speed rather than silently producing an enormous crossing time. The
# relation is strictly positive everywhere, so this can only be reached by a
# non-finite or absurd gradient, which is a data fault worth surfacing.
MIN_USABLE_SPEED_MS = 1e-6


class ToblerTimeError(RuntimeError):
    """Raised with an actionable message when a walking time cannot be formed."""


def tobler_speed_kmh(gradient: float) -> float:
    """Tobler walking speed at one gradient, in km/h.

    Args:
        gradient: dimensionless gradient (rise/run). +10 per cent is +0.10.

    Returns:
        Walking speed in km/h. Strictly positive, peaking at
        :data:`TOBLER_PEAK_SPEED_KMH` when ``gradient`` is
        :data:`TOBLER_PEAK_GRADIENT`.

    Raises:
        ToblerTimeError: if the gradient is not finite. Nothing is substituted.
    """
    value = float(gradient)
    if not np.isfinite(value):
        raise ToblerTimeError(
            "a non-finite gradient has no Tobler speed; the edge must stay "
            "unresolved rather than receive a substituted value.")
    return TOBLER_PEAK_SPEED_KMH * float(
        np.exp(-TOBLER_DECAY * abs(value - TOBLER_PEAK_GRADIENT)))


def tobler_speed_kmh_array(gradients: Sequence[float]) -> np.ndarray:
    """:func:`tobler_speed_kmh` over an array, with the same refusal."""
    values = np.asarray(gradients, dtype=float)
    if values.size and not np.isfinite(values).all():
        raise ToblerTimeError(
            f"{int((~np.isfinite(values)).sum())} of {values.size} gradients "
            "are not finite; no Tobler speed is formed for this edge.")
    return TOBLER_PEAK_SPEED_KMH * np.exp(
        -TOBLER_DECAY * np.abs(values - TOBLER_PEAK_GRADIENT))


def tobler_speed_ms(gradient: float) -> float:
    """Tobler walking speed at one gradient, in m/s."""
    return tobler_speed_kmh(gradient) * KMH_TO_MS


def tobler_speed_ms_array(gradients: Sequence[float]) -> np.ndarray:
    """Tobler walking speed over an array, in m/s."""
    return tobler_speed_kmh_array(gradients) * KMH_TO_MS


#: Level-ground speed, derived from the relation rather than written down.
LEVEL_SPEED_KMH = tobler_speed_kmh(0.0)
LEVEL_SPEED_MS = tobler_speed_ms(0.0)


# --------------------------------------------------------------------------- #
# integration over the adopted represented-length cells
# --------------------------------------------------------------------------- #
def walking_time_over_cells(
    gradients: Sequence[float], weights_m: Sequence[float]
) -> Dict[str, Any]:
    """Edge walking time from the local gradients and their represented lengths.

    This is the time counterpart of the D-20 effort integration and it consumes
    the identical inputs: one gradient per estimate, and the non-overlapping
    represented length that estimate is responsible for.

        t_i      = w_i / v(g_i)
        t_edge   = sum_i t_i

    Nothing is averaged first. The caller owns eligibility: every pair handed
    over is assumed to have already passed the terrain-cost build's gates.

    Args:
        gradients: eligible local gradients g_i, dimensionless.
        weights_m: represented length w_i in metres for each gradient, in the
            same order. These are the adopted cells, so they already cover the
            edge exactly once.

    Returns:
        A dict with ``walking_time_s``, the per-cell ``times_s`` and
        ``speeds_ms`` arrays, ``covered_length_m`` (the weights' own sum, so a
        caller can re-check coverage against the edge), the slowest and fastest
        local speed, and ``mean_speed_ms`` defined as covered length over total
        time - a harmonic mean, which is the only average that reproduces the
        integrated time.

    Raises:
        ToblerTimeError: on mismatched lengths, an empty input, a non-positive
            or non-finite weight, or a speed at or below
            :data:`MIN_USABLE_SPEED_MS`.
    """
    slopes = np.asarray(gradients, dtype=float)
    weights = np.asarray(weights_m, dtype=float)
    if slopes.size != weights.size:
        raise ToblerTimeError(
            f"{slopes.size} gradients but {weights.size} represented lengths; "
            "each gradient must carry exactly one length.")
    if slopes.size == 0:
        raise ToblerTimeError(
            "no gradient estimates were supplied, so no walking time exists "
            "for this edge.")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ToblerTimeError(
            "a represented length is zero, negative or not finite; the "
            "adopted cells must all be positive.")

    speeds = tobler_speed_ms_array(slopes)
    if np.any(speeds <= MIN_USABLE_SPEED_MS):
        raise ToblerTimeError(
            "a local Tobler speed is not usable, so no walking time is formed.")

    times = weights / speeds
    total_time = float(times.sum())
    covered = float(weights.sum())
    return {
        "walking_time_s": total_time,
        "times_s": times,
        "speeds_ms": speeds,
        "estimates": int(slopes.size),
        "covered_length_m": covered,
        "min_speed_ms": float(speeds.min()),
        "max_speed_ms": float(speeds.max()),
        # Harmonic, by construction: covered / sum(w_i / v_i). An arithmetic
        # mean of v_i would not reproduce the integrated time.
        "mean_speed_ms": covered / total_time if total_time > 0 else None,
    }


def walking_time_whole_edge(
    gradient: float, length_m: float
) -> Dict[str, Any]:
    """Walking time for an edge carried by a single gradient over its length.

    The time counterpart of the adopted short-edge endpoint fallback, where one
    estimate already spans the edge and the multi-cell integration is therefore
    deliberately not used. Implemented as the one-cell case of
    :func:`walking_time_over_cells` so the two paths cannot drift apart.
    """
    if not np.isfinite(float(length_m)) or float(length_m) <= 0.0:
        raise ToblerTimeError(
            f"an edge length of {length_m} m has no walking time.")
    return walking_time_over_cells([float(gradient)], [float(length_m)])


def seconds_to_minutes(seconds: Optional[float]) -> Optional[float]:
    """Seconds to minutes, passing missing values straight through."""
    if seconds is None:
        return None
    value = float(seconds)
    if not np.isfinite(value):
        return None
    return value / 60.0


# --------------------------------------------------------------------------- #
# a runnable self-check, so the relation can be demonstrated without pytest
# --------------------------------------------------------------------------- #
def self_check() -> int:
    """Print the published landmarks of the relation and verify them."""
    peak = tobler_speed_kmh(TOBLER_PEAK_GRADIENT)
    level = tobler_speed_kmh(0.0)
    print("Tobler (1993) hiking function: v(g) = 6 * exp(-3.5 * |g + 0.05|)")
    print(f"  peak           g = {TOBLER_PEAK_GRADIENT:+.2f}  "
          f"v = {peak:.4f} km/h = {peak * KMH_TO_MS:.4f} m/s")
    print(f"  level          g = {0.0:+.2f}  "
          f"v = {level:.4f} km/h = {level * KMH_TO_MS:.4f} m/s")
    for g in (0.05, 0.10, 0.20, -0.10, -0.20, -0.45, 0.45):
        v = tobler_speed_kmh(g)
        print(f"  {'uphill  ' if g > 0 else 'downhill'}   g = {g:+.2f}  "
              f"v = {v:.4f} km/h = {v * KMH_TO_MS:.4f} m/s")

    assert abs(peak - TOBLER_PEAK_SPEED_KMH) < 1e-12, "peak is not 6 km/h"
    assert abs(level - 5.0367) < 1e-3, "level speed is not about 5.04 km/h"
    assert tobler_speed_kmh(0.10) < level, "uphill must be slower than level"
    assert tobler_speed_kmh(-0.30) < level, "steep downhill must be slower"
    grid = np.linspace(-0.45, 0.45, 9001)
    assert abs(grid[int(np.argmax(tobler_speed_kmh_array(grid)))]
               - TOBLER_PEAK_GRADIENT) < 1e-3, "peak is not at -0.05"

    cells = walking_time_over_cells([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
    expected = 30.0 / LEVEL_SPEED_MS
    assert abs(cells["walking_time_s"] - expected) < 1e-9
    print(f"\n  30 m of level walking over three 10 m cells: "
          f"{cells['walking_time_s']:.3f} s "
          f"({seconds_to_minutes(cells['walking_time_s']):.3f} min)")
    print("  self-check passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tobler walking speed and represented-length walking time.")
    parser.add_argument("--self-check", action="store_true",
                        help="print the published landmarks and verify them")
    parser.add_argument("--gradient", type=float, default=None,
                        help="print the Tobler speed at one gradient")
    args = parser.parse_args(argv)
    if args.gradient is not None:
        v = tobler_speed_kmh(args.gradient)
        print(f"g = {args.gradient:+.4f}  v = {v:.4f} km/h "
              f"= {v * KMH_TO_MS:.4f} m/s")
        return 0
    return self_check()


if __name__ == "__main__":
    raise SystemExit(main())
