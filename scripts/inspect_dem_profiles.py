"""Week 5A - inspect the raw 1 m LiDAR DEM along real OSM walking geometry.

A DIAGNOSTIC, not a model. It answers one question for a human: does the raw
1 m DEM look sensible along ordinary urban walking routes, and where does it
clearly need separate treatment?

Deliberately NOT here
---------------------
No smoothing, no gap filling, no interpolation over NoData, no gradient
threshold, no walking effort, no Minetti, no Tobler, no terrain-aware routing,
and no change to the Week 4 pipeline. Elevations are reported exactly as the
raster stores them. Choosing a smoothing distance is a later decision that this
script exists to inform, not to pre-empt.

What it does
------------
For each selected OSM segment it follows the ACTUAL stored geometry - every
vertex, not the straight line between endpoints - resamples it at about 1 m in
EPSG:7856, and reads the DEM at each sample.

The four tiles are treated as one coverage. Each sample records which tile it
came from, so a discontinuity at a tile seam is visible rather than hidden.

Usage
-----
    python scripts/inspect_dem_profiles.py --list
    python scripts/inspect_dem_profiles.py --dem data/dem
    python scripts/inspect_dem_profiles.py --dem a.tif b.tif c.tif d.tif
    python scripts/inspect_dem_profiles.py --dem data/dem --edge 123,456

Where to put the tiles
----------------------
`data/dem/` is git-ignored and is the expected local home for them. Nothing
requires them to live there: --dem takes any directory or list of files, so the
tiles can stay wherever they already are.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import LineString  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402

from src import osm_network  # noqa: E402
from src.config import PROJECT_ROOT as CONFIG_ROOT, get_output_paths, load_config  # noqa: E402

WGS84 = "EPSG:4326"

# The DEM tiles are supplied in EPSG:7856 (GDA2020 / MGA Zone 56), which is
# already this project's metric CRS: the routing buffer (A-07) and the Stage 4
# snapping distances are both measured in it. Sampling in the same system means
# no new CRS enters the project.
METRIC_CRS = "EPSG:7856"

# One sample per metre, matching the DEM's own 1 m posting. Sampling finer
# would invent detail the raster does not have; coarser would discard it.
SAMPLE_SPACING_M = 1.0

DEFAULT_DEM_DIR = "data/dem"
OUTPUT_SUBDIR = "dem_diagnostics"

# A REPORTING threshold only. Adjacent 1 m samples differing by more than this
# get counted so a human can go and look at them. It is not a smoothing
# decision, not a filter, and nothing is altered or excluded because of it. The
# full jump distribution is reported alongside the count so the number is never
# load-bearing.
DEFAULT_JUMP_REPORT_M = 0.5


@dataclass
class Example:
    """One OSM segment chosen for inspection, with the reason it was chosen."""

    key: str
    label: str
    u: int
    v: int
    category: str
    reason: str


# --------------------------------------------------------------------------- #
# The selection.
#
# Every one of these was found by querying the real cached graph, and every one
# lies ENTIRELY inside the official Glebe suburb polygon with stored OSM
# geometry (not just two endpoints). Together they cover an ordinary street, a
# street with an independent gradient claim, a plain footway, a long flight of
# steps, and a bridge.
# --------------------------------------------------------------------------- #
EXAMPLES: Tuple[Example, ...] = (
    Example(
        key="westmoreland", label="Westmoreland Street",
        u=10991256956, v=1831874041, category="ordinary residential street",
        reason="The control. 284 m of ordinary asphalt residential road, the "
               "most common thing a walking route is made of. If the DEM looks "
               "rough here, the roughness is the DEM and not the terrain.",
    ),
    Example(
        key="greek", label="Greek Street",
        u=5125821744, v=11233630673, category="street with a gradient claim",
        reason="The only edge inside Glebe carrying a NUMERIC OSM incline "
               "(-1.5). That gives an independent statement about the gradient "
               "to compare the DEM against - the one case where the DEM can be "
               "checked against something other than itself.",
    ),
    Example(
        key="footway", label="Footway (unnamed, 350 m)",
        u=9282094114, v=9768504300, category="pedestrian footway",
        reason="The longest plain footway wholly inside Glebe: 350 m of "
               "asphalt with 13 stored geometry vertices, so following the "
               "real geometry rather than the endpoint chord actually matters "
               "here.",
    ),
    Example(
        key="pederson_steps", label="Ernest Pederson Steps",
        u=2748637003, v=2748637005, category="steps",
        reason="66 concrete steps over 50 m - the steepest sustained "
               "pedestrian feature in Glebe. Whether a 1 m DEM resolves "
               "individual treads or smooths them into a ramp decides whether "
               "steps need separate treatment at all.",
    ),
    Example(
        key="tram_bridge", label="Glebe Tram Stop footbridge",
        u=7808737477, v=14069911362, category="bridge (and steps)",
        reason="A 97 m bridge=yes footway that is also tagged steps, and it is "
               "the pedestrian access to Glebe Light Rail - stops 203781 and "
               "203782 in our own candidate set. A bare-earth DEM samples the "
               "ground UNDER a deck, so this is where a spurious dip should "
               "appear if one is going to.",
    ),
)


class DemError(RuntimeError):
    """Raised with an actionable message when the DEM cannot be inspected."""


# --------------------------------------------------------------------------- #
# coordinate handling
# --------------------------------------------------------------------------- #
def to_metric(geometry, source_crs: str = WGS84, metric_crs: str = METRIC_CRS):
    """Project a WGS84 geometry into the DEM's CRS.

    Raises:
        DemError: if any projected coordinate is non-finite, which means the
            geometry falls outside the projection's valid domain - in practice
            latitude and longitude the wrong way round.
    """
    transform = Transformer.from_crs(source_crs, metric_crs, always_xy=True).transform
    projected = shapely_transform(transform, geometry)
    for x, y in projected.coords:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise DemError(
                f"A coordinate of this geometry does not project into "
                f"{metric_crs}. Check that latitude and longitude have not "
                "been swapped."
            )
    return projected


def edge_geometry(graph, u: int, v: int, key: int = 0) -> Tuple[LineString, bool]:
    """The stored OSM geometry of one edge, in WGS84.

    Returns:
        Tuple of (line, has_stored_geometry). When OSMnx stored no geometry the
        edge is a straight line between its endpoints, and the flag says so -
        a profile along a fabricated chord means something different from one
        along real surveyed geometry.

    Raises:
        DemError: if the edge is not in the graph.
    """
    if not graph.has_edge(u, v, key):
        raise DemError(
            f"Edge ({u}, {v}, {key}) is not in the graph. Use --list to see "
            "the selected examples, or pick an edge from the cached graph."
        )
    data = graph.edges[u, v, key]
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry, True
    return LineString([
        (graph.nodes[u]["x"], graph.nodes[u]["y"]),
        (graph.nodes[v]["x"], graph.nodes[v]["y"]),
    ]), False


def sample_positions(
    line_metric: LineString, spacing_m: float = SAMPLE_SPACING_M
) -> List[Tuple[float, float, float]]:
    """Points every ``spacing_m`` along a projected line, following its shape.

    The line is walked by arc length, so every vertex is honoured. The final
    point is always included even when the length is not a whole multiple of
    the spacing, so the profile reaches the far end of the segment.

    Returns:
        List of (distance_along_m, x, y).
    """
    if spacing_m <= 0:
        raise DemError(f"spacing must be positive, got {spacing_m}")
    length = line_metric.length
    positions: List[Tuple[float, float, float]] = []
    distance = 0.0
    while distance < length:
        point = line_metric.interpolate(distance)
        positions.append((distance, point.x, point.y))
        distance += spacing_m
    end = line_metric.interpolate(length)
    positions.append((length, end.x, end.y))
    return positions


# --------------------------------------------------------------------------- #
# the DEM coverage
# --------------------------------------------------------------------------- #
class DemCoverage:
    """Several adjacent GeoTIFF tiles, read as one elevation surface.

    The tiles are NOT merged into a single array. Each sample is routed to the
    tile whose bounds contain it and that tile is recorded against the sample,
    so a discontinuity at a tile seam shows up in the output instead of being
    averaged away by a mosaic.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        import rasterio

        if not paths:
            raise DemError(
                "No DEM tiles given. Pass --dem <directory> or --dem a.tif "
                "b.tif ..., or put the tiles in data/dem/."
            )
        self.paths = [Path(p) for p in paths]
        self.datasets = []
        for path in self.paths:
            if not path.exists():
                raise DemError(f"{path} does not exist.")
            self.datasets.append(rasterio.open(path))
        self._check_consistency()

    def _check_consistency(self) -> None:
        """Refuse a set of tiles that are not one coverage."""
        crs_values = {str(dataset.crs) for dataset in self.datasets}
        if len(crs_values) != 1:
            raise DemError(f"The tiles have different CRSs: {sorted(crs_values)}.")
        resolutions = {
            (round(dataset.res[0], 6), round(dataset.res[1], 6))
            for dataset in self.datasets
        }
        if len(resolutions) != 1:
            raise DemError(
                f"The tiles have different resolutions: {sorted(resolutions)}."
            )

    @property
    def crs(self) -> str:
        return str(self.datasets[0].crs)

    @property
    def resolution(self) -> Tuple[float, float]:
        return tuple(self.datasets[0].res)

    def describe(self) -> Dict[str, Any]:
        bounds = [dataset.bounds for dataset in self.datasets]
        return {
            "tiles": len(self.datasets),
            "crs": self.crs,
            "resolution_m": self.resolution,
            "coverage_bounds": {
                "left": min(b.left for b in bounds),
                "bottom": min(b.bottom for b in bounds),
                "right": max(b.right for b in bounds),
                "top": max(b.top for b in bounds),
            },
            "files": [p.name for p in self.paths],
            "nodata": [dataset.nodata for dataset in self.datasets],
        }

    def tile_for(self, x: float, y: float) -> Optional[int]:
        """Index of the tile containing a point, or None if outside coverage."""
        for index, dataset in enumerate(self.datasets):
            bounds = dataset.bounds
            if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
                return index
        return None

    def sample(
        self, xs: Sequence[float], ys: Sequence[float]
    ) -> Tuple[List[Optional[float]], List[Optional[str]]]:
        """Read the raw elevation at each point.

        Returns:
            Tuple of (elevations, tile names). An elevation is None when the
            point falls outside the coverage or the raster says NoData there.
            Nothing is interpolated, filled or substituted.
        """
        elevations: List[Optional[float]] = []
        tiles: List[Optional[str]] = []
        for x, y in zip(xs, ys):
            index = self.tile_for(x, y)
            if index is None:
                elevations.append(None)
                tiles.append(None)
                continue
            dataset = self.datasets[index]
            value = float(next(dataset.sample([(x, y)]))[0])
            nodata = dataset.nodata
            if nodata is not None and value == nodata:
                elevations.append(None)
            elif not math.isfinite(value):
                elevations.append(None)
            else:
                elevations.append(value)
            tiles.append(self.paths[index].name)
        return elevations, tiles

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()


def discover_tiles(inputs: Sequence[str]) -> List[Path]:
    """Turn --dem arguments into a list of tile paths.

    Accepts any mix of directories and files, so the tiles can stay wherever
    they already live.
    """
    paths: List[Path] = []
    for item in inputs:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = (CONFIG_ROOT / candidate).resolve()
        if candidate.is_dir():
            paths.extend(sorted(
                p for p in candidate.iterdir()
                if p.suffix.lower() in {".tif", ".tiff"}
            ))
        else:
            paths.append(candidate)
    return paths


# --------------------------------------------------------------------------- #
# profiles and diagnostics
# --------------------------------------------------------------------------- #
def ensure_directory(path: Path) -> Path:
    """Create a directory and its parents, tolerating one that already exists.

    Bug fix: an earlier version created the output directory only as a side
    effect of saving the first PNG, but the CSV for each profile is written
    first. On a machine where data/cache/dem_diagnostics/ did not already
    exist, the very first write failed.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def tag_values(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def build_profile(
    graph, coverage: "DemCoverage", u: int, v: int, key: int = 0,
    spacing_m: float = SAMPLE_SPACING_M,
) -> pd.DataFrame:
    """Sample the DEM along one edge's real geometry.

    Returns:
        One row per sample: distance along the segment, projected and
        geographic coordinates, the RAW elevation, and which tile it came from.
        No smoothing and no gap filling.
    """
    line_wgs84, has_geometry = edge_geometry(graph, u, v, key)
    line_metric = to_metric(line_wgs84)
    positions = sample_positions(line_metric, spacing_m)

    xs = [x for _d, x, _y in positions]
    ys = [y for _d, _x, y in positions]
    elevations, tiles = coverage.sample(xs, ys)

    back = Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True).transform
    longitudes, latitudes = [], []
    for x, y in zip(xs, ys):
        longitude, latitude = back(x, y)
        longitudes.append(longitude)
        latitudes.append(latitude)

    frame = pd.DataFrame({
        "distance_m": [round(d, 3) for d, _x, _y in positions],
        "x_7856": xs, "y_7856": ys,
        "latitude": latitudes, "longitude": longitudes,
        "elevation_m_ahd": elevations,
        "dem_tile": tiles,
    })
    frame.attrs["has_stored_geometry"] = has_geometry
    return frame


def profile_diagnostics(
    frame: pd.DataFrame, graph, u: int, v: int, key: int = 0,
    jump_report_m: float = DEFAULT_JUMP_REPORT_M,
) -> Dict[str, Any]:
    """Summarise one profile for human inspection. Nothing is judged."""
    data = graph.edges[u, v, key]
    elevations = frame["elevation_m_ahd"]
    valid = elevations.dropna()

    jumps = elevations.diff().abs().dropna()
    diagnostics: Dict[str, Any] = {
        "edge": f"{u} -> {v}",
        "osm_length_m": round(float(data.get("length", 0.0)), 2),
        "sampled_length_m": round(float(frame["distance_m"].max()), 2),
        "samples": int(len(frame)),
        "has_stored_geometry": bool(frame.attrs.get("has_stored_geometry")),
        "geometry_vertices": len(list(
            data["geometry"].coords)) if data.get("geometry") is not None else 2,
        "valid_samples": int(len(valid)),
        "nodata_or_outside_samples": int(elevations.isna().sum()),
        "tiles_touched": sorted({t for t in frame["dem_tile"] if t}),
        "highway": tag_values(data, "highway"),
        "name": tag_values(data, "name"),
        "surface": tag_values(data, "surface"),
        "incline_osm": tag_values(data, "incline"),
        "bridge": tag_values(data, "bridge"),
        "tunnel": tag_values(data, "tunnel"),
        "step_count": tag_values(data, "step_count"),
    }
    if len(valid):
        diagnostics.update({
            "elevation_min_m": round(float(valid.min()), 3),
            "elevation_max_m": round(float(valid.max()), 3),
            "elevation_range_m": round(float(valid.max() - valid.min()), 3),
        })
    if len(jumps):
        diagnostics.update({
            "max_adjacent_jump_m": round(float(jumps.max()), 3),
            "p95_adjacent_jump_m": round(float(jumps.quantile(0.95)), 3),
            "median_adjacent_jump_m": round(float(jumps.median()), 3),
            f"jumps_over_{jump_report_m}m": int((jumps > jump_report_m).sum()),
        })
    return diagnostics


def plot_profile(frame: pd.DataFrame, diagnostics: Dict[str, Any],
                 example_label: str, out_path: Path) -> None:
    """Draw one raw elevation profile. Raw points only - no fitted line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.plot(frame["distance_m"], frame["elevation_m_ahd"],
              linewidth=1.0, color="#1f78b4")
    axis.scatter(frame["distance_m"], frame["elevation_m_ahd"],
                 s=5, color="#1f78b4", alpha=0.55, zorder=3)

    missing = frame[frame["elevation_m_ahd"].isna()]
    if len(missing):
        for distance in missing["distance_m"]:
            axis.axvline(distance, color="#e31a1c", alpha=0.35, linewidth=0.8)

    axis.set_xlabel("distance along the OSM geometry (m)")
    axis.set_ylabel("raw DEM elevation (m AHD)")
    tags = " ".join(
        f"{k}={diagnostics[k]}" for k in
        ("highway", "surface", "incline_osm", "bridge", "step_count")
        if diagnostics.get(k)
    )
    axis.set_title(f"{example_label}\n{tags}", fontsize=10, pad=10)
    axis.grid(True, linewidth=0.4, alpha=0.4)
    figure.text(
        0.5, 0.01,
        "Raw 1 m DEM samples, unsmoothed. Diagnostic only: no gap filling, no "
        "gradient model, no walking effort.",
        ha="center", fontsize=7.5, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    ensure_directory(out_path.parent)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
def describe_examples() -> List[str]:
    lines = ["Selected OSM segments for Week 5A DEM inspection", ""]
    for index, example in enumerate(EXAMPLES, start=1):
        lines.append(f"{index}. {example.label}  [{example.category}]")
        lines.append(f"   edge {example.u} -> {example.v}")
        lines.append(f"   why: {example.reason}")
        lines.append("")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the raw 1 m DEM along real OSM walking geometry."
    )
    parser.add_argument("--dem", nargs="+", default=[DEFAULT_DEM_DIR],
                        help="DEM tile files, or a directory of them "
                             f"(default: {DEFAULT_DEM_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="print the selected segments and exit; no DEM needed")
    parser.add_argument("--edge", action="append", default=[],
                        help="also inspect an ad-hoc edge, as u,v")
    parser.add_argument("--spacing-m", type=float, default=SAMPLE_SPACING_M,
                        help=f"sample spacing (default {SAMPLE_SPACING_M} m)")
    parser.add_argument("--jump-report-m", type=float,
                        default=DEFAULT_JUMP_REPORT_M,
                        help="report adjacent samples differing by more than "
                             "this. A reporting aid, not a smoothing decision.")
    options = parser.parse_args(argv)

    if options.list:
        for line in describe_examples():
            print(line)
        return 0

    config = load_config()
    graph_path = (CONFIG_ROOT / get_output_paths(config)["walk_graph"]).resolve()
    if not graph_path.exists():
        print(f"ERROR: {graph_path} is missing. Run "
              "scripts/run_week4_foundation.py first.", file=sys.stderr)
        return 1

    tiles = discover_tiles(options.dem)
    try:
        coverage = DemCoverage(tiles)
    except DemError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    description = coverage.describe()
    print("DEM coverage")
    for key, value in description.items():
        print(f"  {key:<18} {value}")
    if METRIC_CRS.split(":")[-1] not in description["crs"]:
        print(f"  WARNING: the tiles report {description['crs']}, not "
              f"{METRIC_CRS}. Sampling assumes {METRIC_CRS}; check before "
              "trusting these profiles.")
    print("")

    graph = osm_network.load_graph(graph_path)
    # Created up front: the per-profile CSV is written before the first PNG,
    # so relying on the plot to create it left the first write to fail.
    out_dir = ensure_directory(
        (CONFIG_ROOT / "data" / "cache" / OUTPUT_SUBDIR).resolve()
    )

    selected: List[Example] = list(EXAMPLES)
    for index, raw in enumerate(options.edge, start=1):
        try:
            u_text, v_text = raw.split(",")
            selected.append(Example(
                key=f"adhoc{index}", label=f"ad-hoc edge {raw}",
                u=int(u_text), v=int(v_text), category="ad hoc",
                reason="supplied on the command line",
            ))
        except ValueError:
            print(f"ERROR: --edge expects u,v - got {raw!r}", file=sys.stderr)
            return 1

    all_diagnostics: List[Dict[str, Any]] = []
    for example in selected:
        try:
            frame = build_profile(graph, coverage, example.u, example.v,
                                  spacing_m=options.spacing_m)
        except DemError as error:
            print(f"SKIPPED {example.label}: {error}")
            continue
        diagnostics = profile_diagnostics(
            frame, graph, example.u, example.v,
            jump_report_m=options.jump_report_m,
        )
        diagnostics["key"] = example.key
        diagnostics["label"] = example.label
        diagnostics["category"] = example.category
        all_diagnostics.append(diagnostics)

        frame.to_csv(out_dir / f"profile_{example.key}.csv", index=False)
        plot_profile(frame, diagnostics, example.label,
                     out_dir / f"profile_{example.key}.png")

        print("=" * 78)
        print(f"{example.label}  [{example.category}]")
        print("=" * 78)
        for field_name in (
            "edge", "highway", "name", "surface", "incline_osm", "bridge",
            "step_count", "osm_length_m", "sampled_length_m",
            "geometry_vertices", "has_stored_geometry", "samples",
            "valid_samples", "nodata_or_outside_samples", "elevation_min_m",
            "elevation_max_m", "elevation_range_m", "max_adjacent_jump_m",
            "p95_adjacent_jump_m", "median_adjacent_jump_m",
            f"jumps_over_{options.jump_report_m}m", "tiles_touched",
        ):
            if diagnostics.get(field_name) not in (None, "", []):
                print(f"  {field_name:<28} {diagnostics[field_name]}")
        print("")

    if all_diagnostics:
        summary_path = out_dir / "dem_profile_diagnostics.csv"
        pd.DataFrame(all_diagnostics).to_csv(summary_path, index=False)
        print(f"Wrote {out_dir.relative_to(CONFIG_ROOT)}/  "
              f"({len(all_diagnostics)} profiles, CSV + PNG each, plus "
              f"{summary_path.name})")
    coverage.close()

    print("")
    print("Raw samples only. No smoothing distance has been chosen, no gaps "
          "filled, and")
    print("no walking effort computed. Read the profiles before deciding any "
          "of that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
