"""Local vector basemap for the unified map - PRESENTATION ONLY (Basemap Phase A).

Reads the cache written once by ``scripts/build_basemap_cache.py``
(``data/cache/basemap/glebe_basemap.json``: streets, laneways, footpaths and
road names from the project's own OSM routing graph; buildings / parks /
water when a local OSM export was supplied) and draws it UNDER the
analytical layers of ``MapView``. It never reads the routing graph, the cost
table, the stops or any scene, it makes no network request, and nothing it
draws is written into the scene export.

Visual hierarchy (z-order):
    0.10  land background            0.25 water   0.30 parks   0.40 buildings
    0.50  footpaths / steps          0.60 minor streets (service, pedestrian)
    0.70  medium streets             0.80 major streets        0.95 road labels
    ... then the validated layers: boundary (2), walking routes (4-6), stops (7+), origin (11)

If the cache is missing or malformed, ``Basemap.load`` returns None and the
window keeps its validated analytical presentation (message in the status strip).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .walk_engine import PROJECT_ROOT, plots

CACHE_SCHEMA = "unified-basemap-cache-1"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "cache" / "basemap" / "glebe_basemap.json"
# V-4.2: a wider VISUAL cache (Inner West, from the local Overpass responses) used by the launcher when present;
# the routing graph and the study area are untouched - this changes what is drawn under the map, never what is computed
WIDE_CACHE = PROJECT_ROOT / "data" / "cache" / "basemap" / "inner_west_basemap.json"


def preferred_cache() -> Path:
    """The launcher's default: the wide Inner West cache when it exists, else the validated Glebe cache."""
    return WIDE_CACHE if WIDE_CACHE.exists() else DEFAULT_CACHE
UNAVAILABLE_MESSAGE = "Local basemap unavailable — using analytical map only."

# restrained cartography: pale land, white streets with grey casing, grey footpaths
COLOUR_LAND = "#f6f5f1"
COLOUR_WATER = "#d6e6f0"
COLOUR_PARK = "#e3eedb"
COLOUR_BUILDING = "#e7e4de"
COLOUR_BUILDING_EDGE = "#d9d5cd"
COLOUR_PATH = "#d9d7d2"
COLOUR_CASING = "#cfcdc7"
COLOUR_STREET = "#ffffff"
COLOUR_LABEL = "#6f6f6f"
LABEL_HALO = "#f6f5f1"

# road class -> (casing width, fill width) in points; labels only for these classes
ROAD_STYLE = {"major": (4.6, 3.0), "medium": (3.2, 1.9), "minor": (1.9, 1.0)}
ROAD_ZORDER = {"minor": 0.60, "medium": 0.70, "major": 0.80}
LABEL_CLASSES = ("major", "medium")
LABEL_FONTSIZE = 6.6
LABEL_MIN_SEGMENT_M = 70.0       # a street chain shorter than this is never labelled
LABEL_REPEAT_M = 520.0           # a long street gets another label about this far along
LABEL_MAX = 60                   # never more labels than this in one view (no label explosion on zoom)
RELABEL_MIN_CHANGE = 0.04        # re-place labels only when the extent moved / scaled by more than 4 %


class Basemap:
    """Parsed once, drawn once per Axes; keeps no reference to any analytical object."""

    def __init__(self, cache: Dict[str, Any], path: Optional[Path] = None) -> None:
        self.path = path
        self.schema = cache.get("schema")
        self.sources = cache.get("sources") or {}
        self.unavailable: List[str] = list(cache.get("unavailable_layers") or [])
        self.counts: Dict[str, int] = dict(cache.get("counts") or {})
        self.load_seconds: Optional[float] = None
        self.draw_seconds: Optional[float] = None
        layers = cache.get("layers") or {}
        # projected polylines by road class (EPSG:7856, the map's own CRS)
        self.roads: Dict[str, List[np.ndarray]] = {"major": [], "medium": [], "minor": []}
        self.named: Dict[Tuple[str, str], List[np.ndarray]] = {}
        for road in layers.get("roads") or []:
            xy = self._project(road["coords"])
            if xy is None:
                continue
            cls = road.get("class") if road.get("class") in self.roads else "minor"
            self.roads[cls].append(xy)
            name = str(road.get("name") or "").strip()
            if name and cls in LABEL_CLASSES:
                self.named.setdefault((name, cls), []).append(xy)
        self.paths: List[np.ndarray] = []
        self.steps: List[np.ndarray] = []
        for path in layers.get("paths") or []:
            xy = self._project(path["coords"])
            if xy is None:
                continue
            (self.steps if path.get("class") == "steps" else self.paths).append(xy)
        self.polygons: Dict[str, List[Tuple[np.ndarray, List[np.ndarray]]]] = {}
        for layer in ("water", "parks", "buildings"):
            items = []
            for poly in layers.get(layer) or []:
                exterior = self._project(poly.get("exterior") or [])
                if exterior is None:
                    continue
                holes = [h for h in (self._project(ring) for ring in poly.get("interiors") or []) if h is not None]
                items.append((exterior, holes))
            self.polygons[layer] = items
        # street chains (same-name ways joined end to end) are computed ONCE here and reused by every relabel
        self.chains: Dict[Tuple[str, str], List[Tuple[np.ndarray, float]]] = {
            key: [(chain, self._length(chain)) for chain in self._chains(list(pieces))]
            for key, pieces in self.named.items()}
        self.artists: List[Any] = []
        self.labels: List[Any] = []
        self.labels_placed = 0
        self.label_extent: Optional[Tuple[float, float, float, float]] = None
        self.relabels = 0

    # ----- loading ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["Basemap"]:
        """The cached basemap, or None (missing / malformed / wrong schema) - never raises."""
        target = Path(path) if path else DEFAULT_CACHE
        started = time.perf_counter()
        try:
            cache = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(cache, dict) or cache.get("schema") != CACHE_SCHEMA or "layers" not in cache:
                return None
            basemap = cls(cache, target)
        except (OSError, ValueError, TypeError, KeyError):
            return None
        basemap.load_seconds = time.perf_counter() - started
        return basemap

    @staticmethod
    def _project(coords: List[List[float]]) -> Optional[np.ndarray]:
        if not coords or len(coords) < 2:
            return None
        arr = np.asarray(coords, dtype=float)
        return plots.project(arr[:, 0], arr[:, 1])

    def data_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Metric bounds of every drawn line (the visual coverage; nothing analytical)."""
        pieces = [xy for lines in self.roads.values() for xy in lines] + self.paths + self.steps
        if not pieces:
            return None
        xs = np.concatenate([xy[:, 0] for xy in pieces])
        ys = np.concatenate([xy[:, 1] for xy in pieces])
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def summary(self) -> str:
        parts = [f"{self.counts.get('roads', 0)} streets", f"{self.counts.get('paths', 0)} footpaths"]
        for layer in ("buildings", "parks", "water"):
            if self.counts.get(layer):
                parts.append(f"{self.counts[layer]} {layer}")
        missing = ", ".join(self.unavailable) if self.unavailable else "none"
        return "Local basemap: " + ", ".join(parts) + f"; not available locally: {missing}."

    # ----- drawing ------------------------------------------------------------------ #
    def draw(self, axis, extent: Tuple[float, float, float, float]) -> List[Any]:
        """Draw every layer once, clipped to a modest buffer around the map extent."""
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.patches import Rectangle

        started = time.perf_counter()
        left, bottom, right, top = extent
        pad = 0.15 * max(right - left, top - bottom)
        clip = (left - pad, bottom - pad, right + pad, top + pad)
        data = self.data_bounds()
        if data is not None:                              # V-4.2: the land patch covers the whole drawn basemap
            clip = (min(clip[0], data[0]), min(clip[1], data[1]), max(clip[2], data[2]), max(clip[3], data[3]))
        self.artists = []
        axis.set_facecolor(COLOUR_LAND)
        self.land = Rectangle((clip[0], clip[1]), clip[2] - clip[0], clip[3] - clip[1], facecolor=COLOUR_LAND,
                              edgecolor="none", zorder=0.10)
        axis.add_patch(self.land)
        self.artists.append(self.land)
        for layer, colour, edge, z in (("water", COLOUR_WATER, COLOUR_WATER, 0.25), ("parks", COLOUR_PARK, COLOUR_PARK, 0.30),
                                       ("buildings", COLOUR_BUILDING, COLOUR_BUILDING_EDGE, 0.40)):
            polys = [ext for ext, _holes in self.polygons.get(layer, [])]
            if polys:
                coll = PolyCollection(polys, facecolors=colour, edgecolors=edge, linewidths=0.4, zorder=z)
                axis.add_collection(coll)
                self.artists.append(coll)
        paths = list(self.paths)
        if paths:
            coll = LineCollection(paths, colors=COLOUR_PATH, linewidths=0.7, linestyles=(0, (2.5, 2.0)), zorder=0.50)
            axis.add_collection(coll)
            self.artists.append(coll)
        steps = list(self.steps)
        if steps:
            coll = LineCollection(steps, colors=COLOUR_PATH, linewidths=1.6, linestyles=(0, (1.0, 1.4)), zorder=0.50)
            axis.add_collection(coll)
            self.artists.append(coll)
        for cls in ("minor", "medium", "major"):
            lines = list(self.roads[cls])
            if not lines:
                continue
            casing, fill = ROAD_STYLE[cls]
            z = ROAD_ZORDER[cls]
            outer = LineCollection(lines, colors=COLOUR_CASING, linewidths=casing, zorder=z, capstyle="round",
                                   joinstyle="round")
            inner = LineCollection(lines, colors=COLOUR_STREET, linewidths=fill, zorder=z + 0.05, capstyle="round",
                                   joinstyle="round")
            axis.add_collection(outer)
            axis.add_collection(inner)
            self.artists += [outer, inner]
        self.draw_labels(axis, clip, extent)
        self.label_extent = tuple(extent)
        self.draw_seconds = time.perf_counter() - started
        return self.artists

    def relabel(self, axis, extent: Tuple[float, float, float, float]) -> bool:
        """After a pan / zoom: re-place the road labels for the new extent from the cached chains.

        Visual only: no file is read, no geometry recomputed, the old label artists are removed and
        at most LABEL_MAX new ones are placed. Skipped when the extent changed by less than
        RELABEL_MIN_CHANGE (continuous drags), so the cost stays a few milliseconds per real move.
        """
        if self.label_extent is not None:
            ow, oh = self.label_extent[2] - self.label_extent[0], self.label_extent[3] - self.label_extent[1]
            nw, nh = extent[2] - extent[0], extent[3] - extent[1]
            moved = max(abs(extent[0] - self.label_extent[0]), abs(extent[1] - self.label_extent[1]),
                        abs(extent[2] - self.label_extent[2]), abs(extent[3] - self.label_extent[3]))
            if moved < RELABEL_MIN_CHANGE * max(ow, oh) and abs(nw - ow) < RELABEL_MIN_CHANGE * ow \
                    and abs(nh - oh) < RELABEL_MIN_CHANGE * oh:
                return False
        for text in self.labels:
            try:
                text.remove()
            except (ValueError, AttributeError):
                pass
            if text in self.artists:
                self.artists.remove(text)
        self.labels = []
        left, bottom, right, top = extent
        pad = 0.15 * max(right - left, top - bottom)
        clip = (left - pad, bottom - pad, right + pad, top + pad)
        self.draw_labels(axis, clip, extent)
        self.label_extent = tuple(extent)
        self.relabels += 1
        return True

    @staticmethod
    def _inside(xy: np.ndarray, clip: Tuple[float, float, float, float]) -> bool:
        return bool((xy[:, 0].max() >= clip[0]) and (xy[:, 0].min() <= clip[2])
                    and (xy[:, 1].max() >= clip[1]) and (xy[:, 1].min() <= clip[3]))

    # ----- road labels (conservative) ------------------------------------------------- #
    def draw_labels(self, axis, clip: Tuple[float, float, float, float],
                    extent: Optional[Tuple[float, float, float, float]] = None) -> List[Any]:
        """One label per named street chain (major / medium), repeated only on long streets,
        rotated along the street, never overlapping another label; skipped when too short."""
        import matplotlib.patheffects as pe

        self.labels = []
        placed: List[Tuple[float, float, float]] = []            # (x, y, radius_m) of placed labels
        metres_per_point = self._metres_per_point(axis, extent)   # sized for the given extent, deterministic
        candidates = []
        for (name, cls), chains in self.chains.items():
            for chain, length in chains:
                if length < LABEL_MIN_SEGMENT_M or not self._inside(chain, clip):
                    continue
                text_m = len(name) * LABEL_FONTSIZE * 0.58 * metres_per_point
                if length < 1.3 * text_m:
                    continue
                n_labels = max(1, int(length // LABEL_REPEAT_M)) if cls == "major" else 1
                for k in range(n_labels):
                    at = (k + 0.5) / n_labels * length
                    pos, angle = self._point_along(chain, at)
                    candidates.append((0 if cls == "major" else 1, -length, name, pos, angle, text_m))
        candidates.sort(key=lambda c: (c[0], c[1]))
        for _rank, _neg_len, name, (x, y), angle, text_m in candidates:
            if len(self.labels) >= LABEL_MAX:
                break
            radius = 0.5 * text_m + 12.0
            if any(math.hypot(x - px, y - py) < radius + pr for px, py, pr in placed):
                continue
            if not (clip[0] <= x <= clip[2] and clip[1] <= y <= clip[3]):
                continue
            text = axis.text(x, y, name, fontsize=LABEL_FONTSIZE, color=COLOUR_LABEL, ha="center", va="center",
                             rotation=angle, rotation_mode="anchor", zorder=0.95, clip_on=True,
                             path_effects=[pe.withStroke(linewidth=2.2, foreground=LABEL_HALO)])
            self.labels.append(text)
            placed.append((x, y, radius))
        self.labels_placed = len(self.labels)
        self.artists += self.labels
        return self.labels

    @staticmethod
    def _metres_per_point(axis, extent: Optional[Tuple[float, float, float, float]] = None) -> float:
        fig = axis.get_figure()
        bbox = axis.get_position()
        width_pt = bbox.width * fig.get_figwidth() * 72.0
        x0, x1 = (extent[0], extent[2]) if extent is not None else axis.get_xlim()
        return abs(x1 - x0) / max(width_pt, 1.0)

    @staticmethod
    def _length(xy: np.ndarray) -> float:
        return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))

    @staticmethod
    def _chains(pieces: List[np.ndarray]) -> List[np.ndarray]:
        """Join way pieces of one street name end-to-end (greedy, by shared endpoints)."""
        remaining = [p for p in pieces]
        chains: List[np.ndarray] = []
        while remaining:
            chain = remaining.pop(0)
            grew = True
            while grew:
                grew = False
                for i, piece in enumerate(remaining):
                    if np.allclose(chain[-1], piece[0], atol=0.5):
                        chain = np.vstack([chain, piece[1:]])
                    elif np.allclose(chain[-1], piece[-1], atol=0.5):
                        chain = np.vstack([chain, piece[::-1][1:]])
                    elif np.allclose(chain[0], piece[-1], atol=0.5):
                        chain = np.vstack([piece[:-1], chain])
                    elif np.allclose(chain[0], piece[0], atol=0.5):
                        chain = np.vstack([piece[::-1][:-1], chain])
                    else:
                        continue
                    remaining.pop(i)
                    grew = True
                    break
            chains.append(chain)
        return chains

    @staticmethod
    def _point_along(xy: np.ndarray, distance: float) -> Tuple[Tuple[float, float], float]:
        """The point at ``distance`` along the chain and the local direction (degrees, reading left-to-right)."""
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        i = int(np.searchsorted(cum, distance, side="right") - 1)
        i = max(0, min(i, len(seg) - 1))
        t = 0.0 if seg[i] == 0 else (distance - cum[i]) / seg[i]
        p = xy[i] + t * (xy[i + 1] - xy[i])
        # direction over a window around the point, so a wiggly street still gets a readable angle
        j0, j1 = max(0, i - 1), min(len(xy) - 1, i + 2)
        dx, dy = xy[j1, 0] - xy[j0, 0], xy[j1, 1] - xy[j0, 1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return (float(p[0]), float(p[1])), angle

    def remove(self) -> None:
        for artist in self.artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self.artists = []
        self.labels = []
