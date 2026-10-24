"""Read-only catalogue endpoints for the web client (Mobile Web Phase 1).

    GET /api/stops      the 37 candidate physical stops (identity, mode, WGS84 point, snap facts)
    GET /api/services   every selectable route + direction (ServiceIntentTables.choices), with its stop ids
    GET /api/boundary   the official Glebe suburb polygon (data/boundaries/glebe_suburb.geojson, tracked)
    GET /api/basemap    the project's own vector basemap cache as GeoJSON LineStrings (streets / paths),
                        so the browser map needs no external tile server and matches the desktop look
    GET /api/stairs     every mapped stair segment (mapped_steps.json geometry) with the stair model's
                        vertical change where the Standard cache costs that edge
    GET /api/contours   contour lines for the study area when scripts/build_contours.py has produced them,
                        else {"available": false, ...} - never an invented terrain layer
    GET /api/replays    the named server-side replay sources (recorded feeds; the browser never sends a path)

Everything here is a re-shaping of files and tables the desktop already loads.  Nothing is computed: no
distance, effort, time, catchability or stair cost.  Presentation data (basemap, contours) is kept separate from
the analytical caches and is never read by an analytical module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import schemas
from .engines import PROJECT_ROOT, EngineHost

BOUNDARY_PATH = PROJECT_ROOT / "data" / "boundaries" / "glebe_suburb.geojson"
CONTOUR_PATH = PROJECT_ROOT / "data" / "cache" / "contours" / "contours.geojson"
CONTOUR_META_PATH = PROJECT_ROOT / "data" / "cache" / "contours" / "contours_meta.json"

_BASEMAP_CACHE: Dict[str, Any] = {}


def stops(host: EngineHost) -> Dict[str, Any]:
    frame = host.engine.stops
    features: List[Dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        record = row._asdict()
        features.append({
            "type": "Feature",
            "geometry": schemas.point(float(record["stop_lon"]), float(record["stop_lat"])),
            "properties": {
                "stop_id": str(record["stop_id"]), "stop_name": str(record["stop_name"]),
                "mode": str(record.get("mode", "")), "n_routes": _int_or_none(record.get("n_routes")),
                "logical_station_id": str(record.get("logical_station_id", "")),
                "nearest_node": _int_or_none(record.get("nearest_node")),
                "snap_distance_m": _float_or_none(record.get("snap_distance_m")),
                "component_id": str(record.get("component_id", "")),
            },
        })
    return {"type": "FeatureCollection", "features": features, "count": len(features),
            "source": "data/cache/glebe_stops_snapped.csv (Week 4 Stage 4; the 37 candidate physical stops)",
            "provenance": schemas.provenance(schemas.SOURCE_ENGINE, "candidate stops as loaded by the desktop WALK engine")}


def services(host: EngineHost) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    for choice in host.tables.choices():
        out.append({
            "key": choice.key, "route_short_name": str(choice.route_short_name),
            "direction_id": int(choice.direction_id), "route_id": str(choice.route_id), "mode": str(choice.mode),
            "route_direction": str(choice.route_direction), "headsigns": list(choice.headsigns),
            "stop_ids": list(choice.stop_ids), "n_trips": int(choice.n_trips),
            "label": f"{choice.route_short_name} → {choice.route_direction}" if choice.route_direction
                     else str(choice.route_short_name),
        })
    return {"services": out, "count": len(out),
            "source": "results/service_intent/ (static GTFS, built once by scripts/build_service_intent.py)",
            "provenance": schemas.provenance(schemas.SOURCE_ENGINE, "service intent tables as loaded by the desktop COMPARE engine")}


def boundary() -> Dict[str, Any]:
    payload = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    payload["provenance"] = schemas.provenance(schemas.SOURCE_ENGINE, "official NSW suburb polygon (tracked file)")
    return payload


def basemap(path: Optional[Path] = None) -> Dict[str, Any]:
    """The desktop basemap cache (roads / paths polylines, WGS84) as a GeoJSON FeatureCollection."""
    from unified_ui.basemap import preferred_cache                     # presentation module, never analytical
    target = Path(path) if path else preferred_cache()
    key = str(target)
    if key in _BASEMAP_CACHE:
        return _BASEMAP_CACHE[key]
    if not target.exists():
        return {"available": False, "reason": "no local basemap cache; build it with scripts/build_basemap_cache.py",
                "type": "FeatureCollection", "features": []}
    raw = json.loads(target.read_text(encoding="utf-8"))
    features: List[Dict[str, Any]] = []
    for layer in ("roads", "paths"):
        for item in raw.get("layers", {}).get(layer, []) or []:
            coords = item.get("coords") or []
            if len(coords) < 2:
                continue
            features.append({"type": "Feature",
                             "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]},
                             "properties": {"layer": layer, "class": item.get("class"), "name": item.get("name") or "",
                                            "highway": item.get("highway")}})
    payload = {"type": "FeatureCollection", "features": features, "available": True,
               "cache": target.name, "counts": raw.get("counts"), "extent_lonlat": _extent(features),
               "purpose": "visual context only - never read by the analytical modules",
               "provenance": schemas.provenance(schemas.SOURCE_ENGINE, f"desktop basemap cache {target.name}")}
    _BASEMAP_CACHE[key] = payload
    return payload


def stairs(host: EngineHost) -> Dict[str, Any]:
    """Every mapped stair segment, with the stair model's vertical change for the directed edges it costs."""
    from unified_ui.raw_shortest import MappedSteps
    standard = host.standard
    steps = standard.mapped_steps if standard is not None and standard.mapped_steps is not None else MappedSteps.load()
    if steps is None:
        return {"available": False, "reason": "data/cache/mapped_steps.json not present", "type": "FeatureCollection", "features": []}
    rows = standard.stair_rows if standard is not None else {}
    features: List[Dict[str, Any]] = []
    seen = set()
    for (u, v, k), record in steps.by_edge.items():
        undirected = (min(int(u), int(v)), max(int(u), int(v)), int(k))      # one feature per physical stair edge
        if undirected in seen:
            continue
        seen.add(undirected)
        parts = record.get("stair_parts_lonlat") or []
        forward, backward = rows.get((int(u), int(v), int(k))), rows.get((int(v), int(u), int(k)))
        features.append({
            "type": "Feature",
            "geometry": {"type": "MultiLineString",
                         "coordinates": [[[float(lon), float(lat)] for lon, lat in part] for part in parts]},
            "properties": {
                "u": int(u), "v": int(v), "key": int(k), "name": record.get("name") or "",
                "step_count": _float_or_none(record.get("step_count")), "stair_length_m": _float_or_none(record.get("stair_length_m")),
                "handrail": record.get("handrail"), "ramp": record.get("ramp"), "incline_osm": record.get("incline"),
                "marker": [float(record["marker_lon"]), float(record["marker_lat"])] if record.get("marker_lon") is not None else None,
                "vertical_change_abs_m": _abs_rise(forward, backward),
                "confidence_class": (forward or backward or {}).get("confidence_class"),
                "eligible": bool((forward or backward or {}).get("eligible", False)),
                "elevation_source": ("stair model cache (data/cache/stair_model/stair_costs.csv)" if (forward or backward)
                                     else "no stair-model row for this edge: elevation unknown"),
            },
        })
    return {"type": "FeatureCollection", "features": features, "count": len(features), "available": True,
            "direction_rule": "ascent / descent is decided per route from the travel direction of the stair edge "
                              "(POST /api/walk routes[*].stairs); this layer carries the unsigned vertical change only",
            "provenance": schemas.provenance(schemas.SOURCE_ENGINE, "mapped_steps.json + stair_model cache, as loaded by the desktop")}


def contours() -> Dict[str, Any]:
    if not CONTOUR_PATH.exists():
        return {"available": False, "type": "FeatureCollection", "features": [],
                "reason": "contours not built on this server: python scripts/build_contours.py (needs the DEM tiles)",
                "contract": {"minor_interval_m": 5, "index_interval_m": 20, "crs": "EPSG:4326",
                             "properties": ["elevation_m", "index", "label", "kind", "rotate"],
                             "kinds": {"line": "the contour", "label": "a label anchor point on an index contour (text rotated by 'rotate')"}}}
    payload = json.loads(CONTOUR_PATH.read_text(encoding="utf-8"))
    payload["available"] = True
    if CONTOUR_META_PATH.exists():
        payload["meta"] = json.loads(CONTOUR_META_PATH.read_text(encoding="utf-8"))
    payload["provenance"] = schemas.provenance(schemas.SOURCE_ENGINE,
                                               "display contours derived from the DEM by scripts/build_contours.py; presentation only")
    return payload


HILLSHADE_PATH = CONTOUR_PATH.with_name("hillshade.png")
HILLSHADE_META_PATH = CONTOUR_PATH.with_name("hillshade_meta.json")


def hillshade() -> Dict[str, Any]:
    """The relief-shading image's descriptor (Phase 2): corner coordinates + where the PNG is served."""
    if not HILLSHADE_PATH.exists() or not HILLSHADE_META_PATH.exists():
        return {"available": False,
                "reason": "terrain shading not built on this server: python scripts/build_contours.py (needs the DEM tiles)",
                "contract": {"image": "GET /api/hillshade.png", "coordinates": "top-left, top-right, bottom-right, bottom-left (WGS84)"}}
    meta = json.loads(HILLSHADE_META_PATH.read_text(encoding="utf-8"))
    return {"available": True, "url": "/api/hillshade.png", "coordinates": meta["coordinates_wgs84"], "meta": meta,
            "provenance": schemas.provenance(schemas.SOURCE_ENGINE,
                                             "relief shading derived from the DEM by scripts/build_contours.py; presentation only")}


def hillshade_png() -> bytes:
    if not HILLSHADE_PATH.exists():
        raise FileNotFoundError(str(HILLSHADE_PATH))
    return HILLSHADE_PATH.read_bytes()


TERRAIN_DIR = PROJECT_ROOT / "data" / "cache" / "terrain_display"
TERRAIN_META_PATH = TERRAIN_DIR / "terrain_metadata.json"
TERRAIN_FILES = {"elevation_tint.png", "hillshade.png"}


def terrain() -> Dict[str, Any]:
    """The terrain DISPLAY surface descriptor (terrain redesign): elevation tint + soft hillshade images (one set of WGS84
    corners), sparse major contours, the robust elevation range for the legend, the DEM coverage polygon and the audit.
    Presentation only: nothing here is read by an analytical module, and it changes no routing, effort, time, stairs or
    catchability."""
    if not TERRAIN_META_PATH.exists() or not all((TERRAIN_DIR / f).exists() for f in TERRAIN_FILES):
        return {"available": False,
                "reason": "terrain display surface not built on this server: python scripts/build_terrain_display.py (needs the DEM tiles)",
                "contract": {"images": ["GET /api/terrain/elevation_tint.png", "GET /api/terrain/hillshade.png"],
                             "contours": "GET /api/terrain/contours", "coordinates": "top-left, top-right, bottom-right, bottom-left (WGS84)"}}
    meta = json.loads(TERRAIN_META_PATH.read_text(encoding="utf-8"))
    display = meta.get("display", {})
    return {"available": True,
            "elevation_tint_url": "/api/terrain/elevation_tint.png", "hillshade_url": "/api/terrain/hillshade.png",
            "contours_url": "/api/terrain/contours",
            "coordinates": meta["coordinates_wgs84"], "coverage_polygon": meta.get("coverage_polygon_wgs84"),
            "elevation_range_m": meta["elevation_range_m"], "contour_interval_m": display.get("contours", {}).get("interval_m"),
            "contour_levels_m": display.get("contours", {}).get("levels_m"), "display_cell_m": display.get("cell_m"),
            "hillshade_cell_m": display.get("hillshade", {}).get("cell_m"),
            "source": meta.get("source"), "tint_stops": display.get("tint_stops"), "meta": meta,
            "analytical_effect": "none: terrain visualization does not affect routing calculations",
            "provenance": schemas.provenance(schemas.SOURCE_ENGINE,
                                             "terrain display surface derived from the DEM by scripts/build_terrain_display.py; presentation only")}


def terrain_contours() -> Dict[str, Any]:
    path = TERRAIN_DIR / "major_contours.geojson"
    if not path.exists():
        return {"available": False, "type": "FeatureCollection", "features": [],
                "reason": "terrain display surface not built on this server: python scripts/build_terrain_display.py (needs the DEM tiles)"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["available"] = True
    payload["provenance"] = schemas.provenance(schemas.SOURCE_ENGINE,
                                               "sparse major display contours derived from the DEM by scripts/build_terrain_display.py; presentation only")
    return payload


def terrain_png(name: str) -> bytes:
    if name not in TERRAIN_FILES:
        raise FileNotFoundError(name)
    path = TERRAIN_DIR / name
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path.read_bytes()


def replays() -> Dict[str, Any]:
    from .session import REPLAY_SOURCES
    out = []
    for name, source in REPLAY_SOURCES.items():
        out.append({"name": name, "description": source.get("description"),
                    "available": Path(source["trip_updates"]).exists(),
                    "has_vehicle_positions": bool(source.get("vehicle_positions")) and Path(source["vehicle_positions"]).exists()})
    return {"replays": out, "note": "a replay is a recorded snapshot and is never presented as live"}


# ----- helpers ------------------------------------------------------------------------ #
def _int_or_none(value: Any) -> Optional[int]:
    try:
        return None if value is None or value != value else int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return None if value is None or value != value else float(value)
    except (TypeError, ValueError):
        return None


def _abs_rise(forward: Optional[Dict[str, Any]], backward: Optional[Dict[str, Any]]) -> Optional[float]:
    for row in (forward, backward):
        if row is None:
            continue
        value = _float_or_none(row.get("signed_vertical_change_m"))
        if value is not None:
            return round(abs(value), 2)
    return None


def _extent(features: List[Dict[str, Any]]) -> Optional[List[float]]:
    xs = [c[0] for f in features for c in f["geometry"]["coordinates"]]
    ys = [c[1] for f in features for c in f["geometry"]["coordinates"]]
    return [min(xs), min(ys), max(xs), max(ys)] if xs else None
