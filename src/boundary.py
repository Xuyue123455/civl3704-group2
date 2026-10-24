"""Official study-area boundary for candidate stop selection.

The candidate-stop extent is the official NSW Glebe suburb polygon, not a
rectangle. A rectangle around irregular Glebe would still include neighbouring
suburbs and water, which is what contaminated the earlier A-01 extent.

A bounding box is still useful, but only as a cheap pre-filter. It is always
derived from the polygon at run time and never stored, so the two cannot drift
apart.

See scripts/fetch_boundary.py for how the polygon is obtained and
data/boundaries/boundary_provenance.json for its provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from src.config import PROJECT_ROOT

BOUNDARY_DIR = PROJECT_ROOT / "data" / "boundaries"
CANDIDATE_POLYGON_PATH = BOUNDARY_DIR / "glebe_suburb.geojson"
BOUNDARY_PROVENANCE_PATH = BOUNDARY_DIR / "boundary_provenance.json"


class BoundaryUnavailable(RuntimeError):
    """Raised with an actionable message when the boundary cannot be used."""


def load_candidate_polygon(
    path: Path | None = None,
) -> Tuple[BaseGeometry, Dict[str, Any]]:
    """Load and validate the single official candidate-stop polygon.

    Args:
        path: override for the GeoJSON file. Defaults to the project boundary.

    Returns:
        Tuple of (geometry, feature properties).

    Raises:
        BoundaryUnavailable: if the file is missing, does not contain exactly
            one feature, or the geometry is not a valid non-empty polygon.
    """
    path = path or CANDIDATE_POLYGON_PATH
    if not path.exists():
        raise BoundaryUnavailable(
            f"{path} is missing. Run scripts/fetch_boundary.py to obtain the "
            "official Glebe suburb boundary."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if features is None:
        raise BoundaryUnavailable(
            f"{path} has no 'features' key; it is not a GeoJSON FeatureCollection."
        )
    if len(features) != 1:
        raise BoundaryUnavailable(
            f"{path} contains {len(features)} features; exactly one is required. "
            "Re-run scripts/fetch_boundary.py."
        )

    geometry = shape(features[0]["geometry"])
    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise BoundaryUnavailable(
            f"Boundary geometry is {geometry.geom_type}; expected Polygon or "
            "MultiPolygon."
        )
    if geometry.is_empty:
        raise BoundaryUnavailable("The boundary geometry is empty.")
    if not geometry.is_valid:
        raise BoundaryUnavailable(
            "The boundary geometry is invalid (self-intersection or similar). "
            "Inspect it before using it to select candidate stops."
        )

    return geometry, (features[0].get("properties") or {})


def polygon_bbox(geometry: BaseGeometry) -> Dict[str, float]:
    """Derive a bounding box from a polygon, for use as a pre-filter only.

    A dictionary is returned rather than a tuple because different libraries
    order bounding boxes differently; named keys make that impossible to get
    wrong. This mirrors config.get_study_bbox.

    Args:
        geometry: the candidate-stop polygon.

    Returns:
        Dict with keys ``north``, ``south``, ``east``, ``west`` in WGS84 degrees.
    """
    minx, miny, maxx, maxy = geometry.bounds
    return {"north": maxy, "south": miny, "east": maxx, "west": minx}


def load_boundary_provenance(path: Path | None = None) -> Dict[str, Any]:
    """Return the recorded provenance of the boundary, or an empty dict."""
    path = path or BOUNDARY_PROVENANCE_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
