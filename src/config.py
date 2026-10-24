"""Configuration and shared paths for the Week 4 pipeline.

Single source of truth
----------------------
All tunable Week 4 numbers live in JSON, not in Python:

    config.json          local, git-ignored, edit this one
    config.example.json  committed, used as a fallback

If ``config.json`` does not exist we fall back to ``config.example.json``
so that a teammate can clone the repository and run the pipeline
without any setup step.

IMPORTANT
---------
Every value under the ``week4`` key is PROVISIONAL. It exists to let us
test that the data sources connect to each other. None of it is a
research finding. See docs/ASSUMPTIONS.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# --- Directory layout (follows the course template, do not invent new ones) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GTFS_DIR = PROJECT_ROOT / "data" / "gtfs"      # static GTFS .txt files live here
CACHE_DIR = PROJECT_ROOT / "data" / "cache"    # processed outputs live here

CONFIG_PATH = PROJECT_ROOT / "config.json"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config.example.json"


def load_config() -> Dict[str, Any]:
    """Load project configuration, preferring config.json.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: if neither config.json nor config.example.json exists.
    """
    if CONFIG_PATH.exists():
        source = CONFIG_PATH
    elif CONFIG_EXAMPLE_PATH.exists():
        source = CONFIG_EXAMPLE_PATH
        logger.warning(
            "config.json not found - falling back to config.example.json. "
            "Copy it to config.json if you want to change settings locally."
        )
    else:
        raise FileNotFoundError(
            f"No configuration found. Expected {CONFIG_PATH} or {CONFIG_EXAMPLE_PATH}."
        )

    logger.info("Loading configuration from %s", source.name)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_study_bbox(config: Dict[str, Any]) -> Dict[str, float]:
    """Return the superseded Week 4 rectangle (assumption A-01).

    SUPERSEDED for candidate-stop selection. The candidate-stop extent is now
    the official NSW Glebe suburb polygon; see src/boundary.py. The rectangle
    included neighbouring suburbs and water, which is why it was replaced.
    This function is kept only so the existing Stage 1 structure tests keep
    passing, and because a rectangle may still be useful for a coarser
    routing/network extent once that is decided. It must not be used as the
    authoritative candidate-stop geography.

    A dictionary is returned rather than a tuple on purpose: different
    libraries order bounding boxes differently (OSMnx, GeoPandas and
    Shapely all disagree), and named keys make that impossible to get wrong.

    Args:
        config: the dictionary returned by :func:`load_config`.

    Returns:
        Dict with keys ``north``, ``south``, ``east``, ``west`` in WGS84 degrees.
    """
    bounds = config["filters"]["geographic_bounds"]
    return {
        "north": float(bounds["max_latitude"]),
        "south": float(bounds["min_latitude"]),
        "east": float(bounds["max_longitude"]),
        "west": float(bounds["min_longitude"]),
    }


def get_included_modes(config: Dict[str, Any]) -> Dict[str, str]:
    """Return the included route_type -> label mapping for candidate stops.

    The mapping lives in configuration rather than in code so that the mode
    scope can be changed and reviewed in one place. The keys are GTFS extended
    route_type values as text, matching how they are read from routes.txt
    (e.g. "700" bus, "900" light rail), not the basic 0-3 codes.

    Args:
        config: the dictionary returned by :func:`load_config`.

    Returns:
        Dict of route_type -> human-readable mode label.

    Raises:
        KeyError: if filters.mode_policy.included is absent.
        ValueError: if it is present but empty.
    """
    policy = config.get("filters", {}).get("mode_policy")
    if not isinstance(policy, dict) or "included" not in policy:
        raise KeyError(
            "config.json has no filters.mode_policy.included. Copy the block "
            "from config.example.json."
        )
    included = {
        str(key): str(value)
        for key, value in policy["included"].items()
        if not str(key).startswith("_")
    }
    if not included:
        raise ValueError(
            "filters.mode_policy.included is empty, so no stop could ever be "
            "selected. Add at least one route_type."
        )
    return included


def get_excluded_modes(config: Dict[str, Any]) -> Dict[str, str]:
    """Return the excluded route_type -> reason mapping, for reporting only.

    Nothing in the pipeline branches on this. It exists so that an exclusion
    always carries its recorded reason with it.

    Args:
        config: the dictionary returned by :func:`load_config`.

    Returns:
        Dict of route_type -> reason. Empty if none are recorded.
    """
    policy = config.get("filters", {}).get("mode_policy") or {}
    excluded = policy.get("excluded") or {}
    return {
        str(key): str(value)
        for key, value in excluded.items()
        if not str(key).startswith("_")
    }


def get_output_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Return the configured output paths, as written in the configuration.

    The values are relative to :data:`PROJECT_ROOT` and are returned unresolved
    so that the caller decides how to anchor them.

    Args:
        config: the dictionary returned by :func:`load_config`.

    Returns:
        Dict of output name -> path string.
    """
    outputs = config["week4"]["outputs"]
    return {
        str(key): str(value)
        for key, value in outputs.items()
        if not str(key).startswith("_")
    }


def get_test_origin(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the provisional test origin used to validate the pipeline.

    This is NOT a research choice. It is one arbitrary point inside Glebe
    used only to prove that routing runs end to end.

    Args:
        config: the dictionary returned by :func:`load_config`.

    Returns:
        Dict with keys ``name``, ``latitude``, ``longitude``.
    """
    origin = config["week4"]["test_origin"]
    return {
        "name": str(origin["name"]),
        "latitude": float(origin["latitude"]),
        "longitude": float(origin["longitude"]),
    }


def get_routing_network_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Stage 3 routing-network settings.

    ``buffer_metres`` may be ``None``. That is not a bug: no buffer distance
    has been adopted yet, and this function will not invent one. The caller is
    expected to require an explicit value rather than fall back to a default,
    so that a test level never becomes a permanent project assumption by
    accident. See docs/ASSUMPTIONS.md (A-07).

    Args:
        config: the dictionary returned by :func:`load_config`.

    Returns:
        Dict with keys ``buffer_metres`` (float or None), ``buffer_status``,
        ``sensitivity_buffers_metres``, ``metric_crs``, ``network_type``,
        ``simplify``, ``truncate_by_edge`` and ``retain_all``.

    Raises:
        KeyError: if week4.routing_network is absent.
    """
    try:
        settings = config["week4"]["routing_network"]
    except KeyError as error:
        raise KeyError(
            "config.json has no week4.routing_network block. Copy it from "
            "config.example.json."
        ) from error

    buffer_metres = settings.get("buffer_metres")
    return {
        "buffer_metres": None if buffer_metres is None else float(buffer_metres),
        "buffer_status": str(settings.get("buffer_status", "OPEN")),
        "sensitivity_buffers_metres": [
            float(value) for value in settings.get("sensitivity_buffers_metres", [])
        ],
        "metric_crs": str(settings.get("metric_crs", "EPSG:7856")),
        "network_type": str(settings.get("network_type", "walk")),
        "simplify": bool(settings.get("simplify", True)),
        "truncate_by_edge": bool(settings.get("truncate_by_edge", True)),
        "retain_all": bool(settings.get("retain_all", True)),
    }


def ensure_directories() -> None:
    """Create the data directories if they are missing.

    ``data/gtfs`` and ``data/cache`` are kept in git via .gitkeep files,
    but a fresh checkout on another machine may still lack them.
    """
    for directory in (GTFS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
