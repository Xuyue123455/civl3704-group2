"""Request validation and response shapes of the web API (Phase W1).

Plain dictionaries, no framework: the contract (docs/WEB_MIGRATION_CONTRACT.md §9) is expressed as required keys
and simple type checks so the frontend can develop against fixtures before every endpoint is live.  Nothing here
computes anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

PROFILES = ("standard", "step_free")
RUN_MODES = ("ANALYTICAL", "REPLAY", "LIVE")
VEHICLE_STATES = ("POSITION REPORTED", "LAST SEEN", "POSITION UNAVAILABLE", "AMBIGUOUS")   # user-facing names
WEEK7_VEHICLE_ENUM = ("LIVE", "LAST SEEN", "POSITION UNAVAILABLE", "AMBIGUOUS")             # the unchanged enum
OBJECTIVES = ("shortest", "gentlest", "fastest")
OBJECTIVE_KEYS = {"shortest": "distance", "gentlest": "effort", "fastest": "time"}         # API name -> engine key
SOURCE_FIXTURE, SOURCE_ENGINE = "fixture", "engine"

# response keys the frontend may rely on (the contract's "response concept"); adapters and fixtures must carry them
HEALTH_KEYS = ("status", "desktop_gold", "api_version", "graph", "provenance")
WALK_KEYS = ("profile", "origin", "selected_stop", "routes", "route_identity", "mapped_stairs", "model_provenance",
             "provenance")
COMPARE_KEYS = ("route", "direction", "walking_profile", "upcoming_trips", "selected_trip", "next_feasible_trip",
                "candidate_stops", "objective_winners", "selected_stop_status", "realtime_provenance", "provenance")
TRACK_KEYS = ("trip", "boarding_stop", "walking", "timing", "vehicle", "planned_route", "refresh", "provenance")
ROUTE_KEYS = ("distance_m", "walking_time_s", "effort_j_per_kg", "geometry", "edges")
CANDIDATE_KEYS = ("stop_id", "stop_name", "eligibility", "routes", "catchability", "effort_rank", "feasible_rank",
                  "highlights")


class ValidationError(ValueError):
    """A request that does not meet the contract (HTTP 400)."""


def _require(body: Dict[str, Any], key: str, kind=None) -> Any:
    if not isinstance(body, dict) or key not in body:
        raise ValidationError(f"missing field: {key}")
    value = body[key]
    if kind is not None and not isinstance(value, kind):
        raise ValidationError(f"field {key} must be {getattr(kind, '__name__', kind)}")
    return value


def validate_origin(body: Dict[str, Any]) -> Tuple[float, float]:
    origin = _require(body, "origin", dict)
    try:
        lat, lon = float(origin["lat"]), float(origin["lon"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError("origin must carry numeric lat and lon") from error
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValidationError("origin out of range")
    return lat, lon


def validate_profile(value: Any) -> str:
    if value not in PROFILES:
        raise ValidationError(f"profile must be one of {PROFILES}")
    return str(value)


def validate_walk_request(body: Dict[str, Any]) -> Dict[str, Any]:
    lat, lon = validate_origin(body)
    stop_id = str(_require(body, "stop_id"))
    profile = validate_profile(body.get("profile", "standard"))
    return {"origin": {"lat": lat, "lon": lon}, "stop_id": stop_id, "profile": profile}


def validate_compare_request(body: Dict[str, Any]) -> Dict[str, Any]:
    lat, lon = validate_origin(body)
    route = str(_require(body, "route_short_name"))
    direction = _require(body, "direction_id")
    if direction not in (0, 1, "0", "1"):
        raise ValidationError("direction_id must be 0 or 1")
    profile = validate_profile(body.get("walking_profile", "standard"))
    trip = body.get("selected_trip_id")
    stop = body.get("selected_stop_id")
    return {"origin": {"lat": lat, "lon": lon}, "route_short_name": route, "direction_id": int(direction),
            "walking_profile": profile, "selected_trip_id": None if trip in (None, "") else str(trip),
            "selected_stop_id": None if stop in (None, "") else str(stop)}


def validate_track_request(body: Dict[str, Any]) -> Dict[str, Any]:
    lat, lon = validate_origin(body)
    trip = str(_require(body, "trip_id"))
    stop = str(_require(body, "boarding_stop_id"))
    profile = validate_profile(body.get("walking_profile", "standard"))
    return {"origin": {"lat": lat, "lon": lon}, "trip_id": trip, "boarding_stop_id": stop, "walking_profile": profile}


def provenance(source: str, note: str, **extra: Any) -> Dict[str, Any]:
    """Every response says where its numbers come from: a golden fixture or the live engines.  A fixture is never
    presented as live production output."""
    from . import API_VERSION, DESKTOP_GOLD
    if source not in (SOURCE_FIXTURE, SOURCE_ENGINE):
        raise ValueError(source)
    block = {"source": source, "desktop_gold": DESKTOP_GOLD, "api_version": API_VERSION, "note": note,
             "is_fixture": source == SOURCE_FIXTURE}
    block.update(extra)
    return block


def missing_keys(payload: Dict[str, Any], keys: Tuple[str, ...]) -> List[str]:
    return [k for k in keys if k not in payload]


def line_string(coords: List[List[float]]) -> Dict[str, Any]:
    """GeoJSON LineString from [[lon, lat], ...] (WGS84)."""
    return {"type": "LineString", "coordinates": [[float(lon), float(lat)] for lon, lat in coords]}


def point(lon: float, lat: float) -> Dict[str, Any]:
    return {"type": "Point", "coordinates": [float(lon), float(lat)]}
