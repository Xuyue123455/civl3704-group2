"""Mode-aware realtime feed adapter (Week 11, L1 Inner West Light Rail integration).

One pipeline, several feeds. The validated chain

    fetch -> V1 parse -> V2 events -> service intent -> exact trip -> catchability
          -> Compare -> Track -> auto-refresh

is unchanged; this module only decides WHICH TfNSW feed the two fetches hit,
from the selected service's mode:

    service mode  --+--> bus         Trip Updates v1  /buses            Vehicle Positions v1  /buses
                    +--> light_rail  Trip Updates v2  /lightrail/innerwest   Vehicle Positions v2  /lightrail/innerwest

The mode comes from the project's normalised mode metadata (``config.json``
``filters.mode_policy``: GTFS ``route_type`` 700 -> ``bus``, 900 ->
``light_rail``), as carried by the Week 9 ``ServiceChoice.mode`` and the
candidate-stop table's ``mode`` column - never from a route name.

API key handling is exactly the validated Week 7 chain: ``V1.load_api_key``
reads ``TFNSW_API_KEY`` (environment or .env), ``V1.fetch_feed`` sends it as
the ``Authorization: apikey`` header once and discards it, ``V1.redact``
strips it from any error. Nothing here prints, stores or returns the key.

Feed provenance
---------------
* bus Trip Updates / Vehicle Positions: the V1 / V4A registries (``buses``, VERIFIED).
* light rail v2 Trip Updates: ``https://api.transport.nsw.gov.au/v2/gtfs/realtime/lightrail/innerwest``
  - quoted on the TfNSW Open Data Forum thread "Public Transport - Realtime
  Vehicle Positions v2" (…/t/public-transport-realtime-vehicle-positions-v2/2550)
  and confirmed by the real capture of 2026-10-17 (8 Trip Updates, 8/8 trip_ids
  in the static bundle, all four Glebe / Jubilee platforms present).
* light rail v2 Vehicle Positions: the V4A registry entry ``lightrail_innerwest_v2``
  (VERIFIED), confirmed by the same capture (6 reports, 6/6 trip_ids, ages 3-5 s).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .walk_engine import load_script

MODE_BUS = "bus"
MODE_LIGHT_RAIL = "light_rail"
SUPPORTED_MODES = (MODE_BUS, MODE_LIGHT_RAIL)

LIGHT_RAIL_TRIP_UPDATES_V2 = "https://api.transport.nsw.gov.au/v2/gtfs/realtime/lightrail/innerwest"
LIGHT_RAIL_VEHICLE_POSITIONS_V2 = "https://api.transport.nsw.gov.au/v2/gtfs/vehiclepos/lightrail/innerwest"

# neutral feed metadata (Technical details); URLs carry no secret
FEEDS: Dict[str, Dict[str, Dict[str, str]]] = {
    MODE_BUS: {
        "trip_updates": {"key": "buses", "label": "bus", "version": "v1", "url": ""},            # url from V1
        "vehicle_positions": {"key": "buses", "label": "bus", "version": "v1", "url": ""},      # url from V4A
    },
    MODE_LIGHT_RAIL: {
        "trip_updates": {"key": "lightrail_innerwest_v2", "label": "Inner West light rail", "version": "v2",
                         "url": LIGHT_RAIL_TRIP_UPDATES_V2},
        "vehicle_positions": {"key": "lightrail_innerwest_v2", "label": "Inner West light rail", "version": "v2",
                              "url": LIGHT_RAIL_VEHICLE_POSITIONS_V2},
    },
}
FEED_LABELS = {MODE_BUS: "bus", MODE_LIGHT_RAIL: "light rail"}


def normalise_mode(mode: Any) -> str:
    """'bus' / 'light_rail' from the project's mode metadata; anything else defaults to bus (as before)."""
    text = str(mode or "").strip().lower()
    return text if text in SUPPORTED_MODES else MODE_BUS


def mode_for_choice(choice: Any) -> str:
    """The mode of a Week 9 ServiceChoice (route_type -> mode_policy), never the route name."""
    return normalise_mode(getattr(choice, "mode", None))


def mode_for_trip(tables: Any, trip_id: Any) -> str:
    """The mode of an exact static trip through the service-intent tables (trip_id -> route/direction -> mode)."""
    found = tables.trip_direction(trip_id) if tables is not None and trip_id is not None else None
    if found:
        choices = tables.find(found[0], found[1])
        if choices:
            return mode_for_choice(choices[0])
    return MODE_BUS


def mode_for_stop(stops: Any, stop_id: Any) -> str:
    rows = stops[stops["stop_id"].astype(str) == str(stop_id)]
    return normalise_mode(rows.iloc[0]["mode"]) if len(rows) else MODE_BUS


# ----- the fetchers ------------------------------------------------------------------ #
def _v1():
    return load_script("inspect_gtfs_realtime")


def _redacted_fetch(url: str, timeout: Any = None) -> bytes:
    """ONE bounded GET through V1 with the validated key handling; errors are redacted."""
    v1 = _v1()
    api_key = v1.load_api_key()
    arguments = {} if timeout is None else {"timeout": timeout}
    try:
        return v1.fetch_feed(url, api_key, **arguments).payload
    except v1.RealtimeError as error:
        raise v1.RealtimeError(v1.redact(error, api_key)) from None


def trip_update_url(mode: Any) -> str:
    mode = normalise_mode(mode)
    if mode == MODE_LIGHT_RAIL:
        return LIGHT_RAIL_TRIP_UPDATES_V2
    return _v1().resolve_feed("buses").url


def vehicle_position_url(mode: Any) -> str:
    mode = normalise_mode(mode)
    if mode == MODE_LIGHT_RAIL:
        return LIGHT_RAIL_VEHICLE_POSITIONS_V2
    return load_script("inspect_vehicle_positions").resolve_vehicle_feed("buses").url


def trip_update_fetcher(mode: Any) -> Callable[[], bytes]:
    """The Trip Update fetcher of this mode (bus: the validated V3 fetch; light rail: the v2 feed)."""
    mode = normalise_mode(mode)
    if mode == MODE_BUS:
        return load_script("demo_week7_realtime").fetch_payload          # unchanged bus behaviour

    def fetch_light_rail() -> bytes:
        return _redacted_fetch(LIGHT_RAIL_TRIP_UPDATES_V2)
    fetch_light_rail.feed_mode = MODE_LIGHT_RAIL                        # type: ignore[attr-defined]
    return fetch_light_rail


def vehicle_position_fetcher(mode: Any) -> Callable[[], bytes]:
    """The Vehicle Position fetcher of this mode (bus: the validated V4B fetch; light rail: the v2 feed)."""
    mode = normalise_mode(mode)
    if mode == MODE_BUS:
        return load_script("demo_week7_live_vehicles").fetch_vehicle_payload   # unchanged bus behaviour

    def fetch_light_rail() -> bytes:
        return _redacted_fetch(LIGHT_RAIL_VEHICLE_POSITIONS_V2)
    fetch_light_rail.feed_mode = MODE_LIGHT_RAIL                        # type: ignore[attr-defined]
    return fetch_light_rail


def feed_metadata(mode: Any) -> Dict[str, Any]:
    """Neutral facts for Technical details: mode, feed keys, versions, URLs (no secret)."""
    mode = normalise_mode(mode)
    entry = {k: dict(v) for k, v in FEEDS[mode].items()}
    try:
        entry["trip_updates"]["url"] = trip_update_url(mode)
        entry["vehicle_positions"]["url"] = vehicle_position_url(mode)
    except Exception:                                              # registries unavailable: keep the static text
        pass
    return {"mode": mode, "label": FEED_LABELS[mode], **entry}


def feed_lines(mode: Any) -> list:
    """Technical-details lines: 'Feed mode: light_rail' / 'Trip updates: lightrail_innerwest_v2 (v2)' ..."""
    meta = feed_metadata(mode)
    return [f"Feed mode: {meta['mode']}",
            f"  Trip updates feed: {meta['trip_updates']['key']} ({meta['trip_updates']['version']})",
            f"  Vehicle positions feed: {meta['vehicle_positions']['key']} ({meta['vehicle_positions']['version']})"]
