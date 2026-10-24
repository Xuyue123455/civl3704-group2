"""Week 7 V4A - TfNSW live Vehicle Positions: inspection and identifier audit.

WHAT THIS IS
------------
RECONNAISSANCE, not visualisation. The longer-term idea is to show live
vehicles moving on the existing Glebe map, but before anything is drawn this
script establishes what the real TfNSW Vehicle Position feeds actually contain
and whether their identifiers can be connected - safely, by exact match - to
what this project already has:

    VehiclePosition  ->  Trip Update  ->  static GTFS  ->  a Glebe service

V2 already showed that a TfNSW realtime ``route_id`` and a static ``route_id``
can live in different identifier namespaces. So every link here is MEASURED.
Nothing is assumed, and no identifier is parsed, split or reinterpreted to
guess what it means: a realtime route_id such as "2459_433" is an opaque token
in this file.

WHAT IT DOES NOT DO
-------------------
It draws no vehicle, stores no history, builds no trajectory, and does not
touch ``scripts/demo_week7_realtime.py``, the Week 5 routing code or any Week 5
number. One snapshot is not a trajectory. See D-27 in docs/ASSUMPTIONS.md.

WHAT IT REUSES
--------------
``scripts/inspect_gtfs_realtime.py``     V1: the API-key loader, the single
                                         bounded request, the error and
                                         redaction behaviour, the protobuf
                                         decode, Sydney local time, the
                                         presence-aware field reader and the
                                         gitignored raw-capture helper
``scripts/show_glebe_live_arrivals.py``  V2: the Glebe candidate stops, the
                                         static GTFS loader, the Trip Update
                                         event builder and its adopted
                                         route-resolution chain
``src/boundary.py``                      the official Glebe suburb polygon,
                                         the same file Stage 2 used

There is no second decoder, no second key loader and no second networking
convention in this file.

PRESENCE
--------
GTFS-Realtime is proto2. A genuine zero is data: ``speed`` 0, ``bearing`` 0,
``current_stop_sequence`` 0, and even latitude/longitude 0 are real values,
while an ABSENT field is absent. Every field is therefore read through V1's
``field_or_none``, which asks ``HasField`` first.

Usage
-----
    python scripts/inspect_vehicle_positions.py
    python scripts/inspect_vehicle_positions.py --feed buses --sample 5
    python scripts/inspect_vehicle_positions.py --feed lightrail_innerwest_v2
    python scripts/inspect_vehicle_positions.py --audit
    python scripts/inspect_vehicle_positions.py --list-feeds
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import boundary  # noqa: E402


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The finished ingestion layer (V1) and the finished Glebe join (V2), used as
# they stand. Neither file is modified by this work.
v2 = load_script("show_glebe_live_arrivals")
v1 = v2.v1

RealtimeError = v1.RealtimeError
MissingApiKey = v1.MissingApiKey
field_or_none = v1.field_or_none
to_sydney = v1.to_sydney
redact = v1.redact
enum_name = v1.enum_name

FeedDefinition = v1.FeedDefinition
STATUS_VERIFIED = v1.STATUS_VERIFIED
STATUS_SUPERSEDED = v1.STATUS_SUPERSEDED
STATUS_DEPRECATED = "DEPRECATED_DO_NOT_USE"

# --------------------------------------------------------------------------- #
# feed registry - every URL carries its provenance
#
# These are VEHICLE POSITION feeds. They are a different API family from the
# Trip Update feeds V1 registers, so they are registered here rather than by
# editing V1, whose contents V2 and V3 already depend on.
# --------------------------------------------------------------------------- #
VEHICLE_POSITION_BASE_V1 = "https://api.transport.nsw.gov.au/v1/gtfs/vehiclepos/"
VEHICLE_POSITION_BASE_V2 = "https://api.transport.nsw.gov.au/v2/gtfs/vehiclepos/"

PROVENANCE_BUS = (
    "Endpoint supplied in the V4A task specification as verified against "
    "TfNSW Open Data / TfNSW Open Data Forum sources, and independently "
    "corroborated on 2026-09-10 by two public pages that quote the same URL "
    "verbatim: the TfNSW Open Data Forum thread 'Vehicle position API "
    "documentation' "
    "(https://opendataforum.transport.nsw.gov.au/t/"
    "vehicle-position-api-documentation/1009), and the Transitland Atlas feed "
    "registry entry for opendata.transport.nsw.gov.au. The TfNSW resource "
    "page itself is behind an Open Data Hub login and was NOT read here."
)

PROVENANCE_LIGHT_RAIL_V2 = (
    "Endpoint supplied in the V4A task specification as verified against "
    "TfNSW Open Data / TfNSW Open Data Forum sources, and independently "
    "corroborated on 2026-09-10 by the TfNSW Open Data Forum thread 'Public "
    "Transport - Realtime Vehicle Positions v2' "
    "(https://opendataforum.transport.nsw.gov.au/t/"
    "public-transport-realtime-vehicle-positions-v2/2550), which quotes this "
    "exact v2 vehiclepos URL for the Inner West Light Rail and describes it "
    "as the replacement for the v1 path."
)

VEHICLE_FEEDS: Dict[str, FeedDefinition] = {
    "buses": FeedDefinition(
        key="buses",
        label="Buses - Vehicle Positions (v1)",
        url=VEHICLE_POSITION_BASE_V1 + "buses",
        modes="Sydney and surrounds bus network",
        status=STATUS_VERIFIED,
        provenance=PROVENANCE_BUS,
    ),
    "lightrail_innerwest_v2": FeedDefinition(
        key="lightrail_innerwest_v2",
        label="Inner West Light Rail - Vehicle Positions (v2)",
        url=VEHICLE_POSITION_BASE_V2 + "lightrail/innerwest",
        modes="Inner West Light Rail (L1)",
        status=STATUS_VERIFIED,
        provenance=PROVENANCE_LIGHT_RAIL_V2,
        caveat=(
            "This project's Glebe candidate light-rail stops have no verified "
            "Trip Update feed yet, so a vehicle seen here cannot currently be "
            "tied to a Glebe light-rail service the way a bus can."
        ),
    ),
    "lightrail_innerwest_v1": FeedDefinition(
        key="lightrail_innerwest_v1",
        label="Inner West Light Rail - Vehicle Positions (v1, DEPRECATED)",
        url=None,                       # deliberately not fetchable
        modes="Inner West Light Rail (L1)",
        status=STATUS_DEPRECATED,
        provenance=(
            "The v1 path " + VEHICLE_POSITION_BASE_V1
            + "lightrail/innerwest is published in third-party feed "
            "registries, but TfNSW has superseded it with the v2 Vehicle "
            "Positions feed for this mode."
        ),
        caveat=(
            "Registered ONLY so that asking for it fails loudly instead of "
            "silently returning data that would be presented as current. Use "
            "lightrail_innerwest_v2."
        ),
    ),
}

DEFAULT_FEED = "buses"
BUS_MODE_KEYS = ("buses",)

# The Trip Update feed this audit compares against. It is V1's registered,
# verified bus feed - not a second URL.
TRIP_UPDATE_FEED = v2.BUS_FEED

DEFAULT_SAMPLE = 3
MAX_SAMPLE = 20
EXAMPLE_LIMIT = 10          # bounded examples of any unmatched identifier set

LATITUDE_RANGE = (-90.0, 90.0)
LONGITUDE_RANGE = (-180.0, 180.0)

RULE = "-" * 72


class VehiclePositionError(RealtimeError):
    """Raised with an actionable message when V4A cannot continue."""


# --------------------------------------------------------------------------- #
# feed selection
# --------------------------------------------------------------------------- #
def resolve_vehicle_feed(key: str, url_override: Optional[str] = None
                         ) -> FeedDefinition:
    """The registered Vehicle Position feed, or an explicit refusal."""
    if key not in VEHICLE_FEEDS:
        raise VehiclePositionError(
            f"Unknown vehicle-position feed {key!r}. Registered feeds: "
            f"{', '.join(sorted(VEHICLE_FEEDS))}. Run with --list-feeds to "
            "see each endpoint and where it came from.")
    definition = VEHICLE_FEEDS[key]
    if url_override:
        return FeedDefinition(
            key=definition.key, label=definition.label, url=url_override,
            modes=definition.modes, status="SUPPLIED_BY_CALLER",
            provenance=("URL supplied on the command line by the operator; "
                        "not verified by this script."),
            caveat=definition.caveat)
    if not definition.usable:
        raise VehiclePositionError(
            f"{definition.label} has no usable endpoint here. "
            f"{definition.caveat} Provenance: {definition.provenance}")
    return definition


# --------------------------------------------------------------------------- #
# one vehicle, read with presence semantics
# --------------------------------------------------------------------------- #
@dataclass
class VehicleRecord:
    """One FeedEntity carrying a VehiclePosition, field by field.

    Every field is Optional and every Optional None means the feed did NOT
    supply it. A zero anywhere below is a measurement.

    ``realtime_route_id`` and ``static_route_id`` are two identifiers in two
    namespaces and are never merged - the same rule V2.1 adopted.
    """

    entity_id: str
    trip_id: Optional[str] = None
    realtime_route_id: Optional[str] = None
    start_date: Optional[str] = None
    start_time: Optional[str] = None
    direction_id: Optional[int] = None
    trip_schedule_relationship: Optional[str] = None

    vehicle_id: Optional[str] = None
    vehicle_label: Optional[str] = None
    has_license_plate: bool = False      # counted, deliberately not retained

    position_present: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bearing: Optional[float] = None
    speed: Optional[float] = None
    odometer: Optional[float] = None

    timestamp: Optional[int] = None
    current_stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None
    current_status: Optional[str] = None
    congestion_level: Optional[str] = None
    occupancy_status: Optional[str] = None

    unknown_fields: bool = False         # e.g. unparsed TfNSW extensions

    # resolved through the adopted chain, never by reading the route_id string
    trip_in_static: Optional[bool] = None
    static_route_id: Optional[str] = None
    route_short_name: Optional[str] = None
    route_long_name: Optional[str] = None
    route_type: Optional[str] = None
    trip_headsign: Optional[str] = None

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def coordinate_valid(self) -> Optional[bool]:
        """True/False for a supplied coordinate, None when none was supplied.

        An out-of-range coordinate is REPORTED, never clamped and never
        quietly dropped.
        """
        if not self.has_coordinates:
            return None
        return (LATITUDE_RANGE[0] <= self.latitude <= LATITUDE_RANGE[1]
                and LONGITUDE_RANGE[0] <= self.longitude
                <= LONGITUDE_RANGE[1])

    def age_seconds(self, observed_at: int) -> Optional[float]:
        if self.timestamp is None:
            return None
        return float(int(observed_at) - int(self.timestamp))


def iter_vehicle_entities(message: Any) -> Iterator[Tuple[Any, Any]]:
    """Every FeedEntity that actually carries a VehiclePosition."""
    for entity in message.entity:
        if entity.HasField("vehicle"):
            yield entity, entity.vehicle


def has_unknown_fields(message: Any) -> bool:
    """Whether the standard bindings left anything undecoded.

    TfNSW publishes custom GTFS-Realtime extensions. This inspector does not
    depend on them: it only NOTICES that something extra was present.

    Reading unknown fields is not supported by every protobuf runtime, so an
    unsupported runtime reports "nothing noticed" rather than failing - the
    diagnostic is optional by design and nothing depends on it.
    """
    try:
        from google.protobuf.unknown_fields import UnknownFieldSet

        return len(UnknownFieldSet(message)) > 0
    except (ImportError, AttributeError, TypeError, ValueError,
            NotImplementedError):
        return False


def read_vehicle(entity: Any, vehicle: Any) -> VehicleRecord:
    """One VehiclePosition, read through V1's presence-aware reader."""
    from google.transit import gtfs_realtime_pb2

    record = VehicleRecord(entity_id=entity.id)

    trip = field_or_none(vehicle, "trip")
    if trip is not None:
        record.trip_id = _text(field_or_none(trip, "trip_id"))
        record.realtime_route_id = _text(field_or_none(trip, "route_id"))
        record.start_date = _text(field_or_none(trip, "start_date"))
        record.start_time = _text(field_or_none(trip, "start_time"))
        direction = field_or_none(trip, "direction_id")
        record.direction_id = (int(direction) if direction is not None
                               else None)
        record.trip_schedule_relationship = enum_name(
            gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.DESCRIPTOR,
            field_or_none(trip, "schedule_relationship"))

    descriptor = field_or_none(vehicle, "vehicle")
    if descriptor is not None:
        record.vehicle_id = _text(field_or_none(descriptor, "id"))
        record.vehicle_label = _text(field_or_none(descriptor, "label"))
        # Presence only. A licence plate is personal-ish operational data and
        # nothing in this project needs its value, so it is never retained.
        record.has_license_plate = (
            field_or_none(descriptor, "license_plate") is not None)

    position = field_or_none(vehicle, "position")
    if position is not None:
        record.position_present = True
        record.latitude = _number(field_or_none(position, "latitude"))
        record.longitude = _number(field_or_none(position, "longitude"))
        record.bearing = _number(field_or_none(position, "bearing"))
        record.speed = _number(field_or_none(position, "speed"))
        record.odometer = _number(field_or_none(position, "odometer"))

    stamp = field_or_none(vehicle, "timestamp")
    record.timestamp = int(stamp) if stamp is not None else None
    sequence = field_or_none(vehicle, "current_stop_sequence")
    record.current_stop_sequence = (int(sequence) if sequence is not None
                                    else None)
    record.stop_id = _text(field_or_none(vehicle, "stop_id"))
    record.current_status = enum_name(
        gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus.DESCRIPTOR,
        field_or_none(vehicle, "current_status"))
    record.congestion_level = enum_name(
        gtfs_realtime_pb2.VehiclePosition.CongestionLevel.DESCRIPTOR,
        field_or_none(vehicle, "congestion_level"))
    record.occupancy_status = enum_name(
        getattr(gtfs_realtime_pb2.VehiclePosition, "OccupancyStatus",
                gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus)
        .DESCRIPTOR,
        field_or_none(vehicle, "occupancy_status"))
    record.unknown_fields = (has_unknown_fields(vehicle)
                            or has_unknown_fields(entity))
    return record


def _text(value: Any) -> Optional[str]:
    """A supplied string stays a string; an EMPTY supplied string stays too."""
    return None if value is None else str(value)


def _number(value: Any) -> Optional[float]:
    """A supplied number stays a number - including a genuine 0.0."""
    return None if value is None else float(value)


def read_vehicles(message: Any) -> List[VehicleRecord]:
    return [read_vehicle(entity, vehicle)
            for entity, vehicle in iter_vehicle_entities(message)]


# --------------------------------------------------------------------------- #
# the adopted route-resolution chain, applied to a vehicle's trip_id
# --------------------------------------------------------------------------- #
def static_tables(static: Optional[Any]) -> Tuple[Dict[str, Any],
                                                  Dict[str, Any]]:
    trips = ({} if static is None or not static.has_trips else
             {row.trip_id: row for row in static.trips.itertuples(
                 index=False)})
    routes = ({} if static is None or not static.has_routes else
              {row.route_id: row for row in static.routes.itertuples(
                  index=False)})
    return trips, routes


def resolve_route(record: VehicleRecord, trips: Dict[str, Any],
                  routes: Dict[str, Any]) -> VehicleRecord:
    """THE ADOPTED CHAIN, unchanged from V2.1, applied to a VehiclePosition.

        vehicle trip_id -> trips.txt -> that trip's static route_id
                        -> routes.txt -> route_short_name / long name / type

    The realtime route_id is never used to look up routes.txt and is never
    taken apart to guess a route number. Both identifiers are kept, separately.
    """
    if not trips:
        return record
    row = trips.get(record.trip_id) if record.trip_id else None
    record.trip_in_static = row is not None
    if row is not None:
        record.static_route_id = (row.route_id or "") or None
        record.trip_headsign = (row.trip_headsign or "") or None
    route = routes.get(record.static_route_id) if record.static_route_id \
        else None
    if route is not None:
        record.route_short_name = (route.route_short_name or "") or None
        record.route_long_name = (route.route_long_name or "") or None
        record.route_type = (route.route_type or "") or None
    return record


def route_label(record: VehicleRecord) -> str:
    """The human-readable route line, or an explicit refusal to guess one."""
    if record.route_short_name:
        return f"Route {record.route_short_name}"
    if record.realtime_route_id:
        return (f"Route unresolved  (raw realtime route_id "
                f"{record.realtime_route_id}, not a route number)")
    return "Route -"


# --------------------------------------------------------------------------- #
# geography - diagnostics only, and never a new study area
# --------------------------------------------------------------------------- #
@dataclass
class GlebeGeometry:
    """The OFFICIAL Glebe polygon, loaded from the file the project already
    uses, plus its own envelope.

    The envelope is the polygon's bounding box - a strictly looser diagnostic,
    labelled as such. No study area is redefined here, and the Week 5 map
    extent is deliberately NOT used as a filter: that extent includes a
    drawing margin chosen for presentation, which is not a measurement
    boundary.
    """

    polygon: Any
    bbox: Dict[str, float]
    properties: Dict[str, Any] = field(default_factory=dict)

    def contains(self, latitude: float, longitude: float) -> bool:
        from shapely.geometry import Point

        return bool(self.polygon.contains(Point(longitude, latitude)))

    def in_envelope(self, latitude: float, longitude: float) -> bool:
        # src.boundary.polygon_bbox names its keys, deliberately, so a
        # north/south/east/west box cannot be read in the wrong order.
        return (self.bbox["south"] <= latitude <= self.bbox["north"]
                and self.bbox["west"] <= longitude <= self.bbox["east"])


def load_glebe_geometry(path: Optional[Path] = None
                        ) -> Optional[GlebeGeometry]:
    """The official polygon, or None with the reason left to the caller."""
    try:
        geometry, properties = boundary.load_candidate_polygon(path)
    except boundary.BoundaryUnavailable:
        return None
    return GlebeGeometry(polygon=geometry,
                         bbox=boundary.polygon_bbox(geometry),
                         properties=properties)


# --------------------------------------------------------------------------- #
# the snapshot summary
# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile, stated plainly rather than interpolated."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1,
                       int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


@dataclass
class VehicleFeedSummary:
    """What one Vehicle Position snapshot contains, counted."""

    gtfs_realtime_version: Optional[str] = None
    incrementality: Optional[str] = None
    header_timestamp: Optional[int] = None
    observed_at: Optional[int] = None
    age_seconds: Optional[float] = None

    entity_count: int = 0
    vehicle_entity_count: int = 0
    other_entity_count: int = 0

    with_position: int = 0
    with_coordinates: int = 0
    with_valid_coordinates: int = 0
    invalid_coordinate_examples: List[str] = field(default_factory=list)
    invalid_coordinates: int = 0

    with_trip_id: int = 0
    with_vehicle_id: int = 0
    with_vehicle_label: int = 0
    with_license_plate: int = 0
    with_bearing: int = 0
    with_speed: int = 0
    with_odometer: int = 0
    with_timestamp: int = 0
    with_stop_id: int = 0
    with_current_stop_sequence: int = 0
    with_current_status: int = 0
    with_congestion_level: int = 0
    with_occupancy_status: int = 0
    with_unknown_fields: int = 0

    missing_trip_id: int = 0
    missing_vehicle_id: int = 0
    duplicate_trip_ids: Dict[str, int] = field(default_factory=dict)
    duplicate_vehicle_ids: Dict[str, int] = field(default_factory=dict)

    latitude_min: Optional[float] = None
    latitude_max: Optional[float] = None
    longitude_min: Optional[float] = None
    longitude_max: Optional[float] = None

    inside_glebe_polygon: Optional[int] = None
    inside_glebe_envelope: Optional[int] = None

    ages: List[float] = field(default_factory=list)

    @property
    def age_min(self) -> Optional[float]:
        return min(self.ages) if self.ages else None

    @property
    def age_median(self) -> Optional[float]:
        return float(statistics.median(self.ages)) if self.ages else None

    @property
    def age_p95(self) -> Optional[float]:
        return percentile(self.ages, 0.95)

    @property
    def age_max(self) -> Optional[float]:
        return max(self.ages) if self.ages else None


def summarise_vehicles(message: Any, records: Sequence[VehicleRecord],
                       observed_at: Optional[int] = None,
                       glebe: Optional[GlebeGeometry] = None
                       ) -> VehicleFeedSummary:
    """Count one snapshot. Nothing here modifies a record."""
    from google.transit import gtfs_realtime_pb2

    header = message.header
    observed = int(observed_at if observed_at is not None else time.time())
    summary = VehicleFeedSummary(
        gtfs_realtime_version=field_or_none(header, "gtfs_realtime_version"),
        incrementality=enum_name(
            gtfs_realtime_pb2.FeedHeader.Incrementality.DESCRIPTOR,
            field_or_none(header, "incrementality")),
        header_timestamp=field_or_none(header, "timestamp"),
        observed_at=observed)
    if summary.header_timestamp is not None:
        summary.age_seconds = float(observed - int(summary.header_timestamp))

    summary.entity_count = len(message.entity)
    summary.vehicle_entity_count = len(records)
    summary.other_entity_count = (summary.entity_count
                                  - summary.vehicle_entity_count)

    trip_counts: Dict[str, int] = {}
    vehicle_counts: Dict[str, int] = {}
    latitudes: List[float] = []
    longitudes: List[float] = []
    inside_polygon = 0
    inside_envelope = 0

    for record in records:
        if record.position_present:
            summary.with_position += 1
        if record.has_coordinates:
            summary.with_coordinates += 1
            if record.coordinate_valid:
                summary.with_valid_coordinates += 1
                latitudes.append(record.latitude)
                longitudes.append(record.longitude)
                if glebe is not None:
                    if glebe.in_envelope(record.latitude, record.longitude):
                        inside_envelope += 1
                        if glebe.contains(record.latitude, record.longitude):
                            inside_polygon += 1
            else:
                summary.invalid_coordinates += 1
                if len(summary.invalid_coordinate_examples) < EXAMPLE_LIMIT:
                    summary.invalid_coordinate_examples.append(
                        f"{record.entity_id}: "
                        f"{record.latitude}, {record.longitude}")

        if record.trip_id is not None:
            summary.with_trip_id += 1
            trip_counts[record.trip_id] = trip_counts.get(record.trip_id,
                                                          0) + 1
        else:
            summary.missing_trip_id += 1
        if record.vehicle_id is not None:
            summary.with_vehicle_id += 1
            vehicle_counts[record.vehicle_id] = vehicle_counts.get(
                record.vehicle_id, 0) + 1
        else:
            summary.missing_vehicle_id += 1

        summary.with_vehicle_label += record.vehicle_label is not None
        summary.with_license_plate += record.has_license_plate
        summary.with_bearing += record.bearing is not None
        summary.with_speed += record.speed is not None
        summary.with_odometer += record.odometer is not None
        summary.with_stop_id += record.stop_id is not None
        summary.with_current_stop_sequence += (
            record.current_stop_sequence is not None)
        summary.with_current_status += record.current_status is not None
        summary.with_congestion_level += record.congestion_level is not None
        summary.with_occupancy_status += record.occupancy_status is not None
        summary.with_unknown_fields += record.unknown_fields
        if record.timestamp is not None:
            summary.with_timestamp += 1
            summary.ages.append(record.age_seconds(observed))

    summary.duplicate_trip_ids = {key: count for key, count
                                  in trip_counts.items() if count > 1}
    summary.duplicate_vehicle_ids = {key: count for key, count
                                     in vehicle_counts.items() if count > 1}
    if latitudes:
        summary.latitude_min, summary.latitude_max = (min(latitudes),
                                                      max(latitudes))
        summary.longitude_min, summary.longitude_max = (min(longitudes),
                                                        max(longitudes))
    if glebe is not None:
        summary.inside_glebe_polygon = inside_polygon
        summary.inside_glebe_envelope = inside_envelope
    return summary


# --------------------------------------------------------------------------- #
# the identifier audit - the point of V4A
# --------------------------------------------------------------------------- #
@dataclass
class GlebeVehicleSample:
    """One Glebe-relevant service that a live vehicle could be tied to."""

    trip_id: str
    route: str
    vehicle_ids: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bearing: Optional[float] = None
    speed: Optional[float] = None
    vehicle_timestamp: Optional[int] = None
    vehicle_age_s: Optional[float] = None
    glebe_stops: List[str] = field(default_factory=list)
    ambiguous: bool = False


@dataclass
class IdentifierAudit:
    """Every measured overlap, with the unmatched side kept, not discarded."""

    vehicle_fetched_at: Optional[int] = None
    trip_update_fetched_at: Optional[int] = None

    # A. VehiclePositions <-> Trip Updates
    vehicle_trip_ids: int = 0
    trip_update_trip_ids: int = 0
    trip_id_intersection: int = 0
    vehicle_only_examples: List[str] = field(default_factory=list)
    trip_update_only_examples: List[str] = field(default_factory=list)

    # B. VehiclePositions -> static trips.txt
    vehicle_trip_ids_in_static: int = 0
    vehicle_trip_id_unmatched_examples: List[str] = field(default_factory=list)

    # C. realtime route_id -> static routes.txt (direct; expected to fail)
    vehicle_route_ids: int = 0
    vehicle_route_ids_in_static: int = 0
    vehicle_route_id_examples: List[str] = field(default_factory=list)

    # D. the adopted chain
    resolved_to_static_trip: int = 0
    resolved_to_static_route: int = 0
    resolved_to_short_name: int = 0

    # Glebe relevance
    glebe_trip_ids: int = 0
    glebe_trip_ids_with_vehicle: int = 0
    glebe_trip_ids_without_vehicle: List[str] = field(default_factory=list)
    glebe_samples: List[GlebeVehicleSample] = field(default_factory=list)
    ambiguous_glebe_trips: Dict[str, int] = field(default_factory=dict)

    static_available: bool = False
    trip_updates_available: bool = False

    @property
    def fetch_gap_seconds(self) -> Optional[float]:
        if (self.vehicle_fetched_at is None
                or self.trip_update_fetched_at is None):
            return None
        return abs(float(self.trip_update_fetched_at
                         - self.vehicle_fetched_at))

    @property
    def pct_vehicle_in_trip_updates(self) -> Optional[float]:
        if not self.vehicle_trip_ids:
            return None
        return 100.0 * self.trip_id_intersection / self.vehicle_trip_ids

    @property
    def pct_trip_updates_with_vehicle(self) -> Optional[float]:
        if not self.trip_update_trip_ids:
            return None
        return 100.0 * self.trip_id_intersection / self.trip_update_trip_ids

    @property
    def pct_vehicle_in_static(self) -> Optional[float]:
        if not self.vehicle_trip_ids:
            return None
        return 100.0 * self.vehicle_trip_ids_in_static / self.vehicle_trip_ids

    @property
    def pct_glebe_trips_with_vehicle(self) -> Optional[float]:
        if not self.glebe_trip_ids:
            return None
        return 100.0 * self.glebe_trip_ids_with_vehicle / self.glebe_trip_ids


def bounded(values: Sequence[str], limit: int = EXAMPLE_LIMIT) -> List[str]:
    return sorted(values)[:limit]


def clock(epoch: Optional[int]) -> str:
    """A wall-clock ``HH:MM:SS`` taken from V1's own Sydney rendering.

    This SHORTENS a string V1 already produced; it converts no timezone. If
    V1 could not produce its usual format the full string is returned rather
    than a guess.
    """
    if epoch is None:
        return "-"
    text = to_sydney(int(epoch))
    if not text:
        return "-"
    parts = text.split(" ")
    if len(parts) >= 2 and parts[1].count(":") == 2:
        return parts[1]
    return text


def audit_identifiers(records: Sequence[VehicleRecord],
                      trip_update_message: Optional[Any] = None,
                      stops: Optional[pd.DataFrame] = None,
                      static: Optional[Any] = None,
                      vehicle_fetched_at: Optional[int] = None,
                      trip_update_fetched_at: Optional[int] = None
                      ) -> IdentifierAudit:
    """Measure every link. Nothing here assumes one.

    The Glebe side reuses V2's own ``build_events``: the Glebe-relevant trips
    are exactly the trips V2 already says serve a Glebe candidate bus stop, so
    this audit cannot drift from what the Week 7 GUI shows.
    """
    audit = IdentifierAudit(vehicle_fetched_at=vehicle_fetched_at,
                            trip_update_fetched_at=trip_update_fetched_at)
    trips, routes = static_tables(static)
    audit.static_available = bool(trips)

    vehicle_trips: Dict[str, List[VehicleRecord]] = {}
    vehicle_route_ids: Set[str] = set()
    for record in records:
        if record.trip_id is not None:
            vehicle_trips.setdefault(record.trip_id, []).append(record)
        if record.realtime_route_id is not None:
            vehicle_route_ids.add(record.realtime_route_id)
        resolve_route(record, trips, routes)
        if record.trip_in_static:
            audit.resolved_to_static_trip += 1
        if record.static_route_id:
            audit.resolved_to_static_route += 1
        if record.route_short_name:
            audit.resolved_to_short_name += 1

    audit.vehicle_trip_ids = len(vehicle_trips)
    audit.vehicle_route_ids = len(vehicle_route_ids)
    audit.vehicle_route_id_examples = bounded(vehicle_route_ids, 5)
    if routes:
        audit.vehicle_route_ids_in_static = len(
            vehicle_route_ids & set(routes))
    if trips:
        matched = set(vehicle_trips) & set(trips)
        audit.vehicle_trip_ids_in_static = len(matched)
        audit.vehicle_trip_id_unmatched_examples = bounded(
            set(vehicle_trips) - matched)

    if trip_update_message is None:
        return audit
    audit.trip_updates_available = True

    update_trips: Set[str] = set()
    for _entity, update, _stop in v2.iter_stop_time_updates(
            trip_update_message):
        trip = field_or_none(update, "trip")
        trip_id = field_or_none(trip, "trip_id") if trip is not None else None
        if trip_id is not None:
            update_trips.add(str(trip_id))
    audit.trip_update_trip_ids = len(update_trips)
    shared = set(vehicle_trips) & update_trips
    audit.trip_id_intersection = len(shared)
    audit.vehicle_only_examples = bounded(set(vehicle_trips) - update_trips)
    audit.trip_update_only_examples = bounded(update_trips
                                              - set(vehicle_trips))

    if stops is None:
        return audit

    # The Glebe side, straight from V2.
    events = v2.build_events(trip_update_message, stops, static)
    glebe_trips: Dict[str, List[Any]] = {}
    for event in events:
        if event.trip_id is not None:
            glebe_trips.setdefault(str(event.trip_id), []).append(event)
    audit.glebe_trip_ids = len(glebe_trips)

    without: List[str] = []
    for trip_id, trip_events in sorted(glebe_trips.items()):
        vehicles = vehicle_trips.get(trip_id, [])
        if not vehicles:
            without.append(trip_id)
            continue
        audit.glebe_trip_ids_with_vehicle += 1
        if len(vehicles) > 1:
            audit.ambiguous_glebe_trips[trip_id] = len(vehicles)
        first = vehicles[0]
        sample = GlebeVehicleSample(
            trip_id=trip_id,
            route=route_label(first),
            vehicle_ids=[record.vehicle_id or "-" for record in vehicles],
            entity_ids=[record.entity_id for record in vehicles],
            latitude=first.latitude, longitude=first.longitude,
            bearing=first.bearing, speed=first.speed,
            vehicle_timestamp=first.timestamp,
            vehicle_age_s=(first.age_seconds(vehicle_fetched_at)
                           if vehicle_fetched_at is not None else None),
            glebe_stops=[
                f"{event.stop_id} {event.stop_name}"
                + (f"  {clock(event.sort_epoch)}"
                   if event.sort_epoch is not None else "")
                for event in trip_events[:3]],
            ambiguous=len(vehicles) > 1)
        if len(audit.glebe_samples) < EXAMPLE_LIMIT:
            audit.glebe_samples.append(sample)
    audit.glebe_trip_ids_without_vehicle = bounded(without)
    return audit


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def show(value: Any) -> str:
    """Absent stays visibly absent. Never 0, never '', never a guess."""
    return "-" if value is None else str(value)


def show_number(value: Optional[float], places: int = 2) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def format_feed_block(definition: FeedDefinition,
                      summary: VehicleFeedSummary,
                      fetch: Optional[Any]) -> List[str]:
    lines = [
        "Feed",
        f"  key            {definition.key}",
        f"  label          {definition.label}",
        f"  endpoint       {definition.status}",
        f"  version        {show(summary.gtfs_realtime_version)}"
        f"   incrementality {show(summary.incrementality)}",
        f"  updated        {show(to_sydney(summary.header_timestamp))}",
        f"  age            {show_number(summary.age_seconds, 0)} s "
        "(observation time - feed header timestamp)",
    ]
    if definition.caveat:
        lines.append(f"  CAVEAT         {definition.caveat}")
    lines.append("")
    lines.append("Transport")
    if fetch is None:
        lines.append("  (decoded from a local payload; no request was made)")
    else:
        lines += [
            f"  HTTP status    {fetch.http_status}",
            f"  content-type   {fetch.content_type}",
            f"  response size  {fetch.byte_count} bytes",
            f"  fetch duration {fetch.elapsed_s:.3f} s",
        ]
    return lines


def format_counts(summary: VehicleFeedSummary) -> List[str]:
    total = max(1, summary.vehicle_entity_count)

    def share(count: int) -> str:
        return f"{count:>7}  ({100.0 * count / total:5.1f} %)"

    lines = [
        "Entities",
        f"  total entities             {summary.entity_count}",
        f"  with vehicle position      {summary.vehicle_entity_count}",
        f"  other entity kinds         {summary.other_entity_count}",
        "",
        f"  with position block        {share(summary.with_position)}",
        f"  with latitude+longitude    {share(summary.with_coordinates)}",
        f"  with VALID coordinates     "
        f"{share(summary.with_valid_coordinates)}",
        f"  with trip_id               {share(summary.with_trip_id)}",
        f"  with vehicle_id            {share(summary.with_vehicle_id)}",
        f"  with vehicle label         {share(summary.with_vehicle_label)}",
        f"  with bearing               {share(summary.with_bearing)}",
        f"  with speed                 {share(summary.with_speed)}",
        f"  with odometer              {share(summary.with_odometer)}",
        f"  with vehicle timestamp     {share(summary.with_timestamp)}",
        f"  with stop_id               {share(summary.with_stop_id)}",
        f"  with current_stop_sequence "
        f"{share(summary.with_current_stop_sequence)}",
        f"  with current_status        {share(summary.with_current_status)}",
        f"  with congestion_level      "
        f"{share(summary.with_congestion_level)}",
        f"  with occupancy_status      "
        f"{share(summary.with_occupancy_status)}",
        f"  with licence plate field   "
        f"{share(summary.with_license_plate)}  (presence only; value not read)",
        f"  carrying unparsed fields   "
        f"{share(summary.with_unknown_fields)}  (e.g. TfNSW extensions; "
        "nothing here depends on them)",
        "",
        f"  MISSING trip_id            {summary.missing_trip_id}",
        f"  MISSING vehicle_id         {summary.missing_vehicle_id}",
        f"  duplicate trip_ids         {len(summary.duplicate_trip_ids)}"
        + ("   e.g. " + ", ".join(
            f"{key} x{count}" for key, count
            in sorted(summary.duplicate_trip_ids.items())[:5])
           if summary.duplicate_trip_ids else ""),
        f"  duplicate vehicle_ids      {len(summary.duplicate_vehicle_ids)}"
        + ("   e.g. " + ", ".join(
            f"{key} x{count}" for key, count
            in sorted(summary.duplicate_vehicle_ids.items())[:5])
           if summary.duplicate_vehicle_ids else ""),
    ]
    if summary.duplicate_trip_ids:
        lines.append("  NOTE: a trip_id carrying more than one simultaneous "
                     "VehiclePosition is an")
        lines.append("        AMBIGUITY. It is reported, not resolved: no "
                     "vehicle is chosen here.")
    return lines


def format_geography(summary: VehicleFeedSummary) -> List[str]:
    lines = [
        "Geographic bounds (valid coordinates only)",
        f"  latitude   min {show_number(summary.latitude_min, 6)}"
        f"   max {show_number(summary.latitude_max, 6)}",
        f"  longitude  min {show_number(summary.longitude_min, 6)}"
        f"   max {show_number(summary.longitude_max, 6)}",
        f"  invalid coordinates        {summary.invalid_coordinates}"
        "  (reported, never clamped)",
    ]
    for example in summary.invalid_coordinate_examples:
        lines.append(f"    {example}")
    if summary.inside_glebe_polygon is not None:
        lines += [
            "",
            "  Glebe diagnostics (NOT a redefinition of the study area)",
            f"    inside the official Glebe suburb polygon: "
            f"{summary.inside_glebe_polygon}",
            f"    inside that polygon's bounding box:       "
            f"{summary.inside_glebe_envelope}",
            "    The Week 5 map extent is a drawing margin chosen for "
            "presentation, so it is",
            "    not used as a measurement boundary here.",
        ]
    lines += [
        "",
        "  A whole-network feed is EXPECTED to extend far beyond Glebe.",
    ]
    return lines


def format_freshness(summary: VehicleFeedSummary) -> List[str]:
    return [
        "Vehicle-position age  (observation time - VehiclePosition.timestamp)",
        f"  vehicles with a timestamp  {summary.with_timestamp}",
        f"  min     {show_number(summary.age_min, 1)} s",
        f"  median  {show_number(summary.age_median, 1)} s",
        f"  p95     {show_number(summary.age_p95, 1)} s   "
        "(nearest-rank, not interpolated)",
        f"  max     {show_number(summary.age_max, 1)} s",
        "  No stale/fresh threshold is invented: TfNSW publishes none that "
        "this project has read,",
        "  so an age is reported and never used to call a vehicle stale.",
    ]


def format_vehicle(record: VehicleRecord, index: int,
                   observed_at: Optional[int]) -> List[str]:
    position = ("-" if not record.has_coordinates else
                f"{record.latitude:.6f}, {record.longitude:.6f}"
                + ("" if record.coordinate_valid else "   INVALID RANGE"))
    lines = [
        f"[{index}]",
        f"  vehicle id        {show(record.vehicle_id)}",
        f"  label             {show(record.vehicle_label)}",
        f"  trip_id           {show(record.trip_id)}",
        f"  realtime route_id {show(record.realtime_route_id)}"
        f"   static route_id {show(record.static_route_id)}",
        f"  route             {route_label(record)}"
        + (f"  ({record.route_long_name})" if record.route_long_name else ""),
        f"  position          {position}",
        f"  bearing           {show_number(record.bearing, 1)}"
        f"   speed {show_number(record.speed, 2)}"
        f"   odometer {show_number(record.odometer, 1)}",
        f"  vehicle timestamp {show(to_sydney(record.timestamp))}",
        f"  vehicle age       "
        f"{show_number(record.age_seconds(observed_at) if observed_at else None, 1)} s",
        f"  status            {show(record.current_status)}"
        f"   stop_id {show(record.stop_id)}"
        f"   stop_sequence {show(record.current_stop_sequence)}",
        f"  occupancy         {show(record.occupancy_status)}"
        f"   congestion {show(record.congestion_level)}",
    ]
    return lines


def format_audit(audit: IdentifierAudit) -> List[str]:
    lines = ["Identifier audit  (one snapshot of each feed)"]
    if audit.fetch_gap_seconds is not None:
        lines.append(f"  the two snapshots were fetched "
                     f"{audit.fetch_gap_seconds:.1f} s apart")
    lines += [
        "",
        "  A. VehiclePositions  <->  bus Trip Updates",
    ]
    if not audit.trip_updates_available:
        lines.append("     no Trip Update snapshot was taken, so this link "
                     "was not measured")
    else:
        lines += [
            f"     unique VehiclePosition trip_ids   "
            f"{audit.vehicle_trip_ids}",
            f"     unique Trip Update trip_ids       "
            f"{audit.trip_update_trip_ids}",
            f"     exact intersection                "
            f"{audit.trip_id_intersection}",
            f"     VP trip_ids seen in Trip Updates  "
            f"{show_number(audit.pct_vehicle_in_trip_updates, 1)} %",
            f"     Trip Update trips with a vehicle  "
            f"{show_number(audit.pct_trip_updates_with_vehicle, 1)} %",
            f"     VP-only examples      "
            f"{', '.join(audit.vehicle_only_examples) or '-'}",
            f"     TripUpdate-only examples "
            f"{', '.join(audit.trip_update_only_examples) or '-'}",
        ]
    lines += [
        "",
        "  B. VehiclePosition trip_id  ->  static trips.txt",
    ]
    if not audit.static_available:
        lines.append("     static trips.txt was not loaded, so this link was "
                     "not measured")
    else:
        lines += [
            f"     matched                           "
            f"{audit.vehicle_trip_ids_in_static} of {audit.vehicle_trip_ids}"
            f"  ({show_number(audit.pct_vehicle_in_static, 1)} %)",
            f"     unmatched examples    "
            f"{', '.join(audit.vehicle_trip_id_unmatched_examples) or '-'}",
            "",
            "  C. VehiclePosition realtime route_id  ->  static routes.txt "
            "(DIRECT)",
            f"     unique realtime route_ids         "
            f"{audit.vehicle_route_ids}",
            f"     direct matches in routes.txt      "
            f"{audit.vehicle_route_ids_in_static}",
            f"     examples              "
            f"{', '.join(audit.vehicle_route_id_examples) or '-'}",
        ]
        if audit.vehicle_route_ids and not audit.vehicle_route_ids_in_static:
            lines += [
                "     NOTE: zero direct overlap, as V2.1 already measured for "
                "Trip Updates.",
                "     These are two identifier namespaces. The raw token is "
                "never taken apart to",
                "     guess a route number; the chain in D is used instead.",
            ]
        lines += [
            "",
            "  D. the adopted chain: trip_id -> trips.txt -> route_id -> "
            "routes.txt",
            f"     vehicle records resolving to a trip   "
            f"{audit.resolved_to_static_trip}",
            f"     resolving to a static route           "
            f"{audit.resolved_to_static_route}",
            f"     obtaining a route_short_name          "
            f"{audit.resolved_to_short_name}",
        ]
    if not audit.trip_updates_available:
        return lines

    lines += [
        "",
        "  Glebe relevance  (V2's own Glebe candidate bus stop matching)",
        f"     Trip Update trips serving a Glebe stop  {audit.glebe_trip_ids}",
        f"     of those, with a VehiclePosition        "
        f"{audit.glebe_trip_ids_with_vehicle}"
        f"  ({show_number(audit.pct_glebe_trips_with_vehicle, 1)} %)",
        f"     without one (reported, not discarded)   "
        f"{', '.join(audit.glebe_trip_ids_without_vehicle) or '-'}",
    ]
    if audit.ambiguous_glebe_trips:
        lines.append(f"     AMBIGUOUS: {len(audit.ambiguous_glebe_trips)} "
                     "Glebe trip_id(s) carry more than one simultaneous")
        lines.append("     VehiclePosition. Both are kept; none is chosen.")
    if audit.glebe_samples:
        lines.append("")
        lines.append("  Glebe services with a live vehicle")
        for index, sample in enumerate(audit.glebe_samples, start=1):
            position = ("-" if sample.latitude is None
                        or sample.longitude is None else
                        f"{sample.latitude:.6f}, {sample.longitude:.6f}")
            lines += [
                f"    [{index}] {sample.route}",
                f"         trip_id     {sample.trip_id}",
                f"         vehicle_id  {', '.join(sample.vehicle_ids)}"
                + ("   AMBIGUOUS" if sample.ambiguous else ""),
                f"         position    {position}"
                f"   bearing {show_number(sample.bearing, 1)}"
                f"   speed {show_number(sample.speed, 2)}",
                f"         vehicle at  {show(to_sydney(sample.vehicle_timestamp))}"
                f"   age {show_number(sample.vehicle_age_s, 1)} s",
                f"         Glebe stops {'; '.join(sample.glebe_stops) or '-'}",
            ]
    return lines


def format_report(definition: FeedDefinition, summary: VehicleFeedSummary,
                  records: Sequence[VehicleRecord], sample: int,
                  fetch: Optional[Any] = None,
                  audit: Optional[IdentifierAudit] = None) -> str:
    lines = [RULE, "Week 7 V4A - TfNSW live vehicle positions", RULE]
    lines += format_feed_block(definition, summary, fetch)
    lines += [""] + format_counts(summary)
    lines += [""] + format_geography(summary)
    lines += [""] + format_freshness(summary)

    wanted = max(0, min(int(sample), MAX_SAMPLE))
    if wanted:
        lines += ["", f"Sample vehicles  (first {wanted} of "
                      f"{len(records)})"]
        for index, record in enumerate(records[:wanted], start=1):
            lines += [""] + format_vehicle(record, index,
                                           summary.observed_at)
    if audit is not None:
        lines += ["", RULE] + format_audit(audit)
    lines += [
        "",
        RULE,
        "This is ONE snapshot. It is not a trajectory, nothing is stored, and "
        "no vehicle is drawn.",
        "V4A measures whether the identifiers connect. It changes no Week 5 "
        "method and no GUI.",
        RULE,
    ]
    return "\n".join(lines)


def format_registry() -> str:
    lines = ["Registered Vehicle Position feeds", RULE]
    for key in sorted(VEHICLE_FEEDS):
        definition = VEHICLE_FEEDS[key]
        lines += [
            f"{definition.key}",
            f"  label      {definition.label}",
            f"  url        {definition.url or '(none registered)'}",
            f"  modes      {definition.modes}",
            f"  status     {definition.status}",
            f"  provenance {definition.provenance}",
        ]
        if definition.caveat:
            lines.append(f"  caveat     {definition.caveat}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 V4A: inspect a live TfNSW GTFS-Realtime Vehicle "
                    "Positions feed and MEASURE whether its identifiers "
                    "connect to the Trip Update feed, the static GTFS "
                    "snapshot and this project's Glebe candidate stops. "
                    "Reconnaissance only - nothing is drawn or stored.")
    parser.add_argument("--feed", default=DEFAULT_FEED,
                        help=f"registered feed key (default {DEFAULT_FEED})")
    parser.add_argument("--url", default=None,
                        help="override the endpoint (recorded as caller-"
                             "supplied, not verified)")
    parser.add_argument("--list-feeds", action="store_true",
                        help="print the registry and its provenance, then "
                             "exit")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"vehicles to print in full "
                             f"(default {DEFAULT_SAMPLE}, max {MAX_SAMPLE})")
    parser.add_argument("--audit", action="store_true",
                        help="also fetch ONE bus Trip Update snapshot and "
                             "measure the identifier overlap")
    parser.add_argument("--no-static", action="store_true",
                        help="skip the static GTFS join entirely")
    parser.add_argument("--parse-file", default=None,
                        help="decode a saved vehicle-position .pb payload "
                             "instead of fetching")
    parser.add_argument("--parse-trip-file", default=None,
                        help="decode a saved Trip Update .pb payload for the "
                             "audit instead of fetching")
    parser.add_argument("--save-raw", action="store_true",
                        help="write the response body under data/cache/ "
                             "(gitignored). Never contains the key.")
    options = parser.parse_args(argv)

    if options.list_feeds:
        print(format_registry())
        return 0

    api_key: Optional[str] = None
    try:
        definition = resolve_vehicle_feed(options.feed, options.url)
        static = None if options.no_static else v2.load_static_gtfs()
        stops = None if options.no_static else v2.load_candidate_stops()

        # ---- the vehicle-position snapshot ----------------------------- #
        if options.parse_file:
            source = Path(options.parse_file)
            if not source.exists():
                raise VehiclePositionError(f"{source} does not exist.")
            payload = source.read_bytes()
            fetch = None
        else:
            api_key = v1.load_api_key()
            print(f"Requesting {definition.url}")
            print("Authorization header: apikey <redacted>")
            fetch = v1.fetch_feed(definition.url, api_key)
            payload = fetch.payload
            if options.save_raw:
                written = v1.save_raw(payload, definition)
                print(f"Raw payload written to {written}")
        vehicle_observed_at = int(time.time())
        message = v1.parse_feed(payload)
        records = read_vehicles(message)

        # ---- the Trip Update snapshot, close in time -------------------- #
        trip_message = None
        trip_observed_at = None
        if options.audit:
            if options.parse_trip_file:
                trip_source = Path(options.parse_trip_file)
                if not trip_source.exists():
                    raise VehiclePositionError(
                        f"{trip_source} does not exist.")
                trip_payload = trip_source.read_bytes()
            elif definition.key in BUS_MODE_KEYS or options.url:
                trip_definition = v1.resolve_feed(TRIP_UPDATE_FEED)
                api_key = api_key or v1.load_api_key()
                print(f"Requesting {trip_definition.url}")
                trip_payload = v1.fetch_feed(trip_definition.url,
                                             api_key).payload
            else:
                trip_payload = None
                print(f"No Trip Update counterpart is requested for "
                      f"{definition.key}: the verified Trip Update feed in "
                      "this project is the BUS one, and forcing it onto "
                      "another mode would measure nothing.")
            if trip_payload is not None:
                trip_observed_at = int(time.time())
                trip_message = v1.parse_feed(trip_payload)

        glebe = load_glebe_geometry()
        summary = summarise_vehicles(message, records, vehicle_observed_at,
                                     glebe)
        audit = None
        if options.audit or not options.no_static:
            audit = audit_identifiers(
                records, trip_message, stops, static,
                vehicle_fetched_at=vehicle_observed_at,
                trip_update_fetched_at=trip_observed_at)
    except MissingApiKey as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except RealtimeError as error:
        print(f"ERROR: {redact(error, api_key)}", file=sys.stderr)
        return 1

    print(redact(format_report(definition, summary, records, options.sample,
                               fetch, audit), api_key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
