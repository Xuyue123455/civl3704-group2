"""Week 7 V1 - fetch and inspect a live TfNSW GTFS-Realtime Trip Update feed.

WHAT THIS PROVES, AND WHAT IT DOES NOT
--------------------------------------
This script proves ONE thing: that this project can authenticate to the
Transport for NSW Open Data API, retrieve a live GTFS-Realtime Trip Update
feed over HTTPS, and decode it with the official GTFS-Realtime protobuf
bindings.

It does **NOT** claim that any realtime entity has been matched to a Glebe
candidate stop, to the Week 4 static GTFS snapshot, or to any Week 5 routing
result. No join is attempted here, and none of the numbers printed below may
be read as a statement about Glebe accessibility. That work is V2.

Nothing in Week 4 or Week 5 is read, written or modified by this script.

AUTHENTICATION
--------------
The API key is read from the ``TFNSW_API_KEY`` environment variable (or a
local ``.env``), exactly as ``scripts/fetch_gtfs.py`` already does. It is sent
only in the ``Authorization: apikey <key>`` request header. It is never
printed, never written to a file, never included in a saved payload, and every
line this script prints is passed through a redaction guard before it reaches
the terminal.

FIELD PRESENCE
--------------
GTFS-Realtime is proto2: a TripUpdate may carry an absolute event ``time``, a
relative ``delay``, both, or neither, and any of ``trip_id``, ``route_id``,
``start_date`` and the rest may simply be absent. Every field below is read
through ``HasField`` and reported as ``-`` when absent. A missing realtime
field is NEVER turned into zero, into "on time", or into any other value.

ENDPOINT PROVENANCE
-------------------
See ``FEEDS`` below. Each registered feed records where its URL came from.
Only feeds whose URL has been confirmed against TfNSW's own published resource
page are marked ``VERIFIED``; nothing is guessed from memory.

Usage
-----
    python scripts/inspect_gtfs_realtime.py
    python scripts/inspect_gtfs_realtime.py --feed buses --sample 5
    python scripts/inspect_gtfs_realtime.py --list-feeds
    python scripts/inspect_gtfs_realtime.py --parse-file some_captured.pb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402

SYDNEY_TZ = "Australia/Sydney"

API_KEY_VARIABLE = "TFNSW_API_KEY"
AUTH_HEADER_TEMPLATE = "apikey {key}"        # TfNSW: "Authorization: apikey ak123"

# Connect and read timeouts, in seconds. Explicit, and deliberately short: a
# live classroom demonstration must fail fast and say why, not hang.
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 30.0

DEFAULT_SAMPLE = 3
MAX_SAMPLE = 20
DEFAULT_STOP_TIME_UPDATES = 3

RAW_SUBDIR = "gtfs_realtime"                 # under data/cache/, gitignored

# Redaction placeholder. If this ever appears in the output, the guard worked.
REDACTED = "<redacted>"


# --------------------------------------------------------------------------- #
# feed registry - every URL carries its provenance
# --------------------------------------------------------------------------- #
STATUS_VERIFIED = "VERIFIED"
STATUS_SUPERSEDED = "SUPERSEDED_BY_V2"
STATUS_UNVERIFIED = "ENDPOINT_NOT_VERIFIED"

TRIP_UPDATE_BASE_URI = "https://api.transport.nsw.gov.au/v1/gtfs/realtime/"

PROVENANCE_RESOURCE_PAGE = (
    "TfNSW Open Data Hub, dataset 'Public Transport - Realtime Trip Update', "
    "API resource page "
    "https://opendata.transport.nsw.gov.au/dataset/"
    "public-transport-realtime-trip-update/resource/"
    "d936a1c6-1fe6-4985-9df2-326e91036f80 - publishes the base URI "
    f"{TRIP_UPDATE_BASE_URI} and the per-mode paths."
)


@dataclass(frozen=True)
class FeedDefinition:
    """One registered GTFS-Realtime Trip Update feed.

    ``url`` is None when the endpoint has NOT been confirmed against an
    authoritative TfNSW page. Selecting such a feed is an error, not a guess.
    """

    key: str
    label: str
    url: Optional[str]
    modes: str
    status: str
    provenance: str
    caveat: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.url)


FEEDS: Dict[str, FeedDefinition] = {
    "buses": FeedDefinition(
        key="buses",
        label="Buses - Trip Updates (v1)",
        url=TRIP_UPDATE_BASE_URI + "buses",
        modes="Sydney and surrounds bus network",
        status=STATUS_VERIFIED,
        provenance=(
            PROVENANCE_RESOURCE_PAGE
            + " The page lists 'GET /buses' under that base URI. The same URL "
            "is already present in this repository's config.json as "
            "gtfs_realtime.trip_updates_url."
        ),
    ),
    "lightrail_innerwest_v1": FeedDefinition(
        key="lightrail_innerwest_v1",
        label="Inner West Light Rail - Trip Updates (v1)",
        url=TRIP_UPDATE_BASE_URI + "lightrail/innerwest",
        modes="Inner West Light Rail (L1)",
        status=STATUS_SUPERSEDED,
        provenance=(
            PROVENANCE_RESOURCE_PAGE
            + " The page lists 'GET /lightrail/innerwest' under that base URI "
            "AND marks it superseded by the Realtime Trip Update v2 dataset."
        ),
        caveat=(
            "TfNSW marks this v1 path as superseded by the Realtime Trip "
            "Update v2 dataset. It is registered here because the path is "
            "published, not because it is the current feed. Do not present "
            "its contents as the current Inner West Light Rail feed."
        ),
    ),
    "trip_update_v2": FeedDefinition(
        key="trip_update_v2",
        label="Realtime Trip Update v2 (Metro, Sydney Trains, Inner West LR)",
        url=None,
        modes="Sydney Metro, Sydney Trains, Inner West Light Rail",
        status=STATUS_UNVERIFIED,
        provenance=(
            "TfNSW dataset 'Public Transport - Realtime Trip Update v2' "
            "https://opendata.transport.nsw.gov.au/data/dataset/"
            "public-transport-realtime-trip-update-v2 exists and is the "
            "current feed for these modes, but its API resource page requires "
            "an Open Data Hub login, so the exact endpoint URL could not be "
            "confirmed from a public page and is therefore NOT recorded here."
        ),
        caveat=(
            "No URL is registered. Supply the endpoint from your logged-in "
            "Open Data Hub account (--url), or add it to config.json under "
            "gtfs_realtime.trip_update_v2_url once you have read it from the "
            "resource page."
        ),
    ),
}

DEFAULT_FEED = "buses"


class RealtimeError(RuntimeError):
    """Raised with an actionable message when the inspection cannot continue."""


class MissingApiKey(RealtimeError):
    """Raised when TFNSW_API_KEY is not set."""


# --------------------------------------------------------------------------- #
# secret handling
# --------------------------------------------------------------------------- #
def load_api_key(environment: Optional[Dict[str, str]] = None) -> str:
    """The TfNSW API key, from the environment or a local .env.

    Raises:
        MissingApiKey: with instructions the user can follow, and WITHOUT
            echoing anything secret.
    """
    if environment is None:
        try:                       # optional: .env is how Week 4 already works
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True), override=False)
        except ImportError:
            pass
        environment = dict(os.environ)

    key = (environment.get(API_KEY_VARIABLE) or "").strip()
    if not key:
        raise MissingApiKey(
            f"{API_KEY_VARIABLE} is not set, so the TfNSW API cannot be "
            "called.\n"
            "\n"
            "Set it for THIS Anaconda Prompt session only (Windows):\n"
            f"    set {API_KEY_VARIABLE}=your-key-here\n"
            "\n"
            "or, in PowerShell:\n"
            f"    $env:{API_KEY_VARIABLE} = \"your-key-here\"\n"
            "\n"
            "or add a line to a local .env file, which is gitignored:\n"
            f"    {API_KEY_VARIABLE}=your-key-here\n"
            "\n"
            "The key comes from your application in the TfNSW Open Data Hub. "
            "Never put it in config.json, in any committed file, or in a "
            "screenshot."
        )
    return key


def redact(text: Any, api_key: Optional[str]) -> str:
    """Remove the API key from anything about to be printed.

    A belt-and-braces guard. No code path is supposed to put the key into a
    message; this makes it impossible for one to do so by accident.
    """
    rendered = str(text)
    if not api_key:
        return rendered
    return rendered.replace(api_key, REDACTED)


def authorization_header(api_key: str) -> Dict[str, str]:
    """The single header TfNSW requires: ``Authorization: apikey <key>``."""
    return {"Authorization": AUTH_HEADER_TEMPLATE.format(key=api_key)}


# --------------------------------------------------------------------------- #
# feed selection
# --------------------------------------------------------------------------- #
def resolve_feed(key: str, url_override: Optional[str] = None
                 ) -> FeedDefinition:
    """The registered feed, optionally with a caller-supplied URL.

    Raises:
        RealtimeError: for an unknown key, or for a registered feed whose
            endpoint has not been verified and for which no URL was supplied.
    """
    if key not in FEEDS:
        raise RealtimeError(
            f"Unknown feed {key!r}. Registered feeds: "
            f"{', '.join(sorted(FEEDS))}. Run with --list-feeds to see each "
            "endpoint and where it came from.")
    definition = FEEDS[key]
    if url_override:
        return FeedDefinition(
            key=definition.key, label=definition.label, url=url_override,
            modes=definition.modes, status="SUPPLIED_BY_CALLER",
            provenance=("URL supplied on the command line by the operator; "
                        "not verified by this script."),
            caveat=definition.caveat)
    if not definition.usable:
        raise RealtimeError(
            f"{definition.label} has no verified endpoint URL.\n"
            f"{definition.caveat}\n"
            f"Provenance: {definition.provenance}")
    return definition


def describe_feeds() -> str:
    lines = ["Registered GTFS-Realtime Trip Update feeds", ""]
    for key in sorted(FEEDS):
        definition = FEEDS[key]
        lines.append(f"  {key}")
        lines.append(f"    label      {definition.label}")
        lines.append(f"    modes      {definition.modes}")
        lines.append(f"    url        {definition.url or '(not registered)'}")
        lines.append(f"    status     {definition.status}")
        lines.append(f"    provenance {definition.provenance}")
        if definition.caveat:
            lines.append(f"    caveat     {definition.caveat}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# network - kept apart from parsing so parsing can be tested offline
# --------------------------------------------------------------------------- #
@dataclass
class FetchResult:
    """One HTTP response, reduced to what the report needs."""

    url: str
    http_status: int
    content_type: str
    payload: bytes
    elapsed_s: float

    @property
    def byte_count(self) -> int:
        return len(self.payload)


def fetch_feed(url: str, api_key: str,
               timeout: Tuple[float, float] = (CONNECT_TIMEOUT_S,
                                               READ_TIMEOUT_S),
               session: Any = None) -> FetchResult:
    """One GET against the feed. One attempt, explicit timeouts, no retry loop.

    Raises:
        RealtimeError: for a rejected key, a timeout, any non-200 status, or
            an ordinary network failure - each with an actionable message and
            with the API key redacted.
    """
    caller = session or requests
    try:
        response = caller.get(url, headers=authorization_header(api_key),
                              timeout=timeout)
    except requests.Timeout as error:
        raise RealtimeError(
            f"The request to {url} timed out after "
            f"{timeout[0]:.0f} s connect / {timeout[1]:.0f} s read. The feed "
            "was not retried. Check the network, then run the command again."
        ) from error
    except requests.RequestException as error:
        raise RealtimeError(
            f"Could not reach {url}: {redact(error, api_key)}") from error

    elapsed = float(getattr(response, "elapsed", None).total_seconds()
                    if getattr(response, "elapsed", None) is not None else 0.0)
    status = int(response.status_code)
    body = response.content or b""
    content_type = response.headers.get("Content-Type", "")

    if status in (401, 403):
        raise RealtimeError(
            f"TfNSW returned HTTP {status}: the API key was rejected.\n"
            f"Check that {API_KEY_VARIABLE} holds the key from your TfNSW "
            "Open Data Hub application, and that the application is "
            "SUBSCRIBED to the Realtime Trip Update API - a key that works "
            "for the static Timetables API is not automatically subscribed "
            "to realtime.")
    if status == 404:
        raise RealtimeError(
            f"TfNSW returned HTTP 404 for {url}. The endpoint path may have "
            "changed or been superseded. Re-check it on the Open Data Hub "
            "resource page before presenting this feed.")
    if status != 200:
        preview = redact(body[:200], api_key)
        raise RealtimeError(
            f"TfNSW returned HTTP {status}, expected 200. First 200 bytes of "
            f"the response: {preview}")
    if not body:
        raise RealtimeError(
            f"TfNSW returned HTTP 200 for {url} but the body was empty, so "
            "there is nothing to decode. This is a feed-side condition, not a "
            "parsing failure.")

    return FetchResult(url=url, http_status=status, content_type=content_type,
                       payload=body, elapsed_s=elapsed)


# --------------------------------------------------------------------------- #
# parsing - pure, offline-testable
# --------------------------------------------------------------------------- #
def parse_feed(payload: bytes):
    """Decode a GTFS-Realtime FeedMessage with the official bindings.

    Raises:
        RealtimeError: if the bindings are missing, or the bytes are not a
            decodable FeedMessage.
    """
    try:
        from google.transit import gtfs_realtime_pb2
    except ImportError as error:
        raise RealtimeError(
            "The official GTFS-Realtime bindings are not installed. Install "
            "them into the project environment:\n"
            "    conda activate civl3704\n"
            "    pip install gtfs-realtime-bindings") from error

    message = gtfs_realtime_pb2.FeedMessage()
    try:
        message.ParseFromString(payload)
    except Exception as error:          # protobuf raises DecodeError subclasses
        raise RealtimeError(
            f"The response is not a decodable GTFS-Realtime FeedMessage "
            f"({type(error).__name__}: {error}). The first bytes were "
            f"{payload[:32]!r}. An HTML error page decoded as protobuf fails "
            "exactly like this.") from error
    return message


def field_or_none(message: Any, name: str) -> Any:
    """A proto2 optional field's value, or None when it is genuinely absent.

    This is the whole point of the exercise: an absent ``delay`` is not zero
    and an absent ``time`` is not "on time".
    """
    try:
        present = message.HasField(name)
    except ValueError:                  # repeated fields have no presence
        return None
    return getattr(message, name) if present else None


def enum_name(descriptor, value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    try:
        return descriptor.values_by_number[value].name
    except (KeyError, AttributeError):
        return str(value)


def to_sydney(epoch_seconds: Optional[int]) -> Optional[str]:
    """A POSIX timestamp as an explicit Australia/Sydney wall-clock string.

    Returns None if the timestamp is absent, and a plain marker if the
    timezone database is unavailable, rather than silently pretending UTC is
    local time.
    """
    if epoch_seconds is None:
        return None
    moment = datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        local = moment.astimezone(ZoneInfo(SYDNEY_TZ))
    except Exception:
        return (f"{moment.isoformat()} (UTC; {SYDNEY_TZ} unavailable - "
                "install tzdata to show Sydney local time)")
    return local.strftime("%Y-%m-%d %H:%M:%S %Z%z")


def stop_time_update_summary(update: Any) -> Dict[str, Any]:
    """One StopTimeUpdate, reporting only what is actually present."""
    from google.transit import gtfs_realtime_pb2

    row: Dict[str, Any] = {
        "stop_id": field_or_none(update, "stop_id"),
        "stop_sequence": field_or_none(update, "stop_sequence"),
        "schedule_relationship": enum_name(
            gtfs_realtime_pb2.TripUpdate.StopTimeUpdate
            .ScheduleRelationship.DESCRIPTOR,
            field_or_none(update, "schedule_relationship")),
    }
    for event in ("arrival", "departure"):
        stop_event = field_or_none(update, event)
        if stop_event is None:
            row[event] = None
            continue
        row[event] = {
            "time": field_or_none(stop_event, "time"),
            "time_local": to_sydney(field_or_none(stop_event, "time")),
            "delay": field_or_none(stop_event, "delay"),
            "uncertainty": field_or_none(stop_event, "uncertainty"),
        }
    return row


def trip_update_summary(entity: Any, stop_time_updates: int
                        ) -> Dict[str, Any]:
    """One TripUpdate entity, reporting only what is actually present."""
    from google.transit import gtfs_realtime_pb2

    update = entity.trip_update
    trip = update.trip if update.HasField("trip") else None
    vehicle = field_or_none(update, "vehicle")
    return {
        "entity_id": entity.id,
        "trip_id": field_or_none(trip, "trip_id") if trip else None,
        "route_id": field_or_none(trip, "route_id") if trip else None,
        "direction_id": field_or_none(trip, "direction_id") if trip else None,
        "start_date": field_or_none(trip, "start_date") if trip else None,
        "start_time": field_or_none(trip, "start_time") if trip else None,
        "schedule_relationship": enum_name(
            gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.DESCRIPTOR,
            field_or_none(trip, "schedule_relationship")) if trip else None,
        "vehicle_id": field_or_none(vehicle, "id") if vehicle else None,
        "vehicle_label": field_or_none(vehicle, "label") if vehicle else None,
        "timestamp": field_or_none(update, "timestamp"),
        "timestamp_local": to_sydney(field_or_none(update, "timestamp")),
        "trip_delay": field_or_none(update, "delay"),
        "stop_time_update_count": len(update.stop_time_update),
        "stop_time_updates": [
            stop_time_update_summary(item)
            for item in update.stop_time_update[:stop_time_updates]
        ],
    }


@dataclass
class FeedSummary:
    """Everything the report prints, derived from one decoded FeedMessage."""

    gtfs_realtime_version: Optional[str] = None
    incrementality: Optional[str] = None
    header_timestamp: Optional[int] = None
    header_timestamp_local: Optional[str] = None
    entity_count: int = 0
    trip_update_count: int = 0
    vehicle_count: int = 0
    alert_count: int = 0
    other_count: int = 0
    samples: List[Dict[str, Any]] = field(default_factory=list)
    observed_at: Optional[int] = None
    age_seconds: Optional[float] = None


def summarise_feed(message: Any, sample: int = DEFAULT_SAMPLE,
                   stop_time_updates: int = DEFAULT_STOP_TIME_UPDATES,
                   now_epoch: Optional[float] = None) -> FeedSummary:
    """Count the feed and take a BOUNDED sample of its TripUpdate entities.

    ``sample`` is clamped to ``MAX_SAMPLE``: a live feed can carry thousands
    of entities and a demonstration must not print them all.
    """
    from google.transit import gtfs_realtime_pb2

    header = message.header
    summary = FeedSummary(
        gtfs_realtime_version=field_or_none(header, "gtfs_realtime_version"),
        incrementality=enum_name(
            gtfs_realtime_pb2.FeedHeader.Incrementality.DESCRIPTOR,
            field_or_none(header, "incrementality")),
        header_timestamp=field_or_none(header, "timestamp"),
    )
    summary.header_timestamp_local = to_sydney(summary.header_timestamp)

    wanted = max(0, min(int(sample), MAX_SAMPLE))
    for entity in message.entity:
        if entity.HasField("trip_update"):
            summary.trip_update_count += 1
            if len(summary.samples) < wanted:
                summary.samples.append(
                    trip_update_summary(entity, stop_time_updates))
        elif entity.HasField("vehicle"):
            summary.vehicle_count += 1
        elif entity.HasField("alert"):
            summary.alert_count += 1
        else:
            summary.other_count += 1
        summary.entity_count += 1

    observed = int(now_epoch if now_epoch is not None else time.time())
    summary.observed_at = observed
    if summary.header_timestamp is not None:
        summary.age_seconds = float(observed - int(summary.header_timestamp))
    return summary


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
RULE = "-" * 72


def show(value: Any) -> str:
    """Absent stays visibly absent. Never 0, never 'on time', never ''."""
    return "-" if value is None else str(value)


def format_stop_time_update(row: Dict[str, Any], indent: str = "        "
                            ) -> List[str]:
    lines = [f"{indent}stop_id {show(row['stop_id'])}"
             f"  stop_sequence {show(row['stop_sequence'])}"
             f"  schedule_relationship {show(row['schedule_relationship'])}"]
    for event in ("arrival", "departure"):
        payload = row.get(event)
        if payload is None:
            lines.append(f"{indent}  {event:<9} (field absent)")
            continue
        lines.append(
            f"{indent}  {event:<9} time {show(payload['time'])}"
            f"  delay {show(payload['delay'])}"
            f"  uncertainty {show(payload['uncertainty'])}")
        if payload["time_local"] is not None:
            lines.append(f"{indent}            local {payload['time_local']}")
    return lines


def format_report(definition: FeedDefinition, fetch: Optional[FetchResult],
                  summary: FeedSummary, source_label: str = "") -> str:
    lines = [
        RULE,
        "Week 7 V1 - live GTFS-Realtime Trip Update ingestion check",
        RULE,
        "Feed",
        f"  key            {definition.key}",
        f"  label          {definition.label}",
        f"  modes          {definition.modes}",
        f"  url            {show(definition.url)}",
        f"  endpoint       {definition.status}",
    ]
    if definition.caveat:
        lines.append(f"  caveat         {definition.caveat}")
    lines.append("")

    lines.append("Transport")
    if fetch is None:
        lines.append(f"  source         {source_label or 'local file'} "
                     "(no HTTP request was made)")
    else:
        lines += [
            f"  HTTP status    {fetch.http_status} OK",
            f"  content type   {show(fetch.content_type or None)}",
            f"  response size  {fetch.byte_count} bytes",
            f"  fetch duration {fetch.elapsed_s:.3f} s",
        ]
    lines.append("")

    lines += [
        "Feed header",
        f"  gtfs_realtime_version  {show(summary.gtfs_realtime_version)}",
        f"  incrementality         {show(summary.incrementality)}",
        f"  timestamp (raw POSIX)  {show(summary.header_timestamp)}",
        f"  timestamp (Sydney)     {show(summary.header_timestamp_local)}",
        "",
        "Entities",
        f"  total FeedEntity       {summary.entity_count}",
        f"  with trip_update       {summary.trip_update_count}",
        f"  with vehicle           {summary.vehicle_count}",
        f"  with alert             {summary.alert_count}",
        f"  with none of the above {summary.other_count}",
        "",
        "Observation age",
    ]
    if summary.age_seconds is None:
        lines.append("  the feed header carries no timestamp, so no age can "
                     "be stated.")
    else:
        lines += [
            f"  observed at (raw)      {show(summary.observed_at)}",
            f"  observed at (Sydney)   {show(to_sydney(summary.observed_at))}",
            f"  feed age               {summary.age_seconds:.0f} s "
            "(observation time minus feed header timestamp)",
            "  This is a measured age, not a verdict. No freshness threshold "
            "is applied:",
            "  none was verified against TfNSW documentation.",
        ]
    lines.append("")

    lines.append(f"Sample of TripUpdate entities "
                 f"({len(summary.samples)} of {summary.trip_update_count}, "
                 f"bounded)")
    if not summary.samples:
        lines.append("  none present in this feed at this moment.")
    for index, row in enumerate(summary.samples):
        lines += [
            f"  [{index + 1}] entity id {show(row['entity_id'])}",
            f"      trip_id {show(row['trip_id'])}"
            f"  route_id {show(row['route_id'])}"
            f"  direction_id {show(row['direction_id'])}",
            f"      start_date {show(row['start_date'])}"
            f"  start_time {show(row['start_time'])}"
            f"  schedule_relationship {show(row['schedule_relationship'])}",
            f"      vehicle_id {show(row['vehicle_id'])}"
            f"  vehicle_label {show(row['vehicle_label'])}",
            f"      trip timestamp {show(row['timestamp'])}"
            f"  local {show(row['timestamp_local'])}",
            f"      trip-level delay {show(row['trip_delay'])}",
            f"      stop_time_update count {row['stop_time_update_count']}"
            f"  (showing {len(row['stop_time_updates'])})",
        ]
        for item in row["stop_time_updates"]:
            lines += format_stop_time_update(item)
    lines += [
        "",
        "Scope",
        "  This confirms authenticated retrieval and protobuf decoding only.",
        "  No realtime entity has been matched to a Glebe candidate stop or",
        "  to the Week 4 static GTFS snapshot. That is V2.",
        "  A missing field is printed as '-' and is never read as 0 or",
        "  'on time'.",
        RULE,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# optional raw capture
# --------------------------------------------------------------------------- #
def raw_capture_dir(root: Optional[Path] = None) -> Path:
    return (Path(root or CONFIG_ROOT) / "data" / "cache" / RAW_SUBDIR)


def save_raw(payload: bytes, definition: FeedDefinition,
             directory: Optional[Path] = None,
             now_epoch: Optional[float] = None) -> Path:
    """Write the protobuf body for later offline inspection.

    The file holds ONLY the response body: no header, no URL, no key. It lands
    under ``data/cache/``, which .gitignore already excludes, so a captured
    feed cannot be committed by accident.
    """
    target = Path(directory) if directory else raw_capture_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(
        int(now_epoch if now_epoch is not None else time.time()),
        tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"{definition.key}_{stamp}.pb"
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 V1: fetch and inspect a live TfNSW GTFS-Realtime "
                    "Trip Update feed. Proves ingestion and decoding only - "
                    "no join to Glebe stops or static GTFS.")
    parser.add_argument("--feed", default=DEFAULT_FEED,
                        help=f"registered feed key (default {DEFAULT_FEED})")
    parser.add_argument("--url", default=None,
                        help="override the endpoint for this run only")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"TripUpdate entities to show "
                             f"(default {DEFAULT_SAMPLE}, max {MAX_SAMPLE})")
    parser.add_argument("--stop-time-updates", type=int,
                        default=DEFAULT_STOP_TIME_UPDATES,
                        help="stop_time_updates to show per sampled trip")
    parser.add_argument("--list-feeds", action="store_true",
                        help="print the feed registry and its provenance, "
                             "then exit without calling the API")
    parser.add_argument("--parse-file", default=None,
                        help="decode a saved .pb payload instead of fetching "
                             "(no network, no API key)")
    parser.add_argument("--save-raw", action="store_true",
                        help="also write the raw protobuf body under "
                             "data/cache/gtfs_realtime/ (gitignored)")
    options = parser.parse_args(argv)

    if options.list_feeds:
        print(describe_feeds())
        return 0

    api_key: Optional[str] = None
    try:
        definition = resolve_feed(options.feed, options.url)

        if options.parse_file:
            source = Path(options.parse_file)
            if not source.exists():
                raise RealtimeError(f"{source} does not exist.")
            payload = source.read_bytes()
            fetch = None
            source_label = str(source)
        else:
            api_key = load_api_key()
            print(f"Requesting {definition.url}")
            print("Authorization header: apikey <redacted>")
            fetch = fetch_feed(definition.url, api_key)
            payload = fetch.payload
            source_label = ""

        message = parse_feed(payload)
        summary = summarise_feed(message, sample=options.sample,
                                 stop_time_updates=options.stop_time_updates)
    except MissingApiKey as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except RealtimeError as error:
        print(f"ERROR: {redact(error, api_key)}", file=sys.stderr)
        return 1

    print(redact(format_report(definition, fetch, summary, source_label),
                 api_key))

    if options.save_raw and not options.parse_file:
        written = save_raw(payload, definition)
        print(f"Raw payload written to {written}")
        print("It contains the response body only - no URL, no header, no key "
              "- and data/cache/ is gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
