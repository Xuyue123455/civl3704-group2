"""Week 9 Phase 2 - the minimum user intention: a ROUTE plus a DIRECTION.

WHAT THIS IS
------------
The Phase 2 prototype does not ask for a destination. It asks for the
smallest transit intention that still decides which kerb to stand on:

    route_short_name + direction_id          e.g.  433 / 0

This module turns the static tables written by
``scripts/build_service_intent.py`` (``results/service_intent/``) into
human-readable choices, tells the engine which candidate stops a chosen
service boards at, resolves a realtime trip_id to its direction, and can
list the scheduled visits of a service at a stop on a given service date
(the schedule FALLBACK, calendar-checked).

WHY (route_short_name, direction_id) AND NOT THE HEADSIGN
---------------------------------------------------------
Measured on the project's static snapshot (``inspection.json``):

* ``direction_id`` is present on every trip at every candidate stop (0/1);
* ``trip_headsign`` is present on every trip, but 7 of the 19
  (route, direction) pairs carry MORE THAN ONE headsign in the same direction
  (short workings: "Coogee" / "Coogee Beach" / "Kensington" on 370
  direction 0; "Martin Place" / "Pitt St And Barlow St" on 431/438N/470
  direction 1). A headsign is therefore a VARIANT within a direction, not the
  direction itself;
* no headsign is used in both directions of one route, and no candidate stop
  is served by one route in both directions, so (route, direction) is also
  sufficient to separate the two kerbs of every same-name stop pair;
* TfNSW's ``route_direction`` text ("Glebe Point to Coogee") is unique per
  (route, direction) for all 19 pairs, so it is the display label; the
  headsign variants are listed under it.

``route_id`` is 1:1 with ``route_short_name`` in this snapshot, so either
would do; the short name is what people say.

WHAT THIS IS NOT
----------------
Not a journey planner: no destination, no transfers, no city-wide search.
Nothing here is realtime, routing, effort or walking time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402

DEFAULT_DIR = CONFIG_ROOT / "results" / "service_intent"
TRIPS_FILE = "candidate_stop_trips.csv"
SERVICES_FILE = "candidate_stop_services.csv"
CALENDAR_FILE = "service_calendar.csv"
CALENDAR_DATES_FILE = "service_calendar_dates.csv"
INSPECTION_FILE = "inspection.json"
PROVENANCE_FILE = "provenance.json"

SYDNEY_TZ = "Australia/Sydney"
WEEKDAY_COLUMNS = ("monday", "tuesday", "wednesday", "thursday", "friday",
                   "saturday", "sunday")
EXCEPTION_ADDED = "1"
EXCEPTION_REMOVED = "2"

SCHEDULE_BASIS = "schedule_fallback_static_timetable"


class ServiceIntentError(RuntimeError):
    """The intent tables are missing, or a choice does not exist."""


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# GTFS time arithmetic (one implementation, shared with V2 by construction)
# --------------------------------------------------------------------------- #
def parse_gtfs_time(text: Any) -> Optional[int]:
    """``HH:MM:SS`` -> seconds after the service-day origin; HH may exceed 23."""
    if text is None:
        return None
    parts = str(text).strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return None
    if hours < 0 or not (0 <= minutes < 60) or not (0 <= seconds < 60):
        return None
    return hours * 3600 + minutes * 60 + seconds


def service_epoch(service_date: str, seconds_after_origin: int) -> Optional[int]:
    """GTFS 'noon minus twelve hours' origin on the service date, Sydney time."""
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(SYDNEY_TZ)
    except Exception:
        return None
    try:
        day = datetime.strptime(str(service_date).strip(), "%Y%m%d")
    except (TypeError, ValueError):
        return None
    noon = datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=zone)
    origin = noon - timedelta(hours=12)
    return int((origin + timedelta(seconds=int(seconds_after_origin))).timestamp())


def sydney_service_date(epoch: int) -> str:
    """The calendar date in Sydney for an epoch second, as YYYYMMDD."""
    from zoneinfo import ZoneInfo
    return datetime.fromtimestamp(int(epoch), ZoneInfo(SYDNEY_TZ)).strftime("%Y%m%d")


# --------------------------------------------------------------------------- #
# choices
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ServiceChoice:
    """One selectable intention: a route in one direction."""

    route_short_name: str
    direction_id: str
    route_id: str
    mode: str
    route_direction: str
    headsigns: Tuple[str, ...]
    stop_ids: Tuple[str, ...]
    n_trips: int

    @property
    def key(self) -> str:
        return f"{self.route_short_name}/{self.direction_id}"

    @property
    def label(self) -> str:
        """Human-readable, no raw ids: '433  Central Pitt St to Balmain Gladstone Park'."""
        text = self.route_direction or " / ".join(self.headsigns)
        return f"{self.route_short_name}  {text}"

    @property
    def short_label(self) -> str:
        """'433 -> Balmain - Gladstone Park' (the most frequent headsign)."""
        return f"{self.route_short_name} -> {self.headsigns[0] if self.headsigns else self.direction_id}"

    def serves(self, stop_id: Any) -> bool:
        return str(stop_id) in self.stop_ids

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "route_short_name": self.route_short_name,
            "direction_id": self.direction_id, "route_id": self.route_id,
            "mode": self.mode, "route_direction": self.route_direction,
            "headsigns": list(self.headsigns), "label": self.label,
            "stop_ids": list(self.stop_ids), "n_trips": self.n_trips,
        }


class ServiceIntentTables:
    """The committed static intent tables, loaded once."""

    def __init__(self, trips: pd.DataFrame, services: pd.DataFrame,
                 calendar: Optional[pd.DataFrame] = None,
                 calendar_dates: Optional[pd.DataFrame] = None,
                 provenance: Optional[Dict[str, Any]] = None,
                 inspection: Optional[Dict[str, Any]] = None) -> None:
        self.trips = trips.astype({"stop_id": str, "trip_id": str,
                                   "direction_id": str,
                                   "route_short_name": str})
        self.services = services.astype({"stop_id": str, "direction_id": str,
                                         "route_short_name": str})
        self.calendar = calendar
        self.calendar_dates = calendar_dates
        self.provenance = provenance or {}
        self.inspection = inspection or {}
        self._calendar_cache: Dict[Tuple[str, str], bool] = {}   # (service_id, date) -> active
        self._trip_index: Dict[str, Tuple[str, str, str]] = {
            str(r.trip_id): (str(r.route_short_name), str(r.direction_id),
                             str(r.service_id))
            for r in self.trips.drop_duplicates("trip_id").itertuples()}
        self._choices = self._build_choices()

    # ----- loading --------------------------------------------------------- #
    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "ServiceIntentTables":
        directory = Path(directory) if directory else DEFAULT_DIR
        trips_path = directory / TRIPS_FILE
        services_path = directory / SERVICES_FILE
        if not trips_path.exists() or not services_path.exists():
            raise ServiceIntentError(
                f"{directory} does not hold {TRIPS_FILE} and {SERVICES_FILE}. "
                "Run scripts/build_service_intent.py on the machine with the "
                "static GTFS snapshot.")
        trips = pd.read_csv(trips_path, dtype=str).fillna("")
        services = pd.read_csv(services_path, dtype=str).fillna("")
        services["n_trips"] = services["n_trips"].astype(int)

        def optional(name: str) -> Optional[pd.DataFrame]:
            path = directory / name
            return pd.read_csv(path, dtype=str).fillna("") if path.exists() else None

        def optional_json(name: str) -> Dict[str, Any]:
            path = directory / name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        return cls(trips, services, optional(CALENDAR_FILE),
                   optional(CALENDAR_DATES_FILE), optional_json(PROVENANCE_FILE),
                   optional_json(INSPECTION_FILE))

    # ----- choices --------------------------------------------------------- #
    def _build_choices(self) -> List[ServiceChoice]:
        choices: List[ServiceChoice] = []
        grouped = self.services.groupby(["route_short_name", "direction_id"],
                                        sort=True)
        for (short, direction), group in grouped:
            heads = (group.groupby("trip_headsign")["n_trips"].sum()
                     .sort_values(ascending=False))
            route_ids = sorted(group.route_id.unique())
            rdirs = [v for v in group.route_direction.unique() if v]
            trips = self.trips[(self.trips.route_short_name == short)
                               & (self.trips.direction_id == direction)]
            choices.append(ServiceChoice(
                route_short_name=str(short), direction_id=str(direction),
                route_id=route_ids[0] if len(route_ids) == 1 else ";".join(route_ids),
                mode=str(group["mode"].iloc[0]),
                route_direction=rdirs[0] if len(rdirs) == 1 else " | ".join(rdirs),
                headsigns=tuple(str(h) for h in heads.index),
                stop_ids=tuple(sorted(group.stop_id.unique())),
                n_trips=int(trips.trip_id.nunique()),
            ))
        choices.sort(key=lambda c: (_route_sort_key(c.route_short_name), c.direction_id))
        return choices

    def choices(self, mode: Optional[str] = None,
                stop_ids: Optional[Iterable[Any]] = None) -> List[ServiceChoice]:
        """Every (route, direction) that boards at a candidate stop.

        ``stop_ids`` narrows to choices serving at least one of those stops.
        """
        wanted = {str(s) for s in stop_ids} if stop_ids is not None else None
        out = []
        for choice in self._choices:
            if mode is not None and choice.mode != mode:
                continue
            if wanted is not None and not (wanted & set(choice.stop_ids)):
                continue
            out.append(choice)
        return out

    def choice(self, key: str) -> ServiceChoice:
        """'433/0' -> its choice. Raises when it does not exist."""
        for choice in self._choices:
            if choice.key == key:
                return choice
        raise ServiceIntentError(
            f"No service '{key}'. Valid keys: {[c.key for c in self._choices]}")

    def find(self, route_short_name: str,
             direction_id: Optional[Any] = None) -> List[ServiceChoice]:
        return [c for c in self._choices
                if c.route_short_name == str(route_short_name)
                and (direction_id is None or c.direction_id == str(direction_id))]

    # ----- trips ----------------------------------------------------------- #
    def trip_direction(self, trip_id: Any) -> Optional[Tuple[str, str]]:
        """(route_short_name, direction_id) of a static trip, or None.

        The trip_id is used EXACTLY as given; it is never parsed or split.
        """
        row = self._trip_index.get(str(trip_id))
        return (row[0], row[1]) if row else None

    def trip_matches(self, trip_id: Any, choice: ServiceChoice) -> bool:
        found = self.trip_direction(trip_id)
        return found == (choice.route_short_name, choice.direction_id)

    # ----- calendar -------------------------------------------------------- #
    @property
    def has_calendar(self) -> bool:
        return self.calendar is not None

    def service_active_on(self, service_id: str, service_date: str) -> Optional[bool]:
        """calendar.txt + calendar_dates.txt for one service on one date.

        None when the calendar tables are absent (then NOTHING can be said
        about whether a scheduled trip runs, and no fallback is offered).
        """
        if self.calendar is None:
            return None
        cache_key = (str(service_id), str(service_date))
        cached = self._calendar_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            day = datetime.strptime(str(service_date), "%Y%m%d").date()
        except ValueError:
            return None
        active = False
        rows = self.calendar[self.calendar.service_id == str(service_id)]
        for row in rows.itertuples():
            start = datetime.strptime(row.start_date, "%Y%m%d").date()
            end = datetime.strptime(row.end_date, "%Y%m%d").date()
            if start <= day <= end and getattr(row, WEEKDAY_COLUMNS[day.weekday()]) == "1":
                active = True
        if self.calendar_dates is not None:
            exc = self.calendar_dates[(self.calendar_dates.service_id == str(service_id))
                                      & (self.calendar_dates.date == str(service_date))]
            for row in exc.itertuples():
                if row.exception_type == EXCEPTION_ADDED:
                    active = True
                elif row.exception_type == EXCEPTION_REMOVED:
                    active = False
        self._calendar_cache[cache_key] = active
        return active

    def scheduled_visits(self, stop_id: Any, choice: ServiceChoice,
                         service_date: str) -> pd.DataFrame:
        """Static timetable visits of this service at this stop on this date.

        Each row carries ``scheduled_epoch`` (GTFS service-day arithmetic in
        Sydney time) and ``basis = schedule_fallback_static_timetable``.
        Only services the calendar says run on ``service_date`` are kept; an
        empty frame is returned when the calendar is unavailable, because a
        timetable row that may not run today is not a fallback, it is a guess.
        """
        rows = self.trips[(self.trips.stop_id == str(stop_id))
                          & (self.trips.route_short_name == choice.route_short_name)
                          & (self.trips.direction_id == choice.direction_id)].copy()
        if rows.empty or not self.has_calendar:
            return rows.iloc[0:0].assign(scheduled_epoch=pd.Series(dtype="int64"),
                                         basis=pd.Series(dtype="object"))
        keep = []
        for row in rows.itertuples():
            if self.service_active_on(row.service_id, service_date):
                seconds = parse_gtfs_time(row.arrival_time or row.departure_time)
                epoch = service_epoch(service_date, seconds) if seconds is not None else None
                if epoch is not None:
                    keep.append((row.Index, epoch))
        if not keep:
            return rows.iloc[0:0].assign(scheduled_epoch=pd.Series(dtype="int64"),
                                         basis=pd.Series(dtype="object"))
        index, epochs = zip(*keep)
        out = rows.loc[list(index)].copy()
        out["scheduled_epoch"] = list(epochs)
        out["basis"] = SCHEDULE_BASIS
        out["service_date"] = str(service_date)
        return out.sort_values(["scheduled_epoch", "trip_id"]).reset_index(drop=True)


def _route_sort_key(short: str) -> Tuple[int, int, str]:
    """Bus numbers first in numeric order (438 before 438N), lettered lines (L1) after."""
    digits = "".join(ch for ch in short if ch.isdigit())
    lettered = bool(short) and short[0].isalpha()
    return (1 if lettered else 0, int(digits) if digits else 10 ** 6, short)


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def describe_choices(tables: ServiceIntentTables, mode: Optional[str] = None) -> str:
    lines = []
    for choice in tables.choices(mode=mode):
        lines.append(f"{choice.key:8s} {choice.label:60s} "
                     f"{len(choice.stop_ids):2d} stops  {choice.n_trips:4d} trips/wk-sample  "
                     f"headsigns: {', '.join(choice.headsigns)}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default=None, choices=["bus", "light_rail"])
    parser.add_argument("--stop", default=None, help="only choices serving this stop_id")
    parser.add_argument("--dir", default=None)
    args = parser.parse_args(argv)
    tables = ServiceIntentTables.load(Path(args.dir) if args.dir else None)
    stops = [args.stop] if args.stop else None
    for choice in tables.choices(mode=args.mode, stop_ids=stops):
        print(f"{choice.key:8s} {choice.label}")
        print(f"         headsigns: {', '.join(choice.headsigns)}")
        print(f"         boards at: {', '.join(choice.stop_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
