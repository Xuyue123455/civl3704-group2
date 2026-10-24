"""The adapter between the unified COMPARE state and the validated Phase 2 engine.

Nothing here classifies, ranks or selects. Every COMPARE quantity comes from:

* ``service_intent.ServiceIntentTables``  route + direction identity, exact trips
* ``phase2_decision.load_realtime``       one snapshot (LIVE fetch through the Week 7
                                          V1-V3 chain with the API key from the
                                          environment; REPLAY from a saved .pb; none
                                          in ANALYTICAL)
* ``phase2_decision.run_phase2``          the scene: Phase 1 ranking, per-stop trip
                                          inventories, exact-trip grouping, the
                                          NEXT FEASIBLE TRIP, slack, catchability,
                                          feasible-stop winners and Pareto
* ``phase2_decision.apply_trip_selection`` a manual trip, re-derived from the same
                                          scene (no re-routing, no feed access)

The human wording comes from the cleaned Week 9 Compare presentation helpers
(``demo_phase2_decision``: trip rows, disambiguation, provenance-aware
counts, rank lines, winner marker). Only the pure helper functions are used;
the legacy window class is never instantiated.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import feed_adapter
from .walk_engine import WalkEngine, load_script, phase2

intent_mod = phase2.intent_mod                       # service_intent
p2ui = load_script("demo_phase2_decision")           # pure presentation helpers (no window)

MODE_ANALYTICAL = phase2.MODE_ANALYTICAL
MODE_LIVE = phase2.MODE_LIVE
MODE_REPLAY = phase2.MODE_REPLAY


class CompareEngine:
    """One instance per window; realtime is fetched explicitly, never on a timer."""

    def __init__(self, walk: WalkEngine, tables: Any, run_mode: str, replay_path: Optional[Path] = None,
                 margin_s: float = phase2.DEFAULT_MARGIN_S, schedule_fallback: bool = True) -> None:
        self.walk = walk
        self.tables = tables
        self.run_mode = str(run_mode).upper()
        self.replay_path = Path(replay_path) if replay_path else None
        self.margin_s = float(margin_s)
        self.schedule_fallback = bool(schedule_fallback)
        self.loaded_static: Optional[Dict[str, Any]] = None
        self.realtime: Optional[Any] = None           # phase2.RealtimeContext
        self.fetches = 0
        # Week 11: the snapshot is the feed of ONE mode (bus / light_rail); a run for a service of another
        # mode replaces it (never mixes two feeds); counted per mode
        self.realtime_mode: Optional[str] = None
        self.requested_mode: Optional[str] = None
        self.fetches_by_mode: Dict[str, int] = {}

    @classmethod
    def load(cls, walk: WalkEngine, run_mode: str, replay_path: Optional[Path] = None) -> "CompareEngine":
        return cls(walk, intent_mod.ServiceIntentTables.load(), run_mode, replay_path)

    # ----- service intent ------------------------------------------------------ #
    def routes_available(self) -> List[str]:
        seen: List[str] = []
        for choice in self.tables.choices():
            if choice.route_short_name not in seen:
                seen.append(choice.route_short_name)
        return seen

    def directions_for(self, route_short_name: str) -> List[Any]:
        return self.tables.find(route_short_name)

    def choice(self, key: str) -> Any:
        return self.tables.choice(key)

    def choices_for_stop(self, stop_id: str) -> List[Any]:
        return self.tables.choices(stop_ids=[str(stop_id)])

    @staticmethod
    def direction_label(choice: Any) -> str:
        text = choice.route_direction or " / ".join(choice.headsigns)
        return text.replace(" to ", " -> ", 1) if " to " in text else text

    # ----- realtime (explicit, no timers) ---------------------------------------- #
    def mode_of_service(self, service_key: Optional[str]) -> str:
        """The mode of a service intent (route_type -> config mode_policy), through the Week 9 tables."""
        if not service_key:
            return feed_adapter.MODE_BUS
        try:
            return feed_adapter.mode_for_choice(self.tables.choice(service_key))
        except Exception:
            return feed_adapter.MODE_BUS

    def set_feed_mode(self, mode: Optional[str]) -> str:
        """Which mode's feed the next snapshot must be (set by ``run`` from the service, or by the shell)."""
        self.requested_mode = feed_adapter.normalise_mode(mode)
        return self.requested_mode

    def snapshot(self, refresh: bool = False) -> Any:
        """The snapshot for this run; fetched once, again only when asked - or when the requested mode differs
        from the mode the current snapshot was fetched for (LIVE: the other feed replaces it, never mixes)."""
        wanted = getattr(self, "requested_mode", None) or self.realtime_mode or feed_adapter.MODE_BUS
        if self.realtime is not None and not refresh and (self.run_mode != MODE_LIVE
                                                          or self.realtime_mode == wanted):
            return self.realtime
        if self.run_mode != MODE_ANALYTICAL and self.loaded_static is None:
            v3 = load_script("demo_week7_realtime")
            self.loaded_static = v3.load_static_once(want_scheduled=True)
        if self.run_mode == MODE_LIVE:
            self.realtime = phase2.load_realtime(
                self.run_mode, self.replay_path, self.loaded_static,
                fetcher=feed_adapter.trip_update_fetcher(wanted), feed_mode=wanted,
                feed_label=feed_adapter.FEED_LABELS[wanted])
        else:                                          # REPLAY / ANALYTICAL: one file / nothing, as before
            self.realtime = phase2.load_realtime(self.run_mode, self.replay_path, self.loaded_static)
        self.realtime_mode = wanted
        self.fetches += 1
        self.fetches_by_mode[wanted] = self.fetches_by_mode.get(wanted, 0) + 1
        return self.realtime

    # ----- the scene ---------------------------------------------------------------- #
    def run(self, origin: Dict[str, Any], service_key: str, trip_id: Optional[str] = None,
            origin_label: str = "") -> Dict[str, Any]:
        self.set_feed_mode(self.mode_of_service(service_key))
        realtime = self.snapshot()
        return phase2.run_phase2(
            self.walk.context, self.tables, origin["input_latitude"], origin["input_longitude"],
            service_key, realtime, self.walk.cost_table, margin_s=self.margin_s,
            schedule_fallback=self.schedule_fallback, stops=self.walk.stops,
            origin_label=origin_label, trip_id=trip_id)

    @staticmethod
    def select_trip(scene: Dict[str, Any], trip_id: Optional[str]) -> Dict[str, Any]:
        return phase2.apply_trip_selection(scene, trip_id)

    # ----- read-only views over the scene ------------------------------------------ #
    @staticmethod
    def stop_entry(scene: Dict[str, Any], stop_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if scene is None or stop_id is None:
            return None
        return next((s for s in scene["stops"] if s["stop_id"] == str(stop_id)), None)

    @staticmethod
    def winners(scene: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """The feasible-stop walking winners the engine already chose (no recomputation)."""
        summary = scene.get("summary") or {}
        return {"effort": summary.get("lowest_terrain_effort_stop"),
                "time": summary.get("lowest_modelled_walking_time_stop"),
                "distance": summary.get("shortest_walk_stop")}

    @staticmethod
    def default_compare_stop(scene: Dict[str, Any], preferred: Optional[str]) -> Optional[str]:
        """Keep the viewer's stop when the chosen service calls there; else the Gentlest winner."""
        entry = CompareEngine.stop_entry(scene, preferred)
        if entry is not None and entry.get("serves_selected_service"):
            return str(preferred)
        summary = scene.get("summary") or {}
        if summary.get("lowest_terrain_effort_stop"):
            return summary["lowest_terrain_effort_stop"]
        served = [s for s in scene["stops"] if s["serves_selected_service"]]
        return served[0]["stop_id"] if served else None

    def result_from_scene_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A scene stop entry in the WALK result shape, so MapView.draw_routes can draw it."""
        routes = entry.get("routes") or {}
        if not routes.get("distance"):
            return None
        out = {"stop_id": entry["stop_id"], "stop_name": entry["stop_name"], "available": True,
               "routes": {}, "route_identity": entry.get("route_identity")}
        for obj in ("distance", "effort", "time"):
            r = routes.get(obj)
            if r is None:
                continue
            out["routes"][obj] = {"edges": [tuple(e) for e in r["edge_keys"]], "nodes": r["nodes"],
                                  "distance": r["distance_m"], "effort": r["effort_j_per_kg"],
                                  "walking_time_s": r["walking_time_s"]}
        return out

    # ----- wording (reused from the cleaned Week 9 Compare view) ------------------- #
    @staticmethod
    def trip_rows(scene: Dict[str, Any], max_rows: int = 5) -> List[Dict[str, Any]]:
        return p2ui.trip_rows(scene, max_rows)

    @staticmethod
    def trip_labels(scene: Dict[str, Any]) -> Dict[str, Any]:
        return p2ui.trip_display_labels(scene.get("trip_candidates") or [])

    @staticmethod
    def trip_row_text(candidate: Dict[str, Any], label=None) -> str:
        return p2ui.trip_row_text(candidate, label)

    @staticmethod
    def trip_basis(candidate: Dict[str, Any]) -> str:
        return p2ui.trip_basis(candidate)

    @staticmethod
    def feasibility_count_text(counts: Dict[str, Any]) -> str:
        return p2ui.feasibility_count_text(counts)

    @staticmethod
    def selected_trip_text(scene: Dict[str, Any], run_mode: str) -> str:
        return p2ui.selected_trip_text(scene, run_mode)

    @staticmethod
    def trip_at_stop_lines(entry: Dict[str, Any], scene: Dict[str, Any], run_mode: str) -> List[str]:
        return p2ui.trip_at_stop_lines(entry, scene, run_mode)

    @staticmethod
    def rank_lines(scene: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
        return p2ui.rank_lines(scene, entry)

    @staticmethod
    def winner_marker(highlights):
        return p2ui.winner_marker(highlights)

    @staticmethod
    def fmt_slack(seconds: Optional[int]) -> str:
        return p2ui.fmt_slack(seconds)

    @staticmethod
    def clock(sydney_text: Optional[str]) -> str:
        return p2ui.clock(sydney_text)


def feasible_summary_sentence(scene: Dict[str, Any]) -> List[str]:
    """'6 stops you can reach in time' / '6 stops feasible by timetable' + confirmation line."""
    chosen = scene.get("selected_trip")
    if chosen is None:
        if scene.get("trip_candidates"):
            return ["No listed trip is proven catchable from here.",
                    "The walking comparison below ignores catchability."]
        return ["No upcoming trip could be timed for this service."]
    rt = len(chosen.get("catchable_stop_ids") or [])
    tt_only = len(chosen.get("schedule_only_stop_ids") or [])
    lines: List[str] = []
    if rt and tt_only:
        lines.append(f"{rt} stop{'s' if rt != 1 else ''} you can reach in time (realtime) + "
                     f"{tt_only} feasible by timetable only")
    elif rt:
        lines.append(f"{rt} stop{'s' if rt != 1 else ''} you can reach in time")
    elif tt_only:
        lines.append(f"{tt_only} stop{'s' if tt_only != 1 else ''} feasible by timetable")
        lines.append("Realtime confirmation unavailable")
    else:
        lines.append("No stop proven catchable for this trip")
    return lines


def walking_choice_blocks(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gentlest / Fastest / Shortest among the feasible stops, grouped when one stop wins several."""
    winners = CompareEngine.winners(scene)
    order = [("effort", "Gentlest"), ("time", "Fastest"), ("distance", "Shortest")]
    blocks: List[Dict[str, Any]] = []
    for obj, name in order:
        stop_id = winners.get(obj)
        if stop_id is None:
            continue
        existing = next((b for b in blocks if b["stop_id"] == stop_id), None)
        if existing is not None:
            existing["names"].append(name)
            continue
        entry = CompareEngine.stop_entry(scene, stop_id)
        catch = (entry or {}).get("catchability") or {}
        blocks.append({"stop_id": stop_id, "names": [name], "stop_name": (entry or {}).get("stop_name", ""),
                       "effort_j_per_kg": (entry or {}).get("effort_j_per_kg"),
                       "walking_time_s": (entry or {}).get("time_route_walking_time_s"),
                       "distance_m": (entry or {}).get("shortest_distance_m"),
                       "slack_s": catch.get("slack_s"), "status": catch.get("status"),
                       "highlights": list((entry or {}).get("highlights") or [])})
    return blocks


def measure(timings: Dict[str, float], key: str, started: float) -> None:
    timings[key] = time.perf_counter() - started
