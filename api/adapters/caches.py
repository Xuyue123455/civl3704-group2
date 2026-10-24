"""Process-level read-once caches the API installs over two validated primitives (outputs unchanged).

Mobile Web Phase 2, performance.  Profiling the UI-free session showed that, once the window was out of the
request path, most of a request's time went to two derivations that are pure functions of loaded, immutable
objects and that the validated scripts recompute on every call:

* ``src.snapping.project_for_snapping(graph)`` - the whole walking graph re-projected to EPSG:7856 for every
  nearest-node query (the Week 5 docstring says so: "this projects the whole graph on every call"); ~0.9 s per
  call here, two to four calls per request;
* ``inspect_vehicle_positions.static_tables(static)`` - the trips.txt / routes.txt rows re-indexed into dicts on
  every vehicle refresh; ~2.4 s per TRACK request with the full TfNSW snapshot.

Both are memoised here, keyed by the identity of the loaded object (weak references: a reloaded graph or
snapshot is a new key) and guarded by its size (a mutated graph is re-projected).  The memoised function is the
validated function itself - its result for the same input is returned, nothing is re-implemented - so every
analytical value is unchanged; ``tests/test_web_api_contract.py`` (fixture equality) and
``tests/test_mobile_web_phase2.py`` (equivalence with the reference session) run with the caches installed.

The caches are installed by the API process only (``EngineHost``), never by the desktop window or the frozen
scripts; set ``ACCESSIBLE_TRANSIT_API_CACHES=0`` to run the API without them.
"""

from __future__ import annotations

import os
import weakref
from typing import Any, Dict, Tuple

_installed: Dict[str, Any] = {}


def enabled() -> bool:
    return os.environ.get("ACCESSIBLE_TRANSIT_API_CACHES", "1") not in ("0", "false", "no")


def _graph_key(graph: Any, metric_crs: str) -> Tuple[int, int, str]:
    return int(graph.number_of_nodes()), int(graph.number_of_edges()), str(metric_crs)


class _Memo:
    """id -> (weak reference, size key, value).  The loaded objects are dataclasses / graphs without a stable
    hash, so the memo is keyed by identity and checked against a weak reference: a reloaded object (new
    identity) or a resized one (other key) is never served the old value."""

    def __init__(self) -> None:
        self._entries: Dict[int, Tuple[Any, Any, Any]] = {}

    def get(self, obj: Any, key: Any) -> Any:
        entry = self._entries.get(id(obj))
        if entry is None:
            return None
        ref, stored_key, value = entry
        if ref() is not obj or stored_key != key:
            self._entries.pop(id(obj), None)
            return None
        return value

    def put(self, obj: Any, key: Any, value: Any) -> None:
        try:
            ref = weakref.ref(obj)
        except TypeError:                                   # not weak-referenceable: never cached
            return
        self._entries[id(obj)] = (ref, key, value)


def install() -> Dict[str, Any]:
    """Idempotent; returns the cache statistics dict (hits / misses per cache)."""
    if _installed or not enabled():
        return _installed
    from src import snapping
    from unified_ui import track_engine

    stats = {"projection": {"hits": 0, "misses": 0}, "static_tables": {"hits": 0, "misses": 0}}

    # ----- 1. the projected graph for snapping -------------------------------------------------- #
    original_project = snapping.project_for_snapping
    projected = _Memo()

    def project_for_snapping(graph, metric_crs=snapping.METRIC_CRS):
        key = _graph_key(graph, metric_crs)
        cached = projected.get(graph, key)
        if cached is not None:
            stats["projection"]["hits"] += 1
            return cached
        stats["projection"]["misses"] += 1
        result = original_project(graph, metric_crs)
        projected.put(graph, key, result)
        return result

    snapping.project_for_snapping = project_for_snapping

    # ----- 2. the trips / routes dicts the vehicle chain indexes ------------------------------- #
    v4a = track_engine.v4b.v4a
    original_tables = v4a.static_tables
    tables = _Memo()

    def static_tables(static):
        if static is None:
            return original_tables(static)
        key = (len(static.trips) if getattr(static, "has_trips", False) else -1,
               len(static.routes) if getattr(static, "has_routes", False) else -1)
        cached = tables.get(static, key)
        if cached is not None:
            stats["static_tables"]["hits"] += 1
            return cached
        stats["static_tables"]["misses"] += 1
        result = original_tables(static)
        tables.put(static, key, result)
        return result

    v4a.static_tables = static_tables

    _installed.update({"stats": stats, "originals": {"project_for_snapping": original_project,
                                                      "static_tables": original_tables}})
    return _installed


def stats() -> Dict[str, Any]:
    return dict((_installed.get("stats") or {}))
