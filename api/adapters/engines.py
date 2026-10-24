"""Lazy, process-wide handles on the validated engines (loaded once; nothing recomputed here).

The API never builds a graph, a cost table or a model of its own: it loads exactly what the desktop window loads
(``WalkEngine.load``, ``StandardContext.load``, ``ServiceIntentTables.load``) and hands the objects to the adapters.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (str(PROJECT_ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class EngineHost:
    """One loaded set of engines for the API process.  ``stair_cache_dir`` None = the project default."""

    def __init__(self, stair_cache_dir: Optional[Path] = None, hybrid_stairs: bool = True) -> None:
        self.stair_cache_dir = stair_cache_dir
        self.hybrid_stairs = hybrid_stairs
        self._engine = None
        self._standard = None
        self._standard_loaded = False
        self._tables = None
        self._real_static = None
        self._shape_store = None
        from . import caches
        self.caches = caches.install()               # Phase 2: read-once caches over two validated primitives

    @property
    def engine(self):
        if self._engine is None:
            from unified_ui.walk_engine import WalkEngine
            self._engine = WalkEngine.load()
        return self._engine

    @property
    def standard(self):
        """The Standard hybrid context (None when the flag is off or the stair cache is absent)."""
        if not self._standard_loaded:
            self._standard_loaded = True
            if self.hybrid_stairs:
                from unified_ui.standard_hybrid import StandardContext
                self._standard = StandardContext.load(self.engine, self.stair_cache_dir)
        return self._standard

    @property
    def tables(self):
        if self._tables is None:
            from unified_ui import compare_engine
            self._tables = compare_engine.intent_mod.ServiceIntentTables.load()
        return self._tables

    @property
    def real_static(self):
        """The static GTFS snapshot the TRACK planned route needs (trips.txt -> shape_id -> shapes.txt)."""
        if self._real_static is None:
            from unified_ui.walk_engine import load_script
            v3 = load_script("demo_week7_realtime")
            self._real_static = v3.load_static_once(want_scheduled=False)
        return self._real_static

    @property
    def shape_store(self):
        """One ``TripShapeStore`` over the loaded static snapshot, shared by every TRACK request (Phase 2).

        Planned route geometry is static GTFS (trip_id -> shape_id -> shapes.txt) and the store is a read-once
        cache keyed by the snapshot's own sha256; sharing it is what the desktop window does for its lifetime.
        """
        if self._shape_store is None:
            from types import SimpleNamespace
            from unified_ui import track_engine
            self._shape_store = track_engine.v4d.shape_store_for(SimpleNamespace(static=self.real_static["static"]))
        return self._shape_store

    def graph_facts(self) -> Dict[str, Any]:
        """The invariants the contract names, read from the loaded objects (never typed in)."""
        facts = {"trusted_edges": int(self.engine.context["subgraph"].number_of_edges())}
        standard = self.standard
        if standard is not None:
            summary = standard.summary()
            facts.update({"standard_edges": int(summary["hybrid_edges"]),
                          "stair_edges_added": int(summary["stair_edges_added"]),
                          "eligible_classes": list(summary["eligible_classes"]),
                          "excluded_class_edges": summary.get("excluded_class_edges"),
                          "stair_model_version": summary.get("model_version")})
        else:
            facts.update({"standard_edges": None, "stair_edges_added": None, "eligible_classes": [],
                          "excluded_class_edges": None, "stair_model_version": None})
        return facts
