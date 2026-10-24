"""Week 10 unified UI - WALK (Phase A), COMPARE (Phase B), TRACK (Phase C).

A single-window presentation layer over the validated Week 5 / 7 / 9 engine.
Nothing analytical lives here: origin snapping, the three walking routes,
Minetti effort, Tobler time and route identity are read from
``demo_terrain_route`` (Week 5), ``time_optimal_route`` and
``phase2_decision`` (Week 9) exactly as the validated prototypes use them.

The validated Week 9 UI (``scripts/demo_week9_primary.py`` and
``scripts/demo_phase2_decision.py``) remains the fallback implementation and
is not imported by this package at runtime.

Modules
-------
state.py        the explicit application state (no state hidden in callbacks)
walk_engine.py  the adapter over the existing engine (snap, search, solve, text)
map_view.py     the large map: context, stops, origin, three layered routes
walk_panel.py   the contextual right-hand panel and the Technical details box
shell.py        the one window that composes them and dispatches clicks
compare_engine.py / compare_panel.py   Phase B: the Phase 2 adapter and panel
track_engine.py / track_panel.py       Phase C: the Week 7 tracking adapter (headless) and panel
scene_export.py                        the existing schema-1.1 exporters
"""

MODE_WALK = "WALK"
UI_MODES = ("WALK", "COMPARE", "TRACK")  # Phase A / B / C, all in the one window
FUTURE_MODES = ()                         # kept for callers of earlier phases
