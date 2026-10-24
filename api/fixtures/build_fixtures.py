"""Record the golden API fixtures from the validated engines (Desktop Gold).

    python -m api.fixtures.build_fixtures

Every fixture is the adapter's own response for a validated case, plus a ``recorded`` block (commit, inputs,
generator, environment).  Nothing is typed in by hand; re-running on Desktop Gold must reproduce the same
analytical values (geometry and clocks included for the deterministic inputs below).
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from api import DESKTOP_GOLD, schemas                       # noqa: E402
from api.adapters import compare_adapter, track_adapter, walk_adapter   # noqa: E402
from api.adapters.engines import EngineHost                  # noqa: E402

ORIGIN_O2 = {"lat": -33.876602, "lon": 151.1867514}
ORIGIN_L1 = {"lat": -33.878200, "lon": 151.186500}
SERVICE_DATE = "20261023"                                    # the validated analytical anchor (Week 10 suites)
L1_REPLAY = "l1_2026-10-17_0917"
L1_TRIP, L1_PLATFORM = "49189-6043", "203782"                # the Week 11 L1 fixture's next feasible trip


def recorded(inputs, generator):
    import matplotlib
    return {"desktop_gold": DESKTOP_GOLD, "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator": generator, "inputs": inputs,
            "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
            "note": "golden fixture recorded from the validated engines; a deterministic stand-in, not live output"}


def strip_provenance(payload):
    payload = dict(payload)
    payload.pop("provenance", None)
    return payload


def main() -> None:
    from unified_ui import compare_engine
    host = EngineHost()
    observed = int(compare_engine.intent_mod.service_epoch(SERVICE_DATE, 16 * 3600))
    index = {"walk": {}, "compare": {}, "track": {}}
    out = {}

    # health
    out["health"] = {"recorded": recorded({}, "EngineHost.graph_facts"), "graph": host.graph_facts()}

    # WALK: O2 -> 203723, both profiles
    for profile in ("standard", "step_free"):
        req = schemas.validate_walk_request({"origin": ORIGIN_O2, "stop_id": "203723", "profile": profile})
        key = f"walk_o2_203723_{profile}"
        out[key] = {"recorded": recorded(req, "api.adapters.walk_adapter.walk"),
                    "response": strip_provenance(walk_adapter.walk(host, req))}
        index["walk"][key] = {"match": req}

    # COMPARE: 433 direction 1 from O2, Standard, selected stop 203723, analytical anchor 16:00 on 2026-10-23
    req = schemas.validate_compare_request({"origin": ORIGIN_O2, "route_short_name": "433", "direction_id": 1,
                                            "walking_profile": "standard", "selected_trip_id": None,
                                            "selected_stop_id": "203723"})
    key = "compare_433_1_o2_203723_standard"
    inputs = {**req, "run_mode": "ANALYTICAL", "replay": None, "observed_at": observed, "service_date": SERVICE_DATE}
    out[key] = {"recorded": recorded(inputs, "api.adapters.compare_adapter.compare"),
                "response": strip_provenance(compare_adapter.compare(host, req, "ANALYTICAL", None, observed))}
    index["compare"][key] = {"match": {**req, "run_mode": "ANALYTICAL", "replay": None}}

    # TRACK: the recorded L1 capture through REPLAY (realtime-confirmed, POSITION REPORTED)
    req = schemas.validate_track_request({"origin": ORIGIN_L1, "trip_id": L1_TRIP, "boarding_stop_id": L1_PLATFORM,
                                          "walking_profile": "standard"})
    key = "track_l1_replay_2026-10-17"
    inputs = {**req, "run_mode": "REPLAY", "replay": L1_REPLAY, "observed_at": None}
    out[key] = {"recorded": recorded(inputs, "api.adapters.track_adapter.track"),
                "response": strip_provenance(track_adapter.track(host, req, "REPLAY", L1_REPLAY, None))}
    index["track"][key] = {"match": {**req, "run_mode": "REPLAY", "replay": L1_REPLAY}}

    # TRACK: the 433 analytical case (the Desktop Gold TRACK screenshot), next feasible trip at 203723
    compare_fix = out["compare_433_1_o2_203723_standard"]["response"]
    trip = compare_fix["selected_trip"]["trip_id"]
    req = schemas.validate_track_request({"origin": ORIGIN_O2, "trip_id": trip, "boarding_stop_id": "203723",
                                          "walking_profile": "standard"})
    key = "track_433_analytical_o2_203723"
    inputs = {**req, "run_mode": "ANALYTICAL", "replay": None, "observed_at": observed, "service_date": SERVICE_DATE}
    out[key] = {"recorded": recorded(inputs, "api.adapters.track_adapter.track"),
                "response": strip_provenance(track_adapter.track(host, req, "ANALYTICAL", None, observed))}
    index["track"][key] = {"match": {**req, "run_mode": "ANALYTICAL", "replay": None}}

    for name, payload in out.items():
        (HERE / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        print("wrote", name)
    (HERE / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    print("wrote index.json")


if __name__ == "__main__":
    main()
