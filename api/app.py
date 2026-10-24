"""The thin HTTP layer (Phase W1 scaffold, extended for Mobile Web Phase 1): JSON in / JSON out, no framework.

    GET  /api/health
    POST /api/walk
    POST /api/compare
    POST /api/track
    GET  /api/stops  /api/services  /api/boundary  /api/basemap  /api/stairs  /api/contours  /api/terrain  /api/replays   (catalogue)

The analytical endpoints are serialised with one process lock: engine mode drives the UI-free journey session
(adapters/session.py over ``scripts/unified_ui/journey.py``) per request, and the shared engines
(``CompareEngine`` snapshots, the static GTFS snapshot, the shape store) are not written for concurrent access.

Two sources, always declared in ``provenance.source``:

* ``fixture`` (default): deterministic golden responses recorded from Desktop Gold (api/fixtures/*.json) so the
  frontend can be built offline.  A fixture is never presented as live output: ``provenance.is_fixture`` is true
  and the fixture's own ``recorded`` block says which commit and inputs produced it.
* ``engine`` (``ACCESSIBLE_TRANSIT_API_MODE=engine`` or ``ApiApp(mode="engine")``): the adapters run the
  validated engines for the request.

The browser never sees a credential: the API process reads ``TFNSW_API_KEY`` from its own environment only when a
LIVE mode is wired (Phase W6), and no response field is derived from it.

Run locally:  python -m api.app  (serves on 127.0.0.1:8000 with the standard library's WSGI server).
"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from . import API_VERSION, DESKTOP_GOLD, schemas

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MODE_FIXTURE, MODE_ENGINE = "fixture", "engine"
FORBIDDEN_RESPONSE_SUBSTRINGS = ("TFNSW_API_KEY", "apikey ", "Authorization")


def fixture_key(endpoint: str, request: Dict[str, Any]) -> Optional[str]:
    """Which golden fixture answers this request (None when no fixture matches)."""
    index = load_fixture_index()
    for key, spec in index.get(endpoint, {}).items():
        if all(request.get(field) == value for field, value in spec["match"].items()):
            return key
    return None


_INDEX_CACHE: Optional[Dict[str, Any]] = None


def load_fixture_index() -> Dict[str, Any]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        path = FIXTURE_DIR / "index.json"
        _INDEX_CACHE = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _INDEX_CACHE


def load_fixture(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


class ApiApp:
    """Routes requests; owns the lazily loaded engines in engine mode."""

    def __init__(self, mode: Optional[str] = None, host: Optional[Any] = None) -> None:
        self.mode = (mode or os.environ.get("ACCESSIBLE_TRANSIT_API_MODE") or MODE_FIXTURE).lower()
        if self.mode not in (MODE_FIXTURE, MODE_ENGINE):
            raise ValueError(f"mode must be {MODE_FIXTURE} or {MODE_ENGINE}")
        self._host = host
        self._lock = threading.Lock()                       # one engine computation at a time (Matplotlib state)
        self.routes: Dict[Tuple[str, str], Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            ("GET", "/api/health"): self.health,
            ("GET", "/api/ready"): self.ready,
            ("POST", "/api/walk"): self.walk,
            ("POST", "/api/compare"): self.compare,
            ("POST", "/api/track"): self.track,
            ("GET", "/api/stops"): self.stops,
            ("GET", "/api/services"): self.services,
            ("GET", "/api/boundary"): self.boundary,
            ("GET", "/api/basemap"): self.basemap,
            ("GET", "/api/stairs"): self.stairs,
            ("GET", "/api/contours"): self.contours,
            ("GET", "/api/hillshade"): self.hillshade,
            ("GET", "/api/terrain"): self.terrain,
            ("GET", "/api/terrain/contours"): self.terrain_contours,
            ("GET", "/api/replays"): self.replays,
        }
        # binary assets (served as-is; not JSON): the relief-shading image, the terrain display surface images
        self.binary_routes: Dict[Tuple[str, str], Callable[[], Tuple[bytes, str]]] = {
            ("GET", "/api/hillshade.png"): self.hillshade_png,
            ("GET", "/api/terrain/elevation_tint.png"): lambda: self.terrain_png("elevation_tint.png"),
            ("GET", "/api/terrain/hillshade.png"): lambda: self.terrain_png("hillshade.png"),
        }

    @property
    def host(self):
        if self._host is None:
            from .adapters.engines import EngineHost
            self._host = EngineHost()
        return self._host

    # ----- dispatch -------------------------------------------------------------------- #
    def handle_binary(self, method: str, path: str) -> Optional[Tuple[int, bytes, str]]:
        """A binary asset route (None when the path is not one); 503 with an empty body when the file is absent."""
        handler = self.binary_routes.get((method.upper(), path))
        if handler is None:
            return None
        try:
            data, content_type = handler()
        except FileNotFoundError:
            return 503, b"", "application/octet-stream"
        return 200, data, content_type

    def handle(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        handler = self.routes.get((method.upper(), path))
        if handler is None:
            return 404, {"error": "not found", "routes": sorted(f"{m} {p}" for m, p in self.routes)}
        try:
            with self._lock:
                payload = handler(body or {})
        except schemas.ValidationError as error:
            return 400, {"error": str(error)}
        except NotImplementedError as error:
            return 501, {"error": str(error)}
        except FileNotFoundError as error:
            return 503, {"error": f"server data missing: {Path(str(error)).name}", "hint": "run scripts/check_demo_readiness.py"}
        except Exception as error:                                           # noqa: BLE001 - engine failure -> 503, redacted
            return 503, _engine_failure(error)
        payload = _redact(_finite(payload))                            # NaN / inf are not JSON; credential-like text never leaves
        self._assert_no_secret(payload)
        if path == "/api/ready" and payload.get("status") != "ready":
            return 503, payload                                        # a hosting probe reads the status code
        return 200, payload

    @staticmethod
    def _assert_no_secret(payload: Dict[str, Any]) -> None:
        text = json.dumps(payload)
        for needle in FORBIDDEN_RESPONSE_SUBSTRINGS:
            if needle in text:
                raise RuntimeError("a response carried a credential-like field; refused")

    # ----- endpoints ------------------------------------------------------------------- #
    def health(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == MODE_ENGINE:
            graph = self.host.graph_facts()
            source, note = schemas.SOURCE_ENGINE, "graph facts read from the loaded engines"
        else:
            graph = load_fixture("health")["graph"]
            source, note = schemas.SOURCE_FIXTURE, "graph facts recorded from Desktop Gold (api/fixtures/health.json)"
        return {"status": "ok", "desktop_gold": DESKTOP_GOLD[:7], "desktop_gold_full": DESKTOP_GOLD,
                "api_version": API_VERSION, "mode": self.mode, "graph": graph,
                "endpoints": sorted(f"{m} {p}" for m, p in self.routes),
                "provenance": schemas.provenance(source, note)}

    def ready(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        """Hosting readiness (deployment audit): alive, the runtime assets the web request path reads, the engines loadable
        (engine mode: loading them here IS the warm-up), the realtime configuration status.  Never a secret value."""
        from .adapters import readiness
        return readiness.report(self)

    def _answer(self, endpoint: str, request: Dict[str, Any], compute: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        if self.mode == MODE_ENGINE:
            return compute()
        key = fixture_key(endpoint, request)
        if key is None:
            raise NotImplementedError(f"no golden fixture matches this {endpoint} request in fixture mode; "
                                      f"available: {sorted(load_fixture_index().get(endpoint, {}))}")
        fixture = load_fixture(key)
        payload = dict(fixture["response"])
        payload["provenance"] = schemas.provenance(schemas.SOURCE_FIXTURE,
                                                   f"golden fixture {key}: recorded from Desktop Gold, not computed for this request",
                                                   fixture=key, recorded=fixture["recorded"])
        return payload

    def walk(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = schemas.validate_walk_request(body)
        from .adapters import walk_adapter
        return self._answer("walk", request, lambda: walk_adapter.walk(self.host, request))

    def compare(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = schemas.validate_compare_request(body)
        run_mode, replay, observed_at = _mode_fields(body)
        from .adapters import compare_adapter
        return self._answer("compare", {**request, "run_mode": run_mode, "replay": replay},
                            lambda: compare_adapter.compare(self.host, request, run_mode, replay, observed_at))

    def track(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = schemas.validate_track_request(body)
        run_mode, replay, observed_at = _mode_fields(body)
        from .adapters import track_adapter
        return self._answer("track", {**request, "run_mode": run_mode, "replay": replay},
                            lambda: track_adapter.track(self.host, request, run_mode, replay, observed_at))

    # ----- catalogue (read-only re-shaping of what the desktop loads; fixture mode serves the same files) -------- #
    def stops(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.stops(self.host)

    def services(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.services(self.host)

    def boundary(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.boundary()

    def basemap(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.basemap()

    def stairs(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.stairs(self.host)

    def contours(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.contours()

    def hillshade(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.hillshade()

    def hillshade_png(self) -> Tuple[bytes, str]:
        from .adapters import catalog
        return catalog.hillshade_png(), "image/png"

    def terrain(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.terrain()

    def terrain_contours(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.terrain_contours()

    def terrain_png(self, name: str) -> Tuple[bytes, str]:
        from .adapters import catalog
        return catalog.terrain_png(name), "image/png"

    def replays(self, _body: Dict[str, Any]) -> Dict[str, Any]:
        from .adapters import catalog
        return catalog.replays()


def _finite(value: Any) -> Any:
    """Replace float NaN / inf with null everywhere in a payload (a CSV-backed empty name or an unknown measure).

    Python's json.dumps writes ``NaN``, which is not JSON: a browser's ``response.json()`` then fails and the client
    sees no payload at all (the spatial audit found this on Standard routes over an unnamed stair)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _redact(value: Any) -> Any:
    """Strip the forbidden substrings from every string in a payload (e.g. the V1 chain's MissingApiKey instructions,
    which name the environment variable, surface in realtime provenance when LIVE falls back to the timetable)."""
    if isinstance(value, str):
        for needle in FORBIDDEN_RESPONSE_SUBSTRINGS:
            if needle in value:
                value = value.replace(needle, "[redacted]")
        return value
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _engine_failure(error: Exception) -> Dict[str, Any]:
    """A LIVE fetch without a server-side credential or network, or any other engine failure, as a 503 whose text
    can never carry a credential (the forbidden substrings are stripped before the payload leaves)."""
    name = type(error).__name__
    if name == "MissingApiKey":
        return {"error": "realtime unavailable: the API server holds no TfNSW credential, so LIVE cannot fetch; "
                         "use run_mode REPLAY or ANALYTICAL", "type": name, "realtime": "unavailable"}
    detail = (str(error).splitlines() or [""])[0][:300]
    for needle in FORBIDDEN_RESPONSE_SUBSTRINGS:
        detail = detail.replace(needle, "[redacted]")
    return {"error": f"engine failure: {detail}", "type": name, "realtime": "unknown"}


def _mode_fields(body: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[int]]:
    run_mode = str(body.get("run_mode", "ANALYTICAL")).upper()
    if run_mode not in schemas.RUN_MODES:
        raise schemas.ValidationError(f"run_mode must be one of {schemas.RUN_MODES}")
    replay = body.get("replay")
    observed_at = body.get("observed_at")
    if observed_at is not None:
        try:
            observed_at = int(observed_at)
        except (TypeError, ValueError) as error:
            raise schemas.ValidationError("observed_at must be an epoch integer") from error
    return run_mode, (None if replay in (None, "") else str(replay)), observed_at


# ----- WSGI --------------------------------------------------------------------------- #
def make_wsgi(app: Optional[ApiApp] = None):
    app = app or ApiApp()

    def application(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        body: Dict[str, Any] = {}
        if method == "POST":
            try:
                length = int(environ.get("CONTENT_LENGTH") or 0)
                raw = environ["wsgi.input"].read(length) if length else b""
                body = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except (ValueError, UnicodeDecodeError) as error:
                status, payload = 400, {"error": f"invalid JSON body: {error}"}
                return _respond(start_response, status, payload)
        binary = app.handle_binary(method, path)
        if binary is not None:
            status, data, content_type = binary
            start_response(f"{status} {'OK' if status == 200 else 'Service Unavailable'}",
                           [("Content-Type", content_type), ("Content-Length", str(len(data))), ("Cache-Control", "no-store")])
            return [data]
        status, payload = app.handle(method, path, body)
        return _respond(start_response, status, payload)
    return application


def _respond(start_response, status: int, payload: Dict[str, Any]):
    reasons = {200: "OK", 400: "Bad Request", 404: "Not Found", 501: "Not Implemented", 503: "Service Unavailable"}
    data = json.dumps(payload).encode("utf-8")
    start_response(f"{status} {reasons.get(status, 'OK')}", [("Content-Type", "application/json"),
                                                            ("Content-Length", str(len(data))),
                                                            ("Cache-Control", "no-store")])
    return [data]


def serve(host: str = "127.0.0.1", port: int = 8000, mode: Optional[str] = None) -> None:
    from wsgiref.simple_server import WSGIRequestHandler, make_server

    class QuietHandler(WSGIRequestHandler):
        def log_message(self, format, *args):                       # noqa: A002 - stdlib signature
            print(f"{self.address_string()} {format % args}")

    app = ApiApp(mode)
    with make_server(host, port, make_wsgi(app), handler_class=QuietHandler) as server:
        print(f"Accessible Transit API ({app.mode} mode) on http://{host}:{port}/api/health  "
              f"(engine mode: ACCESSIBLE_TRANSIT_API_MODE=engine; the web client proxies /api here)")
        server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Accessible Transit web API")
    parser.add_argument("--mode", choices=[MODE_FIXTURE, MODE_ENGINE], default=None,
                        help="fixture (golden responses, offline) or engine (the validated engines); "
                             "default from ACCESSIBLE_TRANSIT_API_MODE, else fixture")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port, args.mode)
