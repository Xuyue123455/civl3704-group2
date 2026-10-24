"""Production serving (deployment readiness): ONE origin for the built frontend and the API.

    python -m api.serve                       # engine mode, web/dist + /api on http://0.0.0.0:8080 (waitress when installed)
    python -m api.serve --port 8080 --warm-up # run health + a WALK / COMPARE / TRACK in-process before accepting traffic
    python -m api.serve --dist web/dist --threads 4

Layout the browser sees (no CORS, no mixed content, no second URL, no localhost assumption):

    https://<public-url>/            index.html and the hashed assets of `npm run build` (web/dist)
    https://<public-url>/api/...     the JSON API (api.app) - the same WSGI callable the dev proxy forwarded to

Serving: `waitress` (pure Python, Windows + Linux, threaded) when importable; otherwise the standard library's
single-threaded wsgiref server with a loud warning (local checks only).  The analytical endpoints are serialised by the
API's own process lock either way, so threads only overlap catalogue / static / readiness requests.  Static files are
read once into memory at startup (web/dist is ~1.2 MB), hashed assets get immutable caching, index.html never caches,
unknown non-API paths fall back to index.html (the client has one route).  Nothing analytical is touched; the TfNSW
key stays in the API process's environment and is never written anywhere.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = PROJECT_ROOT / "web" / "dist"
O2 = {"lat": -33.876602, "lon": 151.1867514}        # the validated warm-up origin (Desktop Gold seed)

mimetypes.add_type("application/x-protobuf", ".pbf")
mimetypes.add_type("application/javascript", ".js")


def load_static(dist: Path) -> Dict[str, Tuple[bytes, str, str]]:
    """path -> (body, content type, cache-control) for every file under web/dist (read once)."""
    files: Dict[str, Tuple[bytes, str, str]] = {}
    if not dist.exists():
        return files
    for path in dist.rglob("*"):
        if not path.is_file():
            continue
        rel = "/" + path.relative_to(dist).as_posix()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        cache = "public, max-age=31536000, immutable" if rel.startswith("/assets/") else "no-cache"
        files[rel] = (path.read_bytes(), content_type, cache)
    return files


def make_site(api_app, dist: Path = DEFAULT_DIST) -> Callable:
    """The same-origin WSGI callable: /api/* -> the API, everything else -> web/dist (SPA fallback to index.html)."""
    from .app import make_wsgi
    api = make_wsgi(api_app)
    static = load_static(dist)
    index = static.get("/index.html")

    def site(environ, start_response):
        path = environ.get("PATH_INFO", "/") or "/"
        if path == "/api" or path.startswith("/api/"):
            return api(environ, start_response)
        method = environ.get("REQUEST_METHOD", "GET").upper()
        if method not in ("GET", "HEAD"):
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain; charset=utf-8"), ("Allow", "GET, HEAD")])
            return [b"method not allowed"]
        entry = static.get(path) or (index if ("." not in path.rsplit("/", 1)[-1]) else None)
        if entry is None:
            start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"not found"]
        body, content_type, cache = entry
        start_response("200 OK", [("Content-Type", content_type), ("Content-Length", str(len(body))), ("Cache-Control", cache),
                                  ("X-Content-Type-Options", "nosniff")])
        return [b""] if method == "HEAD" else [body]
    return site


def warm_up(api_app) -> Dict[str, float]:
    """Load the engines and run one WALK / COMPARE / TRACK in-process so the first visitor never pays the cold start."""
    timings: Dict[str, float] = {}
    t = time.perf_counter(); status, ready = api_app.handle("GET", "/api/ready"); timings["ready"] = round(time.perf_counter() - t, 2)
    if status != 200 or ready.get("status") != "ready":
        timings["ready_status"] = ready.get("status", status)
        return timings
    if api_app.mode != "engine":
        return timings
    t = time.perf_counter(); api_app.handle("POST", "/api/walk", {"origin": O2, "stop_id": "203723", "profile": "standard"}); timings["walk"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    status, compare = api_app.handle("POST", "/api/compare", {"origin": O2, "route_short_name": "433", "direction_id": 1, "walking_profile": "standard",
                                                               "selected_trip_id": None, "selected_stop_id": "203723", "run_mode": "ANALYTICAL"})
    timings["compare"] = round(time.perf_counter() - t, 2)
    trip = (compare.get("selected_trip") or {}).get("trip_id") if status == 200 else None
    if trip:
        t = time.perf_counter()
        api_app.handle("POST", "/api/track", {"origin": O2, "trip_id": trip, "boarding_stop_id": "203723", "walking_profile": "standard", "run_mode": "ANALYTICAL"})
        timings["track"] = round(time.perf_counter() - t, 2)
    for endpoint in ("/api/services", "/api/stops", "/api/basemap", "/api/stairs", "/api/terrain", "/api/replays"):
        api_app.handle("GET", endpoint)
    return timings


def serve(host: str = "0.0.0.0", port: int = 8080, mode: Optional[str] = "engine", dist: Path = DEFAULT_DIST, threads: int = 4,
          do_warm_up: bool = False) -> None:
    from .app import ApiApp
    os.environ.setdefault("MPLBACKEND", "Agg")
    app = ApiApp(mode)
    if not (dist / "index.html").exists():
        print(f"WARNING: {dist} has no index.html - run `npm run build` in web/ first; only /api will answer", file=sys.stderr)
    if do_warm_up:
        print("warm-up:", warm_up(app), flush=True)
    site = make_site(app, dist)
    try:
        from waitress import serve as waitress_serve
    except ImportError:
        from wsgiref.simple_server import make_server
        print("WARNING: waitress is not installed; serving with the single-threaded wsgiref server (local checks only; "
              "pip install waitress for production)", file=sys.stderr)
        with make_server(host, port, site) as server:
            print(f"Accessible Transit ({app.mode} mode) on http://{host}:{port}/  (api: /api/health, /api/ready)", flush=True)
            server.serve_forever()
        return
    print(f"Accessible Transit ({app.mode} mode) on http://{host}:{port}/  via waitress, {threads} threads  (api: /api/health, /api/ready)", flush=True)
    waitress_serve(site, host=host, port=port, threads=threads, ident=None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--mode", choices=["engine", "fixture"], default=os.environ.get("ACCESSIBLE_TRANSIT_API_MODE", "engine"))
    parser.add_argument("--dist", default=str(DEFAULT_DIST))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("WEB_THREADS", "4")))
    parser.add_argument("--warm-up", action="store_true", help="load the engines and run one WALK / COMPARE / TRACK before serving")
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.mode, Path(args.dist), args.threads, args.warm_up)
    return 0


if __name__ == "__main__":
    sys.exit(main())
