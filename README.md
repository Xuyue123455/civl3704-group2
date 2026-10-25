# Accessible Transit (Glebe) - Render deployment bundle

Built from source commit `be0de68907d88eaa6abbd2f7160445a437766071` by `scripts/build_render_stage.py` (dev repo) out of the verified staging bundle.
This directory is a self-contained production runtime: the JSON API + the built frontend served from ONE process /
origin. It contains no secret, no raw DEM, no raw GTFS, no node_modules, no development results.

    build:   pip install -r requirements.txt
    start:   python -m api.serve --port $PORT --warm-up
    health:  GET /api/ready   (200 ready / 503 not ready; never a secret)
    alive:   GET /api/health

Environment variables: `PORT` (Render), `TFNSW_API_KEY` (LIVE only; set in the Render dashboard, `sync: false` in
render.yaml), `ACCESSIBLE_TRANSIT_API_MODE=engine`, `WEB_THREADS=4`, `MPLBACKEND=Agg` (all in render.yaml).

Deployment steps: RENDER_DEPLOYMENT.md. Contents and hashes: DEPLOYMENT_MANIFEST.json.
