# Render deployment (Blueprint, native Python) - step by step

This directory is the complete production bundle: `api/` (JSON API + the same-origin server), `src/` + `scripts/` (the
engine modules the request path imports), `web/dist` (the built frontend), `data/` + `results/service_intent` +
`tests/fixtures/week11_l1` (runtime assets, planned-shape cache, terrain display, replay recording), `requirements.txt`,
`.python-version`, `render.yaml`, `DEPLOYMENT_MANIFEST.json`. It runs without the development repository, without
`.env`, and writes nothing at run time. **No secret is in this directory; the TfNSW key is entered only in the Render
dashboard.**

Service contract (from `render.yaml`):

| | |
|---|---|
| runtime | native Python (`.python-version` = 3.12 → Render installs the latest 3.12.x) |
| build command | `pip install -r requirements.txt` |
| start command | `python -m api.serve --port $PORT --warm-up` (binds 0.0.0.0, uses Render's `PORT`, serves `/` and `/api` from one process, warms the engines before accepting traffic) |
| health check | `GET /api/ready` → 200 only when every required runtime asset is present and the engines loaded; 503 otherwise |
| region | `singapore` - Render has no Australian region; Singapore is the closest (Oregon / Ohio / Virginia / Frankfurt are the others). Latency to Sydney is not measured here; expect noticeably more than a local server |
| plan | `free` for the first trial (see "Memory / CPU risk" below) |
| auto deploy | `autoDeployTrigger: off` - deploy by hand from the dashboard |
| env vars | `TFNSW_API_KEY` (`sync: false`: asked for in the dashboard, LIVE only), `ACCESSIBLE_TRANSIT_API_MODE=engine`, `WEB_THREADS=4`, `MPLBACKEND=Agg`; `PORT` comes from Render |

## 1. GitHub (private deploy repository)

On Windows, PowerShell, inside this directory (`C:\Users\<you>\Desktop\accessible-transit-render-stage`):

```powershell
git init -b main
git add -A
git status                              # sanity: no .env, no *.tif, no node_modules (the .gitignore keeps local noise out)
git commit -m "Accessible Transit Render deployment bundle (source commit <paste from DEPLOYMENT_MANIFEST.json>)"
```

Create the repository on GitHub: **New repository → name `accessible-transit-render-deploy` → Private → no README /
.gitignore / licence (the bundle has its own)**. Then:

```powershell
git remote add origin https://github.com/<your-github-user>/accessible-transit-render-deploy.git
git push -u origin main
```

Size: ≈ 49 MB, 158 files; the largest file is `data/gtfs/trips.txt` (31.7 MB), under GitHub's 100 MB per-file limit
(GitHub warns above 50 MB; nothing here reaches it). No Git LFS needed.

## 2. Render

1. Sign in at render.com (GitHub sign-in is simplest; grant access to the private repo when asked).
2. Dashboard → **New → Blueprint**.
3. Connect the private repository `accessible-transit-render-deploy`, branch `main`.
4. Render reads `render.yaml` and shows one web service `accessible-transit-glebe` (runtime Python, region Singapore,
   plan free). Keep the name or edit it (it becomes `<name>.onrender.com`).
5. It prompts for **`TFNSW_API_KEY`** (declared `sync: false`): paste the TfNSW Open Data Hub key. This is the only
   place the value ever goes. (If the prompt is skipped, set it later under the service → Environment.)
6. Region: Singapore is preselected by the Blueprint; it cannot be changed after creation.
7. Plan: free for the first trial; see the risk note before a marked demo.
8. **Apply / Deploy**. The build installs `requirements.txt` (numpy / pandas / scipy / shapely / pyproj / geopandas /
   osmnx 2.1.1 / networkx / requests / protobuf / gtfs-realtime-bindings / python-dotenv / waitress; binary wheels,
   a few minutes).
9. Wait for the first deploy to turn **Live**: the start command warms the engines before listening (≈ 8–18 s on a
   1 CPU machine; on the free 0.1 CPU plan this can take **a few minutes** - Render keeps probing `/api/ready` until
   it answers 200).
10. The service page shows the HTTPS URL `https://accessible-transit-glebe.onrender.com` (or your name).

## 3. Validation

11. Open `https://<name>.onrender.com/api/ready`: expect `"status": "ready"`, `"required_missing": []`,
    `"engines": {"loaded": true}`, `"shape_cache": {"shape_ids_needed": 39, "cached": 39}`,
    `"realtime": {"api_key_configured": true, "replays_available": ["l1_2026-10-17_0917"]}` (the key's value never appears).
12. On a phone (mobile data, not the home Wi-Fi) open `https://<name>.onrender.com/`.
13. Test: Explore (tap origin, tap a stop, three routes) · Standard ↔ Step-free · stairs (segments listed, ↑/↓ badges)
    · Terrain (Layers → Terrain: tint + shading + 10 m contours + legend) · Decide (Compare nearby stops) · objective
    selector (Shortest / Gentlest / Fastest) · Service → Direction → Trip selector · Go (Track this trip) · LIVE (status
    line "Live · TfNSW feeds"; Technical details → feed status `ok`) · Replay (data-source menu → Replay → "Replay ·
    not live"; Explore → Decide → Go on the 2026-10-17 L1 recording).
14. Record the URL, the `source_commit` from `DEPLOYMENT_MANIFEST.json`, the region / plan and the `/api/ready` output.

## 4. Memory / CPU risk on the free plan (read before a marked demo)

- Measured peak of this app: **≈ 368 MB RSS** (Linux VM, 1 CPU) after warm-up. Render free = **0.1 CPU / 512 MB RAM**.
  The margin is ≈ 140 MB: enough for a trial, not a guarantee. Watch the service's Memory graph after the first
  COMPARE / TRACK; an **Out of memory** restart means the free plan cannot hold the engines.
- Free services **spin down after 15 minutes without traffic** and take **about a minute plus the warm-up** to come
  back - on 0.1 CPU the warm-up itself is slow. Before a demo, open `/api/ready` several minutes early and keep the tab
  alive, or move off the free plan.
- If OOM / timeouts happen: **do not change the analytics to fit the plan.** Use a bigger compute plan (`1c-2g`:
  1 CPU / 2 GB, or at least `0.5c-512mb` for CPU) or another provider from the requirements list in
  `docs/WEB_DEPLOYMENT_READINESS.md` of the dev repo. The bundle and `render.yaml` stay the same; only `plan` changes.
- Free instance hours: 750 per month per workspace; a spun-down service does not consume them.

## 5. Redeploy / rollback

Rebuild the bundle in the dev repo (`scripts/build_railway_stage.py` → `scripts/build_render_stage.py`), copy the new
contents over this directory, `git add -A && git commit && git push`, then **Manual Deploy → Deploy latest commit** in
Render (auto deploy is off on purpose). Rollback = Manual Deploy of a previous commit / deploy from the Events list.

## 6. What this bundle deliberately does not contain

`.env` / any key, `config.json`, the 1 m DEM tiles, `stop_times.txt`, `stops.txt`, the GTFS zip, `node_modules`,
screenshots, development results, tests other than the replay recording, desktop-only scripts, the dev-repo docs.
Everything the request path opens at run time is here (traced, not guessed); the fresh-venv Linux test in the dev repo
(`scripts/validate_stage_isolated.py --python <venv python>`) proves it runs with nothing else.
