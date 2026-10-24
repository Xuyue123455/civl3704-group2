# Week 11 L1 (Inner West Light Rail) realtime fixtures

## Current generation (used by the tests)

Real TfNSW GTFS-Realtime v2 snapshots captured ONCE on the Windows desktop on
**2026-10-17 09:17:01 AEDT** with `scripts/week11_capture_light_rail.py`, right after
the static GTFS refresh of 2026-10-17 (bundle sha256 `b5d36487404b9f3f…`), so the
realtime trip_ids and the static tables are the SAME timetable generation:

* `l1_trip_updates.pb` — `https://api.transport.nsw.gov.au/v2/gtfs/realtime/lightrail/innerwest`
  (22 entities / 22 Trip Updates, 506 stop_time_updates with absolute times, realtime
  route_id `IWLR-191`; 22/22 trip_ids in `trips.txt`, 0 inactive on the capture date)
* `l1_vehicle_positions.pb` — `https://api.transport.nsw.gov.au/v2/gtfs/vehiclepos/lightrail/innerwest`
  (7 Vehicle Positions, 7/7 with trip_id, 7/7 in static, report ages 6–22 s at capture)
* `bus_433_replay_golden.json` — the bus non-regression golden: Walk result, Compare scene,
  manual selection and Track result produced by the **cb2c5a8 code** (the pre-Week-11 files
  swapped in) on the Track suite's synthetic 433 snapshot (service date 2026-10-23) against
  the 2026-10-17 static bundle.

The `.json` sidecars carry the capture epoch (`captured_at`), the source URL, the byte
count and sha256 — never a key. The protobuf bytes are unmodified. These are RECORDED
fixtures for offline regression (REPLAY); they are not current or live data and the
application never labels them live.

## Historical (not used by the tests)

`historical_2026-10-17_0102/` — the first capture (2026-10-17 01:02 AEDT, 8 Trip Updates /
6 Vehicle Positions) made against the OLD 2026-08-28 bundle whose L1 calendar had ended on
2026-10-06. Its trip_ids (`48887-53xx`) belong to the previous timetable generation and do
not exist in the refreshed `trips.txt`; kept only as the feasibility-spike evidence.
