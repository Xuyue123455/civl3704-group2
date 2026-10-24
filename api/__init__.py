"""Accessible Transit — thin Python web API (Phase W1 scaffold).

The API exposes the validated desktop engines (WALK / COMPARE / TRACK) as JSON + GeoJSON.  It contains no
analytical formula: every number comes from ``scripts/unified_ui`` (the Desktop Gold code) or from a golden
fixture recorded from it.  See docs/WEB_MIGRATION_CONTRACT.md.
"""

API_VERSION = "0.1.0"
DESKTOP_GOLD = "901232d39fa0f87eb479b4fcf7d0c3d74f2fb870"
FUNCTIONAL_AUTHORITY = "c86afab4b92ece45b0ed109e72b7179bfc1bff06"
