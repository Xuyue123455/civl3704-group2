"""Week 4 foundation package for the Accessible Transit Stop Selection project.

Modules
-------
config        : paths and provisional Week 4 parameters
boundary      : official Glebe suburb polygon, the candidate-stop extent
gtfs_loader   : static GTFS -> candidate transit stops      (Stage 2)
osm_network   : OpenStreetMap -> walkable graph             (Stage 3)
snapping      : GTFS stops / user origin -> graph nodes     (Stage 4)
routing       : distance-only shortest paths                (Stage 5)

Terrain-aware cost modelling (DEM, slope, Minetti, Tobler) and
GTFS-Realtime are intentionally NOT part of this package yet.
They are scheduled for later weeks.
"""

__all__ = [
    "config",
    "boundary",
    "gtfs_loader",
    "osm_network",
    "snapping",
    "routing",
]
