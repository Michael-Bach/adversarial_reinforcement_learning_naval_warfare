"""
Coordinate transforms between CMANO (lat/lon/compass) and NavalEnv (XY/math-radians).

Local frame convention (matches NavalEnv):
  origin  = scenario centre (lat0, lon0)
  X axis  = East  (metres positive)
  Y axis  = North (metres positive)
  course  = mathematical radians: 0 = East, CCW positive
  speed   = metres per minute

CMANO convention:
  position = decimal degrees latitude / longitude
  heading  = compass bearing: 0 = North, CW positive, degrees
  speed    = knots
"""

from __future__ import annotations

import math

# WGS-84 approximation: metres per degree of latitude (effectively constant)
_M_PER_DEG_LAT = 111_319.9


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """
    Convert (lat, lon) → local (x_m, y_m) relative to origin (lat0, lon0).

    Uses equirectangular projection; accurate to ~0.1 % within a 200×200 km box.
    """
    cos_lat0 = math.cos(math.radians(lat0))
    x_m = (lon - lon0) * cos_lat0 * _M_PER_DEG_LAT
    y_m = (lat - lat0) * _M_PER_DEG_LAT
    return x_m, y_m


def xy_to_latlon(x_m: float, y_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert local (x_m, y_m) → (lat, lon)."""
    cos_lat0 = math.cos(math.radians(lat0))
    lon = lon0 + x_m / (cos_lat0 * _M_PER_DEG_LAT)
    lat = lat0 + y_m / _M_PER_DEG_LAT
    return lat, lon


def compass_to_math_rad(heading_deg: float) -> float:
    """Compass bearing (0=North, CW, degrees) → math convention (0=East, CCW, radians)."""
    return math.radians(90.0 - heading_deg)


def math_rad_to_compass(course_rad: float) -> float:
    """Math convention (0=East, CCW, radians) → compass bearing (0=North, CW, degrees)."""
    return (90.0 - math.degrees(course_rad)) % 360.0


def knots_to_mpm(kts: float) -> float:
    """Knots → metres per minute."""
    return kts * 1852.0 / 60.0


def mpm_to_knots(mpm: float) -> float:
    """Metres per minute → knots."""
    return mpm * 60.0 / 1852.0
