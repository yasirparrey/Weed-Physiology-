"""
EPA over-the-top (OTT) dicamba label requirements (2026–2027 federal labels).

Sources:
- https://www.epa.gov/ingredients-used-pesticide-products/registration-dicamba-use-dicamba-tolerant-crops
- Kansas follows federal labels (no additional state temperature cap as of 2026).

Kansas applicators should still verify the current product label, state registration,
and Kansas Mesonet inversion observations before spraying.
"""

from dataclasses import dataclass
from datetime import timedelta

# Wind speed (mph) — EPA requires 3–10 mph at application time.
WIND_SPEED_MIN_MPH = 3.0
WIND_SPEED_MAX_MPH = 10.0

# Temperature (°F) — forecast high on application day OR the following day.
TEMP_PROHIBITED_F = 95.0
TEMP_LIMITED_MIN_F = 85.0
TEMP_LIMITED_MAX_F = 95.0  # exclusive upper bound for the 50% rule

# Sunrise/sunset timing — applications prohibited within these windows.
SUNRISE_BUFFER = timedelta(hours=1)
SUNSET_BUFFER = timedelta(hours=2)

# Rain — no application within 48 hours of forecast rainfall.
RAIN_LOOKAHEAD_HOURS = 48

# Kansas Mesonet inversion strength (°F difference between 10 m and 2 m air temp).
# Mesonet scale: <1 none, 1–5 mild, >5 strong. Any measurable inversion is prohibited.
MESONET_INVERSION_THRESHOLD_F = 1.0


@dataclass(frozen=True)
class RegulationThresholds:
    """Configurable thresholds mirroring EPA federal label defaults."""

    wind_min_mph: float = WIND_SPEED_MIN_MPH
    wind_max_mph: float = WIND_SPEED_MAX_MPH
    temp_prohibited_f: float = TEMP_PROHIBITED_F
    temp_limited_min_f: float = TEMP_LIMITED_MIN_F
    temp_limited_max_f: float = TEMP_LIMITED_MAX_F
    sunrise_buffer: timedelta = SUNRISE_BUFFER
    sunset_buffer: timedelta = SUNSET_BUFFER
    rain_lookahead_hours: int = RAIN_LOOKAHEAD_HOURS
    mesonet_inversion_threshold_f: float = MESONET_INVERSION_THRESHOLD_F


DEFAULT_THRESHOLDS = RegulationThresholds()
