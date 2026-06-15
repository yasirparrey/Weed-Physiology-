from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SprayStatus(str, Enum):
  """Overall spray suitability for a given hour."""

  ALLOWED = "allowed"
  LIMITED = "limited"  # 50% acreage rule applies (85–95 °F forecast)
  PROHIBITED = "prohibited"


@dataclass
class ConditionCheck:
  """Result of evaluating a single EPA label criterion."""

  name: str
  passed: bool
  value: str
  requirement: str
  notes: str = ""


@dataclass
class HourlyAssessment:
  """Spray suitability assessment for one forecast hour."""

  timestamp: datetime
  status: SprayStatus
  temperature_f: Optional[float]
  wind_speed_mph: Optional[float]
  dew_point_f: Optional[float]
  cloud_cover_pct: Optional[float]
  precipitation_mm: Optional[float]
  sunrise: datetime
  sunset: datetime
  checks: list[ConditionCheck] = field(default_factory=list)
  inversion_detected: bool = False
  inversion_source: str = "forecast"
  mesonet_station: Optional[str] = None
  mesonet_inversion_strength_f: Optional[float] = None

  @property
  def is_sprayable(self) -> bool:
    return self.status in (SprayStatus.ALLOWED, SprayStatus.LIMITED)

  @property
  def failed_checks(self) -> list[ConditionCheck]:
    return [c for c in self.checks if not c.passed]


@dataclass
class DailySummary:
  """Day-level summary including temperature-based acreage rules."""

  date: datetime
  forecast_high_f: Optional[float]
  next_day_high_f: Optional[float]
  temperature_status: SprayStatus
  sprayable_hours: int
  limited_hours: int
  prohibited_hours: int
  total_hours: int
  best_windows: list[tuple[datetime, datetime]] = field(default_factory=list)
  notes: list[str] = field(default_factory=list)


@dataclass
class KansasLocation:
  name: str
  latitude: float
  longitude: float
  mesonet_station: Optional[str] = None


# Common Kansas field locations with nearest Mesonet inversion stations.
KANSAS_LOCATIONS: dict[str, KansasLocation] = {
  "Manhattan": KansasLocation("Manhattan", 39.1836, -96.5717, "Manhattan"),
  "Wichita": KansasLocation("Wichita", 37.6872, -97.3301, "Viola"),
  "Hutchinson": KansasLocation("Hutchinson", 38.0608, -97.9298, "Hutchinson 10SW"),
  "Dodge City": KansasLocation("Dodge City", 37.7528, -100.0171, "Garden City"),
  "Garden City": KansasLocation("Garden City", 37.9717, -100.8727, "Garden City"),
  "Parsons": KansasLocation("Parsons", 37.3403, -95.2611, "Parsons"),
  "Colby": KansasLocation("Colby", 39.3958, -101.0524, "Colby"),
  "Hays": KansasLocation("Hays", 38.8792, -99.3268, "Hays"),
  "Olathe": KansasLocation("Olathe", 38.8814, -94.8191, "Olathe"),
  "Salina": KansasLocation("Salina", 38.8403, -97.6114, "Gypsum"),
  "Emporia": KansasLocation("Emporia", 38.4039, -96.1817, "Elmdale 1SE"),
  "Liberal": KansasLocation("Liberal", 37.0431, -100.9210, "Richfield"),
}
