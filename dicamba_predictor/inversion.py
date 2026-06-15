from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .mesonet import MesonetObservation
from .regulations import RegulationThresholds


@dataclass
class InversionAssessment:
  detected: bool
  source: str
  strength_f: Optional[float] = None
  reasons: list[str] | None = None


def assess_forecast_inversion(
  *,
  timestamp: datetime,
  temperature_f: Optional[float],
  dew_point_f: Optional[float],
  wind_speed_mph: Optional[float],
  cloud_cover_pct: Optional[float],
  sunrise: datetime,
  sunset: datetime,
  thresholds: RegulationThresholds,
) -> InversionAssessment:
  """
  Estimate temperature inversion risk from forecast meteorology.

  EPA labels prohibit spraying during inversions. Extension guidance and
  Kansas Mesonet research indicate inversions are most common with:
  - calm winds (< 3 mph)
  - clear skies (low cloud cover)
  - near-surface moisture (dew point close to air temperature)
  - early morning and late evening hours
  """
  reasons: list[str] = []

  if wind_speed_mph is not None and wind_speed_mph < thresholds.wind_min_mph:
    reasons.append(f"Calm wind ({wind_speed_mph:.1f} mph) — inversion and low-wind risk")

  if temperature_f is not None and dew_point_f is not None:
    spread = temperature_f - dew_point_f
    if spread <= 4.0:
      reasons.append(f"Small temperature–dew point spread ({spread:.1f} °F)")

  if cloud_cover_pct is not None and cloud_cover_pct <= 25:
    dawn_start = sunrise - timedelta(hours=2)
    dawn_end = sunrise + thresholds.sunrise_buffer
    dusk_start = sunset - thresholds.sunset_buffer
    dusk_end = sunset + timedelta(hours=1)

    in_dawn = dawn_start <= timestamp <= dawn_end
    in_dusk = dusk_start <= timestamp <= dusk_end
    if in_dawn or in_dusk:
      reasons.append("Clear skies during dawn/dusk inversion-prone period")

  detected = len(reasons) >= 2 or (
    len(reasons) == 1 and wind_speed_mph is not None and wind_speed_mph < thresholds.wind_min_mph
  )

  return InversionAssessment(
    detected=detected,
    source="forecast",
    strength_f=None,
    reasons=reasons,
  )


def assess_mesonet_inversion(
  observation: Optional[MesonetObservation],
  thresholds: RegulationThresholds,
) -> InversionAssessment:
  """Use Kansas Mesonet 2 m vs 10 m temperature difference for live inversion checks."""
  if observation is None:
    return InversionAssessment(detected=False, source="mesonet_unavailable")

  strength = observation.inversion_strength_f
  detected = strength >= thresholds.mesonet_inversion_threshold_f
  label = "none"
  if strength >= 5:
    label = "strong"
  elif strength >= 1:
    label = "mild"

  reasons = []
  if detected:
    reasons.append(
      f"Mesonet reports {label} inversion ({strength:.1f} °F difference at 10 m vs 2 m)"
    )

  return InversionAssessment(
    detected=detected,
    source="mesonet",
    strength_f=strength,
    reasons=reasons,
  )
