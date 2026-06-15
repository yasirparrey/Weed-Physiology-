from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from .inversion import assess_forecast_inversion, assess_mesonet_inversion
from .mesonet import MesonetObservation, fetch_mesonet_inversion_data, get_nearest_mesonet_station
from .models import (
  ConditionCheck,
  DailySummary,
  HourlyAssessment,
  KansasLocation,
  KANSAS_LOCATIONS,
  SprayStatus,
)
from .regulations import DEFAULT_THRESHOLDS, RegulationThresholds
from .weather import (
  WeatherForecast,
  fetch_weather_forecast,
  get_daily_high_temperatures,
  get_sun_times,
  has_forecast_rain_within,
)


class DicambaSprayPredictor:
  """
  Predict legal dicamba spray windows for Kansas using EPA label criteria.

  Evaluates:
  - Temperature (95 °F hard stop; 85–95 °F 50% acreage rule)
  - Wind speed (3–10 mph)
  - Sunrise/sunset buffers (≥1 hr after sunrise, ≥2 hr before sunset)
  - Temperature inversions (forecast heuristics + Kansas Mesonet observations)
  - 48-hour rainfall forecast
  """

  def __init__(self, thresholds: RegulationThresholds | None = None):
    self.thresholds = thresholds or DEFAULT_THRESHOLDS

  def predict(
    self,
    location: KansasLocation | str,
    forecast_days: int = 3,
    use_mesonet: bool = True,
  ) -> tuple[WeatherForecast, list[HourlyAssessment], list[DailySummary]]:
    loc = KANSAS_LOCATIONS[location] if isinstance(location, str) else location
    forecast = fetch_weather_forecast(loc.latitude, loc.longitude, forecast_days=forecast_days)

    mesonet_obs: Optional[MesonetObservation] = None
    if use_mesonet:
      try:
        mesonet_obs = get_nearest_mesonet_station(
          loc.mesonet_station, loc.latitude, loc.longitude
        )
      except Exception:
        mesonet_obs = None

    daily_highs = get_daily_high_temperatures(forecast)
    hourly_assessments: list[HourlyAssessment] = []

    for _, row in forecast.hourly.iterrows():
      ts: datetime = row["time"].to_pydatetime()
      day = ts.date()
      sunrise, sunset = get_sun_times(loc.latitude, loc.longitude, day, forecast.timezone)
      next_day = day + timedelta(days=1)
      next_day_high = daily_highs.get(next_day)
      day_high = daily_highs.get(day)

      temp_status = self._temperature_status(day_high, next_day_high)

      assessment = self._assess_hour(
        timestamp=ts,
        temperature_f=_safe_float(row.get("temperature_2m")),
        wind_speed_mph=_safe_float(row.get("wind_speed_10m")),
        dew_point_f=_safe_float(row.get("dew_point_2m")),
        cloud_cover_pct=_safe_float(row.get("cloud_cover")),
        precipitation_mm=_safe_float(row.get("precipitation")),
        sunrise=sunrise,
        sunset=sunset,
        hourly_df=forecast.hourly,
        temp_status=temp_status,
        mesonet_obs=mesonet_obs,
      )
      hourly_assessments.append(assessment)

    daily_summaries = self._summarize_days(hourly_assessments, daily_highs)
    return forecast, hourly_assessments, daily_summaries

  def _temperature_status(
    self,
    day_high: Optional[float],
    next_day_high: Optional[float],
  ) -> SprayStatus:
    highs = [h for h in (day_high, next_day_high) if h is not None]
    if not highs:
      return SprayStatus.ALLOWED

    if any(h >= self.thresholds.temp_prohibited_f for h in highs):
      return SprayStatus.PROHIBITED

    if any(self.thresholds.temp_limited_min_f <= h < self.thresholds.temp_limited_max_f for h in highs):
      return SprayStatus.LIMITED

    return SprayStatus.ALLOWED

  def _assess_hour(
    self,
    *,
    timestamp: datetime,
    temperature_f: Optional[float],
    wind_speed_mph: Optional[float],
    dew_point_f: Optional[float],
    cloud_cover_pct: Optional[float],
    precipitation_mm: Optional[float],
    sunrise: datetime,
    sunset: datetime,
    hourly_df: pd.DataFrame,
    temp_status: SprayStatus,
    mesonet_obs: Optional[MesonetObservation],
  ) -> HourlyAssessment:
    checks: list[ConditionCheck] = []

    # Temperature-based day rule
    if temp_status == SprayStatus.PROHIBITED:
      checks.append(
        ConditionCheck(
          name="Temperature forecast",
          passed=False,
          value="≥ 95 °F on application or following day",
          requirement="No spray when forecast high ≥ 95 °F (day of or day after)",
          notes="EPA 95 °F hard stop",
        )
      )
    elif temp_status == SprayStatus.LIMITED:
      checks.append(
        ConditionCheck(
          name="Temperature forecast",
          passed=True,
          value="85–95 °F on application or following day",
          requirement="50% untreated acreage limit; wait 2 days for remainder",
          notes="EPA 50% rule applies for the day",
        )
      )
    else:
      checks.append(
        ConditionCheck(
          name="Temperature forecast",
          passed=True,
          value="< 85 °F forecast highs",
          requirement="No temperature-based acreage restriction",
        )
      )

    # Sunrise / sunset window
    earliest = sunrise + self.thresholds.sunrise_buffer
    latest = sunset - self.thresholds.sunset_buffer
    in_window = earliest <= timestamp <= latest
    checks.append(
      ConditionCheck(
        name="Sunrise/sunset window",
        passed=in_window,
        value=timestamp.strftime("%H:%M"),
        requirement=(
          f"Between {earliest.strftime('%H:%M')} and {latest.strftime('%H:%M')} "
          f"(≥1 hr after sunrise, ≥2 hr before sunset)"
        ),
      )
    )

    # Wind speed
    wind_ok = (
      wind_speed_mph is not None
      and self.thresholds.wind_min_mph <= wind_speed_mph <= self.thresholds.wind_max_mph
    )
    wind_value = f"{wind_speed_mph:.1f} mph" if wind_speed_mph is not None else "N/A"
    checks.append(
      ConditionCheck(
        name="Wind speed",
        passed=wind_ok,
        value=wind_value,
        requirement=f"{self.thresholds.wind_min_mph:.0f}–{self.thresholds.wind_max_mph:.0f} mph",
        notes="Too calm (<3 mph) risks inversions; too strong (>10 mph) increases drift",
      )
    )

    # 48-hour rain prohibition
    rain_expected, rain_total = has_forecast_rain_within(
      hourly_df, timestamp, self.thresholds.rain_lookahead_hours
    )
    checks.append(
      ConditionCheck(
        name="48-hour rainfall",
        passed=not rain_expected,
        value=f"{rain_total:.2f} in forecast" if rain_expected else "No measurable rain forecast",
        requirement="No application within 48 hours of forecast rainfall",
      )
    )

    # Inversion — forecast heuristic
    forecast_inv = assess_forecast_inversion(
      timestamp=timestamp,
      temperature_f=temperature_f,
      dew_point_f=dew_point_f,
      wind_speed_mph=wind_speed_mph,
      cloud_cover_pct=cloud_cover_pct,
      sunrise=sunrise,
      sunset=sunset,
      thresholds=self.thresholds,
    )

    # Inversion — live Mesonet (used for current/near-current hours)
    mesonet_inv = assess_mesonet_inversion(mesonet_obs, self.thresholds)
    use_mesonet_for_hour = (
      mesonet_obs is not None
      and abs((timestamp - mesonet_obs.observed_at).total_seconds()) < 3 * 3600
    )
    inversion_detected = (
      mesonet_inv.detected if use_mesonet_for_hour else forecast_inv.detected
    )
    inversion_source = mesonet_inv.source if use_mesonet_for_hour else forecast_inv.source
    inv_reasons = (mesonet_inv.reasons or []) if use_mesonet_for_hour else (forecast_inv.reasons or [])

    checks.append(
      ConditionCheck(
        name="Temperature inversion",
        passed=not inversion_detected,
        value="Inversion detected" if inversion_detected else "No inversion indicated",
        requirement="Do not spray during a temperature inversion",
        notes="; ".join(inv_reasons) if inv_reasons else "",
      )
    )

    # Overall status
    hard_fail = any(
      not c.passed
      for c in checks
      if c.name
      in ("Sunrise/sunset window", "Wind speed", "48-hour rainfall", "Temperature inversion")
    ) or temp_status == SprayStatus.PROHIBITED

    if hard_fail:
      status = SprayStatus.PROHIBITED
    elif temp_status == SprayStatus.LIMITED:
      status = SprayStatus.LIMITED
    else:
      status = SprayStatus.ALLOWED

    return HourlyAssessment(
      timestamp=timestamp,
      status=status,
      temperature_f=temperature_f,
      wind_speed_mph=wind_speed_mph,
      dew_point_f=dew_point_f,
      cloud_cover_pct=cloud_cover_pct,
      precipitation_mm=precipitation_mm,
      sunrise=sunrise,
      sunset=sunset,
      checks=checks,
      inversion_detected=inversion_detected,
      inversion_source=inversion_source,
      mesonet_station=mesonet_obs.station if mesonet_obs else None,
      mesonet_inversion_strength_f=mesonet_obs.inversion_strength_f if mesonet_obs else None,
    )

  def _summarize_days(
    self,
    assessments: list[HourlyAssessment],
    daily_highs: dict[date, float],
  ) -> list[DailySummary]:
    by_day: dict[date, list[HourlyAssessment]] = {}
    for a in assessments:
      by_day.setdefault(a.timestamp.date(), []).append(a)

    summaries: list[DailySummary] = []
    sorted_days = sorted(by_day.keys())

    for i, day in enumerate(sorted_days):
      day_assessments = by_day[day]
      next_day = sorted_days[i + 1] if i + 1 < len(sorted_days) else day + timedelta(days=1)
      next_high = daily_highs.get(next_day)

      temp_status = self._temperature_status(daily_highs.get(day), next_high)

      sprayable = [a for a in day_assessments if a.status == SprayStatus.ALLOWED]
      limited = [a for a in day_assessments if a.status == SprayStatus.LIMITED]
      prohibited = [a for a in day_assessments if a.status == SprayStatus.PROHIBITED]

      best_windows = _find_contiguous_windows(sprayable + limited)

      notes: list[str] = []
      if temp_status == SprayStatus.PROHIBITED:
        notes.append("Do not spray — forecast high ≥ 95 °F on this day or the next.")
      elif temp_status == SprayStatus.LIMITED:
        notes.append(
          "50% acreage rule: only half of untreated DT acres in the county may be sprayed; "
          "wait at least 2 days before treating the remainder."
        )

      summaries.append(
        DailySummary(
          date=datetime.combine(day, datetime.min.time()),
          forecast_high_f=daily_highs.get(day),
          next_day_high_f=next_high,
          temperature_status=temp_status,
          sprayable_hours=len(sprayable),
          limited_hours=len(limited),
          prohibited_hours=len(prohibited),
          total_hours=len(day_assessments),
          best_windows=best_windows,
          notes=notes,
        )
      )

    return summaries


def predict_spray_windows(
  location: str = "Manhattan",
  forecast_days: int = 3,
) -> tuple[WeatherForecast, list[HourlyAssessment], list[DailySummary]]:
  """Convenience wrapper for quick predictions."""
  predictor = DicambaSprayPredictor()
  return predictor.predict(location, forecast_days=forecast_days)


def _find_contiguous_windows(
  assessments: list[HourlyAssessment],
) -> list[tuple[datetime, datetime]]:
  if not assessments:
    return []

  sorted_hours = sorted(assessments, key=lambda a: a.timestamp)
  windows: list[tuple[datetime, datetime]] = []
  start = sorted_hours[0].timestamp
  prev = start

  for assessment in sorted_hours[1:]:
    if (assessment.timestamp - prev) <= timedelta(hours=1, minutes=5):
      prev = assessment.timestamp
      continue
    windows.append((start, prev))
    start = assessment.timestamp
    prev = assessment.timestamp

  windows.append((start, prev))
  return windows


def _safe_float(value) -> Optional[float]:
  if value is None or (isinstance(value, float) and pd.isna(value)):
    return None
  return float(value)
