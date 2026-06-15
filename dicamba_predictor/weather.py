from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from astral import Observer
from astral.sun import sun

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
KANSAS_TIMEZONE = "America/Chicago"


@dataclass
class WeatherForecast:
  latitude: float
  longitude: float
  timezone: str
  hourly: pd.DataFrame
  daily: pd.DataFrame


def _build_hourly_dataframe(payload: dict, timezone: str) -> pd.DataFrame:
  hourly = payload["hourly"]
  df = pd.DataFrame(hourly)
  df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(timezone)
  return df


def _build_daily_dataframe(payload: dict, timezone: str) -> pd.DataFrame:
  daily = payload["daily"]
  df = pd.DataFrame(daily)
  df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(timezone)
  return df


def fetch_weather_forecast(
  latitude: float,
  longitude: float,
  forecast_days: int = 7,
  timezone: str = KANSAS_TIMEZONE,
) -> WeatherForecast:
  """Fetch hourly and daily weather from Open-Meteo (NWS-backed models)."""
  params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": ",".join(
      [
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "dew_point_2m",
        "cloud_cover",
        "precipitation",
        "precipitation_probability",
        "relative_humidity_2m",
      ]
    ),
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": timezone,
    "forecast_days": forecast_days,
  }
  response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
  response.raise_for_status()
  payload = response.json()

  return WeatherForecast(
    latitude=latitude,
    longitude=longitude,
    timezone=timezone,
    hourly=_build_hourly_dataframe(payload, timezone),
    daily=_build_daily_dataframe(payload, timezone),
  )


def get_sun_times(
  latitude: float,
  longitude: float,
  day: date,
  timezone: str = KANSAS_TIMEZONE,
) -> tuple[datetime, datetime]:
  """Return localized sunrise and sunset for a Kansas location."""
  observer = Observer(latitude=latitude, longitude=longitude)
  tz = ZoneInfo(timezone)
  times = sun(observer, date=day, tzinfo=tz)
  return times["sunrise"], times["sunset"]


def get_daily_high_temperatures(
  forecast: WeatherForecast,
) -> dict[date, float]:
  highs: dict[date, float] = {}
  for _, row in forecast.daily.iterrows():
    highs[row["time"].date()] = float(row["temperature_2m_max"])
  return highs


def has_forecast_rain_within(
  hourly: pd.DataFrame,
  start: datetime,
  hours: int,
) -> tuple[bool, float]:
  """Check if measurable rain is forecast within the next N hours."""
  end = start + timedelta(hours=hours)
  window = hourly[(hourly["time"] >= start) & (hourly["time"] < end)]
  if window.empty:
    return False, 0.0

  precip = window["precipitation"].fillna(0.0)
  total_in = float(precip.sum())
  # Treat any measurable precipitation in the 48-hour window as a prohibition.
  return total_in > 0.0, total_in
