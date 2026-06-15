from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

MESONET_INVERSION_URL = "https://mesonet.k-state.edu/agriculture/inversion/"


@dataclass
class MesonetObservation:
  station: str
  temp_2m_f: float
  temp_10m_f: float
  inversion_strength_f: float
  wind_speed_mph: float
  wind_direction: str
  observed_at: datetime
  latitude: float
  longitude: float

  @property
  def has_inversion(self) -> bool:
    return self.inversion_strength_f >= 1.0


def _parse_mesonet_timestamp(raw: str) -> datetime:
  # Example: "2026-06-14 21:15:00 CST" — treat as America/Chicago local time.
  from zoneinfo import ZoneInfo

  cleaned = raw.replace(" CST", "").replace(" CDT", "").strip()
  naive = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
  return naive.replace(tzinfo=ZoneInfo("America/Chicago"))


def fetch_mesonet_inversion_data(timeout: int = 20) -> dict[str, MesonetObservation]:
  """Fetch live inversion observations embedded in the Kansas Mesonet page."""
  import json

  response = requests.get(MESONET_INVERSION_URL, timeout=timeout)
  response.raise_for_status()

  match = re.search(r"var stationData = (\{.*?\});\s*\n", response.text, re.DOTALL)
  if not match:
    raise ValueError("Could not parse Kansas Mesonet inversion station data.")

  station_data = json.loads(match.group(1))

  observations: dict[str, MesonetObservation] = {}
  for station, values in station_data.items():
    temp_2m = float(values["temp2m"])
    temp_10m = float(values["temp10m"])
    inv = float(values.get("inv", temp_10m - temp_2m))
    observations[station] = MesonetObservation(
      station=station,
      temp_2m_f=temp_2m,
      temp_10m_f=temp_10m,
      inversion_strength_f=inv,
      wind_speed_mph=float(values.get("wind_spd2m", 0)),
      wind_direction=str(values.get("wind_comp2m", "")),
      observed_at=_parse_mesonet_timestamp(values["timestamp"]),
      latitude=float(values["lat_corr"]),
      longitude=float(values["lon_corr"]),
    )
  return observations


def get_nearest_mesonet_station(
  station_name: Optional[str],
  latitude: float,
  longitude: float,
  observations: Optional[dict[str, MesonetObservation]] = None,
) -> Optional[MesonetObservation]:
  """Return a Mesonet observation for the named station or the nearest site."""
  data = observations or fetch_mesonet_inversion_data()

  if station_name and station_name in data:
    return data[station_name]

  if not data:
    return None

  def distance_sq(obs: MesonetObservation) -> float:
    return (obs.latitude - latitude) ** 2 + (obs.longitude - longitude) ** 2

  return min(data.values(), key=distance_sq)
