#!/usr/bin/env python3
"""Streamlit web app for Kansas dicamba spray window prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dicamba_predictor import DicambaSprayPredictor
from dicamba_predictor.models import KANSAS_LOCATIONS, SprayStatus

st.set_page_config(
  page_title="Kansas Dicamba Spray Predictor",
  page_icon="🌾",
  layout="wide",
)

st.title("Kansas Dicamba Spray Window Predictor")
st.caption(
  "EPA 2026–2027 over-the-top dicamba label criteria • "
  "Weather via Open-Meteo • Inversions via Kansas Mesonet"
)

with st.sidebar:
  location = st.selectbox("Kansas location", sorted(KANSAS_LOCATIONS.keys()), index=0)
  forecast_days = st.slider("Forecast days", 1, 7, 3)
  use_mesonet = st.checkbox("Use Kansas Mesonet inversion data", value=True)
  run = st.button("Run prediction", type="primary")

st.markdown(
  """
  ### EPA spray conditions evaluated
  | Criterion | Requirement |
  |-----------|-------------|
  | **Temperature** | No spray if forecast high ≥ **95°F** on application day or the next day |
  | **Temperature (50% rule)** | If forecast is **85–95°F**, only 50% of untreated acres per county; wait 2 days for remainder |
  | **Wind speed** | **3–10 mph** at application time |
  | **Sunrise / sunset** | No spray within **1 hour after sunrise** or **2 hours before sunset** |
  | **Temperature inversion** | No spray during inversions (Mesonet 2 m vs 10 m temps + forecast indicators) |
  | **Rainfall** | No spray within **48 hours** of forecast rainfall |
  """
)

if run:
  with st.spinner("Fetching weather and inversion data..."):
    try:
      predictor = DicambaSprayPredictor()
      forecast, hourly, daily = predictor.predict(
        location,
        forecast_days=forecast_days,
        use_mesonet=use_mesonet,
      )
    except Exception as exc:
      st.error(f"Failed to load data: {exc}")
      st.stop()

  loc = KANSAS_LOCATIONS[location]

  col1, col2, col3 = st.columns(3)
  allowed = sum(1 for h in hourly if h.status == SprayStatus.ALLOWED)
  limited = sum(1 for h in hourly if h.status == SprayStatus.LIMITED)
  prohibited = sum(1 for h in hourly if h.status == SprayStatus.PROHIBITED)
  col1.metric("Allowed hours", allowed)
  col2.metric("Limited hours (50% rule)", limited)
  col3.metric("Prohibited hours", prohibited)

  st.subheader("Daily summary")
  daily_rows = []
  for s in daily:
    windows = ", ".join(
      f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
      for start, end in s.best_windows[:3]
    ) or "None"
    daily_rows.append(
      {
        "Date": s.date.strftime("%Y-%m-%d"),
        "Forecast High (°F)": s.forecast_high_f,
        "Next Day High (°F)": s.next_day_high_f,
        "Temp Rule": s.temperature_status.value.upper(),
        "Allowed Hrs": s.sprayable_hours,
        "Limited Hrs": s.limited_hours,
        "Prohibited Hrs": s.prohibited_hours,
        "Best Windows (CT)": windows,
      }
    )
  st.dataframe(pd.DataFrame(daily_rows), use_container_width=True)

  for s in daily:
    for note in s.notes:
      if s.temperature_status == SprayStatus.PROHIBITED:
        st.error(f"{s.date.strftime('%Y-%m-%d')}: {note}")
      elif s.temperature_status == SprayStatus.LIMITED:
        st.warning(f"{s.date.strftime('%Y-%m-%d')}: {note}")

  st.subheader("Hourly spray suitability")
  hourly_rows = []
  for h in hourly:
    hourly_rows.append(
      {
        "Time (CT)": h.timestamp.strftime("%Y-%m-%d %H:%M"),
        "Status": h.status.value,
        "Temp (°F)": h.temperature_f,
        "Wind (mph)": h.wind_speed_mph,
        "Cloud (%)": h.cloud_cover_pct,
        "Inversion": "Yes" if h.inversion_detected else "No",
        "Failed Checks": ", ".join(c.name for c in h.failed_checks) or "—",
      }
    )

  df = pd.DataFrame(hourly_rows)

  def _color_status(val: str) -> str:
    colors = {
      "allowed": "background-color: #d4edda",
      "limited": "background-color: #fff3cd",
      "prohibited": "background-color: #f8d7da",
    }
    return colors.get(val, "")

  st.dataframe(
    df.style.map(_color_status, subset=["Status"]),
    use_container_width=True,
    height=400,
  )

  st.subheader("Charts")
  chart_df = pd.DataFrame(
    {
      "time": [h.timestamp for h in hourly],
      "temperature_f": [h.temperature_f for h in hourly],
      "wind_mph": [h.wind_speed_mph for h in hourly],
      "sprayable": [1 if h.is_sprayable else 0 for h in hourly],
    }
  ).set_index("time")

  c1, c2 = st.columns(2)
  c1.line_chart(chart_df["temperature_f"], height=250)
  c1.caption("Temperature (°F)")
  c2.line_chart(chart_df["wind_mph"], height=250)
  c2.caption("Wind speed (mph) — green zone is 3–10 mph")

  st.info(
    f"Location: {loc.name} ({loc.latitude:.4f}, {loc.longitude:.4f}). "
    "This tool supports decisions only; always follow the product label, "
    "Kansas regulations, and on-site weather observations."
  )
else:
  st.info("Select a Kansas location in the sidebar and click **Run prediction**.")
