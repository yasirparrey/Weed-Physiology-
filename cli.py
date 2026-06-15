#!/usr/bin/env python3
"""Command-line interface for the Kansas dicamba spray predictor."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from dicamba_predictor import DicambaSprayPredictor
from dicamba_predictor.models import KANSAS_LOCATIONS, SprayStatus


def _status_icon(status: SprayStatus) -> str:
  return {
    SprayStatus.ALLOWED: "✓",
    SprayStatus.LIMITED: "~",
    SprayStatus.PROHIBITED: "✗",
  }[status]


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Predict dicamba spray windows in Kansas using EPA label criteria and weather data."
  )
  parser.add_argument(
    "--location",
    default="Manhattan",
    choices=sorted(KANSAS_LOCATIONS.keys()),
    help="Kansas location for the forecast",
  )
  parser.add_argument(
    "--days",
    type=int,
    default=3,
    help="Number of forecast days (default: 3)",
  )
  parser.add_argument(
    "--json",
    action="store_true",
    help="Output machine-readable JSON",
  )
  parser.add_argument(
    "--no-mesonet",
    action="store_true",
    help="Skip Kansas Mesonet live inversion observations",
  )
  args = parser.parse_args(argv)

  predictor = DicambaSprayPredictor()
  try:
    forecast, hourly, daily = predictor.predict(
      args.location,
      forecast_days=args.days,
      use_mesonet=not args.no_mesonet,
    )
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1

  if args.json:
    payload = {
      "location": args.location,
      "generated_at": datetime.now().isoformat(),
      "daily_summaries": [
        {
          "date": s.date.date().isoformat(),
          "forecast_high_f": s.forecast_high_f,
          "next_day_high_f": s.next_day_high_f,
          "temperature_status": s.temperature_status.value,
          "sprayable_hours": s.sprayable_hours,
          "limited_hours": s.limited_hours,
          "prohibited_hours": s.prohibited_hours,
          "best_windows": [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in s.best_windows
          ],
          "notes": s.notes,
        }
        for s in daily
      ],
      "hourly": [
        {
          "time": h.timestamp.isoformat(),
          "status": h.status.value,
          "temperature_f": h.temperature_f,
          "wind_speed_mph": h.wind_speed_mph,
          "inversion_detected": h.inversion_detected,
          "failed_checks": [c.name for c in h.failed_checks],
        }
        for h in hourly
      ],
    }
    print(json.dumps(payload, indent=2))
    return 0

  loc = KANSAS_LOCATIONS[args.location]
  print(f"\nDicamba Spray Prediction — {loc.name}, Kansas")
  print("Based on EPA 2026–2027 federal OTT dicamba label requirements")
  print("=" * 72)

  for summary in daily:
    day = summary.date.strftime("%A, %B %d, %Y")
    high = f"{summary.forecast_high_f:.0f}°F" if summary.forecast_high_f else "N/A"
    print(f"\n{day}")
    print(f"  Forecast high: {high}  |  Temp rule: {summary.temperature_status.value.upper()}")
    print(
      f"  Hours: {summary.sprayable_hours} allowed, "
      f"{summary.limited_hours} limited (50% rule), "
      f"{summary.prohibited_hours} prohibited"
    )
    for note in summary.notes:
      print(f"  Note: {note}")
    if summary.best_windows:
      print("  Best spray windows:")
      for start, end in summary.best_windows[:5]:
        print(f"    {start.strftime('%H:%M')} – {end.strftime('%H:%M')}")

  print("\nHourly detail (Central Time):")
  print(f"{'Time':<18} {'Status':<12} {'Temp':>6} {'Wind':>8} {'Inversion':>10}")
  print("-" * 60)
  for h in hourly:
    temp = f"{h.temperature_f:.0f}°F" if h.temperature_f is not None else "  N/A"
    wind = f"{h.wind_speed_mph:.1f}" if h.wind_speed_mph is not None else "N/A"
    inv = "YES" if h.inversion_detected else "no"
    print(
      f"{h.timestamp.strftime('%m-%d %H:%M'):<18} "
      f"{_status_icon(h.status)} {h.status.value:<10} "
      f"{temp:>6} {wind:>6} mph {inv:>10}"
    )

  print("\nDisclaimer: Decision support only. Verify product label, state rules,")
  print("and field conditions before spraying. Label is the law.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
