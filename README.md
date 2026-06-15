# Kansas Dicamba Spray Predictor

A Python application that predicts **legal dicamba spray windows** for Kansas fields using **EPA 2026–2027 federal label requirements** and live weather data.

## What it evaluates

Based on [EPA OTT dicamba registration requirements](https://www.epa.gov/ingredients-used-pesticide-products/registration-dicamba-use-dicamba-tolerant-crops):

| Criterion | EPA requirement |
|-----------|-----------------|
| **Temperature** | No spray if forecast high ≥ **95°F** on application day or the following day |
| **50% acreage rule** | If forecast is **85–95°F**, only 50% of untreated DT acres per county; wait ≥2 days for remainder |
| **Wind speed** | **3–10 mph** at application time |
| **Sunrise / sunset** | No spray within **1 hour after sunrise** or **2 hours before sunset** |
| **Temperature inversion** | No spray during inversions |
| **Rainfall** | No spray within **48 hours** of forecast rainfall |

Kansas follows federal labels (no additional state temperature cap as of 2026). Always verify the current product label before spraying.

## Data sources

- **Weather forecast**: [Open-Meteo](https://open-meteo.com/) (NWS-backed models), hourly temperature, wind, dew point, cloud cover, precipitation
- **Inversion observations**: [Kansas Mesonet](https://mesonet.k-state.edu/agriculture/inversion/) (2 m vs 10 m air temperature difference)
- **Sunrise/sunset**: Calculated with the `astral` library for each Kansas location

## Quick start

```bash
pip install -r requirements.txt

# Command line — Manhattan, KS, 3-day forecast
python3 cli.py --location Manhattan --days 3

# JSON output for integration
python3 cli.py --location Wichita --days 5 --json

# Web dashboard
streamlit run app.py
```

## Kansas locations

Manhattan, Wichita, Hutchinson, Dodge City, Garden City, Parsons, Colby, Hays, Olathe, Salina, Emporia, Liberal

## Project structure

```
dicamba_predictor/
  regulations.py   # EPA threshold constants
  weather.py         # Open-Meteo forecast client
  mesonet.py         # Kansas Mesonet inversion parser
  inversion.py       # Inversion risk assessment
  predictor.py       # Main spray window logic
  models.py          # Data classes and Kansas locations
app.py               # Streamlit web UI
cli.py               # Command-line interface
```

## Example: programmatic use

```python
from dicamba_predictor import DicambaSprayPredictor

predictor = DicambaSprayPredictor()
forecast, hourly, daily = predictor.predict("Manhattan", forecast_days=3)

for hour in hourly:
    if hour.is_sprayable:
        print(
            hour.timestamp,
            hour.status.value,
            f"{hour.temperature_f:.0f}°F",
            f"{hour.wind_speed_mph:.1f} mph",
        )
```

## Disclaimer

This tool is **decision support only**. The pesticide label is the law. Before spraying:

1. Confirm the product is registered for use in Kansas
2. Check on-site wind, temperature, and inversion conditions
3. Consult [Kansas Mesonet inversions](https://mesonet.k-state.edu/agriculture/inversion/)
4. Maintain required application records per the EPA label
