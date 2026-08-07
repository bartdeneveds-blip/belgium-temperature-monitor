from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.opendata import Client

LAT = float(os.getenv("LATITUDE", "50.5279"))
LON = float(os.getenv("LONGITUDE", "4.5284"))
TZ = ZoneInfo(os.getenv("TIMEZONE", "Europe/Brussels"))
DAYS = int(os.getenv("FORECAST_DAYS", "10"))
CLIM = Path("data/climatology_1961_1990.csv")
OUT = Path("data/belgium_temperature_anomaly.csv")
GRIB = Path("data/latest_ecmwf.grib2")
META = Path("data/run_metadata.json")

# Forecast steps cover today plus the configured horizon. 00 UTC IFS run.
steps = list(range(0, min(DAYS * 24, 144) + 1, 3))
if DAYS * 24 > 144:
    steps += list(range(150, DAYS * 24 + 1, 6))

client = Client(source="ecmwf", model="ifs", resol="0p25")
client.retrieve(time=0, type="fc", stream="oper", step=steps, param="2t", target=str(GRIB))

# cfgrib exposes forecast valid times; nearest grid point to Fleurus is used.
ds = xr.open_dataset(GRIB, engine="cfgrib", backend_kwargs={"indexpath": ""})
da = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest") - 273.15
valid = pd.to_datetime(ds["valid_time"].values, utc=True)
values = np.asarray(da.values).reshape(-1)
forecast = pd.Series(values, index=valid, name="temperature_c").sort_index()

# Convert timestamps to Europe/Brussels before taking the daily maximum.
forecast.index = forecast.index.tz_convert(TZ)
daily = forecast.resample("1D").max().dropna()

clim = pd.read_csv(CLIM).set_index("month_day")
rows = []
for stamp, value in daily.items():
    key = stamp.strftime("%m-%d")
    norm = float(clim.loc[key, "historic_norm_c"])
    rows.append({
        "date": stamp.strftime("%Y-%m-%d"),
        "forecast_max_c": round(float(value), 2),
        "historic_norm_c": round(norm, 2),
        "anomaly_c": round(float(value) - norm, 2),
        "status": "forecast",
        "latitude": LAT,
        "longitude": LON,
    })
new = pd.DataFrame(rows)

# Replace overlapping forecast dates, retain non-overlapping archive rows.
if OUT.exists():
    old = pd.read_csv(OUT, dtype={"date": str})
    old = old.loc[~old["date"].isin(new["date"])]
    combined = pd.concat([old, new], ignore_index=True)
else:
    combined = new
combined = combined.sort_values("date").drop_duplicates("date", keep="last")
combined.to_csv(OUT, index=False)

run_time = datetime.now(timezone.utc).isoformat()
META.write_text(json.dumps({
    "updated_at_utc": run_time,
    "latitude": LAT,
    "longitude": LON,
    "timezone": str(TZ),
    "method": "Daily max from available 3/6-hourly IFS 2t forecast samples; anomaly versus smoothed ERA5 1961-1990 norm",
}, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(combined)} rows to {OUT}")
