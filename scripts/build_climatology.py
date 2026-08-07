from __future__ import annotations
import os
from pathlib import Path
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

LAT = float(os.getenv("LATITUDE", "50.5279"))
LON = float(os.getenv("LONGITUDE", "4.5284"))
OUT = Path("data/climatology_1961_1990.csv")
CACHE = Path("data/era5_1961_1990_point.nc")

# Small bounding box; nearest 0.25-degree grid point is selected later.
AREA = [LAT + 0.15, LON - 0.15, LAT - 0.15, LON + 0.15]

if not CACHE.exists():
    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [str(y) for y in range(1961, 1991)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", request, str(CACHE))

ds = xr.open_dataset(CACHE)
var = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars)[0]
time_name = "valid_time" if "valid_time" in ds.coords else "time"
da = ds[var].sel(latitude=LAT, longitude=LON, method="nearest") - 273.15
series = da.to_series()
series.index = pd.to_datetime(series.index)

# Daily maximum in UTC, matching a reproducible meteorological day definition.
daily_max = series.resample("1D").max().dropna()
frame = daily_max.rename("daily_max_c").to_frame()
frame["month_day"] = frame.index.strftime("%m-%d")

# Reuters describes a smoothed daily norm. This implementation uses a circular
# 31-day window around each calendar day across 1961-1990.
calendar = pd.date_range("2000-01-01", "2000-12-31", freq="D")
keys = calendar.strftime("%m-%d").tolist()
positions = {k: i for i, k in enumerate(keys)}
rows = []
for key in keys:
    center = positions[key]
    window = {keys[(center + offset) % len(keys)] for offset in range(-15, 16)}
    values = frame.loc[frame["month_day"].isin(window), "daily_max_c"]
    rows.append({
        "month_day": key,
        "historic_norm_c": round(float(values.mean()), 2),
        "sample_count": int(values.count()),
        "reference_period": "1961-1990",
    })
OUT.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {len(rows)} climatology rows to {OUT}")
