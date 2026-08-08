from __future__ import annotations

import calendar
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


LATITUDE = float(os.getenv("LATITUDE", "50.5279"))
LONGITUDE = float(os.getenv("LONGITUDE", "4.5284"))

YEAR = int(
    os.getenv(
        "BACKFILL_YEAR",
        str(datetime.now(timezone.utc).year),
    )
)

LAG_DAYS = int(os.getenv("ERA5_LAG_DAYS", "5"))

CLIMATOLOGY_FILE = Path(
    "data/climatology_1961_1990.csv"
)

OUTPUT_FILE = Path(
    "data/belgium_temperature_anomaly.csv"
)

CACHE_DIR = Path("data/backfill_cache")

MAX_ATTEMPTS = 5

AREA = [
    LATITUDE + 0.15,
    LONGITUDE - 0.15,
    LATITUDE - 0.15,
    LONGITUDE + 0.15,
]


def latest_available_date() -> pd.Timestamp:
    today_utc = datetime.now(timezone.utc).date()
    cutoff = today_utc - timedelta(days=LAG_DAYS)

    if cutoff.year < YEAR:
        return pd.Timestamp(f"{YEAR}-01-01")

    if cutoff.year > YEAR:
        return pd.Timestamp(f"{YEAR}-12-31")

    return pd.Timestamp(cutoff)


def retrieve_month(
    client: cdsapi.Client,
    year: int,
    month: int,
    last_day: int,
) -> Path:
    target = CACHE_DIR / (
        f"era5_{year}_{month:02d}.nc"
    )

    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [
            f"{day:02d}"
            for day in range(1, last_day + 1)
        ],
        "time": [
            f"{hour:02d}:00"
            for hour in range(24)
        ],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            target.unlink(missing_ok=True)

            print(
                f"{year}-{month:02d}: downloadpoging "
                f"{attempt}/{MAX_ATTEMPTS}",
                flush=True,
            )

            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                str(target),
            )

            if not target.exists():
                raise RuntimeError(
                    "Copernicus leverde geen bestand op"
                )

            if target.stat().st_size == 0:
                raise RuntimeError(
                    "Copernicus leverde een leeg bestand op"
                )

            print(
                f"{year}-{month:02d}: download voltooid",
                flush=True,
            )

            return target

        except Exception as error:
            target.unlink(missing_ok=True)

            print(
                f"{year}-{month:02d}: poging mislukt: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

            if attempt == MAX_ATTEMPTS:
                raise

            wait_seconds = attempt * 60

            print(
                f"Nieuwe poging over {wait_seconds} seconden",
                flush=True,
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Download voor {year}-{month:02d} mislukt"
    )


def process_month(
    path: Path,
) -> pd.Series:
    with xr.open_dataset(path) as dataset:
        if "t2m" in dataset.data_vars:
            variable = "t2m"
        else:
            variable = list(dataset.data_vars)[0]

        temperature = dataset[variable].sel(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            method="nearest",
        )

        temperature = temperature - 273.15
        temperature = temperature.squeeze(drop=True)

        series = temperature.to_series()

    if isinstance(series.index, pd.MultiIndex):
        frame = series.reset_index()

        if "valid_time" in frame.columns:
            series = frame.groupby(
                "valid_time"
            )[variable].mean()

        elif "time" in frame.columns:
            series = frame.groupby(
                "time"
            )[variable].mean()

        else:
            raise RuntimeError(
                "Geen tijdsdimensie gevonden"
            )

    series.index = pd.to_datetime(
        series.index,
        utc=True,
    )

    return series.sort_index().resample("1D").max()


def download_current_year() -> pd.Series:
    cutoff = latest_available_date()

    print(
        f"ERA5T-backfill van {YEAR}-01-01 "
        f"tot en met {cutoff.date()}",
        flush=True,
    )

    client = cdsapi.Client()
    monthly_series = []

    for month in range(1, cutoff.month + 1):
        days_in_month = calendar.monthrange(
            YEAR,
            month,
        )[1]

        if month == cutoff.month:
            last_day = cutoff.day
        else:
            last_day = days_in_month

        path = retrieve_month(
            client,
            YEAR,
            month,
            last_day,
        )

        try:
            series = process_month(path)
            monthly_series.append(series)

            print(
                f"{YEAR}-{month:02d}: "
                f"{len(series)} dagen verwerkt",
                flush=True,
            )

        finally:
            path.unlink(missing_ok=True)

    combined = pd.concat(monthly_series)
    combined = combined.sort_index()
    combined = combined[
        ~combined.index.duplicated(keep="last")
    ]

    start = pd.Timestamp(
        f"{YEAR}-01-01",
        tz="UTC",
    )

    end = cutoff.tz_localize("UTC")

    combined = combined.loc[start:end]

    return combined.dropna()


def build_era5_rows(
    daily_maximum: pd.Series,
) -> pd.DataFrame:
    climatology = pd.read_csv(
        CLIMATOLOGY_FILE,
        dtype={"month_day": str},
    ).set_index("month_day")

    rows = []

    for timestamp, maximum in daily_maximum.items():
        month_day = timestamp.strftime("%m-%d")

        if month_day not in climatology.index:
            raise RuntimeError(
                f"Geen historische norm voor {month_day}"
            )

        norm = float(
            climatology.loc[
                month_day,
                "historic_norm_c",
            ]
        )

        rows.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "forecast_max_c": round(
                    float(maximum),
                    2,
                ),
                "historic_norm_c": round(norm, 2),
                "anomaly_c": round(
                    float(maximum) - norm,
                    2,
                ),
                "status": "era5t",
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
            }
        )

    return pd.DataFrame(rows)


def merge_with_existing(
    era5_rows: pd.DataFrame,
) -> pd.DataFrame:
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(
            OUTPUT_FILE,
            dtype={"date": str},
        )

        existing = existing.loc[
            ~existing["date"].isin(era5_rows["date"])
        ]

        combined = pd.concat(
            [existing, era5_rows],
            ignore_index=True,
        )

    else:
        combined = era5_rows

    combined = combined.sort_values("date")
    combined = combined.drop_duplicates(
        "date",
        keep="last",
    )

    calculated = (
        combined["forecast_max_c"]
        - combined["historic_norm_c"]
    ).round(2)

    if not calculated.equals(
        combined["anomaly_c"].round(2)
    ):
        raise RuntimeError(
            "De anomalieberekening is niet consistent"
        )

    return combined


def main() -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CLIMATOLOGY_FILE.exists():
        raise RuntimeError(
            f"{CLIMATOLOGY_FILE} ontbreekt"
        )

    daily_maximum = download_current_year()

    if daily_maximum.empty:
        raise RuntimeError(
            "Geen ERA5T-dagwaarden ontvangen"
        )

    era5_rows = build_era5_rows(
        daily_maximum
    )

    combined = merge_with_existing(
        era5_rows
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"{len(era5_rows)} ERA5T-rijen gemaakt",
        flush=True,
    )

    print(
        f"Totaal aantal CSV-rijen: {len(combined)}",
        flush=True,
    )

    print(
        f"CSV opgeslagen in {OUTPUT_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
