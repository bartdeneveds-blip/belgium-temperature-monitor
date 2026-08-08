from __future__ import annotations

import os
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


LAT = float(os.getenv("LATITUDE", "50.5279"))
LON = float(os.getenv("LONGITUDE", "4.5284"))

START_YEAR = 1961
END_YEAR = 1990

OUT = Path("data/climatology_1961_1990.csv")
CACHE_DIR = Path("data/era5_cache")

# Een kleine zone rond Fleurus.
# Uit die zone wordt later het dichtstbijzijnde rasterpunt gekozen.
AREA = [
    LAT + 0.15,  # noord
    LON - 0.15,  # west
    LAT - 0.15,  # zuid
    LON + 0.15,  # oost
]


def download_year(client: cdsapi.Client, year: int) -> Path:
    """
    Download één jaar ERA5-uurgegevens.
    """

    target = CACHE_DIR / f"era5_{year}.nc"

    if target.exists() and target.stat().st_size > 0:
        print(f"{year}: bestaand bestand wordt hergebruikt")
        return target

    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [str(year)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"{year}: ERA5-aanvraag wordt ingediend")

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(target),
    )

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(
            f"Download voor {year} heeft geen geldig bestand opgeleverd"
        )

    print(
        f"{year}: download voltooid "
        f"({target.stat().st_size / 1_000_000:.1f} MB)"
    )

    return target


def process_year(path: Path, year: int) -> pd.DataFrame:
    """
    Selecteer het dichtstbijzijnde rasterpunt en bereken
    voor iedere UTC-dag de maximumtemperatuur.
    """

    print(f"{year}: gegevens worden verwerkt")

    with xr.open_dataset(path) as ds:
        if "t2m" in ds.data_vars:
            variable = "t2m"
        else:
            variable = list(ds.data_vars)[0]

        temperature = ds[variable].sel(
            latitude=LAT,
            longitude=LON,
            method="nearest",
        )

        # Kelvin omzetten naar graden Celsius
        temperature = temperature - 273.15

        series = temperature.to_series()

    # Recente CDS-bestanden gebruiken vaak valid_time;
    # oudere bestanden kunnen time gebruiken.
    if isinstance(series.index, pd.MultiIndex):
        if "valid_time" in series.index.names:
            series = series.reset_index().set_index("valid_time")[variable]
        elif "time" in series.index.names:
            series = series.reset_index().set_index("time")[variable]

    series.index = pd.to_datetime(series.index)

    daily_max = series.resample("1D").max().dropna()

    frame = daily_max.rename("daily_max_c").to_frame()
    frame["month_day"] = frame.index.strftime("%m-%d")
    frame["year"] = year

    print(f"{year}: {len(frame)} dagelijkse maxima berekend")

    return frame


def make_climatology(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Bereken voor iedere kalenderdag een historische norm.

    De norm gebruikt een circulair venster van 31 dagen:
    de kalenderdag zelf, de 15 voorafgaande dagen en
    de 15 volgende dagen, over de periode 1961-1990.
    """

    calendar = pd.date_range(
        "2000-01-01",
        "2000-12-31",
        freq="D",
    )

    month_days = calendar.strftime("%m-%d").tolist()
    positions = {
        month_day: position
        for position, month_day in enumerate(month_days)
    }

    rows = []

    for month_day in month_days:
        center = positions[month_day]

        window = {
            month_days[(center + offset) % len(month_days)]
            for offset in range(-15, 16)
        }

        values = daily_data.loc[
            daily_data["month_day"].isin(window),
            "daily_max_c",
        ]

        if values.empty:
            raise RuntimeError(
                f"Geen historische waarden gevonden voor {month_day}"
            )

        rows.append(
            {
                "month_day": month_day,
                "historic_norm_c": round(float(values.mean()), 2),
                "sample_count": int(values.count()),
                "reference_period": "1961-1990",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    yearly_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        path = download_year(client, year)
        frame = process_year(path, year)
        yearly_frames.append(frame)

        # Verwijder het tijdelijke NetCDF-bestand na verwerking.
        # Zo raakt de GitHub-runner niet vol.
        path.unlink(missing_ok=True)
        print(f"{year}: tijdelijk bestand verwijderd")

    daily_data = pd.concat(
        yearly_frames,
        ignore_index=False,
    ).sort_index()

    print(
        f"Totaal aantal dagelijkse maxima: {len(daily_data)}"
    )

    climatology = make_climatology(daily_data)

    if len(climatology) != 366:
        raise RuntimeError(
            f"Er werden {len(climatology)} kalenderdagen berekend "
            "in plaats van 366"
        )

    if climatology["historic_norm_c"].isna().any():
        raise RuntimeError(
            "De historische norm bevat ontbrekende waarden"
        )

    climatology.to_csv(
        OUT,
        index=False,
    )

    print(f"Climatologie opgeslagen in {OUT}")
    print(f"Aantal rijen: {len(climatology)}")
    print(climatology.head())


if __name__ == "__main__":
    main()
