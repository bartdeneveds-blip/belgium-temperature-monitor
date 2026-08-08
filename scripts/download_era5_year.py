from __future__ import annotations

import os
import time
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


# Het te verwerken jaar wordt door GitHub Actions doorgegeven.
YEAR = int(os.environ["YEAR"])

# Rasterpunt bij Wagnelée, Fleurus.
LATITUDE = float(os.getenv("LATITUDE", "50.5279"))
LONGITUDE = float(os.getenv("LONGITUDE", "4.5284"))

# Tijdelijke download en definitieve jaaruitvoer.
CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("data/yearly")

NETCDF_FILE = CACHE_DIR / f"era5_{YEAR}.nc"
OUTPUT_FILE = OUTPUT_DIR / f"era5_daily_{YEAR}.csv"

MAX_ATTEMPTS = 5

# Kleine zone rond het geselecteerde punt:
# noord, west, zuid, oost.
AREA = [
    LATITUDE + 0.15,
    LONGITUDE - 0.15,
    LATITUDE - 0.15,
    LONGITUDE + 0.15,
]


def build_request(year: int) -> dict:
    """Bouw de Copernicus ERA5-aanvraag voor één jaar."""

    return {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [str(year)],
        "month": [
            f"{month:02d}"
            for month in range(1, 13)
        ],
        "day": [
            f"{day:02d}"
            for day in range(1, 32)
        ],
        "time": [
            f"{hour:02d}:00"
            for hour in range(24)
        ],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_era5_year() -> None:
    """Download één jaar, met automatische nieuwe pogingen."""

    client = cdsapi.Client()
    request = build_request(YEAR)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            NETCDF_FILE.unlink(missing_ok=True)

            print(
                f"{YEAR}: ERA5-aanvraag wordt ingediend "
                f"(poging {attempt}/{MAX_ATTEMPTS})",
                flush=True,
            )

            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                str(NETCDF_FILE),
            )

            if not NETCDF_FILE.exists():
                raise RuntimeError(
                    "Copernicus heeft geen bestand opgeleverd"
                )

            if NETCDF_FILE.stat().st_size == 0:
                raise RuntimeError(
                    "Copernicus heeft een leeg bestand opgeleverd"
                )

            print(
                f"{YEAR}: download voltooid, "
                f"{NETCDF_FILE.stat().st_size / 1_000_000:.1f} MB",
                flush=True,
            )

            return

        except Exception as error:
            NETCDF_FILE.unlink(missing_ok=True)

            print(
                f"{YEAR}: poging {attempt} mislukt: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{YEAR}: download na {MAX_ATTEMPTS} "
                    "pogingen definitief mislukt"
                ) from error

            wait_seconds = attempt * 60

            print(
                f"{YEAR}: volgende poging over "
                f"{wait_seconds} seconden",
                flush=True,
            )

            time.sleep(wait_seconds)


def find_temperature_variable(dataset: xr.Dataset) -> str:
    """Zoek de naam van de temperatuurvariabele."""

    if "t2m" in dataset.data_vars:
        return "t2m"

    if not dataset.data_vars:
        raise RuntimeError(
            f"{YEAR}: NetCDF-bestand bevat geen datavariabelen"
        )

    return list(dataset.data_vars)[0]


def process_era5_year() -> None:
    """Bereken de dagelijkse maximumtemperatuur voor één jaar."""

    print(
        f"{YEAR}: gedownloade gegevens worden verwerkt",
        flush=True,
    )

    with xr.open_dataset(NETCDF_FILE) as dataset:
        variable = find_temperature_variable(dataset)

        temperature = dataset[variable].sel(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            method="nearest",
        )

        # Kelvin omzetten naar graden Celsius.
        temperature = temperature - 273.15

        # Verwijder dimensies met lengte één, behalve de tijd.
        temperature = temperature.squeeze(drop=True)

        series = temperature.to_series()

    if isinstance(series.index, pd.MultiIndex):
        index_names = list(series.index.names)

        if "valid_time" in index_names:
            frame = series.reset_index()
            series = frame.groupby("valid_time")[variable].mean()

        elif "time" in index_names:
            frame = series.reset_index()
            series = frame.groupby("time")[variable].mean()

        else:
            raise RuntimeError(
                f"{YEAR}: tijdsdimensie niet gevonden. "
                f"Beschikbare indexen: {index_names}"
            )

    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    # Dagmaximum volgens UTC-kalenderdagen.
    daily_maximum = series.resample("1D").max().dropna()

    expected_days = 366 if pd.Timestamp(
        year=YEAR,
        month=12,
        day=31,
    ).is_leap_year else 365

    if len(daily_maximum) != expected_days:
        raise RuntimeError(
            f"{YEAR}: {len(daily_maximum)} dagelijkse waarden "
            f"gevonden, maar {expected_days} verwacht"
        )

    result = pd.DataFrame(
        {
            "date": daily_maximum.index.strftime("%Y-%m-%d"),
            "year": YEAR,
            "month_day": daily_maximum.index.strftime("%m-%d"),
            "daily_max_c": daily_maximum.round(4).values,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        }
    )

    if result["daily_max_c"].isna().any():
        raise RuntimeError(
            f"{YEAR}: de uitvoer bevat ontbrekende temperaturen"
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"{YEAR}: {len(result)} dagelijkse maxima opgeslagen in "
        f"{OUTPUT_FILE}",
        flush=True,
    )


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        download_era5_year()
        process_era5_year()

    finally:
        # Verwijder het tijdelijke NetCDF-bestand,
        # ook wanneer de verwerking mislukt.
        NETCDF_FILE.unlink(missing_ok=True)

        print(
            f"{YEAR}: tijdelijk NetCDF-bestand verwijderd",
            flush=True,
        )


if __name__ == "__main__":
    main()
