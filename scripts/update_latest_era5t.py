from __future__ import annotations

import calendar
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


LATITUDE = float(
    os.getenv("LATITUDE", "50.5279")
)

LONGITUDE = float(
    os.getenv("LONGITUDE", "4.5284")
)

YEAR = int(
    os.getenv(
        "BACKFILL_YEAR",
        str(datetime.now(timezone.utc).year),
    )
)

ERA5_LAG_DAYS = int(
    os.getenv("ERA5_LAG_DAYS", "5")
)

MAX_ATTEMPTS = 5

CLIMATOLOGY_FILE = Path(
    "data/climatology_1961_1990.csv"
)

COMBINED_FILE = Path(
    "data/belgium_temperature_anomaly.csv"
)

ERA5T_ONLY_FILE = Path(
    "data/belgium_temperature_anomaly_era5t.csv"
)

CACHE_DIR = Path(
    "data/era5t_cache"
)

AREA = [
    LATITUDE + 0.15,
    LONGITUDE - 0.15,
    LATITUDE - 0.15,
    LONGITUDE + 0.15,
]


def target_date() -> pd.Timestamp:
    """
    Neem uit voorzorg de datum van vijf dagen geleden.

    ERA5T loopt enkele dagen achter op de actuele datum.
    """

    today = datetime.now(timezone.utc).date()

    target = today - timedelta(
        days=ERA5_LAG_DAYS
    )

    year_start = datetime(
        YEAR,
        1,
        1,
        tzinfo=timezone.utc,
    ).date()

    year_end = datetime(
        YEAR,
        12,
        31,
        tzinfo=timezone.utc,
    ).date()

    if target < year_start:
        target = year_start

    if target > year_end:
        target = year_end

    return pd.Timestamp(target)


def load_combined_data() -> pd.DataFrame:
    """
    Lees de bestaande gecombineerde CSV.

    Wanneer het bestand nog niet bestaat, wordt een lege
    tabel met de correcte kolommen aangemaakt.
    """

    columns = [
        "date",
        "forecast_max_c",
        "historic_norm_c",
        "anomaly_c",
        "status",
        "latitude",
        "longitude",
    ]

    if not COMBINED_FILE.exists():
        return pd.DataFrame(columns=columns)

    data = pd.read_csv(
        COMBINED_FILE,
        dtype={
            "date": str,
            "status": str,
        },
    )

    missing_columns = set(columns) - set(
        data.columns
    )

    if missing_columns:
        raise RuntimeError(
            "De gecombineerde CSV mist deze kolommen: "
            + ", ".join(sorted(missing_columns))
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    return data


def first_missing_date(
    existing: pd.DataFrame,
    end_date: pd.Timestamp,
) -> pd.Timestamp | None:
    """
    Bepaal de eerste kalenderdag waarvoor nog geen ERA5T-rij
    bestaat.

    Forecast-rijen tellen niet als ERA5T-gegevens.
    """

    start_date = pd.Timestamp(
        year=YEAR,
        month=1,
        day=1,
    )

    expected_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    if existing.empty:
        return start_date

    era5t_dates = existing.loc[
        existing["status"].eq("era5t"),
        "date",
    ]

    era5t_dates = set(
        pd.to_datetime(era5t_dates)
        .dt.normalize()
        .tolist()
    )

    missing_dates = [
        date
        for date in expected_dates
        if date.normalize() not in era5t_dates
    ]

    if not missing_dates:
        return None

    return missing_dates[0]


def month_ranges(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Verdeel de ontbrekende periode in maandstukken.

    Dit houdt de Copernicus-aanvragen klein.
    """

    ranges = []
    current = start_date.normalize()

    while current <= end_date:
        last_calendar_day = calendar.monthrange(
            current.year,
            current.month,
        )[1]

        month_end = pd.Timestamp(
            year=current.year,
            month=current.month,
            day=last_calendar_day,
        )

        period_end = min(
            month_end,
            end_date,
        )

        ranges.append(
            (
                current,
                period_end,
            )
        )

        current = period_end + pd.Timedelta(
            days=1
        )

    return ranges


def build_request(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    """
    Bouw een CDS-request voor één maandfragment.
    """

    if (
        start_date.year != end_date.year
        or start_date.month != end_date.month
    ):
        raise ValueError(
            "Een request mag slechts één maand bevatten"
        )

    return {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [str(start_date.year)],
        "month": [f"{start_date.month:02d}"],
        "day": [
            f"{day:02d}"
            for day in range(
                start_date.day,
                end_date.day + 1,
            )
        ],
        "time": [
            f"{hour:02d}:00"
            for hour in range(24)
        ],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_period(
    client: cdsapi.Client,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Path:
    """
    Download een periode met automatische retries.
    """

    target = CACHE_DIR / (
        f"era5t_{start_date:%Y%m%d}_"
        f"{end_date:%Y%m%d}.nc"
    )

    request = build_request(
        start_date,
        end_date,
    )

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            target.unlink(missing_ok=True)

            print(
                f"ERA5T-aanvraag "
                f"{start_date.date()} tot "
                f"{end_date.date()}, "
                f"poging {attempt}/{MAX_ATTEMPTS}",
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
                f"Download voltooid: "
                f"{target.stat().st_size / 1_000_000:.2f} MB",
                flush=True,
            )

            return target

        except Exception as error:
            target.unlink(missing_ok=True)

            print(
                f"Poging {attempt} mislukt: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    "ERA5T-download definitief mislukt"
                ) from error

            wait_seconds = attempt * 60

            print(
                f"Nieuwe poging over "
                f"{wait_seconds} seconden",
                flush=True,
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "Onverwacht einde van de downloadfunctie"
    )


def find_temperature_variable(
    dataset: xr.Dataset,
) -> str:
    """
    Zoek de temperatuurvariabele in het NetCDF-bestand.
    """

    if "t2m" in dataset.data_vars:
        return "t2m"

    variables = list(dataset.data_vars)

    if not variables:
        raise RuntimeError(
            "Het NetCDF-bestand bevat geen variabelen"
        )

    return variables[0]


def process_download(
    path: Path,
) -> pd.Series:
    """
    Selecteer het rasterpunt en bereken dagelijkse maxima.
    """

    with xr.open_dataset(path) as dataset:
        variable = find_temperature_variable(
            dataset
        )

        temperature = dataset[variable].sel(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            method="nearest",
        )

        temperature = temperature - 273.15
        temperature = temperature.squeeze(
            drop=True
        )

        series = temperature.to_series()

    if isinstance(
        series.index,
        pd.MultiIndex,
    ):
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
                "Geen tijdsdimensie gevonden in ERA5T"
            )

    series.index = pd.to_datetime(
        series.index,
        utc=True,
    )

    series = series.sort_index()

    return (
        series.resample("1D")
        .max()
        .dropna()
    )


def retrieve_missing_days(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Haal alleen de ontbrekende ERA5T-periode op.
    """

    client = cdsapi.Client()
    results = []

    for period_start, period_end in month_ranges(
        start_date,
        end_date,
    ):
        path = download_period(
            client,
            period_start,
            period_end,
        )

        try:
            values = process_download(path)

            if values.empty:
                raise RuntimeError(
                    f"Geen dagwaarden ontvangen voor "
                    f"{period_start.date()} tot "
                    f"{period_end.date()}"
                )

            results.append(values)

            print(
                f"{len(values)} dagwaarden verwerkt "
                f"voor {period_start:%Y-%m}",
                flush=True,
            )

        finally:
            path.unlink(missing_ok=True)

    if not results:
        return pd.Series(
            dtype="float64"
        )

    combined = pd.concat(results)
    combined = combined.sort_index()

    combined = combined[
        ~combined.index.duplicated(
            keep="last"
        )
    ]

    return combined


def build_era5t_rows(
    daily_maximum: pd.Series,
) -> pd.DataFrame:
    """
    Bereken de anomalie tegenover de historische norm.
    """

    climatology = pd.read_csv(
        CLIMATOLOGY_FILE,
        dtype={"month_day": str},
    ).set_index("month_day")

    rows = []

    for timestamp, maximum in daily_maximum.items():
        month_day = timestamp.strftime("%m-%d")

        if month_day not in climatology.index:
            raise RuntimeError(
                f"Geen historische norm gevonden "
                f"voor {month_day}"
            )

        norm = float(
            climatology.loc[
                month_day,
                "historic_norm_c",
            ]
        )

        rows.append(
            {
                "date": timestamp.strftime(
                    "%Y-%m-%d"
                ),
                "forecast_max_c": round(
                    float(maximum),
                    2,
                ),
                "historic_norm_c": round(
                    norm,
                    2,
                ),
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


def merge_data(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Voeg nieuwe ERA5T-waarden toe.

    Een ERA5T-waarde vervangt een eventuele forecast voor
    dezelfde kalenderdag.
    """

    data = existing.copy()

    if not data.empty:
        data["date"] = pd.to_datetime(
            data["date"]
        )

    if not new_rows.empty:
        new_rows = new_rows.copy()

        new_rows["date"] = pd.to_datetime(
            new_rows["date"]
        )

        if not data.empty:
            data = data.loc[
                ~data["date"].isin(
                    new_rows["date"]
                )
            ]

        data = pd.concat(
            [data, new_rows],
            ignore_index=True,
        )

    if data.empty:
        return data

    data = data.sort_values("date")

    data = data.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    expected_anomaly = (
        data["forecast_max_c"]
        - data["historic_norm_c"]
    ).round(2)

    actual_anomaly = data[
        "anomaly_c"
    ].round(2)

    if not expected_anomaly.equals(
        actual_anomaly
    ):
        raise RuntimeError(
            "De anomalieberekening is niet consistent"
        )

    data["date"] = data[
        "date"
    ].dt.strftime("%Y-%m-%d")

    return data


def export_era5t_only(
    combined: pd.DataFrame,
) -> None:
    """
    Maak een aparte Datawrapper-CSV zonder forecasts.
    """

    if combined.empty:
        raise RuntimeError(
            "De gecombineerde CSV is leeg"
        )

    era5t = combined.loc[
        combined["status"].eq("era5t")
    ].copy()

    if era5t.empty:
        raise RuntimeError(
            "Er zijn geen ERA5T-rijen om te exporteren"
        )

    era5t = era5t.sort_values("date")

    era5t.to_csv(
        ERA5T_ONLY_FILE,
        index=False,
    )

    print(
        f"Datawrapper-export bevat "
        f"{len(era5t)} ERA5T-rijen",
        flush=True,
    )

    print(
        f"Eerste datum: {era5t['date'].min()}",
        flush=True,
    )

    print(
        f"Laatste datum: {era5t['date'].max()}",
        flush=True,
    )


def main() -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMBINED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CLIMATOLOGY_FILE.exists():
        raise RuntimeError(
            f"{CLIMATOLOGY_FILE} ontbreekt"
        )

    existing = load_combined_data()
    end_date = target_date()

    start_date = first_missing_date(
        existing,
        end_date,
    )

    if start_date is None:
        print(
            "Geen ontbrekende ERA5T-dagen gevonden",
            flush=True,
        )

        combined = existing.copy()

        if not combined.empty:
            combined["date"] = pd.to_datetime(
                combined["date"]
            )

            combined = combined.sort_values(
                "date"
            )

            combined["date"] = combined[
                "date"
            ].dt.strftime("%Y-%m-%d")

    else:
        print(
            f"Ontbrekende ERA5T-periode: "
            f"{start_date.date()} tot "
            f"{end_date.date()}",
            flush=True,
        )

        daily_maximum = retrieve_missing_days(
            start_date,
            end_date,
        )

        new_rows = build_era5t_rows(
            daily_maximum
        )

        combined = merge_data(
            existing,
            new_rows,
        )

        print(
            f"{len(new_rows)} nieuwe ERA5T-rijen toegevoegd",
            flush=True,
        )

    combined.to_csv(
        COMBINED_FILE,
        index=False,
    )

    export_era5t_only(
        combined
    )

    print(
        f"Gecombineerde CSV: {COMBINED_FILE}",
        flush=True,
    )

    print(
        f"ERA5T-only CSV: {ERA5T_ONLY_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
