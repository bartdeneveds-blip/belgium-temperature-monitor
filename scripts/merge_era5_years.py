from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_DIR = Path("data/yearly")
OUTPUT_FILE = Path("data/climatology_1961_1990.csv")

START_YEAR = 1961
END_YEAR = 1990


def load_yearly_files() -> pd.DataFrame:
    """Lees en controleer alle jaarlijkse ERA5-bestanden."""

    frames = []
    missing_files = []

    for year in range(START_YEAR, END_YEAR + 1):
        path = INPUT_DIR / f"era5_daily_{year}.csv"

        if not path.exists():
            missing_files.append(str(path))
            continue

        frame = pd.read_csv(
            path,
            dtype={"date": str, "month_day": str},
        )

        required_columns = {
            "date",
            "year",
            "month_day",
            "daily_max_c",
            "latitude",
            "longitude",
        }

        missing_columns = required_columns - set(frame.columns)

        if missing_columns:
            raise RuntimeError(
                f"{path} mist deze kolommen: "
                f"{sorted(missing_columns)}"
            )

        expected_days = 366 if pd.Timestamp(
            year=year,
            month=12,
            day=31,
        ).is_leap_year else 365

        if len(frame) != expected_days:
            raise RuntimeError(
                f"{path} bevat {len(frame)} rijen, "
                f"maar {expected_days} werden verwacht"
            )

        if frame["daily_max_c"].isna().any():
            raise RuntimeError(
                f"{path} bevat ontbrekende temperatuurwaarden"
            )

        frames.append(frame)

        print(
            f"{year}: {len(frame)} dagelijkse waarden geladen",
            flush=True,
        )

    if missing_files:
        raise RuntimeError(
            "De volgende jaarbestanden ontbreken:\n"
            + "\n".join(missing_files)
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date")

    expected_total = sum(
        366 if pd.Timestamp(
            year=year,
            month=12,
            day=31,
        ).is_leap_year else 365
        for year in range(START_YEAR, END_YEAR + 1)
    )

    if len(combined) != expected_total:
        raise RuntimeError(
            f"Er zijn {len(combined)} dagelijkse waarden, "
            f"maar {expected_total} werden verwacht"
        )

    if combined["date"].duplicated().any():
        duplicates = combined.loc[
            combined["date"].duplicated(),
            "date",
        ]

        raise RuntimeError(
            "Dubbele datums gevonden: "
            + ", ".join(
                duplicates.dt.strftime("%Y-%m-%d").tolist()
            )
        )

    print(
        f"Totaal: {len(combined)} dagelijkse waarden geladen",
        flush=True,
    )

    return combined


def build_climatology(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bereken voor iedere kalenderdag een historische norm.

    Per kalenderdag wordt een circulair venster gebruikt
    van 31 dagen: de dag zelf, de 15 dagen ervoor en
    de 15 dagen erna, over 1961-1990.
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
            month_days[
                (center + offset) % len(month_days)
            ]
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
                "historic_norm_c": round(
                    float(values.mean()),
                    2,
                ),
                "sample_count": int(values.count()),
                "reference_period": "1961-1990",
            }
        )

    result = pd.DataFrame(rows)

    if len(result) != 366:
        raise RuntimeError(
            f"Er werden {len(result)} normen berekend, "
            "maar 366 werden verwacht"
        )

    if result["historic_norm_c"].isna().any():
        raise RuntimeError(
            "De historische norm bevat lege waarden"
        )

    return result


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_data = load_yearly_files()
    climatology = build_climatology(daily_data)

    climatology.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Climatologie opgeslagen in {OUTPUT_FILE}",
        flush=True,
    )

    print(
        f"Aantal kalenderdagen: {len(climatology)}",
        flush=True,
    )

    print(climatology.head().to_string(index=False))


if __name__ == "__main__":
    main()
