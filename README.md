# Belgium temperature anomaly automation

This repository builds a daily maximum-temperature climatology from Copernicus ERA5 (1961-1990) and updates a CSV from the latest ECMWF IFS open forecast.

## Setup

1. Create a Copernicus Climate Data Store account and accept the ERA5 single-level dataset licence.
2. Add GitHub Actions secrets `CDS_API_URL` and `CDS_API_KEY`.
3. Run **Build Copernicus ERA5 climatology** once from the Actions tab.
4. After `data/climatology_1961_1990.csv` has been committed, run **Daily ECMWF temperature update** manually once.
5. The daily workflow then runs at 07:30 UTC.

## Output

`data/belgium_temperature_anomaly.csv`

## Methodological warning

The daily workflow takes the maximum from the available 3-hourly IFS 2 metre-temperature samples (6-hourly after forecast hour 144). It is not guaranteed to reproduce Reuters exactly and may miss an intrastep maximum. For publication, document the point location, UTC/local-day handling, model run and sampling method.
