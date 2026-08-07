from pathlib import Path
import pandas as pd
p = Path("data/belgium_temperature_anomaly.csv")
df = pd.read_csv(p)
required = {"date", "forecast_max_c", "historic_norm_c", "anomaly_c", "status", "latitude", "longitude"}
missing = required - set(df.columns)
assert not missing, f"Missing columns: {sorted(missing)}"
assert not df.empty, "CSV is empty"
assert df["date"].is_unique, "Duplicate dates"
assert pd.to_datetime(df["date"], errors="coerce").notna().all(), "Invalid dates"
calc = (df["forecast_max_c"] - df["historic_norm_c"]).round(2)
assert (calc == df["anomaly_c"].round(2)).all(), "Anomaly calculation mismatch"
assert df["anomaly_c"].between(-30, 30).all(), "Implausible anomaly"
print(f"Validated {len(df)} rows")
