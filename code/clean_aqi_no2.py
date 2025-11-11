"""
clean_aqi_no2.py
EPA NO2 daily data cleaning.

- Keep only pollutant_standard == "NO2 1-hour 2010"
- Drop unneeded columns
- Aggregate across all sites per date: mean / median / max (from arithmetic_mean)
- Compute mean AQI and mode of first_max_hour
"""

import os
import json
import pandas as pd

RAW = "data/raw"
OUT_DIR = "data/processed"
OUT = os.path.join(OUT_DIR, "aqi_no2_daily.csv")

YEARS = [2020, 2021, 2022]

def load_no2_json(year):
    path = os.path.join(RAW, f"NO2_{year}.json")
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("Data", []))
    df["source_year"] = year
    print(f"[{year}] Loaded: {df.shape}")
    return df

def mode_or_na(s: pd.Series):
    m = s.mode()
    # if all values unique: NaN
    if len(m) == len(s.unique()):
        return pd.NA
    return m.iloc[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    dfs = [load_no2_json(y) for y in YEARS]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[ALL] Combined shape: {df.shape}")

    # Keep only NO2 1-hour 2010
    df = df[df["pollutant_standard"] == "NO2 1-hour 2010"].copy()
    print(f"[Filter] NO2 1-hour 2010 only: {df.shape}")

    # Convert date & numeric
    df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce")
    numeric_cols = ["arithmetic_mean", "aqi", "first_max_hour"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Select columns
    keep = [
        "date_local", "site_number", "poc",
        "pollutant_standard",
        "arithmetic_mean", "first_max_hour",
        "aqi", "validity_indicator", "local_site_name"
    ]
    df = df[keep].copy()

    # City-level daily aggregation
    grouped = (df.groupby("date_local")
                 .agg(
                     NO2_mean_ppb=("arithmetic_mean", "mean"),
                     NO2_median_ppb=("arithmetic_mean", "median"),
                     NO2_max_ppb=("arithmetic_mean", "max"),
                     AQI_mean=("aqi", "mean"),
                     first_max_hour_mode=("first_max_hour", mode_or_na)
                 )
                 .reset_index()
                 .sort_values("date_local"))

    # Summary
    print(f"[Final] Shape: {grouped.shape}")
    print(f"Date range: {grouped['date_local'].min().date()} → {grouped['date_local'].max().date()}")
    nan_count = grouped["first_max_hour_mode"].isna().sum()
    nan_pct = nan_count / len(grouped)
    print(f"[first_max_hour_mode] NaN count: {nan_count} ({nan_pct:.2%})")

    print(grouped.head())

    # Save
    grouped.to_csv(OUT, index=False)
    print(f"[AQS] Saved: {OUT}")

if __name__ == "__main__":
    main()