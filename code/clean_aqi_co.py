"""
clean_aqi_co.py
EPA CO daily data cleaning.

- Keep only pollutant_standard == "CO 8-hour 1971"
- Drop unneeded columns
- Aggregate across all sites per date: mean / median / max (from arithmetic_mean)
"""

import os
import json
import pandas as pd

RAW = "data/raw"
OUT_DIR = "data/processed"
OUT = os.path.join(OUT_DIR, "aqi_co_daily.csv")

YEARS = [2020, 2021, 2022]

def load_co_json(year):
    path = os.path.join(RAW, f"CO_{year}.json")
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("Data", []))
    df["source_year"] = year
    print(f"[{year}] Loaded: {df.shape}")
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dfs = [load_co_json(y) for y in YEARS]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[ALL] Combined shape: {df.shape}")

    df = df[df["pollutant_standard"] == "CO 8-hour 1971"].copy()
    print(f"[Filter] CO 8-hour 2011 only: {df.shape}")

    df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce")
    numeric_cols = ["arithmetic_mean", "aqi"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [
        "date_local", "site_number", "poc",
        "pollutant_standard",
        "arithmetic_mean", "aqi", "validity_indicator", "local_site_name"
    ]
    df = df[keep].copy()

    grouped = (df.groupby("date_local")
                 .agg(
                     CO_mean_ppm=("arithmetic_mean", "mean"),
                     CO_median_ppm=("arithmetic_mean", "median"),
                     CO_max_ppm=("arithmetic_mean", "max"),
                     AQI_mean=("aqi", "mean")
                 )
                 .reset_index()
                 .sort_values("date_local"))

    print(f"[Final] Shape: {grouped.shape}")
    print(f"Date range: {grouped['date_local'].min().date()} → {grouped['date_local'].max().date()}")

    print(grouped.head())

    grouped.to_csv(OUT, index=False)
    print(f"[AQS] Saved: {OUT}")

if __name__ == "__main__":
    main()