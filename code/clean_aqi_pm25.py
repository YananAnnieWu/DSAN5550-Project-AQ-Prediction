"""
clean_aqi_pm25.py
EPA PM2.5 daily data cleaning.

- Keep only pollutant_standard == "PM25 24-hour 2012"
- Drop unneeded columns
- Aggregate across all sites per date: mean / median / max (from arithmetic_mean)
"""


import os
import json
import pandas as pd

RAW = "data/raw"
OUT_DIR = "data/processed"
OUT = os.path.join(OUT_DIR, "aqi_pm25_daily.csv")

YEARS = [2020, 2021, 2022]

def load_pm25_json(year):
    path = os.path.join(RAW, f"PM25_{year}.json")
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("Data", []))
    df["source_year"] = year
    print(f"[{year}] Loaded: {df.shape}")
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dfs = [load_pm25_json(y) for y in YEARS]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[ALL] Combined shape: {df.shape}")

    df = df[df["pollutant_standard"] == "PM25 24-hour 2012"].copy()
    print(f"[Filter] PM2.5 24-hour 2012 only: {df.shape}")

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
                     PM25_mean_ugm3=("arithmetic_mean", "mean"),
                     PM25_median_ugm3=("arithmetic_mean", "median"),
                     PM25_max_ugm3=("arithmetic_mean", "max"),
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