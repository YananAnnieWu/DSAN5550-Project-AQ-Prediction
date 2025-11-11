"""
clean_noaa.py
NOAA daily station data cleaning.

- Build TAVG: (TMAX + TMIN) / 2
- Convert units: temperatures (°F->°C), precipitation (inch->mm)
- Missing value of wind speed AWND: linear interpolate
- Save to data/processed/noaa_clean.csv
"""

import os
import pandas as pd
import numpy as np

RAW = "data/raw/noaa.csv"
OUTDIR = "data/processed"
OUT = os.path.join(OUTDIR, "noaa_clean.csv")

def f_to_c(x):
    """Fahrenheit to Celsius"""
    return (x - 32.0) / 1.8

def inch_to_mm(x):
    """Inch to millimeters"""
    return x * 25.4

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(RAW)
    print(f"[NOAA] Raw shape: {df.shape}")

    # Date
    date_col = "DATE"
    df["DATE"] = pd.to_datetime(df[date_col], errors="coerce")

    # Numeric variables
    for c in ["TMAX", "TMIN", "TAVG", "PRCP", "AWND", "WT01", "WT02", "WT08"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build TAVG
    df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2.0
    print("[NOAA] TAVG constructed from (TMAX + TMIN) / 2.")

    # Unit conversions
    df["TMAX_F"] = df["TMAX"]
    df["TMAX_C"] = f_to_c(df["TMAX_F"])
    df["TMIN_F"] = df["TMIN"]
    df["TMIN_C"] = f_to_c(df["TMIN_F"])
    df["TAVG_F"] = df["TAVG"]
    df["TAVG_C"] = f_to_c(df["TAVG_F"])

    df["PRCP_in"] = df["PRCP"]
    df["PRCP_mm"] = inch_to_mm(df["PRCP_in"])

    df["AWND"] = df["AWND"].interpolate(limit_direction="both")
    print(f"[NOAA] AWND interpolated. Remaining NaN: {df['AWND'].isna().sum()}")

    # Weather type indicators
    for wt in ["WT01", "WT02", "WT08"]:
        if wt in df.columns:
            df[wt] = pd.to_numeric(df[wt], errors="coerce")
            df[wt] = df[wt].fillna(0).astype(int)

    # Data range
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2022-12-31")
    before = len(df)
    df = df[(df["DATE"] >= start) & (df["DATE"] <= end)].copy()
    df = df.sort_values("DATE").reset_index(drop=True)
    print(f"[NOAA] Clipped to {start.date()} ~ {end.date()}: {before} -> {len(df)} rows")

    # Select columns
    keep_cols = [
        "DATE",
        "TMAX_C", "TMIN_C", "TAVG_C",
        "PRCP_mm",
        "AWND",
        "WT01", "WT02", "WT08",
    ]

    df_out = df[keep_cols].copy()

    print(df_out.head(5))

    # Save
    df_out.to_csv(OUT, index=False)
    print(f"[NOAA] Saved: {OUT}")

if __name__ == "__main__":
    main()