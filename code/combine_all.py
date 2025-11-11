"""
combine_all.py
Combine processed NOAA, Mobility, and multiple pollutant daily data.
"""

import os
import re
import glob
import pandas as pd

PROC_DIR = "data/processed"
OUT_PATH = os.path.join(PROC_DIR, "combined_daily.csv")

def load_noaa():
    path = os.path.join(PROC_DIR, "noaa_clean.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"DATE": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def load_mobility():
    path = os.path.join(PROC_DIR, "mobility_la.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def load_pollutants():
    paths = sorted(glob.glob(os.path.join(PROC_DIR, "aqi_*_daily.csv")))
    out = []
    for p in paths:
        df = pd.read_csv(p)
        df = df.rename(columns={"date_local": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        tag_match = re.match(r"aqi_(.+?)_daily\.csv$", os.path.basename(p), re.IGNORECASE)
        tag = tag_match.group(1).upper()

        renamed = {}
        for c in df.columns:
            if c == "date":
                continue
            if c == "AQI_mean":
                renamed[c] = f"AQI_mean_{tag}"
            elif c == "first_max_hour_mode":
                renamed[c] = f"{tag}_first_max_hour_mode"
        if renamed:
            df = df.rename(columns=renamed)
        out.append(df)
    return out

def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    noaa = load_noaa()
    mob = load_mobility()
    pols = load_pollutants()

    pieces = []
    pieces.append(("MOB", mob))
    pieces.append(("NOAA", noaa))
    for i, d in enumerate(pols or [], 1):
        pieces.append((f"POL{i}", d))

    combined = None
    for name, df in pieces:
        if combined is None:
            combined = df.copy()
        else:
            before = len(combined)
            combined = combined.merge(df, on="date", how="inner")

    combined = combined.sort_values("date").reset_index(drop=True)

    print("\n===== SUMMARY =====")
    print(f"Final shape: {combined.shape}")
    if not combined.empty:
        print(f"Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")

    print("\nHead:")
    print(combined.head(5))

    combined.to_csv(OUT_PATH, index=False)
    print(f"\n[SAVE] {OUT_PATH}")

if __name__ == "__main__":
    main()