"""
clean_mobility_la.py
Google Community Mobility data cleaning.

- Filter to United States / California / Los Angeles County
- Keep only date + six mobility percentage columns
"""

import os
import glob
import pandas as pd

RAW = "data/raw"
OUT_DIR = "data/processed"
OUT = os.path.join(OUT_DIR, "mobility_la.csv")

KEEP_COLS = [
    "date",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load and concatenate all years
    files = sorted(glob.glob(os.path.join(RAW, "*_US_Region_Mobility_Report.csv")))

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    mob = pd.concat(dfs, ignore_index=True)

    # Filter to LA County
    sel = (
        (mob["country_region"] == "United States")
        & (mob["sub_region_1"] == "California")
        & (mob["sub_region_2"] == "Los Angeles County")
    )
    mob_la = mob.loc[sel].copy()
    print(f"[Mobility] After LA County filter: {mob_la.shape}")

    # Parse date
    mob_la["date"] = pd.to_datetime(mob_la["date"], errors="coerce")

    # Select columns
    mob_la = mob_la[KEEP_COLS].copy()

    # Rename
    rename_map = {
        "retail_and_recreation_percent_change_from_baseline": "retail_recreation",
        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy",
        "parks_percent_change_from_baseline": "parks",
        "transit_stations_percent_change_from_baseline": "transit",
        "workplaces_percent_change_from_baseline": "workplaces",
        "residential_percent_change_from_baseline": "residential",
    }
    mob_la.rename(columns=rename_map, inplace=True)

    print("[Mobility] Date range:", mob_la["date"].min().date(), "→", mob_la["date"].max().date())

    # Save
    mob_la.to_csv(OUT, index=False)
    print(f"[Mobility] Saved: {OUT}")

if __name__ == "__main__":
    main()