"""
inspect_mobility_la.py
Read-only inspection for Google Community Mobility (LA County only).

- Load 2020/2021/2022 *_US_Region_Mobility_Report.csv from data/raw/
- Filter to United States / California / Los Angeles County
- Concatenate years
- Inspect schema, missingness, date coverage, duplicates by date, and basic stats
- NO cleaning, NO writing outputs
"""

import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PATTERN = os.path.join(RAW_DIR, "*_US_Region_Mobility_Report.csv")

KEEP_COLS = [
    "country_region_code","country_region","sub_region_1","sub_region_2",
    "metro_area","iso_3166_2_code","census_fips_code","place_id","date",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

def print_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No mobility CSVs found under {PATTERN}")
    print_header("1) Files detected")
    for f in files:
        print(" -", os.path.basename(f))

    # Load + select columns (keep superset to avoid surprises)
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
    mob = pd.concat(dfs, ignore_index=True)

    print_header("2) Raw (pre-filter) snapshot")
    print("Shape:", mob.shape)
    print("Columns:", list(mob.columns))
    print(mob.head(3))

    # Filter to LA County, California, United States
    print_header("3) Apply LA County filter")
    sel = pd.Series(True, index=mob.index)

    if "country_region" in mob.columns:
        sel &= (mob["country_region"] == "United States")
    if "sub_region_1" in mob.columns:
        sel &= (mob["sub_region_1"] == "California")
    if "sub_region_2" in mob.columns:
        sel &= (mob["sub_region_2"] == "Los Angeles County")

    mob_la = mob.loc[sel].copy()
    print("After filter shape:", mob_la.shape)

    # Basic region sanity
    for col in ["country_region","sub_region_1","sub_region_2","metro_area","iso_3166_2_code"]:
        if col in mob_la.columns:
            uniq = mob_la[col].dropna().unique()
            print(f" - {col}: {len(uniq)} unique -> {uniq[:5]}")

    # Parse date safely
    print_header("4) Date parsing & coverage")
    mob_la["date"] = pd.to_datetime(mob_la["date"], errors="coerce")
    na_dates = mob_la["date"].isna().sum()
    print("date parse NA:", na_dates)
    mob_la = mob_la.dropna(subset=["date"]).sort_values("date")

    if not mob_la.empty:
        print("Date range:", mob_la["date"].min().date(), "->", mob_la["date"].max().date())
        # Check duplicates per date
        dup_by_date = mob_la.groupby("date").size().rename("rows_per_date")
        multi = (dup_by_date > 1).sum()
        print("Unique dates:", dup_by_date.shape[0])
        print("Dates with >1 row:", int(multi))
        if multi > 0:
            print("Sample multi-row dates:")
            print(dup_by_date[dup_by_date > 1].head(10))

        # Gap check (>1 day)
        unique_dates = dup_by_date.index
        gaps = pd.Series(unique_dates).diff().dropna()
        gap_days = gaps.dt.days
        big_gaps = (gap_days > 1).sum()
        print("Gaps > 1 day:", int(big_gaps))
        if big_gaps:
            idx = np.where(gap_days.values > 1)[0]
            print("First few gaps:")
            for i in idx[:10]:
                prev_d = unique_dates[i]
                next_d = unique_dates[i+1]
                print(f"  {prev_d.date()} -> {next_d.date()} (+{(next_d-prev_d).days} days)")
    else:
        print("No rows after filtering to LA County.")

    # Missingness and basic stats (on LA-only)
    print_header("5) Missingness (LA-only)")
    miss = mob_la.isna().sum().sort_values(ascending=False)
    print(miss.head(20))

    print_header("6) Descriptive stats for mobility columns")
    mob_cols = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    mob_cols = [c for c in mob_cols if c in mob_la.columns]
    if mob_cols:
        # ensure numeric
        for c in mob_cols:
            mob_la[c] = pd.to_numeric(mob_la[c], errors="coerce")
        desc = mob_la[mob_cols].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).T
        print(desc)
    else:
        print("No mobility percent-change columns present.")

    print_header("7) Rows sample (LA-only)")
    print(mob_la.head(3))
    print(mob_la.tail(3))

    print("\n✅ Mobility (LA) inspection completed. No files written.")

if __name__ == "__main__":
    main()