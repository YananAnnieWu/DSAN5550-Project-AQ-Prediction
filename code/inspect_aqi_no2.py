"""
inspect_aqi_no2.py
Read-only inspection for NO2 daily data (2020–2022, Los Angeles County).

- Reads data/raw/NO2_2020.json, NO2_2021.json, NO2_2022.json
- Extracts 'Data' part of each JSON
- Combines into one DataFrame
- Reports date range, pollutant standards, units, missingness, duplicates
- No file writing
"""

import os
import json
import pandas as pd

RAW_DIR = "data/raw"
FILES = [f"NO2_{y}.json" for y in [2020, 2021, 2022]]

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    data = j.get("Data", [])
    return pd.DataFrame(data)

def main():
    dfs = []
    for fn in FILES:
        fp = os.path.join(RAW_DIR, fn)
        if not os.path.exists(fp):
            print(f"[WARN] Missing file: {fp}")
            continue
        print_header(f"Loading {fn}")
        df = load_json(fp)
        print("Shape:", df.shape)
        print("Columns:", list(df.columns[:15]))
        if "parameter" in df.columns:
            print("Parameters (head):", df["parameter"].unique()[:5])
        if "units_of_measure" in df.columns:
            print("Units:", df["units_of_measure"].unique())
        dfs.append(df)

    if not dfs:
        print("No JSONs found, abort.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    print_header("Combined summary")
    print("Total rows:", len(df_all))
    print("Columns:", list(df_all.columns))

    # Parse date
    if "date_local" not in df_all.columns:
        raise ValueError("No 'date_local' column found")
    df_all["date_local"] = pd.to_datetime(df_all["date_local"], errors="coerce")

    print("Date range:", df_all["date_local"].min().date(), "→", df_all["date_local"].max().date())

    # Pollutant standard breakdown
    if "pollutant_standard" in df_all.columns:
        print("\nPollutant standards:")
        print(df_all["pollutant_standard"].value_counts().head(10))

    # Missingness
    print_header("Missingness")
    print(df_all.isna().sum().sort_values(ascending=False).head(15))

    # Station info
    if "local_site_name" in df_all.columns:
        print("\nUnique sites:", df_all["local_site_name"].nunique())
        print("Sample sites:", df_all["local_site_name"].unique()[:10])

    # Duplicates per day-site-POC
    dup_mask = df_all.duplicated(subset=["date_local", "site_number", "poc"], keep=False)
    dup_rows = df_all.loc[dup_mask]
    print(f"\nDuplicates (same date_local + site_number + poc): {len(dup_rows)} rows")
    if len(dup_rows) > 0:
        print(dup_rows[["date_local", "site_number", "poc", "pollutant_standard"]].head(8))

    # Quick stats for arithmetic_mean
    if "arithmetic_mean" in df_all.columns:
        df_all["arithmetic_mean"] = pd.to_numeric(df_all["arithmetic_mean"], errors="coerce")
        print_header("Arithmetic_mean summary")
        print(df_all["arithmetic_mean"].describe(percentiles=[0.01, 0.5, 0.99]))

    print_header("Sample rows")
    print(df_all.head(5))

    print("\n✅ NO2 inspection completed. No files written.")

if __name__ == "__main__":
    main()