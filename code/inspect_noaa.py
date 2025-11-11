"""
inspect_noaa.py
Read-only inspection for NOAA daily station data at data/raw/noaa.csv
No cleaning, no file outputs — prints diagnostics to stdout.
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/noaa.csv"

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"File not found: {RAW_PATH}")

    print_header("1) Load & Basic Info")
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head(5))
    print("\nTail:")
    print(df.tail(5))

    print_header("2) Missingness Overview")
    na_cnt = df.isna().sum().sort_values(ascending=False)
    na_pct = (na_cnt / len(df)).round(4)
    miss = pd.DataFrame({"na_count": na_cnt, "na_pct": na_pct})
    print(miss)

    print_header("3) Numeric Columns — Descriptive Stats")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        desc = df[num_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
        print(desc)
    else:
        print("No numeric columns detected.")

    print_header("4) Categorical Columns — Cardinality & Top Values")
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    # Exclude DATE-like from categorical view for clarity
    date_like = []
    for c in cat_cols:
        lc = c.lower()
        if lc in ("date",) or "date" in lc or "time" in lc:
            date_like.append(c)
    cat_cols = [c for c in cat_cols if c not in date_like]

    if cat_cols:
        for c in cat_cols:
            nunique = df[c].nunique(dropna=True)
            print(f"\n[{c}] unique={nunique}")
            print(df[c].value_counts(dropna=False).head(10))
    else:
        print("No categorical (non-date) text columns to summarize.")

    # Try to parse a DATE column (common NOAA name is 'DATE')
    print_header("5) Date Column Checks")
    date_col = None
    for cand in ["DATE", "date", "Date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # if not found, try heuristic
        maybe = [c for c in df.columns if "date" in c.lower()]
        date_col = maybe[0] if maybe else None

    if date_col is None:
        print("No date-like column found — skipping time-series checks.")
    else:
        print(f"Using date column: {date_col}")
        dts = pd.to_datetime(df[date_col], errors="coerce")
        print(f"Parsed datetime NA count: {(dts.isna()).sum()}")

        # Basic coverage
        dt_min, dt_max = dts.min(), dts.max()
        print(f"Date range: {dt_min}  to  {dt_max}")

        # Sort by date for continuity checks (work on a copy to avoid mutating df)
        order = dts.sort_values()
        unique_dates = pd.Series(order.dropna().unique())
        # Expected daily frequency check: look for gaps of >1 day
        gaps = unique_dates.diff().dropna()
        gap_days = gaps.dt.days if hasattr(gaps, "dt") else (gaps / np.timedelta64(1, "D"))
        big_gaps_idx = np.where(gap_days > 1)[0]

        print(f"Unique dates: {len(unique_dates)}")
        print(f"Potential gaps (>1 day): {len(big_gaps_idx)}")
        if len(big_gaps_idx) > 0:
            print("First 10 gaps (prev_date -> next_date, delta_days):")
            for i in big_gaps_idx[:10]:
                prev_d = unique_dates.iloc[i]
                next_d = unique_dates.iloc[i + 1]
                delta = (next_d - prev_d).days
                print(f"  {prev_d.date()} -> {next_d.date()}  (+{delta} days)")

        # Duplicate dates check
        dup_counts = pd.Series(dts).value_counts()
        dup_dates = dup_counts[dup_counts > 1]
        print(f"Duplicate dates (count > 1): {len(dup_dates)}")
        if len(dup_dates) > 0:
            print("Top 10 duplicated dates:")
            print(dup_dates.head(10))

    print_header("6) Field-Specific Sanity Hints (No Edits)")
    hints = []
    if "PRCP" in df.columns and pd.api.types.is_numeric_dtype(df["PRCP"]):
        neg_prcp = (df["PRCP"] < 0).sum()
        hints.append(f"PRCP: negative values count = {neg_prcp}")
    if "AWND" in df.columns and pd.api.types.is_numeric_dtype(df["AWND"]):
        neg_awnd = (df["AWND"] < 0).sum()
        hi_awnd = (df["AWND"] > 30).sum()  # >30 m/s would be extreme if units are m/s; many NOAA are m/s
        hints.append(f"AWND: negatives = {neg_awnd}, extreme(>30) = {hi_awnd}")
    if "TMAX" in df.columns and "TMIN" in df.columns:
        # Just count cases where TMIN > TMAX (shouldn't happen)
        try:
            tmin_num = pd.to_numeric(df["TMIN"], errors="coerce")
            tmax_num = pd.to_numeric(df["TMAX"], errors="coerce")
            bad_temp = (tmin_num > tmax_num).sum()
            hints.append(f"TMIN>TMAX cases = {bad_temp}")
        except Exception:
            pass
    # Weather type indicators presence
    for wt in ["WT01", "WT02", "WT08"]:
        if wt in df.columns:
            wt_na = df[wt].isna().sum()
            wt_nonzero = (pd.to_numeric(df[wt], errors="coerce").fillna(0) != 0).sum()
            hints.append(f"{wt}: NA={wt_na}, nonzero={wt_nonzero}")

    if hints:
        for h in hints:
            print(" -", h)
    else:
        print("No field-specific hints available for current columns.")

    print_header("7) Station/Name Coverage")
    for col in ["STATION", "NAME"]:
        if col in df.columns:
            print(f"{col}: unique = {df[col].nunique(dropna=True)}")
            print(df[col].value_counts(dropna=False).head(5))

    print_header("8) Sample Rows (Random 5)")
    try:
        print(df.sample(5, random_state=42))
    except ValueError:
        print("Not enough rows for sampling; skipped.")

    print("\n✅ NOAA inspection completed. (No changes made)")

if __name__ == "__main__":
    main()