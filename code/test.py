import os, json, pandas as pd

RAW = "data/raw"
YEARS = [2020, 2021, 2022]

vals = set()
for y in YEARS:
    path = os.path.join(RAW, f"CO_{y}.json")
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("Data", []))
    if "pollutant_standard" in df.columns:
        vals.update(df["pollutant_standard"].dropna().unique().tolist())

print(vals)
