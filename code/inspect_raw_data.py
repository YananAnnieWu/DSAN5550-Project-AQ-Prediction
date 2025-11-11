"""
inspect_raw_data.py
Check the contents and structure of raw data files (no cleaning, no processing).
"""

import os
import json
import pandas as pd

# Set path
RAW_PATH = "data/raw"

# 1. List all files in raw data folder
print("📂 Files in data/raw/:")
for f in os.listdir(RAW_PATH):
    print("   -", f)

# 2. Inspect Google Mobility CSVs
mobility_files = [f for f in os.listdir(RAW_PATH) if f.endswith("_US_Region_Mobility_Report.csv")]
for f in mobility_files:
    path = os.path.join(RAW_PATH, f)
    print(f"\n===== {f} =====")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns[:10]), "...")
    print(df.head(3))

# 3. Inspect NOAA CSV
noaa_path = os.path.join(RAW_PATH, "noaa.csv")
if os.path.exists(noaa_path):
    print(f"\n===== noaa.csv =====")
    wx = pd.read_csv(noaa_path)
    print("Shape:", wx.shape)
    print("Columns:", list(wx.columns[:10]), "...")
    print(wx.head(3))
else:
    print("\n(noaa.csv not found)")

# 4. Inspect LA.json (Air Quality)
aqi_path = os.path.join(RAW_PATH, "LA.json")
if os.path.exists(aqi_path):
    print(f"\n===== LA.json =====")
    with open(aqi_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # print type & length
    print("Type:", type(data))
    if isinstance(data, list):
        print("List length:", len(data))
        print("Sample item keys:", list(data[0].keys()) if data else [])
    elif isinstance(data, dict):
        print("Dict keys:", list(data.keys()))
        # print one sample if has list-like content
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0:
                print(f"Sample under key '{k}':", list(v[0].keys())[:10])
                break
    else:
        print("Unrecognized structure")
else:
    print("\n(LA.json not found)")

print("\n✅ Done inspecting raw data.")