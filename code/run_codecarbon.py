#!/usr/bin/env python3

import os
import sys
from typing import List

from codecarbon import EmissionsTracker

from ml import run_ml, run_ols_significance
from ml_lstm import run_lstm

DATA_PATH = "data/processed/combined_daily.csv"
OUTPUT_DIR = "outputs"
CARBON_DIR = os.path.join(OUTPUT_DIR, "carbon")

DEFAULT_TABULAR_TARGETS: List[str] = [
    "AQI_mean_NO2",
    "CO_mean_ppm",
    "PM25_mean_ugm3",
]

DEFAULT_LSTM_TARGET = "AQI_mean_NO2"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CARBON_DIR, exist_ok=True)


def run_tabular_with_carbon(target: str):
    tracker = EmissionsTracker(
        project_name=f"tabular_{target}",
        output_dir=CARBON_DIR,
        save_to_file=True,
    )
    tracker.start()

    ml_result = run_ml(csv_path=DATA_PATH, target_col=target)

    ols_model = run_ols_significance(csv_path=DATA_PATH, target_col=target)

    emissions_kg = tracker.stop()
    print(f"[CARBON] Tabular run for {target} emitted ~{emissions_kg:.6f} kg CO₂eq.\n")

    return {
        "target": target,
        "emissions_kg": emissions_kg,
        "ml_result": ml_result,
        "ols_model": ols_model,
    }


def run_lstm_with_carbon(target: str):
    tracker = EmissionsTracker(
        project_name=f"lstm_{target}",
        output_dir=CARBON_DIR,
        save_to_file=True,
    )
    tracker.start()

    lstm_result = run_lstm(
        csv_path=DATA_PATH,
        target_col=target,
        save_dir=OUTPUT_DIR,
    )

    emissions_kg = tracker.stop()
    print(f"[CARBON] LSTM run for {target} emitted ~{emissions_kg:.6f} kg CO₂eq.\n")

    return {
        "target": target,
        "emissions_kg": emissions_kg,
        "lstm_result": lstm_result,
    }


def main():
    ensure_dirs()

    mode = None
    if len(sys.argv) >= 2:
        mode = sys.argv[1].lower()

    if mode is None:
        for t in DEFAULT_TABULAR_TARGETS:
            run_tabular_with_carbon(t)
        run_lstm_with_carbon(DEFAULT_LSTM_TARGET)

    elif mode == "tabular":
        for t in DEFAULT_TABULAR_TARGETS:
            run_tabular_with_carbon(t)

    elif mode == "lstm":
        run_lstm_with_carbon(DEFAULT_LSTM_TARGET)

    else:
        print(f"[WARN] Unknown mode: {mode}")
        print("Usage: python run_with_codecarbon.py [tabular|lstm]")
        sys.exit(1)


if __name__ == "__main__":
    main()