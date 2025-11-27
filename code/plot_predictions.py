import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"

TARGETS = [
    "NO2_mean_ppb",
    "CO_mean_ppm",
    "PM25_mean_ugm3",
]

def find_pred_file(target: str):
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("preds_") and f.endswith(f"{target}.csv"):
            return os.path.join(OUTPUT_DIR, f)
    raise FileNotFoundError(f"No prediction CSV for target {target}")

def plot_pred_vs_actual(csv_path: str, target: str):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["y_true"], label="Actual", linewidth=2)
    plt.plot(df["date"], df["y_pred"], label="Predicted", linewidth=2)

    plt.title(f"Predicted vs Actual – {target}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, f"fig_pred_vs_actual_{target}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")

def main():
    for target in TARGETS:
        try:
            csv_path = find_pred_file(target)
            print(f"\n[INFO] Plotting {target} from {csv_path}")
            plot_pred_vs_actual(csv_path, target)
        except Exception as e:
            print(f"[ERROR] {target}: {e}")

if __name__ == "__main__":
    main()