"""
ml_lstm.py
Multivariate sequence model for AQI using mobility + weather (+ target itself).

"""

import os
import math
import json
import random
import numpy as np
import pandas as pd

from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, models


# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


DATE_COL = "date"
TARGET = "AQI_mean_NO2"
CSV_PATH = "data/processed/combined_daily.csv"
SAVE_DIR = "outputs"

WINDOW = 7
TEST_RATIO = 0.15
VAL_RATIO  = 0.15

MOBILITY_FEATURES = [
    "retail_recreation", "grocery_pharmacy", "parks",
    "transit", "workplaces", "residential"
]
WEATHER_FEATURES = [
    "TMAX_C", "TMIN_C", "TAVG_C", "PRCP_mm",
    "AWND", "WT01", "WT02", "WT08"
]


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

def build_feature_list(df: pd.DataFrame) -> List[str]:
    feats = [c for c in MOBILITY_FEATURES + WEATHER_FEATURES if c in df.columns]
    return feats

def chronological_splits(n: int, test_ratio: float, val_ratio: float) -> Tuple[int, int]:
    train_end = int(n * (1 - test_ratio - val_ratio))
    val_end = int(n * (1 - test_ratio))
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    return train_end, val_end

def make_sequences(X2d: np.ndarray, y1d: np.ndarray, window: int):
    X_seq, y_seq = [], []
    for t in range(window, len(X2d)):
        X_seq.append(X2d[t - window:t, :])
        y_seq.append(y1d[t])
    return np.asarray(X_seq), np.asarray(y_seq)

def build_model(n_steps: int, n_feats: int) -> tf.keras.Model:
    inp = layers.Input(shape=(n_steps, n_feats))
    x = layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.GRU(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"]
    )
    return model


def run_lstm(csv_path: str = CSV_PATH,
             target_col: str = TARGET,
             save_dir: str = SAVE_DIR,
             window: int = WINDOW,
             test_ratio: float = TEST_RATIO,
             val_ratio: float = VAL_RATIO) -> dict:

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # seasonality dummies
    df["month"] = df[DATE_COL].dt.month
    df["dow"] = df[DATE_COL].dt.dayofweek
    df = pd.get_dummies(df, columns=["month", "dow"], drop_first=True)
    season_cols = [c for c in df.columns if c.startswith("month_") or c.startswith("dow_")]

    predictors = build_feature_list(df) + [target_col] + season_cols

    X_all = df[predictors].astype("float32").values
    y_all = df[target_col].astype("float32").values.reshape(-1, 1)

    n = len(df)
    train_end, val_end = chronological_splits(n, test_ratio, val_ratio)
    train_idx = slice(0, train_end)
    val_idx = slice(train_end, val_end)
    test_idx = slice(val_end, n)

    X_train_raw, y_train_raw = X_all[train_idx], y_all[train_idx]
    X_val_raw, y_val_raw = X_all[val_idx], y_all[val_idx]
    X_test_raw, y_test_raw = X_all[test_idx], y_all[test_idx]

    x_scaler = StandardScaler().fit(X_train_raw)
    y_scaler = StandardScaler().fit(y_train_raw)

    X_train = x_scaler.transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test  = x_scaler.transform(X_test_raw)

    y_train = y_scaler.transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    Xtr, ytr = make_sequences(X_train, y_train, window)
    Xva, yva = make_sequences(X_val, y_val, window)
    Xte, yte = make_sequences(X_test, y_test, window)

    n_steps = Xtr.shape[1]
    n_feats = Xtr.shape[2]
    model = build_model(n_steps, n_feats)
    model.summary(print_fn=lambda x: print("[MODEL] " + x))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=0)

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=500,
        batch_size=16,
        callbacks=[es, rlrop],
        verbose=0,
        shuffle=False
    )

    # Evaluate
    y_pred_scaled = model.predict(Xte, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = y_scaler.inverse_transform(yte).ravel()

    y_true_full = y_scaler.inverse_transform(y_test).ravel()
    naive = y_true_full[window-1:]
    naive = naive[:len(y_true)]
    print("[Naive] RMSE:", rmse(y_true, naive))

    metrics = {
        "model": "LSTM",
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "window": window,
        "n_train_seq": int(Xtr.shape[0]),
        "n_val_seq": int(Xva.shape[0]),
        "n_test_seq": int(Xte.shape[0]),
        "predictors": len(predictors),
    }

    dates_test = df.iloc[val_end + window : val_end + window + len(y_true)][DATE_COL].values

    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(save_dir, f"metrics_LSTM_{target_col}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    preds_df = pd.DataFrame({
        DATE_COL: dates_test,
        "y_true": y_true,
        "y_pred": y_pred
    })
    preds_path = os.path.join(save_dir, f"preds_LSTM_{target_col}.csv")
    preds_df.to_csv(preds_path, index=False)

    print("\n=== LSTM Results ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics -> {metrics_path}")
    print(f"Saved predictions -> {preds_path}")

    meta = {
        "target": target_col,
        "predictor_columns": predictors,
        "window": window,
        "splits": {"train_end": train_end, "val_end": val_end, "test_start": val_end}
    }
    with open(os.path.join(save_dir, f"lstm_meta_{target_col}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "metrics_csv": metrics_path,
        "preds_csv": preds_path,
        "meta_json": os.path.join(save_dir, f"lstm_meta_{target_col}.json"),
        "metrics": metrics
    }

if __name__ == "__main__":
    run_lstm(CSV_PATH, TARGET, SAVE_DIR, WINDOW, TEST_RATIO, VAL_RATIO)