"""
ml.py
ML for AQI (or any target) using mobility + weather features.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import random

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATE_COL = "date"
DEFAULT_TARGET = "AQI_mean_NO2"

@dataclass
class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float
    best_params: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    feature_importance: Optional[pd.DataFrame] = None


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

def time_based_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    return train, test

def build_feature_list(df: pd.DataFrame) -> List[str]:
    return [
        "retail_recreation", "grocery_pharmacy", "parks",
        "transit", "workplaces", "residential",
        "TMAX_C", "TMIN_C", "TAVG_C", "PRCP_mm",
        "AWND", "WT01", "WT02", "WT08"
    ]

def make_preprocessor(numeric_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_features)],
        remainder="drop"
    )

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def extract_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str]) -> List[str]:
    return list(numeric_features)

def compute_feature_importance(best_est, name, preprocessor, predictors, X_ref, y_ref):
    feat_names = extract_feature_names(preprocessor, predictors)
    model = best_est.named_steps["model"]
    if name in ("Lasso", "Ridge", "ElasticNet", "LinearRegression"):
        coefs = getattr(model, "coef_", None)
        if coefs is not None:
            df = pd.DataFrame({"feature": feat_names, "coefficient": np.ravel(coefs)})
            df["abs_coef_rank"] = df["coefficient"].abs().rank(ascending=False, method="dense")
            return df.sort_values("abs_coef_rank").reset_index(drop=True)
    if name == "RandomForest" and hasattr(model, "feature_importances_"):
        return (
            pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
        )
    if name == "XGBoost":
        try:
            booster = model.get_booster()
            importance = booster.get_score(importance_type="gain")
            mapped = {feat_names[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")}
            df = pd.DataFrame({"feature": list(mapped.keys()), "importance": list(mapped.values())})
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            pass
    if name == "SVR" and getattr(model, "kernel", None) == "linear" and hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        df = pd.DataFrame({"feature": feat_names, "coefficient": coefs})
        df["abs_coef_rank"] = df["coefficient"].abs().rank(ascending=False, method="dense")
        return df.sort_values("abs_coef_rank").reset_index(drop=True)

def run_ml(csv_path: str = "data/processed/combined_daily.csv",
           target_col: str = DEFAULT_TARGET,
           extra_features: Optional[List[str]] = None,
           save_dir: str = "outputs",
           random_state: int = 42) -> Dict[str, Any]:

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    lag1_col = f"{target_col}_lag1"
    df[lag1_col] = df[target_col].shift(1)
    df = df.dropna(subset=[lag1_col]).reset_index(drop=True)

    predictors = build_feature_list(df) + [lag1_col]

    train_df, test_df = time_based_split(df, test_ratio=0.2)

    X_train = train_df[predictors].copy()
    y_train = train_df[target_col].astype(float).values
    X_test = test_df[predictors].copy()
    y_test = test_df[target_col].astype(float).values

    preprocessor = make_preprocessor(predictors)

    models_and_grids = {
        "LinearRegression": (
            Pipeline([
                ("prep", preprocessor),
                ("model", LinearRegression())
            ]),
            {}
        ),
        "Lasso": (
            Pipeline([
                ("prep", preprocessor),
                ("model", Lasso(max_iter=10000, tol=1e-3, random_state=random_state))
            ]),
            {
                "model__alpha": np.logspace(-4, -1, 15)
            }
        ),
        "RandomForest": (
            Pipeline([
                ("prep", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=800,
                    random_state=random_state,
                    n_jobs=-1
                ))
            ]),
            {
                "model__max_depth": [10, 14, 18],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": ["sqrt", 0.5]
            }
        ),
        "XGBoost": (
            Pipeline([
                ("prep", preprocessor),
                ("model", XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0
                ))
            ]),
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        ),
        "Ridge": (
            Pipeline([
                ("prep", preprocessor),
                ("model", Ridge())
            ]),
            {
                "model__alpha": np.logspace(-3, 1, 10)
            }
        ),
        "ElasticNet": (
            Pipeline([
                ("prep", preprocessor),
                ("model", ElasticNet(max_iter=50000, random_state=random_state))
            ]),
            {
                "model__alpha": np.logspace(-3, 1, 10),
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        ),
        "SVR": (
            Pipeline([
                ("prep", preprocessor),
                ("model", SVR())
            ]),
            {
                "model__kernel": ["rbf", "linear"],
                "model__C": [0.1, 1, 10, 50],
                "model__gamma": ["scale", 0.1, 0.01]
            }
        )
    }

    tscv = TimeSeriesSplit(n_splits=10)

    results: List[ModelResult] = []

    for name, (pipe, grid) in models_and_grids.items():
        if grid:
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring="neg_root_mean_squared_error",
                cv=tscv,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            best_params = search.best_params_
        else:
            best_est = pipe.fit(X_train, y_train)
            best_params = {}

        y_pred = best_est.predict(X_test)
        mets = evaluate(y_test, y_pred)

        feat_imp_df = compute_feature_importance(
            best_est=best_est,
            name=name,
            preprocessor=preprocessor,
            predictors=predictors,
            X_ref=X_test,
            y_ref=y_test
        )

        results.append(ModelResult(
            name=name,
            rmse=mets["rmse"],
            mae=mets["mae"],
            r2=mets["r2"],
            best_params=best_params,
            y_true=y_test,
            y_pred=y_pred,
            feature_importance=feat_imp_df
        ))

    results_sorted = sorted(results, key=lambda r: r.rmse)
    best = results_sorted[0]

    metrics_table = pd.DataFrame([{
        "model": r.name, "rmse": r.rmse, "mae": r.mae, "r2": r.r2, "best_params": json.dumps(r.best_params)
    } for r in results_sorted]).sort_values("rmse")

    metrics_path = os.path.join(save_dir, f"metrics_{target_col}.csv")
    metrics_table.to_csv(metrics_path, index=False)

    preds_path = os.path.join(save_dir, f"preds_{best.name}_{target_col}.csv")
    pd.DataFrame({
        "date": test_df[DATE_COL].values,
        "y_true": best.y_true,
        "y_pred": best.y_pred
    }).to_csv(preds_path, index=False)

    if best.feature_importance is not None:
        fi_path = os.path.join(save_dir, f"importance_{best.name}_{target_col}.csv")
        best.feature_importance.to_csv(fi_path, index=False)
    else:
        fi_path = None

    print(f"\n=== Target: {target_col} ===")
    print(metrics_table.to_string(index=False))
    print(f"\nBest model: {best.name}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved predictions -> {preds_path}")
    if fi_path:
        print(f"Saved importance -> {fi_path}")

    return {
        "target": target_col,
        "predictors": predictors,
        "metrics_csv": metrics_path,
        "preds_csv": preds_path,
        "importance_csv": fi_path,
        "leaderboard": metrics_table,
        "best_model": best.name
    }

def run_ols_significance(csv_path: str, target_col: str):
    import statsmodels.api as sm
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    lag_col = f"{target_col}_lag1"
    df[lag_col] = df[target_col].shift(1)
    predictors = build_feature_list(df) + [lag_col]
    df_ols = df.dropna(subset=predictors + [target_col]).reset_index(drop=True)
    X = sm.add_constant(df_ols[predictors])
    y = df_ols[target_col].astype(float)
    ols_model = sm.OLS(y, X).fit()
    print(ols_model.summary())
    return ols_model


if __name__ == "__main__":
    # MAIN_TARGET = "NO2_mean_ppb"
    MAIN_TARGET = "CO_mean_ppm"
    # MAIN_TARGET = "PM25_mean_ugm3"

    run_ml("data/processed/combined_daily.csv", target_col=MAIN_TARGET)
    print("\n[INFO] Running OLS significance test...\n")
    run_ols_significance("data/processed/combined_daily.csv", target_col=MAIN_TARGET)