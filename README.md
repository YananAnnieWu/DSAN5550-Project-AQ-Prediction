# DSAN5550 Project - Air Quality Prediction in Los Angeles

## Overview
This project aims to predict daily air quality indicators (e.g., PM2.5, NO₂, CO) in Los Angeles by integrating **meteorological**, **human mobility**, and **pollution** datasets.
The goal is to build short-term machine learning models that capture rapid changes in air quality and explore how mobility patterns (especially during 2020-2022) affected pollution levels.

## Data Sources
- **Air Quality:** EPA Air Quality System (AQS) - Los Angeles station (2020-2022)
- **Weather:** NOAA daily meteorological data
- **Mobility:** Google COVID-19 Community Mobility Reports (Los Angeles County, 2020-2022)

## Project Structure
DSAN5550-Project-AQ-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── code/
│   ├── clean_*.py           # Data cleaning
│   ├── combine_all.py       # Data integration
│   ├── eda_*.py             # Exploratory data analysis
│   ├── ml.py                # Tabular ML models (Ridge, Lasso, XGBoost, etc.)
│   ├── ml_lstm.py           # LSTM deep learning model
│   ├── run_codecarbon.py    # Carbon footprint tracking for all experiments
│   ├── plot_predictions.py  # Predicted vs actual time-series plots
│
├── outputs/
│   ├── metrics_*.csv        # Model performance tables
│   ├── preds_*.csv          # Model predictions
│   ├── importance_*.csv     # Feature importance
│   ├── carbon/              # CodeCarbon emissions log
│   └── *.png                # Plots
│
└── README.md

## Methods
1. All models are trained with:

- Time-series aware train/test split

- 10-fold TimeSeriesSplit cross-validation

- Hyperparameter tuning via GridSearchCV

- Standardized numeric features via ColumnTransformer


2. Models implemented:

- Linear Regression

- Ridge

- Lasso

- ElasticNet

- Random Forest Regressor

- XGBoost Regressor

- Support Vector Regression (SVR)

- LSTM


3. Carbon Footprint Tracking:

- All experiments (tabular models and LSTM) are instrumented using CodeCarbon.

## Author
Yanan (Annie) Wu - Georgetown University  
DSAN 5550: Climate Change and Data Science