"""
Part 8b: Income Forecast Model Training
Trains two RandomForestRegressor models that predict a
"revenue_index" and "expense_index" — growth multipliers relative to a
particular farm's current monthly revenue/expenses, based on the
farm profile + how many months ahead (month_offset) + seasonality.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from timeseries_generator import simulate_timeseries, FORECAST_FEATURE_COLUMNS

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
FC_REV_PATH = os.path.join(MODEL_DIR, "forecast_revenue_model.pkl")
FC_EXP_PATH = os.path.join(MODEL_DIR, "forecast_expense_model.pkl")
FC_SCALER_PATH = os.path.join(MODEL_DIR, "forecast_scaler.pkl")


def train_and_save_forecast_models(n_farms=1000, horizon=36, verbose=True):
    df = simulate_timeseries(n_farms=n_farms, horizon=horizon)

    X = df[FORECAST_FEATURE_COLUMNS]
    y_rev = df["revenue_index"]
    y_exp = df["expense_index"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, yr_train, yr_test, ye_train, ye_test = train_test_split(
        X_scaled, y_rev, y_exp, test_size=0.2, random_state=42
    )

    rev_model = RandomForestRegressor(n_estimators=120, max_depth=9, random_state=42, n_jobs=-1)
    rev_model.fit(X_train, yr_train)
    rev_r2 = r2_score(yr_test, rev_model.predict(X_test))
    rev_mae = mean_absolute_error(yr_test, rev_model.predict(X_test))

    exp_model = RandomForestRegressor(n_estimators=120, max_depth=9, random_state=42, n_jobs=-1)
    exp_model.fit(X_train, ye_train)
    exp_r2 = r2_score(ye_test, exp_model.predict(X_test))
    exp_mae = mean_absolute_error(ye_test, exp_model.predict(X_test))

    joblib.dump(rev_model, FC_REV_PATH)
    joblib.dump(exp_model, FC_EXP_PATH)
    joblib.dump(scaler, FC_SCALER_PATH)

    if verbose:
        print(f"Revenue-index model  R^2: {rev_r2:.3f} | MAE: {rev_mae:.3f}")
        print(f"Expense-index model  R^2: {exp_r2:.3f} | MAE: {exp_mae:.3f}")

    return rev_model, exp_model, scaler


def forecast_models_exist():
    return all(os.path.exists(p) for p in [FC_REV_PATH, FC_EXP_PATH, FC_SCALER_PATH])


def load_forecast_models():
    rev_model = joblib.load(FC_REV_PATH)
    exp_model = joblib.load(FC_EXP_PATH)
    scaler = joblib.load(FC_SCALER_PATH)
    return rev_model, exp_model, scaler


if __name__ == "__main__":
    train_and_save_forecast_models()
