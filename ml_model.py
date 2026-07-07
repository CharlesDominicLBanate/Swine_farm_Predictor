"""
Part 3: Machine Learning Model Training
Random Forest classifier (Low/Moderate/High) + Random Forest regressor (exact
profit margin %). Includes train-test split, evaluation metrics, and
feature importance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from feature_engineering import FEATURE_COLUMNS, add_engineered_features
from data_generator import generate_dataset

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CLF_PATH = os.path.join(MODEL_DIR, "classifier_model.pkl")
REG_PATH = os.path.join(MODEL_DIR, "regressor_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def train_and_save_models(verbose=True):
    df = generate_dataset()
    df = add_engineered_features(df)

    X = df[FEATURE_COLUMNS]
    y_class = df["profitability_class"]
    y_reg = df["profit_margin_pct"]

    le = LabelEncoder()
    y_class_enc = le.fit_transform(y_class)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X_scaled, y_class_enc, y_reg, test_size=0.2, random_state=42, stratify=y_class_enc
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, yc_train)
    clf_acc = accuracy_score(yc_test, clf.predict(X_test))
    clf_cv = cross_val_score(clf, X_scaled, y_class_enc, cv=5).mean()

    reg = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    reg.fit(X_train, yr_train)
    reg_r2 = r2_score(yr_test, reg.predict(X_test))
    reg_mae = mean_absolute_error(yr_test, reg.predict(X_test))

    joblib.dump(clf, CLF_PATH)
    joblib.dump(reg, REG_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)

    if verbose:
        print(f"Classifier test accuracy: {clf_acc:.3f} | CV accuracy: {clf_cv:.3f}")
        print(f"Regressor R^2: {reg_r2:.3f} | MAE: {reg_mae:.2f} pct points")
        importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print("\nTop feature importances (classifier):")
        print(importances.head(6))

    return clf, reg, scaler, le


def models_exist():
    return all(os.path.exists(p) for p in [CLF_PATH, REG_PATH, SCALER_PATH, ENCODER_PATH])


def load_models():
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    return clf, reg, scaler, le


if __name__ == "__main__":
    train_and_save_models()
