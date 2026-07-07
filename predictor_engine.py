"""
Part 4: Predictor Engine
Main entry point called by the UI. Accepts raw input from the user,
runs it through feature engineering, uses the trained models, and produces
a complete, explainable prediction report.
"""

import numpy as np
import pandas as pd

from feature_engineering import FEATURE_COLUMNS, add_engineered_features
from ml_model import models_exist, train_and_save_models, load_models


class SwineFarmPredictor:
    """
    Example usage:

        predictor = SwineFarmPredictor()
        result = predictor.predict({
            "herd_size": 60,
            "years_operating": 5,
            "labor_count": 2,
            "capital": 150000,
            "feed_inventory_kg": 1800,
            "digital_transaction_freq": 10,
            "market_access_score": 65,
            "biosecurity_measures": 1,
            "monthly_revenue": 85000,
            "monthly_expenses": 60000,
            "loan_amount": 30000,
        })
    """

    REQUIRED_FIELDS = [
        "herd_size", "years_operating", "labor_count", "capital", "feed_inventory_kg",
        "digital_transaction_freq", "market_access_score", "biosecurity_measures",
        "monthly_revenue", "monthly_expenses", "loan_amount",
    ]

    CLASS_EXPLANATIONS = {
        "Low": (
            "Your farm's expected income is low based on the current data. "
            "You should review your expenses and look for ways to increase "
            "income before investing additional capital."
        ),
        "Moderate": (
            "Your farm's income is moderate — the risk is not high, but there "
            "is still room for improvement. This is a good time to tighten up "
            "cash flow management and gradually strengthen operations."
        ),
        "High": (
            "Your farm's income is strong! Your current management approach "
            "is effective. You may consider expanding if your liquidity remains stable."
        ),
    }

    def __init__(self, auto_train=True):
        if auto_train and not models_exist():
            train_and_save_models(verbose=False)
        self.clf, self.reg, self.scaler, self.label_encoder = load_models()

    def _build_feature_row(self, raw_input: dict) -> pd.DataFrame:
        missing = [f for f in self.REQUIRED_FIELDS if f not in raw_input]
        if missing:
            raise ValueError(f"The following inputs are missing: {missing}")

        row = {k: raw_input[k] for k in self.REQUIRED_FIELDS}
        df = pd.DataFrame([row])
        df = add_engineered_features(df)
        return df

    def predict(self, raw_input: dict) -> dict:
        df = self._build_feature_row(raw_input)
        X = df[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)

        proba = self.clf.predict_proba(X_scaled)[0]
        pred_idx = int(np.argmax(proba))
        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx]) * 100

        proba_breakdown = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(self.label_encoder.classes_, proba)
        }

        profit_margin_pred = float(self.reg.predict(X_scaled)[0])

        risk_score = float(df["risk_score"].iloc[0])
        utility_score = float(df["expected_utility_score"].iloc[0])
        liquidity_index = float(df["liquidity_health_index"].iloc[0])
        recommendation = df["decision_recommendation"].iloc[0]

        explanation = self.CLASS_EXPLANATIONS.get(pred_class, "")

        return {
            "predicted_class": pred_class,
            "confidence_pct": round(confidence, 1),
            "probability_breakdown": proba_breakdown,
            "profit_margin_pct": round(profit_margin_pred, 1),
            "risk_score": risk_score,
            "expected_utility_score": utility_score,
            "liquidity_health_index": liquidity_index,
            "decision_recommendation": recommendation,
            "explanation_fil": explanation,
        }


if __name__ == "__main__":
    predictor = SwineFarmPredictor()
    sample = {
        "herd_size": 60,
        "years_operating": 5,
        "labor_count": 2,
        "capital": 150000,
        "feed_inventory_kg": 1800,
        "digital_transaction_freq": 10,
        "market_access_score": 65,
        "biosecurity_measures": 1,
        "monthly_revenue": 85000,
        "monthly_expenses": 60000,
        "loan_amount": 30000,
    }
    result = predictor.predict(sample)
    for k, v in result.items():
        print(f"{k}: {v}")
