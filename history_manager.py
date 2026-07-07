import os
import datetime
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(MODEL_DIR, "prediction_history.csv")

HISTORY_COLUMNS = [
    "timestamp", "farm_name",
    "herd_size", "years_operating", "labor_count", "capital", "feed_inventory_kg",
    "digital_transaction_freq", "market_access_score", "biosecurity_measures",
    "monthly_revenue", "monthly_expenses", "loan_amount",
    "predicted_class", "confidence_pct", "profit_margin_pct",
    "risk_score", "expected_utility_score", "liquidity_health_index",
    "decision_recommendation",
]


def log_prediction(data: dict, result: dict, farm_name: str = "") -> None:
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "farm_name": farm_name or "",
        **{k: data.get(k, "") for k in [
            "herd_size", "years_operating", "labor_count", "capital", "feed_inventory_kg",
            "digital_transaction_freq", "market_access_score", "biosecurity_measures",
            "monthly_revenue", "monthly_expenses", "loan_amount",
        ]},
        "predicted_class": result["predicted_class"],
        "confidence_pct": result["confidence_pct"],
        "profit_margin_pct": result["profit_margin_pct"],
        "risk_score": result["risk_score"],
        "expected_utility_score": result["expected_utility_score"],
        "liquidity_health_index": result["liquidity_health_index"],
        "decision_recommendation": result["decision_recommendation"],
    }

    df_row = pd.DataFrame([row], columns=HISTORY_COLUMNS)
    file_exists = os.path.exists(HISTORY_PATH)
    df_row.to_csv(HISTORY_PATH, mode="a", header=not file_exists, index=False)


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(HISTORY_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def clear_history() -> None:
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)


if __name__ == "__main__":
    sample_data = {
        "herd_size": 60, "years_operating": 5, "labor_count": 2, "capital": 150000,
        "feed_inventory_kg": 1800, "digital_transaction_freq": 10, "market_access_score": 65,
        "biosecurity_measures": 1, "monthly_revenue": 85000, "monthly_expenses": 60000,
        "loan_amount": 30000,
    }
    sample_result = {
        "predicted_class": "High", "confidence_pct": 74.4, "profit_margin_pct": 29.4,
        "risk_score": 35.3, "expected_utility_score": 0.0, "liquidity_health_index": 60.6,
        "decision_recommendation": "Maintain",
    }
    log_prediction(sample_data, sample_result, farm_name="Test Farm")
    print(load_history())
