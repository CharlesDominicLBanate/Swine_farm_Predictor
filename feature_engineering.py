import numpy as np
import pandas as pd

UTILITY_RAW_MIN = -1.4
UTILITY_RAW_MAX = 1.1


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cash_flow_ratio"] = df["monthly_revenue"] / df["monthly_expenses"].replace(0, np.nan)
    df["cash_flow_ratio"] = df["cash_flow_ratio"].fillna(0)
    df["profit_margin_pct"] = (
        (df["monthly_revenue"] - df["monthly_expenses"]) / df["monthly_revenue"].replace(0, np.nan)
    ) * 100
    df["profit_margin_pct"] = df["profit_margin_pct"].fillna(0)
    df["loan_dependency_pct"] = (df["loan_amount"] / df["capital"].replace(0, np.nan)) * 100
    df["loan_dependency_pct"] = df["loan_dependency_pct"].fillna(0).clip(0, 100)
    leverage_component = df["loan_dependency_pct"].clip(0, 100)
    liquidity_pressure = (1 - (df["cash_flow_ratio"].clip(0, 3) / 3)) * 100
    expense_pressure = (df["monthly_expenses"] / df["monthly_revenue"].replace(0, np.nan)).fillna(1).clip(0, 2) * 50
    df["risk_score"] = (
        0.4 * leverage_component + 0.35 * liquidity_pressure + 0.25 * expense_pressure
    ).clip(0, 100).round(1)
    net_cash_flow = df["monthly_revenue"] - df["monthly_expenses"]
    relative_gain = net_cash_flow / df["capital"].replace(0, np.nan).clip(lower=1)
    relative_gain = relative_gain.fillna(0)
    loss_aversion_lambda = 2.25
    utility_raw = np.where(
        relative_gain >= 0,
        np.sqrt(np.abs(relative_gain)),
        -loss_aversion_lambda * np.sqrt(np.abs(relative_gain)),
    )
   
    utility_raw_clipped = np.clip(utility_raw, UTILITY_RAW_MIN, UTILITY_RAW_MAX)
    df["expected_utility_score"] = (
        (utility_raw_clipped - UTILITY_RAW_MIN) / (UTILITY_RAW_MAX - UTILITY_RAW_MIN) * 100
    ).round(1)
    df["liquidity_health_index"] = (
        0.5 * (df["cash_flow_ratio"].clip(0, 3) / 3 * 100)
        + 0.3 * (100 - df["loan_dependency_pct"])
        + 0.2 * df["market_access_score"]
    ).clip(0, 100).round(1)
    def recommend(row):
        if row["risk_score"] >= 65:
            return "High Risk"
        if row["profit_margin_pct"] < 5 or row["liquidity_health_index"] < 40:
            return "Cost-Cutting Needed"
        if row["profit_margin_pct"] >= 20 and row["liquidity_health_index"] >= 65:
            return "Expand"
        return "Maintain"
    df["decision_recommendation"] = df.apply(recommend, axis=1)
    def classify_profit(pm):
        if pm < 8:
            return "Low"
        elif pm < 20:
            return "Moderate"
        else:
            return "High"
    df["profitability_class"] = df["profit_margin_pct"].apply(classify_profit)
    return df


FEATURE_COLUMNS = [
    "herd_size", "years_operating", "labor_count", "capital", "feed_inventory_kg",
    "digital_transaction_freq", "market_access_score", "biosecurity_measures",
    "monthly_revenue", "monthly_expenses", "loan_amount",
    "cash_flow_ratio", "profit_margin_pct", "loan_dependency_pct",
    "risk_score", "expected_utility_score", "liquidity_health_index",
]

if __name__ == "__main__":
    from data_generator import generate_dataset
    df = generate_dataset()
    df = add_engineered_features(df)
    df.to_csv("swine_farm_dataset_engineered.csv", index=False)
    print("Class distribution:")
    print(df["profitability_class"].value_counts())
    print("\nDecision recommendation distribution:")
    print(df["decision_recommendation"].value_counts())
