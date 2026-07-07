"""
Part 1: Data Generator
Generates a synthetic dataset of swine farm financial indicators in Bayugan City,
based on Resource-Based View (RBV) Theory: tangible, intangible, and
financial resources as predictors of profitability.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200

BARANGAYS = [
    "Taglatawan", "Ambago", "San Isidro", "Km 3", "Poblacion",
    "Lucban", "Mahayahay", "Ampiling", "Wire"
]
EDUCATION_LEVELS = ["Elementary", "High School", "Vocational", "College"]


def generate_dataset(n=N):
    herd_size = np.random.randint(5, 500, n)
    years_operating = np.random.randint(1, 25, n)
    labor_count = np.clip((herd_size / 40 + np.random.normal(1, 0.5, n)).astype(int), 1, None)

    capital = herd_size * np.random.uniform(1800, 3500, n) + np.random.normal(0, 15000, n)
    capital = np.clip(capital, 20000, None)

    feed_inventory_kg = herd_size * np.random.uniform(15, 45, n)

    digital_transaction_freq = np.random.poisson(8, n) + (herd_size > 100).astype(int) * np.random.poisson(5, n)

    market_access_score = np.clip(np.random.normal(60, 20, n), 0, 100)
    biosecurity = np.random.choice([0, 1], n, p=[0.4, 0.6])
    education_level = np.random.choice(EDUCATION_LEVELS, n, p=[0.15, 0.4, 0.25, 0.2])
    barangay = np.random.choice(BARANGAYS, n)

    monthly_revenue = herd_size * np.random.uniform(900, 1800, n) * (1 + market_access_score / 300)
    expense_ratio = np.random.normal(0.75, 0.15, n)
    expense_ratio = np.clip(expense_ratio, 0.4, 1.3)
    monthly_expenses = monthly_revenue * expense_ratio

    loan_amount = capital * np.random.uniform(0, 0.6, n) * (1 - biosecurity * 0.15)

    df = pd.DataFrame({
        "herd_size": herd_size,
        "years_operating": years_operating,
        "labor_count": labor_count,
        "capital": capital.round(2),
        "feed_inventory_kg": feed_inventory_kg.round(1),
        "digital_transaction_freq": digital_transaction_freq,
        "market_access_score": market_access_score.round(1),
        "biosecurity_measures": biosecurity,
        "education_level": education_level,
        "barangay": barangay,
        "monthly_revenue": monthly_revenue.round(2),
        "monthly_expenses": monthly_expenses.round(2),
        "loan_amount": loan_amount.round(2),
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("swine_farm_dataset.csv", index=False)
    print(f"Generated {len(df)} records -> swine_farm_dataset.csv")
    print(df.head())
