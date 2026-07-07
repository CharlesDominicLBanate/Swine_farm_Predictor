import numpy as np
import pandas as pd

from data_generator import generate_dataset

np.random.seed(7)

HORIZON_MONTHS = 36

MONTH_SEASONALITY = {
    1: 1.15, 2: 0.95, 3: 0.95, 4: 1.05, 5: 1.10, 6: 0.90,
    7: 0.85, 8: 0.85, 9: 0.95, 10: 1.00, 11: 1.10, 12: 1.30,
}

EXPENSE_SEASONALITY_FACTOR = 0.3

SNAPSHOT_FEATURES = [
    "herd_size", "years_operating", "labor_count", "capital", "feed_inventory_kg",
    "digital_transaction_freq", "market_access_score", "biosecurity_measures",
]


def simulate_timeseries(n_farms=1000, horizon=HORIZON_MONTHS, seed=7):
    rng = np.random.default_rng(seed)
    base = generate_dataset(n_farms)
    rows = []
    for _, farm in base.iterrows():
        start_month = int(rng.integers(1, 13))
        biosecurity = farm["biosecurity_measures"]
        market_access = farm["market_access_score"]

        base_growth = rng.normal(0.004, 0.003)
        monthly_growth_rate = (
            base_growth + 0.0025 * biosecurity + 0.0006 * (market_access - 50) / 50
        )

        outbreak_prob = 0.028 if biosecurity == 0 else 0.010

        expense_drift = rng.normal(0.0018, 0.0015)
        for t in range(1, horizon + 1):
            calendar_month = ((start_month - 1 + t - 1) % 12) + 1
            seasonal = MONTH_SEASONALITY[calendar_month] * rng.normal(1.0, 0.03)
            growth_factor = (1 + monthly_growth_rate) ** t
            noise = rng.normal(1.0, 0.05)
            shock = 1.0
            if rng.random() < outbreak_prob:
                shock = rng.uniform(0.35, 0.65)
            revenue_index = max(growth_factor * seasonal * noise * shock, 0.05)

        
            expense_seasonal = 1 + (MONTH_SEASONALITY[calendar_month] - 1) * EXPENSE_SEASONALITY_FACTOR
            expense_seasonal *= rng.normal(1.0, 0.02)
            expense_growth = (1 + expense_drift) ** t
            expense_noise = rng.normal(1.0, 0.04)
            expense_index = max(expense_growth * expense_seasonal * expense_noise, 0.05)

            row = {k: farm[k] for k in SNAPSHOT_FEATURES}
            row.update({
                "month_offset": t,
                "calendar_month": calendar_month,
                "month_sin": np.sin(2 * np.pi * calendar_month / 12),
                "month_cos": np.cos(2 * np.pi * calendar_month / 12),
                "revenue_index": revenue_index,
                "expense_index": expense_index,
            })
            rows.append(row)
    return pd.DataFrame(rows)


FORECAST_FEATURE_COLUMNS = SNAPSHOT_FEATURES + ["month_offset", "month_sin", "month_cos"]


if __name__ == "__main__":
    df = simulate_timeseries(n_farms=50, horizon=12)
    print(df.head(15))
    print(f"\nTotal rows: {len(df)}")
