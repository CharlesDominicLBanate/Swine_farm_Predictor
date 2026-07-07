"""
Part 8c: Income Forecast Engine
The sole entry point called by the UI for the "Income Forecast" feature. It takes
the current farm profile (the same input as SwineFarmPredictor),
and produces a month-by-month projection of revenue, expenses, net income,
and profit margin for the next N months.

Approach: the model predicts a "growth index" (relative to the
user's current monthly revenue/expenses) instead of a direct
value — so the forecast is personalized to the user's actual
numbers, while the learned trend/seasonality/risk pattern comes
from a broad synthetic simulation.
"""

import datetime
import numpy as np
import pandas as pd

from timeseries_generator import FORECAST_FEATURE_COLUMNS, SNAPSHOT_FEATURES, MONTH_SEASONALITY
from forecast_model import forecast_models_exist, train_and_save_forecast_models, load_forecast_models

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


class IncomeForecastEngine:
    REQUIRED_FIELDS = SNAPSHOT_FEATURES + ["monthly_revenue", "monthly_expenses"]

    def __init__(self, auto_train=True):
        if auto_train and not forecast_models_exist():
            train_and_save_forecast_models(verbose=False)
        self.rev_model, self.exp_model, self.scaler = load_forecast_models()

    def forecast(self, farm_data: dict, horizon_months: int = 12, start_date: datetime.date = None) -> list:
        missing = [f for f in self.REQUIRED_FIELDS if f not in farm_data]
        if missing:
            raise ValueError(f"The following inputs are missing for the forecast: {missing}")

        if start_date is None:
            start_date = datetime.date.today()

        base_revenue = float(farm_data["monthly_revenue"])
        base_expenses = float(farm_data["monthly_expenses"])

        # --- build all months as a single batch (faster than
        # repeated single-row predictions) ---
        feat_rows = []
        target_dates = []
        for t in range(1, horizon_months + 1):
            target_date = _add_months(start_date, t)
            calendar_month = target_date.month
            target_dates.append(target_date)

            feat = {k: farm_data[k] for k in SNAPSHOT_FEATURES}
            feat.update({
                "month_offset": t,
                "month_sin": np.sin(2 * np.pi * calendar_month / 12),
                "month_cos": np.cos(2 * np.pi * calendar_month / 12),
            })
            feat_rows.append(feat)

        X = pd.DataFrame(feat_rows)[FORECAST_FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)

        revenue_indices = self.rev_model.predict(X_scaled)
        expense_indices = self.exp_model.predict(X_scaled)

        rows = []
        for t, target_date, revenue_index, expense_index in zip(
            range(1, horizon_months + 1), target_dates, revenue_indices, expense_indices
        ):
            calendar_month = target_date.month
            projected_revenue = base_revenue * float(revenue_index)
            projected_expenses = base_expenses * float(expense_index)
            net_income = projected_revenue - projected_expenses
            profit_margin_pct = (net_income / projected_revenue * 100) if projected_revenue > 0 else 0.0

            rows.append({
                "month_offset": t,
                "year": target_date.year,
                "month_num": calendar_month,
                "month_label": f"{MONTH_NAMES[calendar_month]} {target_date.year}",
                "projected_revenue": round(projected_revenue, 2),
                "projected_expenses": round(projected_expenses, 2),
                "projected_net_income": round(net_income, 2),
                "projected_profit_margin_pct": round(profit_margin_pct, 1),
            })

        return rows


def _add_months(d: datetime.date, months: int) -> datetime.date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    return datetime.date(year, month, 1)


if __name__ == "__main__":
    engine = IncomeForecastEngine()
    sample = {
        "herd_size": 60, "years_operating": 5, "labor_count": 2, "capital": 150000,
        "feed_inventory_kg": 1800, "digital_transaction_freq": 10, "market_access_score": 65,
        "biosecurity_measures": 1, "monthly_revenue": 85000, "monthly_expenses": 60000,
    }
    forecast = engine.forecast(sample, horizon_months=12)
    for row in forecast:
        print(
            f"{row['month_label']:>16} | Revenue: P{row['projected_revenue']:>10,.2f} | "
            f"Expenses: P{row['projected_expenses']:>10,.2f} | "
            f"Net: P{row['projected_net_income']:>10,.2f} | Margin: {row['projected_profit_margin_pct']}%"
        )
