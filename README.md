# 🐖 Swine Farm Profitability Predictor (Streamlit)

A decision-support web app for hog raisers in Bayugan City. Based on the
Resource-Based View (RBV) Theory and Decision Theory, it predicts a farm's
profitability class (Low/Moderate/High), profit margin, risk score, and
decision recommendation — including a monthly income forecast and PDF
report export.

## Features

- **Prediction** — Random Forest classifier + regressor that provides
  profitability class, confidence, profit margin, and decision-theory
  scores (risk, expected utility, liquidity health).
- **Income Forecast** — projection of revenue/expenses/net income for
  the next 3/6/12 months, based on growth trend + seasonality +
  disease-risk simulation.
- **History** — a logged CSV of every prediction, with a trend chart of
  profit margin over time.
- **PDF Export** — a complete prediction report that can be downloaded
  and shown to a cooperative/lending institution.

## Project Structure

```
streamlit_app.py          # main Streamlit UI (entry point)
data_generator.py          # Part 1: synthetic dataset generator
feature_engineering.py     # Part 2: financial ratios + decision-theory features
ml_model.py                 # Part 3: classifier/regressor training
predictor_engine.py         # Part 4: prediction pipeline
timeseries_generator.py     # Part 8a: synthetic monthly panel data
forecast_model.py           # Part 8b: forecast model training
forecast_engine.py          # Part 8c: income forecast pipeline
report_generator.py         # Part 6: PDF report generator
history_manager.py          # Part 7: prediction history (CSV log)
*.pkl                        # trained model files (auto-retrained if missing)
requirements.txt
.streamlit/config.toml      # theme
```

## How to Run Locally

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The browser will automatically open at `http://localhost:8501`. The
already-trained `.pkl` model files are included in the repo, so the first
load is fast. If they are missing or deleted, the app will automatically
retrain on first run (this may take a few seconds).

## Deploying to Streamlit Community Cloud (free)

1. Push this repo to your own GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
3. Click **"New app"**, select the repo and branch, and set
   `streamlit_app.py` as the main file.
4. Click **Deploy**. Within a few minutes, you'll have a live URL.

### Note on the History Log in the cloud

`prediction_history.csv` is stored only on the deployment's local
filesystem. On Streamlit Community Cloud, this may be reset whenever
the app restarts or is redeployed (ephemeral filesystem). If you need
the history to persist permanently in the cloud, migrate it to an
external database (e.g. Supabase, Google Sheets API, or SQLite mounted
on persistent storage) — just let me know if you'd like help with this.

## Modeling Notes / Disclaimers

- The dataset is **synthetic** (generated using `data_generator.py`),
  not sourced from actual historical data of swine farms in Bayugan
  City. It is used for proof-of-concept / thesis demonstration purposes.
- The seasonality pattern in the Income Forecast (higher demand during
  December-January and fiesta season) is an **illustrative/assumed**
  model — it should be validated using real historical price/sales data
  if available.
- All results are projections only and do not replace professional
  financial advice.
