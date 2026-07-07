import datetime
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from predictor_engine import SwineFarmPredictor
from forecast_engine import IncomeForecastEngine
from report_generator import generate_pdf_report, generate_forecast_pdf_report
import history_manager

THEMES = {
    "dark": {
        "BG": "#1B2420",
        "PANEL": "#232F29",
        "ACCENT": "#6FA875",
        "ACCENT_2": "#C9973F",
        "TEXT": "#EDEDE3",
        "TEXT_MUTED": "#93A399",
        "GRID": "#35443C",
        "CLASS_COLORS": {"Low": "#C0533E", "Moderate": "#D9A441", "High": "#6FA875"},
    },
    "light": {
        "BG": "#F5F3EC",
        "PANEL": "#FFFFFF",
        "ACCENT": "#3F7A4C",
        "ACCENT_2": "#B4791F",
        "TEXT": "#1B2420",
        "TEXT_MUTED": "#5B6B60",
        "GRID": "#D8D4C6",
        "CLASS_COLORS": {"Low": "#B23A26", "Moderate": "#B4791F", "High": "#3F7A4C"},
    },
}

st.set_page_config(
    page_title="Swine Farm Profitability Predictor",
    page_icon="🐖",
    layout="wide",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

col_title, col_toggle = st.columns([5, 1])
with col_toggle:
    is_dark = st.toggle("🌙 Dark", value=(st.session_state.theme == "dark"))
    st.session_state.theme = "dark" if is_dark else "light"

p = THEMES[st.session_state.theme]
BG = p["BG"]
PANEL = p["PANEL"]
ACCENT = p["ACCENT"]
ACCENT_2 = p["ACCENT_2"]
TEXT = p["TEXT"]
TEXT_MUTED = p["TEXT_MUTED"]
GRID = p["GRID"]
CLASS_COLORS = p["CLASS_COLORS"]
REC_COLORS = {
    "Expand": ACCENT,
    "Maintain": ACCENT,
    "Cost-Cutting Needed": ACCENT_2,
    "High Risk": CLASS_COLORS["Low"],
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL}; }}
    div[data-testid="stMetric"] {{ background-color: {PANEL}; padding: 10px 14px; border-radius: 10px; }}

   
    label, .stMarkdown, .stCaption, .stMarkdown p,
    h1, h2, h3, h4, h5, h6,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stMetricValue"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] div[role="radiogroup"] label,
    [data-testid="stCheckbox"] label p,
    [data-testid="stCheckbox"] span,
    [data-testid="stNumberInput"] label p,
    [data-testid="stTextInput"] label p,
    [data-testid="stSlider"] label p,
    [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSlider"] div[data-testid="stTickBarMax"],
    [data-testid="stExpander"] summary p,
    [data-testid="stFileUploaderDropzone"] div,
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] p,
    div[data-testid="stDataFrame"] div {{
        color: {TEXT} !important;
    }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

    
    button[data-baseweb="tab"] {{
        color: {TEXT_MUTED} !important;
    }}
    button[data-baseweb="tab"] p {{
        color: inherit !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {ACCENT} !important;
    }}

  
    div[data-testid="stAlert"] {{
        background-color: {PANEL} !important;
        border: 1px solid {GRID} !important;
    }}
    div[data-testid="stAlert"] p {{
        color: {TEXT} !important;
    }}

    
    .stButton button,
    .stFormSubmitButton button,
    .stDownloadButton button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stBaseButton-secondary"] button {{
        background-color: {ACCENT} !important;
        color: #12180F !important;
        border: 1px solid {ACCENT} !important;
    }}
    .stButton button p,
    .stFormSubmitButton button p,
    .stDownloadButton button p,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stDownloadButton"] button p {{
        color: #12180F !important;
    }}
    .stButton button:hover,
    .stFormSubmitButton button:hover,
    .stDownloadButton button:hover {{
        background-color: {ACCENT_2} !important;
        border: 1px solid {ACCENT_2} !important;
        color: #12180F !important;
    }}
    .stButton button:disabled,
    .stFormSubmitButton button:disabled,
    .stDownloadButton button:disabled {{
        background-color: {GRID} !important;
        color: {TEXT_MUTED} !important;
        border: 1px solid {GRID} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with col_title:
    st.title("🐖 Swine Farm Profitability Predictor")
    st.caption("Decision-support tool for hog raisers in Bayugan City — RBV Theory + Decision Theory")


@st.cache_resource(show_spinner="Loading prediction models...")
def load_predictor():
    return SwineFarmPredictor()


@st.cache_resource(show_spinner="Loading forecast models...")
def load_forecast_engine():
    return IncomeForecastEngine()


predictor = load_predictor()
forecast_engine = load_forecast_engine()

for key, default in [("last_result", None), ("last_data", None), ("farm_name", "")]:
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.header("Farm Profile Input")
    st.caption("Enter your farm's data for this month.")

    with st.form("farm_form"):
        farm_name = st.text_input("Farm Name / Owner (optional)", value=st.session_state.farm_name)
        herd_size = st.number_input("Herd Size (number of pigs)", min_value=1, value=60, step=1)
        years_operating = st.number_input("Years Operating", min_value=0, value=5, step=1)
        labor_count = st.number_input("Number of Workers", min_value=1, value=2, step=1)
        capital = st.number_input("Capital / Investment (₱)", min_value=0.0, value=150000.0, step=1000.0)
        feed_inventory_kg = st.number_input("Feed Inventory (kg)", min_value=0.0, value=1800.0, step=10.0)
        digital_transaction_freq = st.number_input("Digital Transactions / Month", min_value=0, value=10, step=1)
        market_access_score = st.slider("Market Access Score", 0, 100, 65)
        biosecurity_measures = st.checkbox("Has Biosecurity Measures", value=True)
        monthly_revenue = st.number_input("Monthly Revenue (₱)", min_value=0.0, value=85000.0, step=1000.0)
        monthly_expenses = st.number_input("Monthly Expenses (₱)", min_value=0.0, value=60000.0, step=1000.0)
        loan_amount = st.number_input("Loan Amount (₱)", min_value=0.0, value=30000.0, step=1000.0)

        submitted = st.form_submit_button("🔍 PREDICT PROFITABILITY", use_container_width=True)

    if submitted:
        data = {
            "herd_size": int(herd_size),
            "years_operating": int(years_operating),
            "labor_count": int(labor_count),
            "capital": float(capital),
            "feed_inventory_kg": float(feed_inventory_kg),
            "digital_transaction_freq": int(digital_transaction_freq),
            "market_access_score": float(market_access_score),
            "biosecurity_measures": 1 if biosecurity_measures else 0,
            "monthly_revenue": float(monthly_revenue),
            "monthly_expenses": float(monthly_expenses),
            "loan_amount": float(loan_amount),
        }
        try:
            result = predictor.predict(data)
        except Exception as e:
            st.error(f"Prediction error: {e}")
        else:
            st.session_state.last_result = result
            st.session_state.last_data = data
            st.session_state.farm_name = farm_name
            try:
                history_manager.log_prediction(data, result, farm_name=farm_name)
            except Exception:
                pass
            st.success("Prediction complete! See the results on the right.")


tab_pred, tab_forecast, tab_history = st.tabs(["🔍 Prediction", "💰 Income Forecast", "🕒 History"])

with tab_pred:
    result = st.session_state.last_result
    data = st.session_state.last_data

    if result is None:
        st.info("Fill in the form on the left and click **'PREDICT PROFITABILITY'** to see the result here.")
    else:
        cls = result["predicted_class"]
        color = CLASS_COLORS.get(cls, ACCENT)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(
                f"<div style='background:{color};padding:18px 24px;border-radius:12px;"
                f"text-align:center;font-weight:700;font-size:20px;color:#12180F;margin-top:6px;'>"
                f"{cls.upper()} PROFITABILITY</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("Confidence", f"{result['confidence_pct']:.1f}%")
        with col3:
            st.metric("Profit Margin", f"{result['profit_margin_pct']:.1f}%")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Probability Breakdown")
            order = ["Low", "Moderate", "High"]
            proba = result["probability_breakdown"]
            labels = [c for c in order if c in proba]
            values = [proba[c] for c in labels]

            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            fig.patch.set_facecolor(PANEL)
            ax.set_facecolor(PANEL)
            bars = ax.barh(labels, values, color=[CLASS_COLORS[c] for c in labels], height=0.55)
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            ax.tick_params(colors=TEXT, labelsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.set_visible(False)
            for bar, val in zip(bars, values):
                ax.text(val + 2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", color=TEXT)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.subheader("Decision-Theory Scores")
            st.write("Risk Score (lower is better)")
            st.progress(min(max(result["risk_score"], 0), 100) / 100)
            st.caption(f"{result['risk_score']:.1f} / 100")

            st.write("Expected Utility Score")
            st.progress(min(max(result["expected_utility_score"], 0), 100) / 100)
            st.caption(f"{result['expected_utility_score']:.1f} / 100")

            st.write("Liquidity Health Index")
            st.progress(min(max(result["liquidity_health_index"], 0), 100) / 100)
            st.caption(f"{result['liquidity_health_index']:.1f} / 100")

        st.divider()

        rec = result["decision_recommendation"]
        rec_color = REC_COLORS.get(rec, ACCENT_2)
        st.subheader("Recommendation")
        st.markdown(
            f"<div style='background:{PANEL};border-radius:10px;padding:14px 18px;"
            f"font-size:20px;font-weight:700;color:{rec_color};'>{rec}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Explanation")
        st.info(result["explanation_fil"])


        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = generate_pdf_report(result, data, tmp.name, farm_name=st.session_state.farm_name)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            "📄 DOWNLOAD AS PDF REPORT",
            data=pdf_bytes,
            file_name=f"swine_farm_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

with tab_forecast:
    if st.session_state.last_data is None:
        st.info("First make a prediction in the sidebar using your farm's actual data before viewing the Income Forecast — to make sure it's based on the right numbers.")
    else:
        horizon_label = st.radio("Horizon", ["3 months", "6 months", "12 months"], index=2, horizontal=True)
        horizon = {"3 months": 3, "6 months": 6, "12 months": 12}[horizon_label]

        st.caption(
            "This is based on your last prediction, demand seasonality, and the "
            "growth pattern of similar farms. It is a projection only, not a guarantee."
        )

        try:
            forecast_rows = forecast_engine.forecast(st.session_state.last_data, horizon_months=horizon)
        except Exception as e:
            st.error(f"Forecast error: {e}")
        else:
            df_forecast = pd.DataFrame(forecast_rows)

            fig, ax = plt.subplots(figsize=(9, 3.2))
            fig.patch.set_facecolor(PANEL)
            ax.set_facecolor(PANEL)
            x = range(len(df_forecast))
            ax.plot(x, df_forecast["projected_revenue"], color=ACCENT, marker="o", markersize=4, linewidth=2, label="Revenue")
            ax.plot(x, df_forecast["projected_expenses"], color="#C0533E", marker="o", markersize=4, linewidth=2, label="Expenses")
            ax.fill_between(
                x, df_forecast["projected_revenue"], df_forecast["projected_expenses"],
                where=(df_forecast["projected_revenue"] >= df_forecast["projected_expenses"]),
                color=ACCENT, alpha=0.15,
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels([m.split(" ")[0][:3] for m in df_forecast["month_label"]], fontsize=8.5)
            ax.tick_params(colors=TEXT, labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
            legend = ax.legend(loc="upper left", frameon=False)
            for t in legend.get_texts():
                t.set_color(TEXT)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            display_df = df_forecast[[
                "month_label", "projected_revenue", "projected_expenses",
                "projected_net_income", "projected_profit_margin_pct",
            ]].copy()
            display_df.columns = ["Month", "Revenue (₱)", "Expenses (₱)", "Net Income (₱)", "Margin (%)"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
                fig.savefig(tmp_chart.name, facecolor=PANEL, dpi=150, bbox_inches="tight")
                chart_path = tmp_chart.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                forecast_pdf_path = generate_forecast_pdf_report(
                    forecast_rows,
                    farm_name=st.session_state.farm_name,
                    horizon_months=horizon,
                    chart_image_path=chart_path,
                    output_path=tmp_pdf.name,
                )
            with open(forecast_pdf_path, "rb") as f:
                forecast_pdf_bytes = f.read()

            st.download_button(
                "📄 DOWNLOAD FORECAST AS PDF",
                data=forecast_pdf_bytes,
                file_name=f"income_forecast_{horizon}mo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

with tab_history:
    df_hist = history_manager.load_history()

    if df_hist.empty:
        st.info("No predictions logged yet. Make a prediction in the sidebar first.")
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{len(df_hist)}** prediction(s) logged")
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                history_manager.clear_history()
                st.rerun()

        chrono = df_hist.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(9, 2.8))
        fig.patch.set_facecolor(PANEL)
        ax.set_facecolor(PANEL)
        x = range(len(chrono))
        ax.plot(x, chrono["profit_margin_pct"], color=ACCENT, marker="o", markersize=4, linewidth=2)
        labels = [ts.strftime("%m/%d") if pd.notna(ts) else "--" for ts in chrono["timestamp"]]
        step = max(1, len(labels) // 12)
        ax.set_xticks(list(x)[::step])
        ax.set_xticklabels(labels[::step], fontsize=8.5)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
        ax.set_title("Profit Margin Trend", color=TEXT, fontsize=11, loc="left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        display_hist = df_hist[[
            "timestamp", "farm_name", "predicted_class", "profit_margin_pct",
            "risk_score", "decision_recommendation",
        ]].head(30).copy()
        display_hist.columns = ["Date", "Farm", "Class", "Margin (%)", "Risk Score", "Recommendation"]
        st.dataframe(display_hist, use_container_width=True, hide_index=True)
