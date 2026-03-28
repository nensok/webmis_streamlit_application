import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json

from utils.preprocessing import prepare_daily_data, validate_data
from utils.prophet_predict import prophet_forecast
from utils.xgb_predict import xgb_forecast
from utils.backtest import backtest_xgb, backtest_prophet

from components.upload_section import upload_data
from components.manual_input_section import manual_input
from components.results_display import display_results


st.set_page_config(page_title="WEBMIS Malaria Forecasting System v1.0", layout="wide")
st.title("WEBMIS Malaria Surveillance Forecasting System")


# -------------------------
# SIDEBAR
# -------------------------

model_type = st.sidebar.selectbox(
    "Select Forecast Model",
    ["Prophet Model", "XGBoost Model"]
)

compare_models = st.sidebar.checkbox(
    "Compare Both Models",
    help="Run Prophet and XGBoost simultaneously and overlay on the same charts."
)

input_type = st.sidebar.radio(
    "Input Method",
    ["Upload Dataset", "Manual Data Entry"]
)


# -------------------------
# LOAD PROPHET MODELS
# -------------------------

@st.cache_resource
def load_models():
    with open("models/daily_cases_model.json", "r") as f:
        daily_model = model_from_json(f.read())
    with open("models/weekly_cases_model.json", "r") as f:
        weekly_model = model_from_json(f.read())
    with open("models/positivity_rate_model.json", "r") as f:
        pos_model = model_from_json(f.read())
    return daily_model, weekly_model, pos_model


daily_model, weekly_model, pos_model = load_models()


# -------------------------
# DATA INPUT
# -------------------------

if input_type == "Upload Dataset":
    df = upload_data()
elif input_type == "Manual Data Entry":
    df = manual_input()

if df is not None:

    st.subheader("Input Data Preview")
    st.dataframe(df.head())

    daily = prepare_daily_data(df)

    # Validation warnings
    warnings = validate_data(daily)
    for w in warnings:
        st.warning(w)

    forecast_days = st.slider("Forecast Horizon (Days)", 7, 365, 30)
    st.caption(
        "XGBoost accuracy degrades significantly beyond ~30 days due to recursive error compounding. "
        "The Weekly tab shows the equivalent number of weeks."
    )

    # -------------------------
    # RUN FORECASTS
    # -------------------------

    run_prophet = model_type == "Prophet Model" or compare_models
    run_xgb = model_type == "XGBoost Model" or compare_models

    results_prophet = None
    results_xgb = None

    with st.spinner("Running forecast..."):
        if run_prophet:
            results_prophet = prophet_forecast(
                daily, daily_model, weekly_model, pos_model, forecast_days
            )
        if run_xgb:
            results_xgb = xgb_forecast(daily, forecast_days)

    results = results_prophet if model_type == "Prophet Model" else results_xgb
    results2 = None
    model_label2 = None
    if compare_models:
        results2 = results_xgb if model_type == "Prophet Model" else results_prophet
        model_label2 = "XGBoost Model" if model_type == "Prophet Model" else "Prophet Model"

    # -------------------------
    # BACK-TEST METRICS
    # -------------------------

    st.subheader("Model Performance (Back-test)")

    with st.spinner("Running back-test..."):
        if run_xgb:
            xgb_metrics, xgb_err = backtest_xgb(daily)
            if xgb_err:
                st.info(f"XGBoost back-test skipped: {xgb_err}")
            elif xgb_metrics:
                st.write("**XGBoost — 30-day holdout back-test** *(lower is better)*")
                st.dataframe(pd.DataFrame(xgb_metrics).T, use_container_width=True)

        if run_prophet:
            prophet_metrics = backtest_prophet(daily, results_prophet)
            if prophet_metrics:
                st.write("**Prophet — in-sample fit** *(dates overlapping model training data)*")
                st.dataframe(pd.DataFrame(prophet_metrics).T, use_container_width=True)
            else:
                st.info(
                    "Prophet back-test: no overlap between your data dates and the model's "
                    "training period — in-sample metrics unavailable."
                )

    # -------------------------
    # RESULTS DISPLAY
    # -------------------------

    display_results(
        results,
        historical=daily,
        model_label=model_type,
        results2=results2,
        model_label2=model_label2,
    )
