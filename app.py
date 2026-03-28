import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json

from utils.preprocessing import prepare_daily_data
from utils.prophet_predict import prophet_forecast
from utils.xgb_predict import xgb_forecast

from components.upload_section import upload_data
from components.manual_input_section import manual_input
from components.results_display import display_results


st.set_page_config(page_title="WEBMIS Malaria Forecasting System", layout="wide")

st.title("WEBMIS Malaria Surveillance Forecasting System")


model_type = st.sidebar.selectbox(
    "Select Forecast Model",
    ["Prophet Model","XGBoost Model"]
)


input_type = st.sidebar.radio(
    "Input Method",
    ["Upload Dataset","Manual Data Entry"]
)


# -------------------------
# LOAD MODELS
# -------------------------

@st.cache_resource
def load_models():

    with open("models/daily_cases_model.json","r") as f:
        daily_model = model_from_json(f.read())

    with open("models/weekly_cases_model.json","r") as f:
        weekly_model = model_from_json(f.read())

    with open("models/positivity_rate_model.json","r") as f:
        pos_model = model_from_json(f.read())

    return daily_model, weekly_model, pos_model


daily_model, weekly_model, pos_model = load_models()


# -------------------------
# DATA INPUT SECTION
# -------------------------

if input_type == "Upload Dataset":

    df = upload_data()

elif input_type == "Manual Data Entry":

    df = manual_input()


if df is not None:

    st.subheader("Input Data Preview")
    st.dataframe(df.head())


    daily = prepare_daily_data(df)


    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        7,
        365,
        30
    )


# -------------------------
# PROPHET FORECAST
# -------------------------

    if model_type == "Prophet Model":

        results = prophet_forecast(
            daily,
            daily_model,
            weekly_model,
            pos_model,
            forecast_days
        )


# -------------------------
# XGBOOST FORECAST
# -------------------------

    if model_type == "XGBoost Model":

        results = xgb_forecast(
            daily,
            forecast_days
        )


    display_results(results)