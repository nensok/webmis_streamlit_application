import pandas as pd
import numpy as np
import joblib


from utils.xgb_features import (
    create_time_features,
    create_lag_features,
    create_rolling_features
)


# -----------------------------
# LOAD MODELS
# -----------------------------

daily_model = joblib.load("models/xgb_daily_cases.pkl")
weekly_model = joblib.load("models/xgb_weekly_cases.pkl")
pos_model = joblib.load("models/xgb_positivity_rate.pkl")


# -----------------------------
# RECURSIVE FORECAST FUNCTION
# -----------------------------

def recursive_forecast(df, model, target, horizon):

    data = df.copy().reset_index(drop=True)

    predictions = []

    for i in range(horizon):

        last_date = data["Date_of_diagnosis"].max()

        next_date = last_date + pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            "Date_of_diagnosis":[next_date]
        })

        data = pd.concat([data,new_row],ignore_index=True)

        # rebuild features
        data = create_time_features(data)
        data = create_lag_features(data,target)
        data = create_rolling_features(data,target)

        # Fill missing lag/rolling values instead of dropping rows
        data = data.ffill()

        # ensure dataset still has rows
        if data.empty:
            raise ValueError("Feature engineering removed all rows")

        # select model features
        features = list(model.feature_names_in_)

        # Ensure all expected columns exist
        for col in features:
            if col not in data.columns:
                data[col] = 0

        # Select ONLY training features
        X = data.loc[data.index[-1], features].to_frame().T

        X = data.iloc[-1:][features]

        pred = model.predict(X)[0]

        data.loc[data.index[-1],target] = pred

        predictions.append(pred)

    forecast = pd.DataFrame({
        "ds":pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon
        ),
        "yhat":predictions
    })

    return forecast


# -----------------------------
# MAIN XGBOOST FORECAST PIPELINE
# -----------------------------

def xgb_forecast(daily_data, forecast_days):

   df = daily_data.copy()

   df["Date_of_diagnosis"] = pd.to_datetime(df["Date_of_diagnosis"])

   df = create_time_features(df)

   df = create_lag_features(df,"positives")
   df = create_rolling_features(df,"positives")

   df = create_lag_features(df,"positivity_rate")
   df = create_rolling_features(df,"positivity_rate")

   df = df.ffill()
   df = df.fillna(0)


    # -------------------------
    # DAILY FORECAST
    # -------------------------

   daily_forecast = recursive_forecast(
        df,
        daily_model,
        "positives",
        forecast_days
    )


    # -------------------------
    # WEEKLY FORECAST
    # -------------------------

   weekly = df.set_index("Date_of_diagnosis").resample("W").sum().reset_index()
   weekly_forecast = recursive_forecast(
        weekly,
        weekly_model,
        "positives",
        12
    )


    # -------------------------
    # POSITIVITY RATE FORECAST
    # -------------------------

   pos_forecast = recursive_forecast(
        df,
        pos_model,
        "positivity_rate",
        forecast_days
    )


   return {
        "daily": daily_forecast,
        "weekly": weekly_forecast,
        "positivity": pos_forecast
    }