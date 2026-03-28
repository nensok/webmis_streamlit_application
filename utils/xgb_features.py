import pandas as pd


# --------------------------------
# TIME FEATURES
# --------------------------------

def create_time_features(df):

    df = df.copy()

    # Ensure datetime
    df["Date_of_diagnosis"] = pd.to_datetime(
        df["Date_of_diagnosis"], errors="coerce"
    )

    # Remove rows with invalid dates
    df = df.dropna(subset=["Date_of_diagnosis"])

    df["day"] = df["Date_of_diagnosis"].dt.day
    df["month"] = df["Date_of_diagnosis"].dt.month
    df["year"] = df["Date_of_diagnosis"].dt.year

    df["dayofweek"] = df["Date_of_diagnosis"].dt.dayofweek
    df["weekofyear"] = (
        df["Date_of_diagnosis"]
        .dt.isocalendar()
        .week
        .astype("int64")
    )

    df["quarter"] = df["Date_of_diagnosis"].dt.quarter

    return df


# --------------------------------
# LAG FEATURES
# --------------------------------

def create_lag_features(df, target, lags=None):

    df = df.copy()

    if lags is None:
        lags = [1,3,7,14,21]

    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    return df


# --------------------------------
# ROLLING WINDOW FEATURES
# --------------------------------

def create_rolling_features(df, target):

    df = df.copy()

    df[f"{target}_roll_mean_7"] = df[target].rolling(7).mean()
    df[f"{target}_roll_mean_14"] = df[target].rolling(14).mean()

    df[f"{target}_roll_std_7"] = df[target].rolling(7).std()

    return df


# --------------------------------
# FULL FEATURE PIPELINE
# --------------------------------

def build_features(df, target):

    df = create_time_features(df)

    df = create_lag_features(df, target)

    df = create_rolling_features(df, target)

    return df