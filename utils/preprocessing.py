import pandas as pd

def prepare_daily_data(df):

    df["Date_of_diagnosis"] = pd.to_datetime(df["Date_of_diagnosis"])
    df["Result_of_diagnosis"] = df["Result_of_diagnosis"].str.lower()

    df["positive"] = df["Result_of_diagnosis"].str.contains("plasmodium").astype(int)

    daily = (
        df.groupby("Date_of_diagnosis")
        .agg(
            tests=("positive","count"),
            positives=("positive","sum")
        )
        .reset_index()
    )

    daily["positivity_rate"] = daily["positives"] / daily["tests"]

    return daily


