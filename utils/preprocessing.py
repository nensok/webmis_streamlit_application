import pandas as pd


def prepare_daily_data(df):
    df = df.copy()
    df["Date_of_diagnosis"] = pd.to_datetime(df["Date_of_diagnosis"])

    # Raw patient records: derive positives from Result_of_diagnosis text
    if "Result_of_diagnosis" in df.columns:
        df["Result_of_diagnosis"] = df["Result_of_diagnosis"].str.lower()
        df["positive"] = df["Result_of_diagnosis"].str.contains("plasmodium", na=False).astype(int)
        daily = (
            df.groupby("Date_of_diagnosis")
            .agg(tests=("positive", "count"), positives=("positive", "sum"))
            .reset_index()
        )
        daily["positivity_rate"] = daily["positives"] / daily["tests"]

    else:
        # Pre-aggregated input (manual entry or aggregated CSV)
        tested_col = "tested" if "tested" in df.columns else "tests"
        daily = df[["Date_of_diagnosis"]].copy()
        daily["tests"] = pd.to_numeric(df[tested_col], errors="coerce").fillna(0).astype(int)
        daily["positives"] = pd.to_numeric(df["positives"], errors="coerce").fillna(0).astype(int)
        daily = daily.groupby("Date_of_diagnosis").sum().reset_index()
        daily["positivity_rate"] = daily["positives"] / daily["tests"].replace(0, float("nan"))
        daily["positivity_rate"] = daily["positivity_rate"].fillna(0)

    return daily


def validate_data(daily):
    warnings = []

    n = len(daily)
    if n < 21:
        warnings.append(
            f"Only {n} days of data found. XGBoost requires at least 21 days for reliable "
            "lag/rolling features — forecasts may be inaccurate."
        )

    # Check for date gaps
    dates = pd.to_datetime(daily["Date_of_diagnosis"]).sort_values().reset_index(drop=True)
    gaps = dates.diff().dt.days.dropna()
    gap_count = int((gaps > 1).sum())
    if gap_count > 0:
        warnings.append(
            f"{gap_count} gap(s) detected in the date sequence. Missing dates silently distort "
            "lag and rolling features used by XGBoost."
        )

    # Implausible values
    if (daily["positives"] < 0).any():
        warnings.append("Negative values detected in 'positives'. Please check your data.")
    if (daily["tests"] < 0).any():
        warnings.append("Negative values detected in 'tests'. Please check your data.")
    if "positivity_rate" in daily.columns and (daily["positivity_rate"] > 1.0).any():
        warnings.append(
            "Positivity rate exceeds 100% for one or more dates. "
            "Check that 'positives' does not exceed 'tests'."
        )
    if (daily["positives"] > daily["tests"]).any():
        warnings.append("Number of positives exceeds number of tests for one or more dates.")

    return warnings
