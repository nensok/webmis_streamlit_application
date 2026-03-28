import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _metrics(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mask = actual != 0
    if mask.any():
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    else:
        mape = float("nan")
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 1) if not np.isnan(mape) else "N/A",
    }


def backtest_xgb(daily, holdout_days=30):
    """
    Hold out the last `holdout_days` rows, forecast on the training portion,
    and compare predictions to actuals. Returns (metrics_dict, error_message).
    """
    from utils.xgb_predict import xgb_forecast

    min_required = holdout_days + 21
    if len(daily) <= min_required:
        return None, (
            f"Need at least {min_required} days of data for a {holdout_days}-day holdout "
            f"(have {len(daily)})."
        )

    train = daily.iloc[:-holdout_days].copy()
    test = daily.iloc[-holdout_days:].copy()

    results = xgb_forecast(train, holdout_days)
    metrics = {}

    # Daily cases
    daily_fc = results["daily"].copy()
    daily_fc["ds"] = pd.to_datetime(daily_fc["ds"])
    test_daily = test[["Date_of_diagnosis", "positives"]].copy()
    test_daily["Date_of_diagnosis"] = pd.to_datetime(test_daily["Date_of_diagnosis"])
    merged = pd.merge(
        daily_fc.rename(columns={"ds": "Date_of_diagnosis", "yhat": "pred"}),
        test_daily,
        on="Date_of_diagnosis",
        how="inner",
    )
    if len(merged) >= 3:
        metrics["Daily Cases"] = _metrics(merged["positives"], merged["pred"])

    # Positivity rate
    pos_fc = results["positivity"].copy()
    pos_fc["ds"] = pd.to_datetime(pos_fc["ds"])
    test_pos = test[["Date_of_diagnosis", "positivity_rate"]].copy()
    test_pos["Date_of_diagnosis"] = pd.to_datetime(test_pos["Date_of_diagnosis"])
    merged_pos = pd.merge(
        pos_fc.rename(columns={"ds": "Date_of_diagnosis", "yhat": "pred"}),
        test_pos,
        on="Date_of_diagnosis",
        how="inner",
    )
    if len(merged_pos) >= 3:
        metrics["Positivity Rate"] = _metrics(merged_pos["positivity_rate"], merged_pos["pred"])

    return (metrics if metrics else None), None


def backtest_prophet(daily, prophet_results):
    """
    Compare Prophet in-sample fitted values against user input actuals for
    any dates that overlap with the model's training period.
    Returns metrics_dict or None if no overlap found.
    """
    metrics = {}

    for key, actual_col in [("daily", "positives"), ("positivity", "positivity_rate")]:
        fc = prophet_results[key][["ds", "yhat"]].copy()
        fc["ds"] = pd.to_datetime(fc["ds"])

        actuals = daily[["Date_of_diagnosis", actual_col]].copy()
        actuals["Date_of_diagnosis"] = pd.to_datetime(actuals["Date_of_diagnosis"])

        merged = pd.merge(
            fc.rename(columns={"ds": "Date_of_diagnosis", "yhat": "pred"}),
            actuals,
            on="Date_of_diagnosis",
            how="inner",
        )

        if len(merged) >= 5:
            label = "Daily Cases" if key == "daily" else "Positivity Rate"
            metrics[label] = _metrics(merged[actual_col], merged["pred"])

    return metrics if metrics else None
