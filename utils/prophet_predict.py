import pandas as pd


def prophet_forecast(daily, daily_model, weekly_model, pos_model, days):


    future_daily = daily_model.make_future_dataframe(periods=days)
    forecast_daily = daily_model.predict(future_daily)


    weekly = daily.set_index("Date_of_diagnosis").resample("W").sum().reset_index()

    future_week = weekly_model.make_future_dataframe(periods=12,freq="W")
    forecast_week = weekly_model.predict(future_week)


    future_pos = pos_model.make_future_dataframe(periods=days)
    forecast_pos = pos_model.predict(future_pos)


    return {
        "daily":forecast_daily,
        "weekly":forecast_week,
        "positivity":forecast_pos
    }