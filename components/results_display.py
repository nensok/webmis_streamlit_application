import streamlit as st
import pandas as pd
import plotly.graph_objects as go


PROPHET_COLOR = "#1f77b4"
XGB_COLOR = "#ff7f0e"


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _forecast_chart(hist_x, hist_y, fc1, fc1_label, fc1_color,
                    fc2=None, fc2_label=None, fc2_color=None,
                    y_title=""):
    fig = go.Figure()

    # Historical line
    if hist_x is not None:
        fig.add_trace(go.Scatter(
            x=hist_x, y=hist_y,
            name="Historical",
            line=dict(color="grey", width=1.5),
            mode="lines",
        ))

    # Uncertainty band for fc1 (Prophet provides yhat_lower/yhat_upper)
    if "yhat_lower" in fc1.columns and "yhat_upper" in fc1.columns:
        r, g, b = _hex_to_rgb(fc1_color)
        fig.add_trace(go.Scatter(
            x=pd.concat([fc1["ds"], fc1["ds"].iloc[::-1]]),
            y=pd.concat([fc1["yhat_upper"], fc1["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{fc1_label} Uncertainty",
            hoverinfo="skip",
        ))

    # Primary forecast line
    fig.add_trace(go.Scatter(
        x=fc1["ds"], y=fc1["yhat"],
        name=fc1_label,
        line=dict(color=fc1_color, width=2),
        mode="lines",
    ))

    # Second model (comparison mode)
    if fc2 is not None and fc2_label is not None:
        if "yhat_lower" in fc2.columns and "yhat_upper" in fc2.columns:
            r, g, b = _hex_to_rgb(fc2_color)
            fig.add_trace(go.Scatter(
                x=pd.concat([fc2["ds"], fc2["ds"].iloc[::-1]]),
                y=pd.concat([fc2["yhat_upper"], fc2["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{fc2_label} Uncertainty",
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=fc2["ds"], y=fc2["yhat"],
            name=fc2_label,
            line=dict(color=fc2_color, width=2),
            mode="lines",
        ))

    fig.update_layout(
        yaxis_title=y_title,
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    return fig


def _show_table(df, rows):
    cols = ["ds", "yhat"]
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        cols += ["yhat_lower", "yhat_upper"]
    display = df[cols].tail(rows).copy()
    rename = {"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}
    display = display.rename(columns=rename)
    st.dataframe(display, use_container_width=True)


def _download_btn(df, filename, key):
    cols = ["ds", "yhat"]
    if "yhat_lower" in df.columns:
        cols += ["yhat_lower", "yhat_upper"]
    st.download_button(
        label="Download CSV",
        data=df[cols].to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def display_results(results, historical=None,
                    model_label="Prophet Model",
                    results2=None, model_label2=None):

    st.header("Forecast Results")

    fc1_color = PROPHET_COLOR if "Prophet" in model_label else XGB_COLOR
    fc2_color = (XGB_COLOR if "Prophet" in model_label else PROPHET_COLOR) if model_label2 else None

    # Weekly historical (resampled)
    hist_weekly_x = hist_weekly_y = None
    if historical is not None:
        hist_weekly = (
            historical.set_index("Date_of_diagnosis")["positives"]
            .resample("W").sum()
            .reset_index()
        )
        hist_weekly_x = hist_weekly["Date_of_diagnosis"]
        hist_weekly_y = hist_weekly["positives"]

    tab1, tab2, tab3 = st.tabs(["Daily Cases", "Weekly Cases", "Positivity Rate"])

    with tab1:
        hist_x = historical["Date_of_diagnosis"] if historical is not None else None
        hist_y = historical["positives"] if historical is not None else None
        fig = _forecast_chart(
            hist_x, hist_y,
            results["daily"], model_label, fc1_color,
            fc2=results2["daily"] if results2 else None,
            fc2_label=model_label2, fc2_color=fc2_color,
            y_title="Daily Cases",
        )
        st.plotly_chart(fig, use_container_width=True)
        _show_table(results["daily"], 30)
        _download_btn(results["daily"], "daily_forecast.csv", key="dl_daily")

    with tab2:
        fig = _forecast_chart(
            hist_weekly_x, hist_weekly_y,
            results["weekly"], model_label, fc1_color,
            fc2=results2["weekly"] if results2 else None,
            fc2_label=model_label2, fc2_color=fc2_color,
            y_title="Weekly Cases",
        )
        st.plotly_chart(fig, use_container_width=True)
        _show_table(results["weekly"], 12)
        _download_btn(results["weekly"], "weekly_forecast.csv", key="dl_weekly")

    with tab3:
        hist_x = historical["Date_of_diagnosis"] if historical is not None else None
        hist_y = historical["positivity_rate"] if historical is not None else None
        fig = _forecast_chart(
            hist_x, hist_y,
            results["positivity"], model_label, fc1_color,
            fc2=results2["positivity"] if results2 else None,
            fc2_label=model_label2, fc2_color=fc2_color,
            y_title="Positivity Rate",
        )
        st.plotly_chart(fig, use_container_width=True)
        _show_table(results["positivity"], 30)
        _download_btn(results["positivity"], "positivity_forecast.csv", key="dl_positivity")
