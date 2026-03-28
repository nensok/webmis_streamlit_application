# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs at `http://localhost:8501` by default.

**Note:** `joblib` is used in `utils/xgb_predict.py` but is missing from `requirements.txt`. Add it if setting up a fresh environment.

## Architecture Overview

This is a Streamlit web app for malaria surveillance forecasting, supporting two interchangeable ML backends (Prophet and XGBoost) across three metrics (daily cases, weekly cases, positivity rate).

### Data Flow

```
User Input (upload CSV/XLSX or manual form entry)
  → components/upload_section.py or components/manual_input_section.py
  → returns pandas DataFrame
  → utils/preprocessing.py::prepare_daily_data()   # aggregates by date, computes positivity rate
  → utils/prophet_predict.py OR utils/xgb_predict.py
  → components/results_display.py                  # tabbed charts + tables
```

### Key Design Points

**Model loading:** Both forecasting modules use `@st.cache_resource` to load pre-trained models once per session. Prophet models are stored as JSON (`models/*.json`), XGBoost as joblib pickles (`models/*.pkl`).

**XGBoost forecasting is recursive:** Each future step re-engineers features (time, lags, rolling stats) using the previous step's prediction appended as new history. See `utils/xgb_features.py` for the feature set (time features + 5 lag features + 4 rolling features).

**Prophet forecasting** uses standard `make_future_dataframe` → `predict` with uncertainty intervals (yhat_lower/yhat_upper). XGBoost produces point predictions only.

**Expected DataFrame schema** after `prepare_daily_data()`:
- Index: `Date_of_diagnosis` (datetime)
- Columns: `tests` (int), `positives` (int), `positivity_rate` (float)

**Forecast output schema** (both models return same structure):
- Dict with keys `"daily"`, `"weekly"`, `"positivity"` — each a DataFrame with at minimum `ds` and `yhat` columns.

### Known Issues

- `utils/xgb_predict.py` uses deprecated `fillna(method="ffill")` (pandas 2.0+ requires `ffill()` directly)
- `utils/preprocessing.py` references `Result_of_diagnosis` column which is absent in manual input data
