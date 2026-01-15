# GPTCOT — Roll-Aware CoT Forward Returns

## Purpose
Build a reproducible, daily-runnable dataset that links weekly Commitments of Traders (CoT) signals to forward Henry Hub price moves while respecting contract rolls. The pipeline answers: given CoT values on date *t*, how does the market move over future horizons while measuring the contract that will be front **at the horizon date** (front-at-horizon, not front-at-signal).

## Input Data (read-only)
- `NGCommitofTraders.csv`: weekly CoT series with columns `Date`, `Total_OI`, `Total_MM_Net`, `Total_Prod_Net`, `Total_Swap_Net`.
- `HenryForwardCurve.csv`: daily forward curve with `Date`, `FrontMonth_Label` (e.g., `Feb-2015`), and forward strip columns. Used **only** to identify which contract is front on a given date.
- `HenryHub_Absolute_History.csv`: daily settlements by contract month with columns `TradeDate` plus contract-month headers `YYYY-MM-01`. This is the authoritative price source.

## Front-at-Horizon Contract Resolution (why it matters)
Traditional front-at-signal logic bakes in roll distortion: returns jump when the front rolls even if prices are unchanged. This pipeline instead:
1. Computes `t_h = t + horizon`.
2. Snaps `t_h` backward to the most recent forward-curve date (`fc_snap_date`).
3. Converts `FrontMonth_Label` to a canonical contract key (`YYYY-MM-01`).
4. Uses that contract column in the absolute history to measure returns. The contract measured is the one that will be front **on the horizon date**, preventing roll artifacts.

## Date Snapping Rules (lookahead-safe)
- **Forward curve date:** snap backward to the last available date on/before `t_h`. No forward-filling into the future.
- **Entry price:** snap forward to the first `TradeDate` on/after `t`. If the price is NaN, search forward up to `max_price_snap_days` trading days.
- **Exit price:** snap backward to the last `TradeDate` on/before `t_h`. If the price is NaN, search backward up to `max_price_snap_days` trading days.
- If a price cannot be found within the search window, the row is marked invalid with a reason code.

## CoT Feature Engineering (no lookahead)
- Rolling z-score over 52 observations (`_z_52`) with min periods configurable (default 10); std=0 yields z=0.
- Rolling percentile ranks over 52 and 156 observations (`_pct_52`, `_pct_156`), computed only from history available up to and including `t`.

## CLI Usage
```
python -m gptcot --info-dir "C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/INFO" --output-dir "C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/GPTCOT/output"
python -m gptcot --info-dir ... --output-dir ... --horizons 7,14 --max-price-snap-days 3 --min-periods 12 --write-parquet --log-level DEBUG
```

## Output
Written to `output/cot_forward_returns.csv` (and optional Parquet). Columns:
- `cot_date`, `horizon_days`, `horizon_date`, `fc_snap_date`, `frontmonth_label`, `target_contract_month`
- `entry_date`, `exit_date`, `entry_price`, `exit_price`, `abs_change`, `pct_change`
- Raw CoT: `Total_OI`, `Total_MM_Net`, `Total_Prod_Net`, `Total_Swap_Net`
- Derived features per series: `{series}_z_52`, `{series}_pct_52`, `{series}_pct_156`
- Validation: `is_valid`, `invalid_reason`

Console summary reports CoT rows, horizons processed, valid/invalid counts, and top invalid reasons.

## Market Analysis Report
- Purpose: produces a static HTML summary with a **Forecasted Market Moves** section driven by a trained model instead of analog buckets.
- Forecast logic: loads `output/price_forecast_model.pkl` when available and feeds it the latest COT z-scores/percentiles, storage deviation vs 5-year norms, and forward-curve shape (prompt vs strip, winter-summer spread) to predict the next 30-day front-month move. The report explains the predicted percent move/direction and highlights the top signals driving each call when supported.
- Data hygiene: the report filters to `is_valid` rows and drops anything before 2013-01-03 (no forward curve coverage); if the trained model is missing, the forecast section notes that no prediction is available.
- Forecast table: shows horizons 7/14/28/30 days (or whatever models exist) with a single combined return/direction column and the factors that drove the forecast; if a model file (e.g., `price_forecast_model_14.pkl`) is missing, that row will show as unavailable.
- Run it: `python -m gptcot market-analysis` (rebuilds `cot_forward_returns.csv` every time; add `--no-force` to reuse the last output). No paths are required; the tool locates `INFO` and writes outputs automatically to `Scripts/MarketAnalysis_Report_Output/cot_market_report.html` with accompanying PNG charts.
- The command refreshes the ICE Henry forward curve/absolute history before building charts; set `GPTCOT_SKIP_ICE_REFRESH=1` if you need to skip and reuse the existing `INFO` data.
- Training option: add `--train-models` to `python -m gptcot market-analysis` to fit horizon-specific models from historical COT, storage, and forward-curve data and write them to `output/price_forecast_model_{horizon}.pkl`. The refactored training loop drops NaN targets before fitting, aligns storage/curve metrics via time-aware `merge_asof`, forward/back-fills feature gaps (with model-time imputation as a backstop), splits chronologically (train < 2024-01-01), performs `TimeSeriesSplit` cross-validation over RandomForest hyperparameters, and prints MAE/RMSE scores to the console as each horizon trains.
- Price & Positioning charts now plot dashed, statistically derived support/resistance levels (clustered swing highs/lows via DBSCAN) on the 12m strip and 1yr spread charts with outlier filtering, accompanied by a brief methodological note in the report.
- Forecast table is simplified to a combined “Predicted Return (% / Direction)” column and now lists the key drivers (top feature importances) behind each horizon’s call; when importances are missing, it notes that there is insufficient signal to identify drivers.
- Price & Positioning now also includes last-12-month charts for the front-month and a seasonal contract (Oct or Mar depending on time of year), each annotated with the same statistically derived support/resistance levels.
- Forward curve section: the report also compares NYMEX Henry Hub forward curves across recent lookbacks (0/7/14/28/63 days), showing both absolute prices and prompt-normalized shape. Toggle with `--include-forward-curve/--no-include-forward-curve`.

## Morning Pack
- Generates a static HTML “Morning Pack” with 12m strip (SMA10), prompt/red-front spread shading, EIA storage vs 5Y/10Y bands, COT percentile charts (52w/156w), and a model-based **Forecasted Market Moves** callout.
- Command: `python -m gptcot market-analysis [--no-force]` — the command auto-detects `INFO` (HenryForwardCurve.csv, EIAtotals.csv) and writes outputs to `Scripts/MarketAnalysis_Report_Output`; by default it rebuilds inputs on each run.
- Methodology: Prices reflect 1-Month Constant Maturity Henry Hub via 12-month forward strip averages; storage stats exclude the current year when calculating 5Y averages and 10Y ranges to avoid look-ahead.

(.venv) PS C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\GPTCOT> .\.venv\Scripts\activate
(.venv) PS C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\GPTCOT> python -m gptcot market-analysis --train-models
