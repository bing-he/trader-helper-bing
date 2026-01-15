# EIA Storage Forecast Stack

## Overview

This workspace contains the refactored EIAGuesser-style storage forecaster, a support playground, diagnostics scripts, and the ingestion tools required to keep everything deterministic and auditable. The goal is to deliver quantile forecasts for each U.S. storage region plus the Lower 48 total without data leakage, while maintaining traceable calibration statistics and reconciliation logic that can be reviewed by traders or data scientists before deployment.

## Ingestion & base inputs

- **`DataWrangler.py`**: consolidates Criterion storage changes, Fundy fundamentals, weather, prices, and other `INFO/` sources into two canonical outputs: `output/Combined_Wide_Data.csv` (daily features indexed by date) and `output/EIAchanges.csv` (weekly storage deltas per region). The script maps states/stations to regions, normalizes column names, writes to `data_wrangler.log`, and ensures the downstream forecaster always sees consistent feature names.
- **`INFO/` archive**: raw CSVs (Criterion*, Fundy, WEATHER, PRICES, locs_list, etc.) live here. The daily features and weekly storage tables that the forecast engines consume ultimately derive from these files, and `eia_forecaster.py` also expects ready-made aggregates under `INFO/CompiledInfo/csv/`.
- **`models/`**: stores CatBoost binary snapshots (`<region>_point`, `<region>_q10`, `<region>_q50`, `<region>_q90`) that `forecaster_refactored.py` can load to skip retraining if desired; the hyperparameters that produced these artifacts live in `best_params_<region>.json`.

## Modeling & forecasting

- **`forecaster_refactored.py`** (production engine)
  * Consumes `output/Combined_Wide_Data.csv` and `output/EIAchanges.csv`, builds weekly Thursday-ended matrices, and normalizes by deriving 7-day sequences per region before training.
  * Trains CatBoost quantile regressors per region with tuned hyperparameters (`best_params_<region>.json`), applies conformal calibration offsets, and reconciles the regional q50 forecasts to match the Lower 48 aggregate using capacity-weighted regional weights (`REGIONAL_WEIGHTS`).
  * Widened intervals ensure reconciliation uncertainty is captured, backtest + calibration metrics are logged, and outputs are delivered to `output/forecaster/` (forecasts, diagnostics, metadata, summary text).
- **`forecaster_unbundled.py`**
  * Offers reusable helper functions that mirror the refactored pipeline (feature building, training, reconciliation) but expose each stage for experimentation or unit testing.
  * Includes an Optuna tuning routine that searches CatBoost hyperparameters across defined trials and returns the best configuration for retraining.
- **`eia_forecaster.py`** (playground)
  * Loads the pre-aggregated combined daily historical + forecast tables under `INFO/CompiledInfo/csv/`, derives weekly driver matrices, and trains a `GradientBoostingRegressor` per change series to produce next-week EIA forecasts for plotting and logging.
  * Validates every required path before running, aligns forecast columns with historical drivers (Fundy_FC_, Weather_FC_), and logs backtest/forecast stats via structured logging plus optional ANSI tables when `tabulate` is installed.
  * Optuna tuning is available by updating the `OPTUNA_TRIALS` constant near the top of the file; when enabled, a TPE sampler search minimizes average validation MAE across all regions and reuses the best parameter set for both forecasting and the diagnostic playground, keeping the flow deterministic (same RNG seeds for Python/NumPy/Optuna).

## Diagnostics & validation

- **`validate_refactoring.py`**: shows why the refactored pipeline avoids look-ahead bias, prevents feature-selection leakage, reconciles with variance weights, and tracks calibration (coverage + trading impact). Run this script whenever you modify preprocessing or modeling to feel confident about the production behavior.
- **`feat_corr.py`**: correlation screening report. Loads `output/Combined_Wide_Data.csv`, deduplicates columns, filters low-variance/high-missing columns, ranks Spearman correlations (SciPy or fallback), and writes histograms/heatmaps plus diagnostic CSVs under `output/corr_report/`.
- **`analysis.py`**: region-wise statistics for the combined feature table; sums columns per region, logs sub-type coverage, and warns about missing columns.

## Outputs & key statistics

- `output/Combined_Wide_Data.csv`: the canonical daily feature matrix produced by `DataWrangler.py`.
- `output/EIAchanges.csv`: weekly storage change table used for targets.
- `output/forecaster/`: consolidated production artifacts:
  * `next_week_forecast.csv`: q10/q50/q90 plus point forecasts and interval membership flags.
  * `backtest_metrics.csv`: cross-validation MAE, bias, half-life, and pinball losses per region (latest run shown below).
  * `calibration_metrics.csv`: empirical coverage vs. ideal 10/50/90 targets (see table).
  * `feature_importance.csv`, `residuals.csv`, `backtest_predictions.csv`: diagnostics for auditing features and residual distributions.
  * `summary_table.txt`: human-readable summary reused by dashboards or Slack reports.
  * `run_config.json`: metadata (input paths, forecast week, reconciliation weights, methodology notes, ensemble targets).

### Latest validation metrics

| Region | CV MAE (Bcf) | Bias | Pinball Loss (Q10 / Q50 / Q90) | Coverage (Q10 / Q50 / Q90) |
| --- | --- | --- | --- | --- |
| Lower 48 | 13.60 | +0.04 | 4.52 / 7.18 / 5.15 | 23.8% / 53.8% / 80.0% |
| East | 4.21 | -0.94 | 1.60 / 2.05 / 1.28 | 20.6% / 55.0% / 80.0% |
| Midwest | 3.86 | -0.46 | 1.67 / 1.97 / 1.27 | 14.4% / 55.6% / 83.8% |
| South Central | 9.51 | +0.22 | 3.69 / 4.72 / 3.37 | 23.8% / 56.9% / 86.9% |
| Mountain | 1.65 | +0.06 | 0.53 / 0.85 / 0.63 | 9.4% / 44.4% / 83.8% |
| Pacific | 2.50 | +0.33 | 0.87 / 1.27 / 0.86 | 13.8% / 49.4% / 78.1% |

Coverage targets remain 10% / 50% / 90%, so the intervals are conservative by design and the pinball metrics track interval sharpness.

## Suggested workflow

1. Refresh the raw sources under `INFO/` (Criterion*, Fundy, WEATHER, PRICES, etc.).
2. Run `python3 DataWrangler.py` to rebuild `output/Combined_Wide_Data.csv` and `output/EIAchanges.csv`.
3. Execute `python3 forecaster_refactored.py` (or `forecaster_unbundled.py`) to train and reconcile the quantile forecasts; check `output/forecaster/run_config.json` for metadata.
4. Run `python3 validate_refactoring.py` whenever you change model/feature logic to confirm no leakage, proper reconciliation, and calibration coverage.
5. Inspect `output/forecaster/backtest_metrics.csv` / `calibration_metrics.csv` for MAE/bias drift; adjust the `half_life` parameter or revisit feature stability if the errors rise.
6. Use `feat_corr.py` and `analysis.py` regularly to spot disappearing features or gaps in region coverage, then refresh the CatBoost binaries in `models/` if the structure shifts.
