from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .exogenous import ExogenousBundle
from .utils.logging import get_logger
from .utils.stats import rolling_pct_rank


class FeatureEngineer:
    """Compute core flow, stress, regime, and target features."""

    def __init__(
        self,
        data: pd.DataFrame,
        active_pipes: List[str],
        config: Config,
        exogenous: Optional[ExogenousBundle] = None,
    ) -> None:
        self.df = data.copy()
        self.pipes = active_pipes
        self.cfg = config
        self.exogenous = exogenous
        self.logger = get_logger(__name__)
        self.feature_meta: Dict[str, object] = {
            "fundy_items": [],
            "fundy_tightness_source": None,
            "exog_coverage": {},
        }

    def _rolling_zscore(self, series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window=window, min_periods=max(3, window // 2)).mean()
        std = series.rolling(window=window, min_periods=max(3, window // 2)).std()
        zscore = (series - mean) / std
        return zscore.replace([np.inf, -np.inf], np.nan)

    def _align_series(self, series: pd.Series) -> pd.Series:
        if series.index.has_duplicates:
            series = series.groupby(series.index).mean()
        return series.reindex(self.df.index)

    def _select_report_date(self) -> pd.Timestamp:
        if "Target_Ret_Henry" in self.df.columns:
            valid = self.df["Target_Ret_Henry"].notna()
            if valid.any():
                return self.df.index[valid][-1]
        return self.df.index[-1]

    def _set_scalar_feature(self, name: str, value: Optional[float], date: pd.Timestamp) -> None:
        if value is None or not np.isfinite(value):
            return
        series = pd.Series(index=self.df.index, dtype=float)
        series.loc[date] = float(value)
        self.df[name] = series

    def _calc_forecast_impulse(
        self,
        hist_series: pd.Series,
        forecast_series: pd.Series,
        anchor_date: pd.Timestamp,
        horizon_days: int = 7,
        min_days: int = 3,
    ) -> Optional[float]:
        if hist_series.empty or forecast_series.empty:
            return None
        forecast_slice = forecast_series.loc[
            anchor_date + pd.Timedelta(days=1) : anchor_date + pd.Timedelta(days=horizon_days)
        ].dropna()
        hist_slice = hist_series.loc[
            anchor_date - pd.Timedelta(days=horizon_days - 1) : anchor_date
        ].dropna()
        if len(forecast_slice) < min_days or len(hist_slice) < min_days:
            return None
        return float(forecast_slice.mean() - hist_slice.mean())

    def _fundy_item_coverage(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby("Item")["Value"].apply(lambda s: s.notna().sum()).sort_values(
            ascending=False
        )

    def _select_fundy_items(self, df: pd.DataFrame, top_n: int) -> Tuple[List[str], pd.Series]:
        coverage = self._fundy_item_coverage(df)
        items = []
        if "Balance" in coverage.index:
            items.append("Balance")
        for item in coverage.index:
            if item == "Balance":
                continue
            items.append(item)
            if len(items) >= top_n:
                break
        return items, coverage

    def _fundy_proxy_series(
        self, df: pd.DataFrame, items: pd.Series
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        if "Balance" in items.index:
            balance = df[df["Item"] == "Balance"]
            series = balance.pivot_table(index="Date", values="Value", aggfunc="mean")["Value"]
            return series, "Balance"

        lower_items = items.index.to_series().str.lower()
        demand_mask = lower_items.str.contains("demand") | lower_items.str.contains("consumption")
        supply_mask = lower_items.str.contains("supply") | lower_items.str.contains("production")
        demand_items = items[demand_mask]
        supply_items = items[supply_mask]
        if demand_items.empty or supply_items.empty:
            return None, None

        demand_item = demand_items.index[0]
        supply_item = supply_items.index[0]

        demand_series = df[df["Item"] == demand_item].pivot_table(
            index="Date", values="Value", aggfunc="mean"
        )["Value"]
        supply_series = df[df["Item"] == supply_item].pivot_table(
            index="Date", values="Value", aggfunc="mean"
        )["Value"]
        # Tightness proxy = demand minus supply when Balance is unavailable.
        proxy = demand_series - supply_series
        return proxy, f"{demand_item} minus {supply_item}"

    def calc_pipe_metrics(self) -> None:
        for pipe in self.pipes:
            flow = self.df[f"Flow_{pipe}"]
            cap = self.df[f"Cap_{pipe}"]
            valid_mask = cap.notna() & (cap > 0) & flow.notna()
            util_raw = pd.Series(index=self.df.index, dtype=float)
            util_raw[valid_mask] = flow[valid_mask] / cap[valid_mask]
            self.df[f"Util_{pipe}"] = util_raw
            self.df[f"Util_Clipped_{pipe}"] = util_raw.clip(lower=0, upper=self.cfg.util_clip_max)
            self.df[f"Util_Anomaly_{pipe}"] = util_raw > self.cfg.util_anomaly_threshold
            spare = pd.Series(index=self.df.index, dtype=float)
            spare[valid_mask] = (cap[valid_mask] - flow[valid_mask]).clip(lower=0)
            self.df[f"Spare_{pipe}"] = spare

    def calc_system_stress(self) -> None:
        util_cols = [f"Util_Clipped_{p}" for p in self.pipes]
        self.df["Stress_P90"] = self.df[util_cols].quantile(0.90, axis=1)
        self.df["Stress_Max"] = self.df[util_cols].max(axis=1)
        flow_cols = [f"Flow_{p}" for p in self.pipes]
        cap_cols = [f"Cap_{p}" for p in self.pipes]
        total_flow = self.df[flow_cols].sum(axis=1, min_count=1)
        total_cap = self.df[cap_cols].sum(axis=1, min_count=1)
        self.df["Stress_Weighted"] = total_flow / total_cap
        self.df["MSI"] = (
            0.5 * self.df["Stress_P90"]
            + 0.3 * self.df["Stress_Max"]
            + 0.2 * self.df["Stress_Weighted"]
        )

    def calc_substitution_failure(self) -> None:
        total_spare = self.df[[f"Spare_{p}" for p in self.pipes]].sum(axis=1, min_count=1)
        total_flow = self.df[[f"Flow_{p}" for p in self.pipes]].sum(axis=1, min_count=1)
        flow_shock = total_flow.diff().abs().rolling(window=self.cfg.fragility_window).std()
        flow_shock = flow_shock.replace(0, 0.01).fillna(1)
        self.df["Substitution_Ratio"] = total_spare / flow_shock
        self.df["Sub_Failure_Score"] = 100 / self.df["Substitution_Ratio"].replace(0, 0.1)

    def calc_rolling_regimes(self) -> None:
        self.df["MSI_Pct_Rank"] = rolling_pct_rank(
            self.df["MSI"],
            window=self.cfg.rolling_window_days,
            min_periods=self.cfg.min_data_period,
        )
        conditions = [
            (self.df["MSI_Pct_Rank"] < self.cfg.pct_slack),
            (self.df["MSI_Pct_Rank"] < self.cfg.pct_transition),
        ]
        choices = ["Slack", "Transition"]
        self.df["Regime"] = np.select(conditions, choices, default="Binding")

    def add_targets(self) -> None:
        self.df["Ret_Henry"] = np.log(self.df["Henry"] / self.df["Henry"].shift(1))
        self.df["Target_Ret_Henry"] = self.df["Ret_Henry"].shift(-1)
        for hub in self.cfg.target_hubs:
            if hub in self.df.columns:
                self.df[f"Basis_{hub}"] = self.df[hub] - self.df["Henry"]
                self.df[f"Target_Delta_Basis_{hub}"] = (
                    self.df[f"Basis_{hub}"].diff().shift(-1)
                )

    def calc_upstream_shocks(self) -> None:
        """Detect upstream supply shocks masked by stable capacity."""
        shocks = pd.DataFrame(index=self.df.index)
        sigma = self.cfg.upstream_shock_sigma
        cap_stable = self.cfg.cap_stable_pct
        for pipe in self.pipes:
            d_flow = self.df[f"Flow_{pipe}"].diff()
            d_cap = self.df[f"Cap_{pipe}"].diff()
            flow_thresh = self.df[f"Flow_{pipe}"].rolling(self.cfg.fragility_window).std()
            flow_thresh = flow_thresh.fillna(0) * -sigma
            cap_series = self.df[f"Cap_{pipe}"]
            valid_mask = cap_series.notna() & (cap_series > 0)
            cond_flow = valid_mask & (d_flow < flow_thresh)
            cap_drop_thresh = cap_series * -cap_stable
            cond_cap = valid_mask & (d_cap > cap_drop_thresh)
            cond_price = self.df["Ret_Henry"] > 0
            shocks[pipe] = cond_flow & cond_cap & cond_price

        self.df["Upstream_Shock_Count"] = shocks.sum(axis=1)
        self.df["Upstream_Shock_Pipe"] = shocks.apply(
            lambda row: row.index[row].tolist()[0] if row.any() else "None",
            axis=1,
        )

    def add_exogenous_features(self) -> None:
        if not self.exogenous:
            self.logger.info("No exogenous bundle supplied; skipping exogenous features.")
            return

        report_date = self._select_report_date()
        index_len = len(self.df.index)
        coverage: Dict[str, Dict[str, int]] = {}

        if self.exogenous.lng_hist is not None:
            lng_daily = self.exogenous.lng_hist.groupby("Date")["Value"].sum(min_count=1)
            lng_daily = lng_daily.sort_index()
            lng_series = self._align_series(lng_daily)
            self.df["lng_gulf_total"] = lng_series
            self.df["lng_gulf_1d_chg"] = lng_series.diff(1)
            self.df["lng_gulf_7d_chg"] = lng_series.diff(7)
            self.df["lng_gulf_30d_z"] = self._rolling_zscore(lng_series, 30)

            forecast_days = 0
            if self.exogenous.lng_forecast is not None:
                lng_fc = self.exogenous.lng_forecast.groupby("Date")["Value"].sum(min_count=1)
                lng_fc = lng_fc.sort_index()
                forecast_days = int(lng_fc.notna().sum())
                impulse = self._calc_forecast_impulse(lng_daily, lng_fc, report_date)
                self._set_scalar_feature(
                    "lng_gulf_forecast_7d_avg_minus_last7d", impulse, report_date
                )

            hist_days = int(lng_series.notna().sum())
            coverage["lng"] = {
                "hist_days": hist_days,
                "forecast_days": forecast_days,
                "missing_days": max(index_len - hist_days, 0),
            }
        else:
            self.logger.warning("LNG inputs missing; LNG features skipped.")

        if self.exogenous.fundy_hist is not None:
            fundy = self.exogenous.fundy_hist.copy()
            fundy["Region"] = fundy["Region"].str.strip()
            fundy_sc = fundy[fundy["Region"].str.lower() == "southcentral"]
            if fundy_sc.empty:
                self.logger.warning("Fundy SouthCentral data missing; fundy features skipped.")
            else:
                selected_items, item_coverage = self._select_fundy_items(
                    fundy_sc, self.cfg.fundy_item_limit
                )
                self.feature_meta["fundy_items"] = selected_items

                pivot = fundy_sc[fundy_sc["Item"].isin(selected_items)].pivot_table(
                    index="Date",
                    columns="Item",
                    values="Value",
                    aggfunc="mean",
                )
                for item in selected_items:
                    if item not in pivot.columns:
                        continue
                    series = self._align_series(pivot[item])
                    col = f"SouthCentral__{item}"
                    self.df[col] = series
                    self.df[f"{col}_7d_chg"] = series.diff(7)

                forecast_days = 0
                if self.exogenous.fundy_forecast is not None:
                    fc = self.exogenous.fundy_forecast.copy()
                    fc["Region"] = fc["Region"].str.strip()
                    fundy_fc = fc[fc["Region"].str.lower() == "southcentral"]
                    if not fundy_fc.empty:
                        fc_pivot = fundy_fc[fundy_fc["Item"].isin(selected_items)].pivot_table(
                            index="Date",
                            columns="Item",
                            values="Value",
                            aggfunc="mean",
                        )
                        forecast_days = int(fc_pivot.notna().sum().sum())
                        for item in selected_items:
                            if item not in pivot.columns or item not in fc_pivot.columns:
                                continue
                            hist_series = pivot[item].sort_index()
                            fc_series = fc_pivot[item].sort_index()
                            impulse = self._calc_forecast_impulse(
                                hist_series, fc_series, report_date
                            )
                            self._set_scalar_feature(
                                f"SouthCentral__{item}_forecast_7d_avg_minus_last7d",
                                impulse,
                                report_date,
                            )

                proxy_series, proxy_source = self._fundy_proxy_series(fundy_sc, item_coverage)
                if proxy_series is not None:
                    proxy_series = self._align_series(proxy_series.sort_index())
                    self.df["southcentral_tightness_proxy"] = proxy_series
                    self.df["southcentral_tightness_proxy_7d_chg"] = proxy_series.diff(7)
                    self.feature_meta["fundy_tightness_source"] = proxy_source

                hist_days = int(pivot.notna().any(axis=1).sum())
                coverage["fundy"] = {
                    "hist_days": hist_days,
                    "forecast_days": forecast_days,
                    "missing_days": max(index_len - hist_days, 0),
                }
        else:
            self.logger.warning("Fundy inputs missing; fundy features skipped.")

        if self.exogenous.ercot_load_hist is not None:
            load_series = self._align_series(self.exogenous.ercot_load_hist)
            self.df["ercot_load_total"] = load_series
            self.df["ercot_load_7d_chg"] = load_series.diff(7)
            self.df["ercot_load_7d_z"] = self._rolling_zscore(load_series, 7)

            forecast_days = 0
            if self.exogenous.ercot_load_forecast is not None:
                load_fc = self.exogenous.ercot_load_forecast.sort_index()
                forecast_days = int(load_fc.notna().sum())
                impulse = self._calc_forecast_impulse(
                    self.exogenous.ercot_load_hist.sort_index(),
                    load_fc,
                    report_date,
                )
                self._set_scalar_feature(
                    "ercot_load_forecast_7d_avg_minus_last7d", impulse, report_date
                )

            hist_days = int(load_series.notna().sum())
            coverage["ercot_load"] = {
                "hist_days": hist_days,
                "forecast_days": forecast_days,
                "missing_days": max(index_len - hist_days, 0),
            }
        else:
            self.logger.warning("ERCOT load inputs missing; load features skipped.")

        if self.exogenous.power_prices_daily is not None:
            prices = self.exogenous.power_prices_daily.sort_index()
            max_lmp = self._align_series(prices["ercot_max_lmp"])
            self.df["ercot_max_lmp"] = max_lmp
            if "ercot_p95_lmp" in prices.columns:
                self.df["ercot_p95_lmp"] = self._align_series(prices["ercot_p95_lmp"])
            self.df["ercot_lmp_7d_z"] = self._rolling_zscore(max_lmp, 7)
            self.df["ercot_lmp_spike_flag"] = self.df["ercot_lmp_7d_z"] > 2.0

            hist_days = int(max_lmp.notna().sum())
            coverage["ercot_power"] = {
                "hist_days": hist_days,
                "forecast_days": 0,
                "missing_days": max(index_len - hist_days, 0),
            }
        else:
            self.logger.warning("ERCOT power inputs missing; power features skipped.")

        self.feature_meta["exog_coverage"] = coverage

    def run(self) -> pd.DataFrame:
        self.logger.info("Engineering features (MSI, regimes, fragility)")
        self.calc_pipe_metrics()
        self.calc_system_stress()
        self.calc_substitution_failure()
        self.calc_rolling_regimes()
        self.add_targets()
        self.calc_upstream_shocks()
        self.add_exogenous_features()
        return self.df.dropna(subset=["MSI_Pct_Rank", "Target_Ret_Henry"])
