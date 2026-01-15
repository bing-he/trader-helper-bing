from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils.logging import get_logger
from .utils.stats import safe_corr, segment_correlation_stability


class SignalEngine:
    """Compute regime transitions and hub vulnerability signals."""

    def __init__(self, df: pd.DataFrame, active_pipes: List[str], config: Config) -> None:
        self.df = df.copy()
        self.pipes = active_pipes
        self.cfg = config
        self.logger = get_logger(__name__)

    def _rolling_zscore_series(
        self, series: pd.Series, window: int, min_samples: int
    ) -> pd.Series:
        mean = series.rolling(window=window, min_periods=min_samples).mean()
        std = series.rolling(window=window, min_periods=min_samples).std()
        zscore = (series - mean) / std
        return zscore.replace([np.inf, -np.inf], np.nan)

    def _latest_zscore(
        self, series: pd.Series, latest_value: float, window: int, min_samples: int
    ) -> Optional[float]:
        if latest_value is None or not np.isfinite(latest_value):
            return None
        hist = series.dropna()
        if len(hist) < min_samples:
            return None
        recent = hist.tail(window)
        mean = float(recent.mean())
        std = float(recent.std())
        if std == 0 or not np.isfinite(std):
            return None
        return float((latest_value - mean) / std)

    def add_regime_transitions(self) -> pd.DataFrame:
        prev = self.df["Regime"].shift(1)
        self.df["Regime_Prev"] = prev
        self.df["Transition_Slack_to_Transition"] = (
            (prev == "Slack") & (self.df["Regime"] == "Transition")
        )
        self.df["Transition_Transition_to_Binding"] = (
            (prev == "Transition") & (self.df["Regime"] == "Binding")
        )
        self.df["Transition_Binding_to_Transition"] = (
            (prev == "Binding") & (self.df["Regime"] == "Transition")
        )
        self.df["Transition_Any"] = (
            self.df["Transition_Slack_to_Transition"]
            | self.df["Transition_Transition_to_Binding"]
            | self.df["Transition_Binding_to_Transition"]
        )
        self.df["Transition_Label"] = np.select(
            [
                self.df["Transition_Slack_to_Transition"],
                self.df["Transition_Transition_to_Binding"],
                self.df["Transition_Binding_to_Transition"],
            ],
            ["Slack->Transition", "Transition->Binding", "Binding->Transition"],
            default="None",
        )
        return self.df

    def add_gulf_pressure_signals(self) -> Dict[str, Optional[float]]:
        window = max(self.cfg.gulf_pressure_min_samples, 30)
        min_samples = self.cfg.gulf_pressure_min_samples

        components: Dict[str, float] = {}
        component_series: Dict[str, pd.Series] = {}
        component_samples: Dict[str, int] = {}

        last_row = self.df.iloc[-1]

        if "lng_gulf_7d_chg" in self.df.columns:
            lng_series = self.df["lng_gulf_7d_chg"]
            lng_z_series = self._rolling_zscore_series(lng_series, window, min_samples)
            lng_impulse = last_row.get("lng_gulf_forecast_7d_avg_minus_last7d", np.nan)
            lng_latest = float(lng_impulse) if np.isfinite(lng_impulse) else float(lng_series.iloc[-1])
            lng_z = self._latest_zscore(lng_series, lng_latest, window, min_samples)
            if lng_z is not None and np.isfinite(lng_z):
                components["lng"] = lng_z
                component_series["lng"] = lng_z_series
                component_samples["lng"] = int(lng_series.dropna().shape[0])

        if "ercot_load_7d_chg" in self.df.columns:
            load_series = self.df["ercot_load_7d_chg"]
            load_z_series = self._rolling_zscore_series(load_series, window, min_samples)
            load_impulse = last_row.get("ercot_load_forecast_7d_avg_minus_last7d", np.nan)
            load_latest = (
                float(load_impulse) if np.isfinite(load_impulse) else float(load_series.iloc[-1])
            )
            load_z = self._latest_zscore(load_series, load_latest, window, min_samples)
            if load_z is not None and np.isfinite(load_z):
                components["load"] = load_z
                component_series["load"] = load_z_series
                component_samples["load"] = int(load_series.dropna().shape[0])

        if "ercot_lmp_7d_z" in self.df.columns:
            lmp_series = self.df["ercot_lmp_7d_z"]
            lmp_latest = float(lmp_series.iloc[-1]) if pd.notna(lmp_series.iloc[-1]) else None
            if lmp_latest is not None and np.isfinite(lmp_latest):
                components["lmp"] = lmp_latest
                component_series["lmp"] = lmp_series
                component_samples["lmp"] = int(lmp_series.dropna().shape[0])

        if "southcentral_tightness_proxy" in self.df.columns:
            fundy_series = self.df["southcentral_tightness_proxy"]
            fundy_z_series = self._rolling_zscore_series(fundy_series, window, min_samples)
            fundy_latest = float(fundy_series.iloc[-1]) if pd.notna(fundy_series.iloc[-1]) else None
            if fundy_latest is not None and np.isfinite(fundy_latest):
                fundy_z = self._latest_zscore(fundy_series, fundy_latest, window, min_samples)
                if fundy_z is not None and np.isfinite(fundy_z):
                    components["fundy"] = fundy_z
                    component_series["fundy"] = fundy_z_series
                    component_samples["fundy"] = int(fundy_series.dropna().shape[0])

        if "lng" in component_series and "load" in component_series:
            corr = safe_corr(component_series["lng"], component_series["load"])
            if np.isfinite(corr) and abs(corr) >= 0.7:
                if abs(components.get("lng", 0)) >= abs(components.get("load", 0)):
                    components.pop("load", None)
                    component_series.pop("load", None)
                    component_samples.pop("load", None)
                else:
                    components.pop("lng", None)
                    component_series.pop("lng", None)
                    component_samples.pop("lng", None)

        clipped = {k: float(np.clip(v, -2.5, 2.5)) for k, v in components.items()}
        component_count = len(clipped)

        score = None
        if component_count >= self.cfg.gulf_pressure_min_components:
            raw_score = float(sum(clipped.values()))
            score = float(np.tanh(raw_score / 3.0) * 3.0)

        score_series = None
        if component_series:
            score_series = sum(
                component_series[key].clip(lower=-2.5, upper=2.5)
                for key in component_series
            )

        sign_consistency = 0.0
        if score_series is not None:
            recent = score_series.dropna().tail(self.cfg.gulf_pressure_sign_window)
            if len(recent) >= max(5, self.cfg.gulf_pressure_sign_window // 2):
                latest_sign = np.sign(recent.iloc[-1])
                if latest_sign != 0:
                    sign_consistency = float((np.sign(recent) == latest_sign).mean())

        min_component_sample = min(component_samples.values()) if component_samples else 0
        sample_score = min(
            1.0, min_component_sample / max(self.cfg.gulf_pressure_min_samples * 2, 1)
        )
        component_score = min(1.0, component_count / 4)
        confidence = round(
            (0.4 * component_score) + (0.4 * sign_consistency) + (0.2 * sample_score), 2
        )

        if component_count < self.cfg.gulf_pressure_min_components:
            confidence = 0.0

        regime = "Neutral"
        if score is not None:
            if score >= self.cfg.gulf_pressure_tight_threshold:
                regime = "Tightening Risk"
            elif score <= self.cfg.gulf_pressure_loose_threshold:
                regime = "Loose"

        self.df["gulf_tightening_pressure_score"] = score
        self.df["gulf_tightening_pressure_regime"] = regime
        self.df["gulf_tightening_pressure_confidence"] = confidence
        self.df["gulf_pressure_component_count"] = component_count
        self.df["gulf_pressure_sign_consistency"] = sign_consistency
        self.df["gulf_pressure_lng_z"] = components.get("lng")
        self.df["gulf_pressure_load_z"] = components.get("load")
        self.df["gulf_pressure_lmp_z"] = components.get("lmp")
        self.df["gulf_pressure_fundy_z"] = components.get("fundy")

        return {
            "score": score,
            "regime": regime,
            "confidence": confidence,
            "component_count": component_count,
            "sign_consistency": sign_consistency,
            "lng_z": components.get("lng"),
            "load_z": components.get("load"),
            "lmp_z": components.get("lmp"),
            "fundy_z": components.get("fundy"),
        }

    def compute_vulnerability_scores(self) -> pd.DataFrame:
        current_date = self.df.index[-1]
        lookback_start = current_date - pd.DateOffset(days=self.cfg.corr_lookback_days)
        recent_df = self.df.loc[lookback_start:]

        current_regime = self.df["Regime"].iloc[-1]
        regime_df = recent_df[recent_df["Regime"] == current_regime]
        regime_filtered = len(regime_df) >= self.cfg.min_corr_sample
        corr_df = regime_df if regime_filtered else recent_df

        basis_cols = [c for c in self.df.columns if c.startswith("Basis_")]
        util_cols = [c for c in self.df.columns if c.startswith("Util_") and "Clipped" not in c]

        today_row = self.df.iloc[-1]
        total_cap = 0.0
        for u in util_cols:
            pipe = u.replace("Util_", "")
            cap_val = today_row.get(f"Cap_{pipe}")
            if cap_val is not None and pd.notna(cap_val) and cap_val > 0:
                total_cap += float(cap_val)

        scores = []
        for b in basis_cols:
            hub_name = b.replace("Basis_", "")
            hub_score = 0.0
            best_driver = None

            target_col = f"Target_Delta_Basis_{hub_name}"
            for u in util_cols:
                pipe = u.replace("Util_", "")
                series_x = corr_df[u].where(corr_df[u] <= self.cfg.util_anomaly_threshold)
                series_y = corr_df[b]
                sample_size = int(pd.concat([series_x, series_y], axis=1).dropna().shape[0])
                if sample_size < self.cfg.min_corr_sample:
                    continue

                corr_val = safe_corr(series_x, series_y)
                if not np.isfinite(corr_val) or corr_val < self.cfg.min_abs_corr:
                    continue

                stability = segment_correlation_stability(
                    series_x,
                    series_y,
                    segments=self.cfg.corr_stability_segments,
                    min_abs_corr=self.cfg.min_abs_corr,
                )
                if stability < self.cfg.min_stability:
                    continue

                util_val = float(today_row.get(u, np.nan))
                cap_val = today_row.get(f"Cap_{pipe}")
                cap_val = float(cap_val) if cap_val is not None and pd.notna(cap_val) else np.nan
                if total_cap > 0 and cap_val > 0:
                    weight = cap_val / total_cap
                else:
                    weight = 1.0 / max(len(util_cols), 1)

                contribution = corr_val * util_val * weight
                hub_score += contribution

                lift = None
                tight_sample = 0
                median_lift = None
                if target_col in corr_df.columns:
                    target_series = corr_df[target_col]
                    cap_series = corr_df.get(f"Cap_{pipe}")
                    if cap_series is None:
                        valid_mask = series_x.notna() & target_series.notna()
                    else:
                        valid_mask = (
                            series_x.notna()
                            & target_series.notna()
                            & cap_series.notna()
                            & (cap_series > 0)
                        )
                    tight_mask = (series_x > self.cfg.tight_util_threshold) & (
                        series_x <= self.cfg.util_anomaly_threshold
                    )
                    base_series = target_series[valid_mask]
                    tight_series = target_series[valid_mask & tight_mask]
                    tight_sample = int(tight_series.shape[0])
                    if tight_sample >= 1 and base_series.shape[0] >= 1:
                        lift = float(tight_series.mean() - base_series.mean())
                        median_lift = float(tight_series.median() - base_series.median())

                if best_driver is None or contribution > best_driver["contribution"]:
                    best_driver = {
                        "pipe": pipe,
                        "corr": corr_val,
                        "stability": stability,
                        "sample": sample_size,
                        "util": util_val,
                        "weight": weight,
                        "contribution": contribution,
                        "lift": lift,
                        "tight_sample": tight_sample,
                        "median_lift": median_lift,
                    }

            confidence = 0.0
            evidence_ok = False
            if best_driver:
                confidence = self._confidence_score(
                    abs(best_driver["corr"]),
                    best_driver["stability"],
                    best_driver["sample"],
                )
                evidence_ok = (
                    best_driver["sample"] >= self.cfg.evidence_min_sample
                    and abs(best_driver["corr"]) >= self.cfg.evidence_min_abs_corr
                    and best_driver["stability"] >= self.cfg.evidence_min_stability
                    and best_driver["tight_sample"] >= self.cfg.evidence_min_tight_samples
                    and best_driver["lift"] is not None
                    and abs(best_driver["lift"]) >= self.cfg.evidence_min_lift
                )

            scores.append(
                {
                    "Hub": hub_name,
                    "Vulnerability_Score": round(hub_score, 4),
                    "Confidence": confidence,
                    "Primary_Driver_Pipe": best_driver["pipe"] if best_driver else "None",
                    "Primary_Driver_Corr": best_driver["corr"] if best_driver else 0.0,
                    "Primary_Driver_Stability": best_driver["stability"] if best_driver else 0.0,
                    "Primary_Driver_Sample": best_driver["sample"] if best_driver else 0,
                    "Primary_Driver_Util": best_driver["util"] if best_driver else 0.0,
                    "Primary_Driver_Weight": best_driver["weight"] if best_driver else 0.0,
                    "Primary_Driver_Lift": best_driver["lift"] if best_driver else 0.0,
                    "Primary_Driver_Median_Lift": best_driver["median_lift"] if best_driver else 0.0,
                    "Primary_Driver_Tight_Sample": best_driver["tight_sample"] if best_driver else 0,
                    "Regime_Filtered": regime_filtered,
                    "Evidence_Eligible": evidence_ok,
                }
            )

        scores_df = pd.DataFrame(scores).sort_values("Vulnerability_Score", ascending=False)
        return scores_df

    def _confidence_score(self, effect_size: float, stability: float, sample_size: int) -> float:
        if sample_size < self.cfg.min_corr_sample or effect_size <= 0 or stability <= 0:
            return 0.0
        size_score = min(1.0, effect_size / 0.6)
        stability_score = min(1.0, stability)
        sample_score = min(1.0, np.sqrt(sample_size / max(self.cfg.min_corr_sample * 2, 1)))
        return round(size_score * stability_score * sample_score, 2)
