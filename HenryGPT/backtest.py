from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils.logging import get_logger


class BacktestEngine:
    """Evaluate regime performance and walk-forward diagnostics."""

    def __init__(self, df: pd.DataFrame, config: Config) -> None:
        self.df = df
        self.cfg = config
        self.logger = get_logger(__name__)

    def evaluate_regimes(self) -> pd.DataFrame:
        stats = self.df.groupby("Regime")["Target_Ret_Henry"].agg(
            ["count", "mean", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]
        )
        stats.columns = ["Count", "Mean_Daily_Ret", "Daily_Vol", "VaR_5%", "Tail_Upside_95%"]
        stats["Ann_Vol"] = stats["Daily_Vol"] * np.sqrt(252)
        pos_days = (
            self.df[self.df["Target_Ret_Henry"] > 0]
            .groupby("Regime")["Target_Ret_Henry"]
            .count()
        )
        stats["Win_Rate"] = pos_days / stats["Count"]
        return stats

    def walk_forward_vol(self) -> Optional[pd.DataFrame]:
        window = self.cfg.walk_forward_window
        step = self.cfg.walk_forward_step
        if len(self.df) <= window + step:
            return None

        preds = []
        actuals = []
        for end in range(window, len(self.df) - 1, step):
            train = self.df.iloc[end - window : end]
            test = self.df.iloc[end : end + step]
            vol_by_regime = train.groupby("Regime")["Target_Ret_Henry"].std()

            for _, row in test.iterrows():
                regime = row["Regime"]
                if regime not in vol_by_regime:
                    continue
                ret = row["Target_Ret_Henry"]
                if pd.isna(ret):
                    continue
                preds.append(float(vol_by_regime.loc[regime]))
                actuals.append(abs(float(ret)))

        if len(preds) < 20:
            return None

        preds_arr = np.array(preds)
        actuals_arr = np.array(actuals)
        if preds_arr.std() == 0 or actuals_arr.std() == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(preds_arr, actuals_arr)[0, 1])

        pred_thresh = float(np.quantile(preds_arr, 0.7))
        act_thresh = float(np.quantile(actuals_arr, 0.7))
        hit_rate = float(
            ((preds_arr >= pred_thresh) & (actuals_arr >= act_thresh)).mean()
        )

        return pd.DataFrame(
            [
                {
                    "OOS_Corr": corr,
                    "OOS_Hit_Rate": hit_rate,
                    "Samples": len(preds_arr),
                }
            ]
        )
