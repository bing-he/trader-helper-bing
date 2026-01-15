from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .utils.logging import get_logger


@dataclass
class DataBundle:
    data: pd.DataFrame
    active_pipes: List[str]
    quality: "DataQualitySummary"


@dataclass
class DataQualitySummary:
    excluded_pipe_days: int
    top_missing_pipes: List[Tuple[str, int]]


class DataManager:
    """Load and align flow, capacity, and price data."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.logger = get_logger(__name__)
        self.flows_raw: pd.DataFrame | None = None
        self.prices_raw: pd.DataFrame | None = None

    def load_data(self) -> None:
        self.logger.info("Loading data from %s and %s", self.cfg.flows_path, self.cfg.prices_path)
        self.flows_raw = pd.read_csv(self.cfg.flows_path)
        self.prices_raw = pd.read_csv(self.cfg.prices_path)
        self._validate_columns()
        self.flows_raw["Date"] = pd.to_datetime(self.flows_raw["Date"])
        self.prices_raw["Date"] = pd.to_datetime(self.prices_raw["Date"])

    def _validate_columns(self) -> None:
        if self.flows_raw is None or self.prices_raw is None:
            raise ValueError("Raw data not loaded before validation.")
        required_flows = {"Date", "loc_name", "Scheduled", "OperationallyAvailable"}
        required_prices = {"Date", "Henry"}
        missing_flows = required_flows - set(self.flows_raw.columns)
        missing_prices = required_prices - set(self.prices_raw.columns)
        if missing_flows:
            raise ValueError(f"Flows data missing columns: {sorted(missing_flows)}")
        if missing_prices:
            raise ValueError(f"Prices data missing columns: {sorted(missing_prices)}")

    def process_vectors(self) -> DataBundle:
        if self.flows_raw is None or self.prices_raw is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df_flow = self.flows_raw.pivot_table(
            index="Date",
            columns="loc_name",
            values="Scheduled",
            aggfunc="mean",
        )
        df_flow.columns = [f"Flow_{c}" for c in df_flow.columns]

        df_cap = self.flows_raw.pivot_table(
            index="Date",
            columns="loc_name",
            values="OperationallyAvailable",
            aggfunc="mean",
        )
        df_cap.columns = [f"Cap_{c}" for c in df_cap.columns]

        df_flow = df_flow.sort_index()
        df_cap = df_cap.sort_index()
        df_cap = df_cap.where(df_cap > 0, np.nan)
        df_cap = df_cap.ffill()

        cap_cols = [c for c in df_cap.columns]
        flow_cols = [c for c in df_flow.columns]

        cap_valid = df_cap.notna()
        cap_valid_flow = cap_valid.copy()
        cap_valid_flow.columns = [col.replace("Cap_", "Flow_") for col in cap_valid_flow.columns]
        df_flow = df_flow.where(cap_valid_flow, np.nan)
        df_flow = df_flow.fillna(0)
        df_flow = df_flow.where(cap_valid_flow, np.nan)

        df_vec = pd.concat([df_flow, df_cap], axis=1)

        active_pipes = []
        for col in flow_cols:
            pipe = col.replace("Flow_", "")
            cap_col = f"Cap_{pipe}"
            valid_days = int(cap_valid[cap_col].sum()) if cap_col in cap_valid else 0
            mean_flow = float(df_flow[col].mean(skipna=True)) if col in df_flow else 0.0
            if (
                valid_days >= self.cfg.active_pipe_min_cap_days
                and mean_flow > self.cfg.active_pipe_min_flow
            ):
                active_pipes.append(pipe)

        excluded_pipe_days = int((~cap_valid).sum().sum())
        missing_by_pipe = (~cap_valid).sum().sort_values(ascending=False)
        top_missing = [
            (col.replace("Cap_", ""), int(count)) for col, count in missing_by_pipe.head(5).items()
        ]
        quality = DataQualitySummary(
            excluded_pipe_days=excluded_pipe_days,
            top_missing_pipes=top_missing,
        )

        prices = self.prices_raw.set_index("Date").sort_index()
        rel_cols = ["Henry"] + [c for c in self.cfg.target_hubs if c in prices.columns]
        prices = prices[rel_cols]

        data = df_vec.join(prices, how="inner").sort_index()
        data = data[data["Henry"] > self.cfg.min_henry_price]

        self.logger.info("Aligned data rows: %d", len(data))
        return DataBundle(data=data, active_pipes=active_pipes, quality=quality)
