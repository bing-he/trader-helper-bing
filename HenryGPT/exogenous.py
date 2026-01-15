from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils.logging import get_logger


@dataclass
class ExogenousBundle:
    lng_hist: Optional[pd.DataFrame] = None
    lng_forecast: Optional[pd.DataFrame] = None
    fundy_hist: Optional[pd.DataFrame] = None
    fundy_forecast: Optional[pd.DataFrame] = None
    ercot_load_hist: Optional[pd.Series] = None
    ercot_load_forecast: Optional[pd.Series] = None
    power_prices_daily: Optional[pd.DataFrame] = None


class ExogenousDataLoader:
    """Load optional LNG, Fundy, ERCOT load, and ERCOT power price datasets."""

    def __init__(self, info_dir: Path) -> None:
        self.info_dir = Path(info_dir)
        self.logger = get_logger(__name__)

    def load(self) -> ExogenousBundle:
        return ExogenousBundle(
            lng_hist=self._load_lng("CriterionLNGHist.csv"),
            lng_forecast=self._load_lng("CriterionLNGForecast.csv"),
            fundy_hist=self._load_fundy("Fundy.csv"),
            fundy_forecast=self._load_fundy("FundyForecast.csv"),
            ercot_load_hist=self._load_gridstat("GridStatLoadHist.csv"),
            ercot_load_forecast=self._load_gridstat("GridStatLoadForecast.csv"),
            power_prices_daily=self._load_power_prices("PowerPrices.csv"),
        )

    def _read_csv_optional(self, filename: str) -> Optional[pd.DataFrame]:
        path = self.info_dir / filename
        if not path.exists():
            self.logger.warning("Optional exogenous file missing: %s", path)
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.logger.warning("Failed reading %s: %s", path, exc)
            return None
        if df.empty:
            self.logger.warning("Optional exogenous file empty: %s", path)
            return None
        return df

    def _load_lng(self, filename: str) -> Optional[pd.DataFrame]:
        df = self._read_csv_optional(filename)
        if df is None:
            return None
        required = {"Date", "Item", "Value"}
        missing = required - set(df.columns)
        if missing:
            self.logger.warning("LNG file %s missing columns: %s", filename, sorted(missing))
            return None
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Item"] = df["Item"].astype(str).str.strip()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Date", "Item"])

        lower = df["Item"].str.lower()
        exclude = lower.str.contains("cove point") | lower.str.contains("elba")
        if exclude.any():
            df = df.loc[~exclude]

        df = df.dropna(subset=["Value"]).sort_values("Date")
        if df.empty:
            self.logger.warning("LNG file %s has no usable rows after filtering.", filename)
            return None
        return df

    def _load_fundy(self, filename: str) -> Optional[pd.DataFrame]:
        df = self._read_csv_optional(filename)
        if df is None:
            return None
        required = {"Date", "Region", "Item", "Value"}
        missing = required - set(df.columns)
        if missing:
            self.logger.warning("Fundy file %s missing columns: %s", filename, sorted(missing))
            return None
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Region"] = df["Region"].astype(str).str.strip()
        df["Item"] = df["Item"].astype(str).str.strip()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Date", "Region", "Item"])
        df = df.sort_values("Date")
        if df.empty:
            self.logger.warning("Fundy file %s has no usable rows after filtering.", filename)
            return None
        return df

    def _load_gridstat(self, filename: str) -> Optional[pd.Series]:
        df = self._read_csv_optional(filename)
        if df is None:
            return None
        date_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else None
        if date_col is None:
            if "Date" in df.columns:
                date_col = "Date"
            else:
                self.logger.warning("GridStat file %s missing date column.", filename)
                return None
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col).sort_index()

        ercot_cols = [col for col in df.columns if "ERCOT" in col.upper()]
        if not ercot_cols:
            self.logger.warning("GridStat file %s has no ERCOT columns.", filename)
            return None

        series = None
        selected = None
        primary = "ERCOT_system_total" if "ERCOT_system_total" in df.columns else None
        if primary:
            series = pd.to_numeric(df[primary], errors="coerce")
            dup_cols = sorted([c for c in df.columns if c.startswith(primary + ".")])
            for col in dup_cols:
                series = series.where(series.notna(), pd.to_numeric(df[col], errors="coerce"))
            if series.notna().sum() > 0:
                selected = primary
            else:
                series = None

        if series is None:
            best_col = None
            best_count = -1
            best_series = None
            for col in ercot_cols:
                if col.endswith(".1") and col[:-2] in df.columns:
                    continue
                candidate = pd.to_numeric(df[col], errors="coerce")
                count = int(candidate.notna().sum())
                if count > best_count:
                    best_col = col
                    best_count = count
                    best_series = candidate
            if best_series is None:
                for col in ercot_cols:
                    candidate = pd.to_numeric(df[col], errors="coerce")
                    count = int(candidate.notna().sum())
                    if count > best_count:
                        best_col = col
                        best_count = count
                        best_series = candidate
            series = best_series
            selected = best_col

        if series is None or series.notna().sum() == 0:
            self.logger.warning("GridStat file %s has no usable ERCOT load series.", filename)
            return None

        self.logger.info("GridStat %s using ERCOT series: %s", filename, selected)
        series = series.sort_index()
        series.name = "ercot_load_total"
        return series

    def _load_power_prices(self, filename: str) -> Optional[pd.DataFrame]:
        df = self._read_csv_optional(filename)
        if df is None:
            return None
        required = {"ISO", "Location", "Date", "Max LMP"}
        missing = required - set(df.columns)
        if missing:
            self.logger.warning("PowerPrices file %s missing columns: %s", filename, sorted(missing))
            return None
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["ISO"] = df["ISO"].astype(str).str.upper().str.strip()
        df["Max LMP"] = pd.to_numeric(df["Max LMP"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df[df["ISO"] == "ERCOT"]
        if df.empty:
            self.logger.warning("PowerPrices file %s has no ERCOT rows.", filename)
            return None

        grouped = df.groupby("Date")["Max LMP"]
        daily = grouped.agg(
            ercot_max_lmp="max",
            ercot_p95_lmp=lambda s: s.quantile(0.95),
        )
        daily = daily.sort_index()
        if daily.empty:
            self.logger.warning("PowerPrices file %s has no usable daily data.", filename)
            return None
        return daily
