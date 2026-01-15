"""Generate a trader-focused static HTML Morning Pack report."""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402

from gptcot.forecasting import predict_returns, train_and_save_models
from gptcot.forward_curve import LOOKBACK_DAYS, build_forward_curve_table, generate_forward_curve_charts, latest_front_label
from gptcot.utils import identify_sr_levels, last_year_series
from gptcot.spread_analysis import generate_spread_chart, normalize_contract_pair, spread_pairs_from_front
from gptcot.config import PipelineConfig
from gptcot.features import compute_cot_features, compute_rolling_percentile, feature_columns
from gptcot.io import (
    cot_path,
    eia_totals_path,
    forward_curve_path,
    get_project_root,
    output_csv_path,
)
from gptcot.pipeline.build_dataset import build_dataset, summarize, write_outputs

DEFAULT_START_DATE = pd.Timestamp("2013-01-03")
CANDIDATE_PERCENTILES: Sequence[int] = tuple(range(70, 100, 5))
DEFAULT_HORIZONS: Sequence[int] = tuple(PipelineConfig._normalize_horizons([7, 14, 28, 30]))
CURVE_COLUMNS = [f"FWD_{i:02d}" for i in range(13)]
PROMPT_PCT_WINDOW = 260  # ~52 weeks of trading days
MIN_ANALOG_COUNT = 20
ANALOG_FACTORS: Sequence[str] = (
    "prompt_spread_pct_52",
    "Total_MM_Net_pct_52",
    "Total_Prod_Net_pct_52",
    "Total_Swap_Net_pct_52",
    "surplus_deficit_pct",
)
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "Total_MM_Net_z_52": "Managed Money positioning (z-score)",
    "Total_Prod_Net_z_52": "Producer hedging (z-score)",
    "Total_Swap_Net_z_52": "Swap dealer positioning (z-score)",
    "Total_MM_Net_pct_52": "Managed Money positioning",
    "Total_Prod_Net_pct_52": "Producer hedging",
    "Total_Swap_Net_pct_52": "Swap dealer positioning",
    "Total_MM_Net_pct_156": "Managed Money positioning (3y)",
    "Total_Prod_Net_pct_156": "Producer hedging (3y)",
    "Total_Swap_Net_pct_156": "Swap dealer positioning (3y)",
    "storage_dev": "storage deviation vs 5Y",
    "prompt_minus_strip": "term structure slope",
    "winter_summer_spread": "winter vs summer spread",
}


@dataclass(frozen=True)
class ThresholdResult:
    """Represents the selected extreme percentile for a series/horizon pair."""

    series: str
    horizon: int
    percentile: int
    high_cutoff: float
    low_cutoff: float
    delta_mean: float
    high_mean: float
    low_mean: float
    high_count: int
    low_count: int

    @property
    def favored_side(self) -> str:
        return "high" if self.delta_mean >= 0 else "low"


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path
    package_root: Path
    info_dir: Path
    output_dir: Path
    report_dir: Path
    processed_dataset: Path
    forward_returns_csv: Path
    template_path: Path


def assign_season(value: object) -> str:
    """Map a date to a meteorological season."""

    if value is None or pd.isna(value):
        return "Unknown"
    ts = pd.to_datetime(value)
    month = ts.month
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "Unknown"


def _percentile_columns() -> List[str]:
    """Return percentile feature columns for all base series."""

    cols: List[str] = []
    for derived in feature_columns().values():
        cols.extend([col for col in derived if col.endswith("_pct_52")])
    return sorted(set(cols))


def filter_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid observations from 2013-01-03 onward and assign seasons."""

    filtered = df.loc[df["is_valid"]].copy()
    filtered = filtered.loc[filtered["cot_date"] >= DEFAULT_START_DATE]
    filtered["season"] = filtered["horizon_date"].apply(assign_season)
    return filtered.reset_index(drop=True)


def detect_extreme_thresholds(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int],
    series_columns: Iterable[str],
) -> List[ThresholdResult]:
    """Identify percentiles that maximize absolute mean return differences."""

    results: List[ThresholdResult] = []
    for series in series_columns:
        for horizon in horizons:
            subset = df.loc[(df["horizon_days"] == horizon) & df[series].notna()]
            if subset.empty:
                continue
            best: ThresholdResult | None = None
            for pct in CANDIDATE_PERCENTILES:
                high_cut = round(pct / 100, 6)
                low_cut = round(1 - high_cut, 6)
                high_group = subset.loc[subset[series] >= high_cut]
                low_group = subset.loc[subset[series] <= low_cut]
                if high_group.empty or low_group.empty:
                    continue
                high_mean = float(high_group["pct_change"].mean())
                low_mean = float(low_group["pct_change"].mean())
                delta = high_mean - low_mean
                score = abs(delta)
                if best is None or score > abs(best.delta_mean):
                    best = ThresholdResult(
                        series=series,
                        horizon=int(horizon),
                        percentile=int(pct),
                        high_cutoff=high_cut,
                        low_cutoff=low_cut,
                        delta_mean=delta,
                        high_mean=high_mean,
                        low_mean=low_mean,
                        high_count=int(len(high_group)),
                        low_count=int(len(low_group)),
                    )
            if best is not None:
                results.append(best)
    return results


def extreme_summary(df: pd.DataFrame, thresholds: List[ThresholdResult]) -> pd.DataFrame:
    """Build a summary table comparing high, low, and normal returns."""

    records: List[dict] = []
    for threshold in thresholds:
        subset = df.loc[(df["horizon_days"] == threshold.horizon) & df[threshold.series].notna()]
        high_mask = subset[threshold.series] >= threshold.high_cutoff
        low_mask = subset[threshold.series] <= threshold.low_cutoff
        normal_mask = ~(high_mask | low_mask)
        high = subset.loc[high_mask, "pct_change"]
        low = subset.loc[low_mask, "pct_change"]
        normal = subset.loc[normal_mask, "pct_change"]
        records.append(
            {
                "series": threshold.series,
                "horizon_days": threshold.horizon,
                "percentile": threshold.percentile,
                "high_cutoff": threshold.high_cutoff,
                "low_cutoff": threshold.low_cutoff,
                "high_mean_return": float(high.mean()) if not high.empty else np.nan,
                "low_mean_return": float(low.mean()) if not low.empty else np.nan,
                "normal_mean_return": float(normal.mean()) if not normal.empty else np.nan,
                "high_count": int(high_mask.sum()),
                "low_count": int(low_mask.sum()),
                "normal_count": int(normal_mask.sum()),
                "delta_mean": threshold.delta_mean,
                "favored_side": threshold.favored_side,
            }
        )
    return pd.DataFrame(records)


def overall_population_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distributional stats by horizon."""

    def pctl(series: pd.Series, percentile: float) -> float:
        clean = series.dropna()
        if clean.empty:
            return np.nan
        return float(np.percentile(clean, percentile))

    grouped = df.groupby("horizon_days")["pct_change"]
    stats = grouped.agg(
        mean="mean",
        median="median",
        count="count",
        p5=lambda s: pctl(s, 5),
        p95=lambda s: pctl(s, 95),
    )
    stats = stats.reset_index().loc[:, ["horizon_days", "mean", "median", "p5", "p95", "count"]]
    return stats.sort_values("horizon_days")


def _format_table(df: pd.DataFrame) -> str:
    """Format a dataframe as HTML."""

    return df.to_html(index=False, float_format=lambda x: f"{float(x):.4f}")


@dataclass(frozen=True)
class BucketSpec:
    """Quantile cuts for a percentile-based factor."""

    factor: str
    low_cut: float
    high_cut: float


FACTOR_LABELS: Dict[str, str] = {
    "prompt_spread_pct_52": "1yr Spread",
    "Total_MM_Net_pct_52": "Managed Money",
    "Total_Prod_Net_pct_52": "Producers",
    "Total_Swap_Net_pct_52": "Swaps",
    "surplus_deficit_pct": "Storage vs 5Y",
}


def _factor_label(factor: str) -> str:
    """Human-friendly label for a factor column."""

    return FACTOR_LABELS.get(factor, factor)


def compute_factor_buckets(
    df: pd.DataFrame, factor_columns: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, BucketSpec]]:
    """
    Assign quantile-based buckets (Low/Mid/High) for each factor.

    Returns the bucketed dataframe and the quantile cuts used for each factor.
    """

    bucketed = df.copy()
    specs: Dict[str, BucketSpec] = {}
    for factor in factor_columns:
        if factor not in bucketed.columns:
            continue
        series = bucketed[factor]
        clean = series.dropna()
        if clean.empty:
            continue
        q1, q2 = clean.quantile([1 / 3, 2 / 3])
        specs[factor] = BucketSpec(factor=factor, low_cut=float(q1), high_cut=float(q2))

        def _assign(val: float) -> str:
            if pd.isna(val):
                return "Missing"
            if val <= q1:
                return "Low"
            if val >= q2:
                return "High"
            return "Mid"

        bucketed[f"{factor}_bucket"] = series.apply(_assign)
    return bucketed, specs


def format_combo_label(combo_cols: Sequence[str], bucket_values: Sequence[str] | str) -> str:
    """Compose a readable label for a bucketed factor combination."""

    if not isinstance(bucket_values, tuple):
        bucket_values = (bucket_values,)
    parts = []
    for col, bucket in zip(combo_cols, bucket_values):
        factor = col.replace("_bucket", "")
        parts.append(f"{_factor_label(factor)}={bucket}")
    return " | ".join(parts)


def compute_analog_performance(
    df: pd.DataFrame,
    factor_columns: Sequence[str],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_count: int = MIN_ANALOG_COUNT,
) -> pd.DataFrame:
    """
    Compute historical analog performance for combinations of factor buckets.

    Group valid forward returns by bucket combinations (pairs and triplets) and
    compute average returns, medians, hit rates, and sample sizes for each horizon.
    """

    bucket_cols = [f"{col}_bucket" for col in factor_columns if f"{col}_bucket" in df.columns]
    combos: List[Tuple[str, ...]] = []
    if len(bucket_cols) >= 3:
        combos.extend(combinations(bucket_cols, 3))
    if len(bucket_cols) >= 2:
        combos.extend(combinations(bucket_cols, 2))
    elif len(bucket_cols) == 1:
        combos.append((bucket_cols[0],))

    records: List[Dict[str, object]] = []
    for horizon in horizons:
        horizon_df = df.loc[df["horizon_days"] == horizon].copy()
        if horizon_df.empty:
            continue
        for combo in combos:
            subset = horizon_df.dropna(subset=list(combo))
            if subset.empty:
                continue
            grouped = subset.groupby(list(combo))
            for buckets, group in grouped:
                count = len(group)
                if count < min_count:
                    continue
                records.append(
                    {
                        "factor_combo": format_combo_label(combo, buckets),
                        "avg_return": float(group["pct_change"].mean()),
                        "median_return": float(group["pct_change"].median()),
                        "hit_rate": float((group["pct_change"] > 0).mean()),
                        "count": int(count),
                        "horizon": int(horizon),
                    }
                )
    return pd.DataFrame(records)


class MarketReportGenerator:
    """Create the Morning Pack charts."""

    def __init__(
        self,
        output_dir: Path,
        logger: logging.Logger,
        sr_outlier_threshold: float = 1.5,
        sr_max_multiple: float | None = 2.0,
    ) -> None:
        self.output_dir = output_dir
        self.logger = logger
        self.sr_outlier_threshold = sr_outlier_threshold
        self.sr_max_multiple = sr_max_multiple
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _date_axis(self, ax: plt.Axes) -> None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.figure.autofmt_xdate()

    def generate_strip_chart(self, curve_df: pd.DataFrame) -> str:
        if "strip_12m" not in curve_df.columns:
            raise ValueError("strip_12m column missing from curve dataframe.")

        fig, ax = plt.subplots(figsize=(9, 5))

        strip = curve_df["strip_12m"].dropna()
        strip_label = f"12m Strip (Current {strip.iloc[-1]:.3f})" if not strip.empty else "12m Strip"
        ax.plot(curve_df["Date"], curve_df["strip_12m"], label=strip_label, color="black", linewidth=1.6)

        if "strip_12m_sma10" in curve_df.columns:
            sma = curve_df["strip_12m_sma10"].dropna()
            sma_label = f"SMA10 ({sma.iloc[-1]:.3f})" if not sma.empty else "SMA10"
            ax.plot(curve_df["Date"], curve_df["strip_12m_sma10"], label=sma_label, color="steelblue", linestyle="--")

        sr_levels = identify_sr_levels(
            curve_df["strip_12m"],
            n_levels=3,
            outlier_threshold=self.sr_outlier_threshold,
            max_multiple_of_current=self.sr_max_multiple,
        )
        for idx, (level, _) in enumerate(sr_levels, start=1):
            ax.axhline(level, linestyle="--", color="gray", alpha=0.6, label=f"SR{idx}: {level:.2f}")
        ax.set_title("12-month Strip (Avg of Front 12 Months)")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._date_axis(ax)
        fig.tight_layout()
        filename = "chart_strip_12m.png"
        fig.savefig(self.output_dir / filename, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return filename

    def generate_front_month_history_chart(self, curve_df: pd.DataFrame) -> Optional[str]:
        if "FWD_00" not in curve_df.columns:
            self.logger.warning("Front month chart skipped: FWD_00 column missing (columns=%s)", list(curve_df.columns))
            return None
        print("Generating front month history chart...")
        fwd = curve_df[["Date", "FWD_00"]].dropna()
        if fwd.empty:
            self.logger.warning("Front month chart skipped: no data in FWD_00 after dropna.")
            return None
        fwd["Date"] = pd.to_datetime(fwd["Date"])
        series = fwd.sort_values("Date").set_index("Date")["FWD_00"]
        series_last_year = last_year_series(series)
        if series_last_year.empty:
            self.logger.warning("Front month chart skipped: no data available in the last-year window.")
            return None

        sr_levels = identify_sr_levels(
            series_last_year,
            n_levels=3,
            outlier_threshold=self.sr_outlier_threshold,
            max_multiple_of_current=self.sr_max_multiple,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(series_last_year.index, series_last_year, color="black", label="Front Month")
        labels = ["SR1", "SR2", "SR3"]
        for (level, _), label in zip(sr_levels, labels):
            ax.axhline(level, linestyle="--", linewidth=1.2, alpha=0.6, label=f"{label}: {level:.2f}")
        ax.set_title("Front Month (Last 12 Months)")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        self._date_axis(ax)
        fig.tight_layout()
        filename = "front_month_history.png"
        file_path = self.output_dir / filename
        fig.savefig(file_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Front month chart saved to {file_path}")
        return filename

    def generate_seasonal_contract_chart(self, curve_df: pd.DataFrame) -> Optional[str]:
        print("Generating seasonal contract chart...")
        today = dt.date.today()
        year = today.year
        feb25 = dt.date(year, 2, 25)
        sep25 = dt.date(year, 9, 25)
        if feb25 <= today < sep25:
            col = "FWD_09"
            title_suffix = "Oct Contract"
        else:
            col = "FWD_02"
            title_suffix = "Mar Contract"
        if col not in curve_df.columns:
            self.logger.warning("Seasonal chart skipped: %s column missing (columns=%s)", col, list(curve_df.columns))
            return None
        fwd = curve_df[["Date", col]].dropna()
        if fwd.empty:
            self.logger.warning("Seasonal chart skipped: no data in %s after dropna.", col)
            return None
        fwd["Date"] = pd.to_datetime(fwd["Date"])
        series = fwd.sort_values("Date").set_index("Date")[col]
        series_last_year = last_year_series(series)
        if series_last_year.empty:
            self.logger.warning("Seasonal chart skipped: no data available in the last-year window for %s.", col)
            return None

        sr_levels = identify_sr_levels(
            series_last_year,
            n_levels=3,
            outlier_threshold=self.sr_outlier_threshold,
            max_multiple_of_current=self.sr_max_multiple,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(series_last_year.index, series_last_year, color="black", label=title_suffix)
        labels = ["SR1", "SR2", "SR3"]
        for (level, _), label in zip(sr_levels, labels):
            ax.axhline(level, linestyle="--", linewidth=1.2, alpha=0.6, label=f"{label}: {level:.2f}")
        ax.set_title(f"{title_suffix} (Last 12 Months)")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        self._date_axis(ax)
        fig.tight_layout()
        filename = "seasonal_contract_history.png"
        file_path = self.output_dir / filename
        fig.savefig(file_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Seasonal contract chart saved to {file_path}")
        return filename

    def generate_prompt_spread_chart(self, curve_df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        latest = curve_df["prompt_spread"].dropna()
        latest_label = f"1yr Spread (Current {latest.iloc[-1]:.3f})" if not latest.empty else "1yr Spread"
        ax.plot(curve_df["Date"], curve_df["prompt_spread"], color="black", linewidth=1.2, label=latest_label)
        ax.axhline(0, color="black", linewidth=1)
        positive = curve_df["prompt_spread"] > 0
        ax.fill_between(
            curve_df["Date"],
            curve_df["prompt_spread"],
            0,
            where=positive,
            color="green",
            alpha=0.2,
            interpolate=True,
            label="> 0",
        )
        ax.fill_between(
            curve_df["Date"],
            curve_df["prompt_spread"],
            0,
            where=~positive,
            color="red",
            alpha=0.2,
            interpolate=True,
            label="< 0",
        )
        sr_levels = identify_sr_levels(
            curve_df["prompt_spread"],
            n_levels=3,
            outlier_threshold=self.sr_outlier_threshold,
            max_multiple_of_current=self.sr_max_multiple,
        )
        for idx, (level, _) in enumerate(sr_levels, start=1):
            ax.axhline(level, linestyle="--", color="gray", alpha=0.6, label=f"SR{idx}: {level:.3f}")
        ax.set_title("1yr Spread (Front vs 12th Month)")
        ax.set_ylabel("Spread")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._date_axis(ax)
        fig.tight_layout()
        filename = "chart_prompt_spread.png"
        fig.savefig(self.output_dir / filename, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return filename

    def generate_storage_chart(self, storage_df: pd.DataFrame) -> Optional[str]:
        if storage_df is None or storage_df.empty:
            return None
        cutoff = storage_df["date"].max() - pd.DateOffset(years=5)
        recent = storage_df.loc[storage_df["date"] >= cutoff].copy()
        if recent.empty:
            return None
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(recent["date"], recent["Lower48"], color="black", label="Current")
        ax.plot(
            recent["date"],
            recent["avg_5y"],
            color="black",
            linestyle="--",
            label="5Y Avg",
        )
        ax.fill_between(
            recent["date"],
            recent["min_10y"],
            recent["max_10y"],
            color="grey",
            alpha=0.2,
            label="10Y Range",
        )
        above_avg = recent["Lower48"] > recent["avg_5y"]
        ax.fill_between(
            recent["date"],
            recent["Lower48"],
            recent["avg_5y"],
            where=above_avg,
            color="green",
            alpha=0.15,
        )
        ax.fill_between(
            recent["date"],
            recent["Lower48"],
            recent["avg_5y"],
            where=~above_avg,
            color="red",
            alpha=0.15,
        )
        ax.set_title("EIA Lower48 Storage vs 5Y Avg and 10Y Range")
        ax.set_ylabel("Bcf")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._date_axis(ax)
        fig.tight_layout()
        filename = "chart_storage.png"
        fig.savefig(self.output_dir / filename, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return filename

    def generate_cot_charts(self, cot_df: pd.DataFrame) -> List[str]:
        outputs: List[str] = []
        cutoff = cot_df["Date"].max() - pd.DateOffset(years=5)
        recent = cot_df.loc[cot_df["Date"] >= cutoff].copy()
        if recent.empty:
            return outputs
        windows = [52]
        smooth_window = 4
        colors = {
            "Total_MM_Net": "red",
            "Total_Prod_Net": "green",
            "Total_Swap_Net": "purple",
        }
        for window in windows:
            fig, ax = plt.subplots(figsize=(9, 5))
            for prefix, color in colors.items():
                col = f"{prefix}_pct_{window}"
                if col not in cot_df.columns:
                    continue
                smoothed = recent[col].rolling(window=smooth_window, min_periods=1).mean()
                ax.plot(
                    recent["Date"],
                    smoothed * 100,
                    color=color,
                    label=f"{prefix.replace('Total_', '')} (SMA{smooth_window})",
                )
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percentile")
            ax.set_title(f"COT Percentile (Rolling {window}w, SMA{smooth_window})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            self._date_axis(ax)
            fig.tight_layout()
            filename = f"chart_cot_{window}w.png"
            fig.savefig(self.output_dir / filename, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            outputs.append(filename)
        return outputs

    def generate_all(
        self,
        curve_df: pd.DataFrame,
        storage_df: Optional[pd.DataFrame],
        cot_df: pd.DataFrame,
    ) -> List[str]:
        charts: List[str] = []
        charts.append(self.generate_strip_chart(curve_df))
        charts.append(self.generate_prompt_spread_chart(curve_df))
        charts.append(self.generate_front_month_history_chart(curve_df))
        charts.append(self.generate_seasonal_contract_chart(curve_df))
        charts.append(self.generate_storage_chart(storage_df) if storage_df is not None else None)
        charts.extend(self.generate_cot_charts(cot_df))
        print("generate_all() produced charts:", charts)
        return [c for c in charts if c]


class MarketAnalysis:
    """Orchestrate Morning Pack analysis and reporting."""

    def __init__(
        self,
        *,
        force: bool = True,
        logger: Optional[logging.Logger] = None,
        overrides: Optional[Dict[str, pd.DataFrame]] = None,
        project_root: Optional[Path] = None,
        package_root: Optional[Path] = None,
        include_forward_curve: bool = True,
        train_models: bool = False,
    ) -> None:
        self.force = force
        self.logger = logger or logging.getLogger(__name__)
        self.include_forward_curve = include_forward_curve
        self.train_models = train_models
        project_root = project_root or get_project_root()
        guessed_package_root = project_root / "GPTCOT"
        package_root = package_root or (guessed_package_root if guessed_package_root.exists() else Path(__file__).resolve().parents[2])
        template_path = package_root / "src" / "gptcot" / "reporting" / "templates" / "market_report.html"
        if not template_path.exists():
            template_path = Path(__file__).resolve().parent / "reporting" / "templates" / "market_report.html"

        self.paths = AnalysisPaths(
            project_root=project_root,
            package_root=package_root,
            info_dir=project_root / "INFO",
            output_dir=package_root / "output",
            report_dir=project_root / "Scripts" / "MarketAnalysis_Report_Output",
            processed_dataset=package_root / "output" / "processed_dataset.csv",
            forward_returns_csv=output_csv_path(package_root / "output"),
            template_path=template_path,
        )
        self.overrides = overrides or {}
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)

    def _should_refresh_ice(self) -> bool:
        """Return True when we should update INFO via the ICE forward-curve script."""

        if os.getenv("GPTCOT_SKIP_ICE_REFRESH"):
            self.logger.info("Skipping ICE refresh because GPTCOT_SKIP_ICE_REFRESH is set.")
            return False
        if self.overrides:
            self.logger.info("Skipping ICE refresh because overrides are supplied: %s", sorted(self.overrides))
            return False
        return True

    def _refresh_info_from_ice(self) -> None:
        """Call the ICE HenryForwardCurve script to refresh INFO inputs before analysis."""

        ice_script = self.paths.project_root / "ICE" / "HenryForwardCurve.py"
        if not ice_script.exists():
            self.logger.warning("ICE forward curve updater not found at %s; skipping refresh.", ice_script)
            return
        self.logger.info("Refreshing ICE forward curve and absolute history via %s", ice_script)
        try:
            result = subprocess.run(
                [sys.executable, str(ice_script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            self.logger.error(
                "ICE refresh failed (rc=%s). STDOUT:\n%s\nSTDERR:\n%s",
                exc.returncode,
                exc.stdout,
                exc.stderr,
            )
            raise
        if result.stdout:
            self.logger.debug("ICE refresh stdout:\n%s", result.stdout.strip())
        if result.stderr:
            self.logger.debug("ICE refresh stderr:\n%s", result.stderr.strip())
        self.logger.info("ICE refresh complete.")

    def _compute_curve_features(self) -> pd.DataFrame:
        if "curve" in self.overrides:
            return self.overrides["curve"].copy()
        curve = pd.read_csv(forward_curve_path(self.paths.info_dir))
        curve["Date"] = pd.to_datetime(curve["Date"])
        curve = curve.sort_values("Date")
        required = [col for col in CURVE_COLUMNS if col in curve.columns]
        if len(required) < 12:
            raise ValueError("Forward curve missing required FWD_00..FWD_11 columns.")
        curve["strip_12m"] = curve[[f"FWD_{i:02d}" for i in range(12)]].mean(axis=1)
        curve["strip_12m_sma10"] = curve["strip_12m"].rolling(window=10, min_periods=1).mean()
        # 1yr spread = prompt minus the 12th month out (FWD_11)
        curve["prompt_spread"] = curve["FWD_00"] - curve["FWD_11"]
        curve["prompt_spread_pct_52"] = compute_rolling_percentile(
            curve["prompt_spread"], window=PROMPT_PCT_WINDOW, min_periods=PROMPT_PCT_WINDOW // 2
        )
        processed_cols = ["Date", "strip_12m", "strip_12m_sma10", "prompt_spread", "prompt_spread_pct_52"]
        # Preserve key seasonal/prompt columns for downstream charts.
        for keep_col in ("FWD_00", "FWD_02", "FWD_09", "FWD_11"):
            if keep_col in curve.columns and keep_col not in processed_cols:
                processed_cols.append(keep_col)
        processed = curve[processed_cols].copy()
        processed.to_csv(self.paths.processed_dataset, index=False)
        return processed

    def _load_curve_features(self) -> pd.DataFrame:
        if not self.force and self.paths.processed_dataset.exists() and "curve" not in self.overrides:
            df = pd.read_csv(self.paths.processed_dataset)
            df["Date"] = pd.to_datetime(df["Date"])
            required_cols = {"FWD_00", "FWD_02", "FWD_09"}
            if required_cols.issubset(set(df.columns)):
                return df
            self.logger.info("Cached curve features missing required columns (%s); recomputing.", required_cols)
        return self._compute_curve_features()

    def _compute_storage_dataset(self) -> Optional[pd.DataFrame]:
        if "storage" in self.overrides:
            return self.overrides["storage"].copy()
        path = eia_totals_path(self.paths.info_dir)
        if not path.exists():
            self.logger.warning("EIAtotals.csv not found at %s; skipping storage chart.", path)
            return None
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Period"])
        df = df.sort_values("date").reset_index(drop=True)
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

        avg_5y: List[float] = []
        min_10y: List[float] = []
        max_10y: List[float] = []
        for _, row in df.iterrows():
            year = row["date"].year
            week = row["weekofyear"]
            hist5 = df.loc[
                (df["weekofyear"] == week) & (df["date"].dt.year < year) & (df["date"].dt.year >= year - 5),
                "Lower48",
            ]
            hist10 = df.loc[
                (df["weekofyear"] == week) & (df["date"].dt.year < year) & (df["date"].dt.year >= year - 10),
                "Lower48",
            ]
            avg_5y.append(float(hist5.mean()) if not hist5.empty else np.nan)
            min_10y.append(float(hist10.min()) if not hist10.empty else np.nan)
            max_10y.append(float(hist10.max()) if not hist10.empty else np.nan)
        df["avg_5y"] = avg_5y
        df["min_10y"] = min_10y
        df["max_10y"] = max_10y
        df["surplus_deficit_pct"] = (df["Lower48"] / df["avg_5y"]) - 1
        return df[["date", "Lower48", "avg_5y", "min_10y", "max_10y", "surplus_deficit_pct"]]

    def _compute_cot_indices(self) -> pd.DataFrame:
        if "cot" in self.overrides:
            return self.overrides["cot"].copy()
        cot_df = pd.read_csv(cot_path(self.paths.info_dir))
        cot_df["Date"] = pd.to_datetime(cot_df["Date"])
        cot_df = compute_cot_features(cot_df, min_periods=10)
        return cot_df

    def _ensure_forward_returns(self) -> pd.DataFrame:
        if "forward_returns" in self.overrides:
            df = self.overrides["forward_returns"].copy()
        else:
            if self.force or not self.paths.forward_returns_csv.exists():
                config = PipelineConfig.from_args(info_dir=self.paths.info_dir, output_dir=self.paths.output_dir)
                dataset = build_dataset(config, logger=self.logger)
                write_outputs(dataset, config, logger=self.logger)
                summary = summarize(dataset)
                self.logger.info(
                    "Forward returns built: %s valid rows of %s", summary["valid_rows"], summary["total_rows"]
                )
            df = pd.read_csv(self.paths.forward_returns_csv)
        df["cot_date"] = pd.to_datetime(df["cot_date"])
        df["horizon_date"] = pd.to_datetime(df["horizon_date"])
        df["is_valid"] = df["is_valid"].astype(bool)
        return df

    def _merge_factor_columns(self, forward_returns: pd.DataFrame, regimes_df: pd.DataFrame) -> pd.DataFrame:
        """Attach missing factor columns (prompt spread, storage) to the forward returns frame."""

        merged = forward_returns.copy()
        missing = [col for col in ANALOG_FACTORS if col not in merged.columns and col in regimes_df.columns]
        if not missing:
            return merged
        factor_frame = regimes_df[["date"] + missing].rename(columns={"date": "cot_date"})
        merged = merged.merge(factor_frame, on="cot_date", how="left")
        return merged

    @staticmethod
    def _select_actionable_analogs(perf_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Pick the most actionable analogs, favoring strong hit rates and returns."""

        if perf_df.empty:
            return pd.DataFrame(columns=["factor_combo", "bias", "avg_return", "hit_rate", "count", "horizon", "score"])

        def bias(row: pd.Series) -> str:
            return "Bullish" if row["avg_return"] > 0 and row["hit_rate"] > 0.6 else "Bearish"

        # Filter to directional edges
        filtered = perf_df.loc[
            ((perf_df["avg_return"] > 0) & (perf_df["hit_rate"] > 0.6))
            | ((perf_df["avg_return"] < 0) & (perf_df["hit_rate"] < 0.4))
        ].copy()

        if filtered.empty:
            return pd.DataFrame(columns=["factor_combo", "bias", "avg_return", "hit_rate", "count", "horizon", "score"])

        filtered["bias"] = filtered.apply(bias, axis=1)
        filtered["score"] = filtered["hit_rate"] + filtered["avg_return"].abs()
        filtered = filtered.sort_values(["score", "count"], ascending=[False, False]).head(top_n)
        return filtered[["factor_combo", "bias", "avg_return", "hit_rate", "count", "horizon", "score"]]

    @staticmethod
    def _format_analog_rows(df: pd.DataFrame) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "factor_combo": row["factor_combo"],
                    "bias": row["bias"],
                    "horizon": f"{int(row['horizon'])}d",
                    "avg_return": f"{row['avg_return'] * 100:.2f}%",
                    "hit_rate": f"{row['hit_rate'] * 100:.1f}%",
                    "count": int(row["count"]),
                }
            )
        return rows

    def _build_unified_dataset(
        self, curve_df: pd.DataFrame, storage_df: Optional[pd.DataFrame], cot_df: pd.DataFrame
    ) -> pd.DataFrame:
        date_index = pd.date_range(curve_df["Date"].min(), curve_df["Date"].max(), freq="D")
        base = pd.DataFrame(index=date_index)
        base = base.join(curve_df.set_index("Date"), how="left")

        if storage_df is not None and not storage_df.empty:
            storage_daily = (
                storage_df.set_index("date")
                .reindex(date_index)
                .sort_index()
                .ffill()
            )
            base = base.join(storage_daily, how="left")

        cot_daily = cot_df.set_index("Date").reindex(date_index).sort_index().ffill()
        base = base.join(cot_daily, how="left")

        if "prompt_spread_pct_52" not in base.columns:
            base["prompt_spread_pct_52"] = compute_rolling_percentile(
                base["prompt_spread"], window=PROMPT_PCT_WINDOW, min_periods=PROMPT_PCT_WINDOW // 2
            )
        if "surplus_deficit_pct" not in base.columns and "Lower48" in base.columns and "avg_5y" in base.columns:
            base["surplus_deficit_pct"] = (base["Lower48"] / base["avg_5y"]) - 1

        base = base.reset_index().rename(columns={"index": "date"})
        return base

    @staticmethod
    def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
        def spread_regime(val: float) -> str:
            if pd.isna(val):
                return "Unknown"
            if val <= 0.1:
                return "Ext Contango"
            if val <= 0.35:
                return "Contango"
            if val <= 0.65:
                return "Neutral"
            if val <= 0.9:
                return "Backwardation"
            return "Ext Backwardation"

        def storage_regime(val: float) -> str:
            if pd.isna(val):
                return "Unknown"
            if val < -0.1:
                return "Ext Deficit"
            if val < -0.02:
                return "Deficit"
            if val <= 0.02:
                return "Neutral"
            if val <= 0.1:
                return "Surplus"
            return "Ext Surplus"

        def cot_regime(val: float) -> str:
            if pd.isna(val):
                return "Unknown"
            if val <= 0.1:
                return "Crowded Short"
            if val <= 0.35:
                return "Short"
            if val <= 0.65:
                return "Neutral"
            if val <= 0.9:
                return "Long"
            return "Crowded Long"

        classified = df.copy()
        classified["spread_regime"] = classified["prompt_spread_pct_52"].apply(spread_regime)
        storage_series = classified["surplus_deficit_pct"] if "surplus_deficit_pct" in classified.columns else pd.Series(np.nan, index=classified.index)
        classified["storage_regime"] = storage_series.apply(storage_regime)
        classified["cot_regime"] = classified["Total_MM_Net_pct_52"].apply(cot_regime)
        classified["combined_regime"] = classified[["cot_regime", "storage_regime", "spread_regime"]].agg(
            " / ".join, axis=1
        )
        return classified

    @staticmethod
    def _hit_rate(series: pd.Series) -> float:
        if series.empty:
            return np.nan
        return float((series > 0).mean())

    @staticmethod
    def _fmt_pct(value: float, digits: int = 1) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.{digits}f}%"

    def _build_narrative(self, digest: Dict[str, object]) -> str:
        current_row = digest["current_setup"].iloc[0]
        best_combo = digest.get("best_combo")
        suggestion = "Hold neutral"

        parts = []
        parts.append(
            f"Current setup: 1yr spread sits in the {current_row.get('1yr Spread Bucket', 'n/a')} tercile "
            f"({self._fmt_pct(current_row.get('1yr Spread %tile', np.nan), digits=1)} of the past year); "
            f"storage is {self._fmt_pct(current_row.get('Storage Surplus/Deficit %', np.nan), digits=1)} vs 5Y "
            f"({current_row.get('Storage Bucket', 'n/a')} bucket); "
            f"Managed Money is {current_row.get('MM Bucket', 'n/a')} "
            f"at {self._fmt_pct(current_row.get('MM Net %tile', np.nan), digits=1)}."
        )

        if best_combo is None:
            parts.append("History indicates limited edge; a neutral stance may be prudent.")
            parts.append("Action: Hold neutral.")
            return "<p>" + "</p><p>".join(parts) + "</p>"

        avg = best_combo["avg_return"]
        hit = best_combo["hit_rate"]
        bias_phrase = "bullish bias; buying strength may be rewarded." if avg > 0 and hit >= 0.7 else (
            "bearish tilt; rallies may be sold." if avg < 0 and hit >= 0.6 else "limited edge; stay balanced."
        )
        if avg > 0 and hit >= 0.7:
            suggestion = "Lean long."
        elif avg < 0 and hit >= 0.6:
            suggestion = "Lean short."

        parts.append(
            f"Historical analogs suggest a {bias_phrase} "
            f"Top analog: {best_combo['factor_combo']} has delivered "
            f"{self._fmt_pct(avg * 100, digits=2)} avg {best_combo['horizon']}d return "
            f"with a {self._fmt_pct(hit * 100, digits=1)} hit rate across {best_combo['count']} samples."
        )
        parts.append(f"Current blend: {digest.get('current_combo_label', '')}.")
        parts.append(f"Action: {suggestion}")
        return "<p>" + "</p><p>".join(parts) + "</p>"

    def compute_analyst_digest(
        self, regimes_df: pd.DataFrame, forward_returns: pd.DataFrame, latest_date: pd.Timestamp
    ) -> Dict[str, object]:
        current_row = regimes_df.loc[regimes_df["date"] <= latest_date].tail(1)
        if current_row.empty:
            raise ValueError("Unable to determine current regime.")
        current = current_row.iloc[0]

        forward_with_factors = self._merge_factor_columns(forward_returns, regimes_df)
        bucketed, bucket_specs = compute_factor_buckets(forward_with_factors, ANALOG_FACTORS)
        analog_perf = compute_analog_performance(bucketed, ANALOG_FACTORS, horizons=DEFAULT_HORIZONS)
        analog_table = self._select_actionable_analogs(analog_perf)

        latest_bucket_row = bucketed.loc[bucketed["cot_date"] == latest_date].head(1)
        bucket_labels: Dict[str, str] = {}
        if not latest_bucket_row.empty:
            for factor in ANALOG_FACTORS:
                bucket_col = f"{factor}_bucket"
                if bucket_col in latest_bucket_row.columns:
                    bucket_labels[factor] = latest_bucket_row.iloc[0].get(bucket_col, "Missing")

        current_setup = pd.DataFrame(
            [
                {
                    "Date": current["date"].date(),
                    "12m Strip (SMA10)": current.get("strip_12m_sma10", np.nan),
                    "1yr Spread": current.get("prompt_spread", np.nan),
                    "1yr Spread %tile": current.get("prompt_spread_pct_52", np.nan) * 100,
                    "1yr Spread Bucket": bucket_labels.get("prompt_spread_pct_52", "Missing"),
                    "Storage Surplus/Deficit %": current.get("surplus_deficit_pct", np.nan) * 100,
                    "Storage Bucket": bucket_labels.get("surplus_deficit_pct", "Missing"),
                    "MM Net %tile": current.get("Total_MM_Net_pct_52", np.nan) * 100,
                    "MM Bucket": bucket_labels.get("Total_MM_Net_pct_52", "Missing"),
                    "Producers %tile": current.get("Total_Prod_Net_pct_52", np.nan) * 100,
                    "Producers Bucket": bucket_labels.get("Total_Prod_Net_pct_52", "Missing"),
                    "Swaps %tile": current.get("Total_Swap_Net_pct_52", np.nan) * 100,
                    "Swaps Bucket": bucket_labels.get("Total_Swap_Net_pct_52", "Missing"),
                }
            ]
        )

        current_combo_cols = []
        current_combo_vals = []
        for factor in ANALOG_FACTORS:
            label = bucket_labels.get(factor)
            if label and label != "Missing":
                current_combo_cols.append(f"{factor}_bucket")
                current_combo_vals.append(label)
        current_combo_label = format_combo_label(current_combo_cols, tuple(current_combo_vals)) if current_combo_cols else "Current mix unavailable"

        best_combo = None
        if not analog_table.empty:
            anchor_row = analog_table.iloc[0]
            best_combo = {
                "factor_combo": anchor_row["factor_combo"],
                "horizon": int(anchor_row["horizon"]),
                "avg_return": float(anchor_row["avg_return"]),
                "hit_rate": float(anchor_row["hit_rate"]),
                "bias": anchor_row["bias"],
                "count": int(anchor_row["count"]),
            }

        analog_table_payload = {"rows": self._format_analog_rows(analog_table)}

        return {
            "current_setup": current_setup,
            "analog_performance": analog_perf,
            "analog_table": analog_table,
            "analog_table_payload": analog_table_payload,
            "current_combo_label": current_combo_label,
            "best_combo": best_combo,
            "bucket_specs": bucket_specs,
        }

    def _render_template(
        self,
        primary_price_charts: List[str],
        secondary_charts: List[str],
        forward_charts: List[str],
        spread_charts: List[str],
        forecast_html: str,
        data_json: str,
    ) -> str:
        template_text = Path(self.paths.template_path).read_text(encoding="utf-8")
        template = Template(template_text)
        return template.render(
            title="Curve Your Enthusiasm",
            primary_price_charts=primary_price_charts,
            secondary_charts=secondary_charts,
            forward_charts=forward_charts,
            spread_charts=spread_charts,
            forecast_html=forecast_html,
            data_json=data_json,
            generated_on=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            methodology_note="Prices reflect 1-Month Constant Maturity Henry Hub via 12-month forward strip averages.",
        )

    @staticmethod
    def _build_importance_chart(predictions: Dict[int, Optional[Dict[str, Any]]], output_dir: Path) -> str:
        """Aggregate feature importances across horizons and return base64 png (empty string if none)."""

        agg: Dict[str, float] = {}
        for result in predictions.values():
            if not result or not result.get("importances"):
                continue
            for name, score in result["importances"]:
                agg[name] = agg.get(name, 0.0) + float(score)
        if not agg:
            return ""

        names = list(agg.keys())
        scores = [agg[n] for n in names]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(names, scores, color="#0b7ba7")
        ax.set_ylabel("Importance (sum)")
        ax.set_title("Top Feature Importances Across Horizons")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / "feature_importances.png"
        fig.savefig(img_path, bbox_inches="tight", facecolor="white")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return encoded

    @staticmethod
    def _build_forecast_html(
        predictions: Dict[int, Optional[Dict[str, Any]]],
        horizons: List[int],
        importance_img_b64: str,
    ) -> str:
        intro = (
            "Our smart assistant looks at what big traders (like hedge funds and producers) are doing, "
            "how full the gas storage tanks are compared to normal, and how steep or flat the price curve is. "
            "It uses these clues to guess whether prices will go up or down over the next few weeks."
        )
        encouragement = "Think of it like weather forecasting, but for gas prices: we use patterns from the past to guess what might happen next."

        def factor_sentence(result: Optional[Dict[str, Any]]) -> str:
            if not result or not result.get("importances"):
                return "Insufficient signal to identify drivers."
            labels = [FEATURE_DESCRIPTIONS.get(name, name) for name, _ in result["importances"][:3]]
            tone = "Bullish" if (result.get("pred_return", 0) >= 0) else "Bearish"
            if len(labels) == 1:
                body = labels[0]
            elif len(labels) == 2:
                body = " and ".join(labels)
            else:
                body = ", ".join(labels[:2]) + f", and {labels[2]}"
            return f"{tone}: The model was most influenced by {body}."

        rows: List[str] = []
        for h in horizons:
            result = predictions.get(h)
            if result:
                pct = result["pred_return"] * 100
                direction = "up" if pct >= 0 else "down"
                pct_text = f"{pct:+.2f}% ({direction})"
            else:
                pct_text = "N/A"
            rows.append(
                f"<tr><td>{h}</td><td>{pct_text}</td><td>{factor_sentence(result)}</td></tr>"
            )

        table = (
            "<table><thead><tr><th>Horizon (days)</th><th>Predicted Return (% / Direction)</th>"
            "<th>Drivers &amp; rationale</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

        feature_section = ""
        if importance_img_b64:
            feature_section = (
                "<h3>What the model cares about most</h3>"
                "<p>These bars show which signals the model thinks matter most right now.</p>"
                f'<img src="data:image/png;base64,{importance_img_b64}" alt="Feature importances" style="max-width:480px;">'
            )

        parts = [intro, encouragement, table, feature_section]
        return "<div>" + "".join(f"<p>{p}</p>" if not p.startswith("<table") and not p.startswith("<h3") else p for p in parts if p) + "</div>"

    def run(self) -> Path:
        if self._should_refresh_ice():
            self._refresh_info_from_ice()
        curve_df = self._load_curve_features()
        storage_df = self._compute_storage_dataset()
        cot_df = self._compute_cot_indices()
        forward_returns = self._ensure_forward_returns()
        # Persist forward_returns to ensure training can read cot_forward_returns.csv
        if self.force or self.train_models or not self.paths.forward_returns_csv.exists():
            forward_returns.to_csv(self.paths.forward_returns_csv, index=False)

        if self.train_models:
            self.logger.info("Training models for horizons %s...", DEFAULT_HORIZONS)
            try:
                train_and_save_models(list(DEFAULT_HORIZONS), self.paths.info_dir, self.paths.output_dir)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("Training models failed: %s", exc)
                print(f" ! Training models failed: {exc}")

        generator = MarketReportGenerator(
            self.paths.report_dir,
            self.logger,
            sr_outlier_threshold=1.5,
            sr_max_multiple=2.0,
        )
        charts = generator.generate_all(curve_df, storage_df, cot_df)

        forward_charts: List[str] = []
        spread_charts: List[str] = []
        if self.include_forward_curve:
            try:
                table = build_forward_curve_table(
                    self.paths.info_dir, lookbacks=LOOKBACK_DAYS, logger=self.logger
                )
                if not table.empty:
                    forward_charts = list(
                        generate_forward_curve_charts(table, output_dir=self.paths.report_dir)
                    )
                    table.to_csv(self.paths.output_dir / "forward_curve_table.csv", index_label="contract_month")
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("Forward curve section skipped due to error: %s", exc)
        try:
            front_label = latest_front_label(self.paths.info_dir)
            if front_label:
                pairs = spread_pairs_from_front(front_label)
                if not pairs:
                    self.logger.warning("No spread pairs derived from front month %s", front_label)
                else:
                    for c1, c2 in pairs:
                        n1, n2 = normalize_contract_pair(c1, c2)
                        fname = generate_spread_chart(n1, n2, self.paths.report_dir, logger=self.logger)
                        if fname:
                            spread_charts.append(fname)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Spread charts skipped due to error: %s", exc)

        predictions = predict_returns(self.paths.info_dir, list(DEFAULT_HORIZONS))
        importance_img_b64 = self._build_importance_chart(predictions, self.paths.output_dir)
        forecast_html = self._build_forecast_html(predictions, list(DEFAULT_HORIZONS), importance_img_b64)

        data_payload = {"forecast": predictions}
        data_json = json.dumps(data_payload, default=str)

        primary_charts = charts[:4]
        secondary_charts = charts[4:]
        print("primary_price_charts passed to template:", primary_charts)
        html = self._render_template(
            primary_charts,
            secondary_charts,
            forward_charts,
            spread_charts,
            forecast_html,
            data_json,
        )
        report_path = self.paths.report_dir / "cot_market_report.html"
        report_path.write_text(html, encoding="utf-8")
        self.logger.info("Report written to %s", report_path)
        return report_path


def run_market_analysis(
    *,
    force: bool = True,
    logger: Optional[logging.Logger] = None,
    overrides: Optional[Dict[str, pd.DataFrame]] = None,
    project_root: Optional[Path] = None,
    package_root: Optional[Path] = None,
    include_forward_curve: bool = True,
    train_models: bool = False,
) -> Path:
    """Entrypoint for CLI use."""

    analysis = MarketAnalysis(
        force=force,
        logger=logger,
        overrides=overrides,
        project_root=project_root,
        package_root=package_root,
        include_forward_curve=include_forward_curve,
        train_models=train_models,
    )
    return analysis.run()
