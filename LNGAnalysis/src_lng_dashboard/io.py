"""Input/output and data loading utilities for LNG dashboard."""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from .normalize import FacilityNormalizer
from .utils import parse_date

LOGGER = logging.getLogger(__name__)


def load_environment(env_path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Load environment variables from a .env file."""
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)
    gemini_key = os.getenv("GEMINI_KEY") or os.getenv("GeminiKey")
    openai_project = os.getenv("OPENAI_PROJECT")
    return {
        "GEMINI_KEY": gemini_key,
        "GeminiKey": gemini_key,  # retain original casing for backward compatibility
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_PROJECT": openai_project,
    }


def _read_lng_file(path: Path, normalizer: FacilityNormalizer, source_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read a single LNG CSV file and normalize facilities."""
    if not path.exists():
        LOGGER.warning("LNG file missing: %s", path)
        return pd.DataFrame(columns=["date", "canonical_facility", "value", "source"]), pd.DataFrame(
            columns=["raw_name", "canonical_name", "source_file"]
        )

    LOGGER.info("Loading LNG data: %s", path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df, mapping = normalizer.normalize_frame(df, source=path.name)
    df = df.rename(columns={"canonical_facility": "facility"})
    df["source"] = source_label
    df = df[["date", "facility", "value", "source"]].sort_values("date")
    return df, mapping


def load_lng_data(
    info_dir: Path, normalizer: Optional[FacilityNormalizer] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load actual and forecast LNG data and return mapping details."""
    normalizer = normalizer or FacilityNormalizer()
    actual_path = info_dir / "CriterionLNGHist.csv"
    forecast_path = info_dir / "CriterionLNGForecast.csv"

    actual_df, actual_map = _read_lng_file(actual_path, normalizer, source_label="actual")
    forecast_df, forecast_map = _read_lng_file(forecast_path, normalizer, source_label="forecast")

    mapping = pd.concat([actual_map, forecast_map], ignore_index=True).drop_duplicates()
    return actual_df, forecast_df, mapping


def check_data_freshness(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    run_day: date,
    max_actual_lag_days: int = 2,
    min_forecast_horizon_days: int = 7,
) -> List[str]:
    """Detect stale or missing data; returns list of warning messages."""
    warnings: List[str] = []
    if not actual_df.empty:
        latest_actual = actual_df["date"].max().date()
        if (run_day - latest_actual).days > max_actual_lag_days:
            msg = f"Actual data may be stale; latest {latest_actual}"
            LOGGER.warning(msg)
            warnings.append(msg)
    else:
        msg = "No actual LNG data available."
        LOGGER.warning(msg)
        warnings.append(msg)

    if forecast_df.empty:
        msg = "Forecast data not available; rendering historical only."
        LOGGER.warning(msg)
        warnings.append(msg)
        return warnings

    forecast_start = forecast_df["date"].min().date()
    forecast_end = forecast_df["date"].max().date()
    if not actual_df.empty:
        latest_actual = actual_df["date"].max().date()
        if forecast_start <= latest_actual:
            msg = "Forecast data overlaps or precedes latest actuals; stitching may be imprecise."
            LOGGER.warning(msg)
            warnings.append(msg)

    if (forecast_end - run_day).days < min_forecast_horizon_days:
        msg = f"Forecast horizon short; ends {forecast_end}"
        LOGGER.warning(msg)
        warnings.append(msg)

    return warnings
