"""CLI entrypoint for the LNG Feedgas Dashboard."""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from LNGAnalysis.src_lng_dashboard.charts import build_all_charts
from LNGAnalysis.src_lng_dashboard.config import DashboardConfig
from LNGAnalysis.src_lng_dashboard.features import (
    TimeSeriesBundle,
    build_forecast_contributions,
    build_timeseries_bundle,
    compute_actual_metrics,
    compute_forecast_metrics,
    select_top_facilities,
)
from LNGAnalysis.src_lng_dashboard.io import check_data_freshness, load_environment, load_lng_data
from LNGAnalysis.src_lng_dashboard.narrative import generate_narrative
from LNGAnalysis.src_lng_dashboard.report import build_html_report
from LNGAnalysis.src_lng_dashboard.utils import (
    current_run_timestamp,
    ensure_directory,
    find_repo_root,
    resolve_paths,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


def _parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LNG feedgas dashboard.")
    parser.add_argument("--info-dir", type=str, default=None, help="Path to INFO directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output root directory")
    parser.add_argument("--lookback-years", type=int, default=5, help="Number of prior years to overlay")
    parser.add_argument("--top-facilities", type=int, default=5, help="Number of facility charts to display")
    parser.add_argument("--use-llm", type=_parse_bool, default=True, help="Use LLM for narrative (true/false)")
    return parser.parse_args()


def prepare_data(config: DashboardConfig, run_ts: datetime) -> tuple[TimeSeriesBundle, pd.DataFrame, list[str], dict]:
    repo_root = find_repo_root(Path(__file__).resolve())
    env_path = repo_root / "Scripts" / ".env"
    env = load_environment(env_path if env_path.exists() else None)
    actual_df, forecast_df, mapping = load_lng_data(config.info_dir)
    warnings = check_data_freshness(actual_df, forecast_df, run_day=run_ts.date())
    bundle = build_timeseries_bundle(actual_df, forecast_df)
    return bundle, mapping, warnings, env


def main() -> None:
    setup_logging()
    args = parse_args()
    info_dir, out_dir = resolve_paths(args.info_dir, args.out_dir)
    config = DashboardConfig.from_args(args, info_dir=info_dir, out_dir=out_dir)
    run_ts = current_run_timestamp()
    LOGGER.info("Starting LNG dashboard run.")

    bundle, mapping, warnings, env = prepare_data(config, run_ts)

    # Metrics
    actual_with_total = pd.concat([bundle.actual, bundle.total_actual], ignore_index=True)
    forecast_with_total = pd.concat([bundle.forecast, bundle.total_forecast], ignore_index=True)
    actual_metrics = compute_actual_metrics(actual_with_total, config)
    forecast_metrics = compute_forecast_metrics(actual_with_total, forecast_with_total, config)
    contributions = build_forecast_contributions(forecast_metrics, horizon=14)
    top_facilities = select_top_facilities(bundle.actual, config.top_facilities)

    # Narrative
    narrative_text = generate_narrative(actual_metrics, contributions, warnings, config, env, config.use_llm)

    # Charts
    current_year = run_ts.year
    charts = build_all_charts(
        bundle=bundle,
        facilities=top_facilities,
        lookback_years=config.lookback_years,
        current_year=current_year,
        template=config.figure_template,
    )

    # HTML
    html = build_html_report(
        run_ts=run_ts,
        config=config,
        charts=charts,
        actual_metrics=actual_metrics,
        forecast_metrics=forecast_metrics,
        forecast_contributions=contributions,
        selected_facilities=top_facilities,
        mapping=mapping,
        warnings=warnings,
        narrative_text=narrative_text,
    )

    # Output
    dated_dir = ensure_directory(config.out_dir / "lng_dashboard" / run_ts.strftime("%Y-%m-%d"))
    latest_dir = ensure_directory(config.out_dir / "lng_dashboard" / "latest")
    dated_path = dated_dir / "lng_dashboard.html"
    latest_path = latest_dir / "lng_dashboard.html"
    dated_path.write_text(html, encoding="utf-8")
    shutil.copyfile(dated_path, latest_path)
    LOGGER.info("Dashboard written to %s and %s", dated_path, latest_path)


if __name__ == "__main__":
    main()
