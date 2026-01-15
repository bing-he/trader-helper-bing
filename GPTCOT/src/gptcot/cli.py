"""Command-line interface for the gptcot pipeline and market analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from gptcot.config import PipelineConfig
from gptcot.market_analysis import run_market_analysis
from gptcot.pipeline import build_dataset, summarize, write_outputs


def _parse_horizons(raw: str | None) -> List[int] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute roll-aware CoT forward returns.")
    parser.add_argument(
        "--info-dir",
        required=True,
        help="Path to input directory containing NGCommitofTraders.csv, HenryForwardCurve.csv, and HenryHub_Absolute_History.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to directory where outputs will be written",
    )
    parser.add_argument(
        "--horizons",
        default="7,14,28,30",
        help="Comma-separated calendar-day horizons (default: 7,14,28,30)",
    )
    parser.add_argument(
        "--max-price-snap-days",
        type=int,
        default=5,
        help="Max trading-day snaps when searching for entry/exit prices (default: 5)",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=10,
        help="Minimum observations for rolling stats (default: 10)",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write a Parquet output alongside the CSV",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args_list = argv if argv is not None else sys.argv[1:]
    if args_list and args_list[0] == "market-analysis":
        market_parser = argparse.ArgumentParser(
            description="Generate a CoT market analysis report (no path parameters required)."
        )
        market_parser.add_argument(
            "--force",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Regenerate the forward-returns CSV before running the analysis "
                "(default: True; use --no-force to reuse cached outputs)."
            ),
        )
        market_parser.add_argument(
            "--include-forward-curve",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Include forward-curve comparison section (enabled by default).",
        )
        market_parser.add_argument(
            "--train-models",
            action="store_true",
            help="Train and save price forecast models before generating the report.",
        )
        market_parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level (default: INFO)",
        )
        ma_args = market_parser.parse_args(args_list[1:])
        logging.basicConfig(
            level=getattr(logging, ma_args.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
        logger = logging.getLogger("gptcot.market_analysis")
        try:
            report_path = run_market_analysis(
                force=ma_args.force,
                logger=logger,
                include_forward_curve=ma_args.include_forward_curve,
                train_models=ma_args.train_models,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Market analysis failed: %s", exc)
            sys.exit(1)
        print(str(report_path.resolve()))
        return

    parser = build_parser()
    args = parser.parse_args(args_list)

    horizons = _parse_horizons(args.horizons)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger("gptcot")

    try:
        config = PipelineConfig.from_args(
            info_dir=args.info_dir,
            output_dir=args.output_dir,
            horizons=horizons,
            max_price_snap_days=args.max_price_snap_days,
            min_periods=args.min_periods,
            write_parquet=args.write_parquet,
            log_level=args.log_level,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Invalid arguments: %s", exc)
        sys.exit(1)

    try:
        df = build_dataset(config, logger=logger)
        write_outputs(df, config, logger=logger)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    summary = summarize(df)
    logger.info(
        "CoT rows: %s | Horizons processed: %s | Valid: %s | Invalid: %s",
        summary["cot_rows"],
        summary["horizons_processed"],
        summary["valid_rows"],
        summary["invalid_rows"],
    )
    if summary["invalid_reasons"]:
        logger.info("Top invalid reasons: %s", summary["invalid_reasons"])
    else:
        logger.info("No invalid rows encountered.")


if __name__ == "__main__":
    main()
