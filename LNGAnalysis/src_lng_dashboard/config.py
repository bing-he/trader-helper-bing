"""Configuration objects for the LNG dashboard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class DashboardConfig:
    """Runtime configuration for the LNG dashboard."""

    info_dir: Path
    out_dir: Path
    lookback_years: int = 5
    top_facilities: int = 5
    use_llm: bool = True
    regime_delta_threshold: float = 500.0
    shock_window_days: int = 180
    shock_quantile: float = 0.9
    trailing_windows: Tuple[int, int] = (30, 60)
    forecast_horizons: Tuple[int, int, int] = (7, 14, 21)
    forecast_maxmin_horizon: int = 14
    figure_template: str = "plotly_white"

    @classmethod
    def from_args(cls, args: argparse.Namespace, info_dir: Path, out_dir: Path) -> "DashboardConfig":
        """Construct a configuration object from CLI arguments."""
        return cls(
            info_dir=info_dir,
            out_dir=out_dir,
            lookback_years=int(args.lookback_years),
            top_facilities=int(args.top_facilities),
            use_llm=bool(args.use_llm),
        )

    def trailing_windows_list(self) -> Iterable[int]:
        """Return trailing windows as a list for iteration."""
        return list(self.trailing_windows)
