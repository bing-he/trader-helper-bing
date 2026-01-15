"""Configuration primitives for the gptcot pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the CoT forward-returns pipeline."""

    info_dir: Path
    output_dir: Path
    horizons: List[int] = field(default_factory=lambda: [7, 14, 28, 30])
    max_price_snap_days: int = 5
    min_periods: int = 10
    write_parquet: bool = False
    log_level: str = "INFO"

    @staticmethod
    def _normalize_horizons(values: Iterable[int]) -> List[int]:
        horizons = sorted({int(v) for v in values})
        if not horizons:
            raise ValueError("At least one horizon must be provided.")
        if any(h <= 0 for h in horizons):
            raise ValueError("Horizon values must be positive integers.")
        return horizons

    @classmethod
    def from_args(
        cls,
        *,
        info_dir: str | Path,
        output_dir: str | Path,
        horizons: Iterable[int] | None = None,
        max_price_snap_days: int = 5,
        min_periods: int = 10,
        write_parquet: bool = False,
        log_level: str = "INFO",
    ) -> "PipelineConfig":
        """Create a config object from CLI-friendly arguments."""

        info_path = Path(info_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        default_horizons = [7, 14, 28, 30]
        resolved_horizons = (
            cls._normalize_horizons(horizons) if horizons is not None else default_horizons
        )
        if max_price_snap_days < 0:
            raise ValueError("max_price_snap_days must be non-negative.")
        if min_periods < 0:
            raise ValueError("min_periods must be non-negative.")
        return cls(
            info_dir=info_path,
            output_dir=output_path,
            horizons=resolved_horizons,
            max_price_snap_days=int(max_price_snap_days),
            min_periods=int(min_periods),
            write_parquet=bool(write_parquet),
            log_level=log_level.upper(),
        )
