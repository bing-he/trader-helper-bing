from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import os

from .utils.logging import get_logger


DEFAULT_TARGET_HUBS: Tuple[str, ...] = (
    "ANR-SE-T",
    "CG-Onshore",
    "FGT-Z3",
    "HSC-HPL Pool",
    "NGPL-TXOK East",
    "NGPL-STX",
    "Pine Prairie",
    "Sonat-Z0 South",
    "TETCO-WLA",
    "TGP-500L",
    "TGP-Z0 South",
    "Transco-65",
    "Transco-85",
)


def _default_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env_files(root_dir: Path, scripts_dir: Path) -> None:
    logger = get_logger(__name__)

    env_candidates = [
        scripts_dir / ".env",
        root_dir / ".env",
        Path.cwd() / ".env",
    ]

    env_path = next((p for p in env_candidates if p.exists()), None)
    if not env_path:
        logger.debug("No .env file found in %s", ", ".join(str(p) for p in env_candidates))
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path)
        logger.info("Loaded environment from %s", env_path)
        return
    except ImportError:
        logger.debug("python-dotenv not installed; using minimal .env parser.")

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
        logger.info("Loaded environment from %s (manual parse)", env_path)
    except Exception as exc:
        logger.warning("Failed to load .env file %s: %s", env_path, exc)


@dataclass
class Config:
    root_dir: Optional[Path] = None
    info_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    flows_path: Optional[Path] = None
    prices_path: Optional[Path] = None
    html_output_path: Optional[Path] = None

    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_project: Optional[str] = None
    gemini_key_source: Optional[str] = None

    rolling_window_days: int = 252 * 2
    min_data_period: int = 252

    pct_slack: float = 0.60
    pct_transition: float = 0.85

    fragility_window: int = 30
    upstream_shock_sigma: float = 1.5
    cap_stable_pct: float = 0.05
    util_clip_max: float = 1.05
    util_anomaly_threshold: float = 1.1
    tight_util_threshold: float = 0.90

    corr_lookback_days: int = 252 * 2
    min_corr_sample: int = 120
    min_abs_corr: float = 0.2
    min_stability: float = 0.6
    corr_stability_segments: int = 4

    active_pipe_min_flow: float = 50.0
    active_pipe_min_cap_days: int = 252
    min_henry_price: float = 0.1

    evidence_min_sample: int = 120
    evidence_min_abs_corr: float = 0.2
    evidence_min_stability: float = 0.6
    evidence_min_transition_samples: int = 30
    evidence_max_charts: int = 5
    evidence_min_tight_samples: int = 30
    evidence_min_lift: float = 0.01

    seasonal_years: int = 5
    gas_year_start_month: int = 4
    seasonal_hub_count: int = 2

    walk_forward_window: int = 252 * 2
    walk_forward_step: int = 21

    enable_llm: bool = True
    high_delta_msi_pct: float = 0.07
    high_delta_fragility: float = 0.20
    high_delta_binding: int = 2
    fundy_item_limit: int = 3

    gulf_pressure_tight_threshold: float = 1.0
    gulf_pressure_loose_threshold: float = -1.0
    gulf_pressure_min_components: int = 2
    gulf_pressure_min_samples: int = 30
    gulf_pressure_sign_window: int = 14
    gulf_pressure_sign_share: float = 0.6
    gulf_pressure_confidence_threshold: float = 0.55
    gulf_pressure_oos_corr_min: float = 0.1
    gulf_pressure_oos_hit_min: float = 0.5

    target_hubs: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGET_HUBS)

    def __post_init__(self) -> None:
        logger = get_logger(__name__)
        root_dir = Path(self.root_dir) if self.root_dir else _default_root_dir()
        scripts_dir = root_dir / "Scripts"
        _load_env_files(root_dir, scripts_dir)

        self.root_dir = root_dir
        self.info_dir = Path(self.info_dir) if self.info_dir else root_dir / "INFO"
        self.output_dir = (
            Path(self.output_dir)
            if self.output_dir
            else scripts_dir / "MarketAnalysis_Report_Output"
        )

        self.flows_path = (
            Path(self.flows_path)
            if self.flows_path
            else self.info_dir / "CriterionHenryFlows.csv"
        )
        self.prices_path = (
            Path(self.prices_path)
            if self.prices_path
            else self.info_dir / "PRICES.csv"
        )
        self.html_output_path = (
            Path(self.html_output_path)
            if self.html_output_path
            else self.output_dir / "HenryFlows.html"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.gemini_api_key is None:
            env_key = os.getenv("GEMINI_API_KEY")
            if env_key:
                self.gemini_api_key = env_key
                self.gemini_key_source = "GEMINI_API_KEY"
            else:
                alias_key = os.getenv("GeminiKey")
                if alias_key:
                    self.gemini_api_key = alias_key
                    self.gemini_key_source = "GeminiKey"
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_project is None:
            self.openai_project = os.getenv("OPENAI_PROJECT")

        if self.gemini_api_key:
            masked = f"{self.gemini_api_key[:4]}...{self.gemini_api_key[-4:]}"
            source = self.gemini_key_source or "unknown"
            logger.debug("Gemini key found via %s (%s)", source, masked)
        elif self.openai_api_key:
            masked = f"{self.openai_api_key[:4]}...{self.openai_api_key[-4:]}"
            logger.debug("OPENAI_API_KEY found (%s)", masked)
        else:
            logger.debug("No AI API keys detected in environment.")
