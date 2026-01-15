from pathlib import Path
import shutil

import pytest

from HenryGPT.config import Config
from HenryGPT.pipeline import run_pipeline


def _copy_required_inputs(src_dir: Path, dest_dir: Path) -> None:
    for filename in ("CriterionHenryFlows.csv", "PRICES.csv"):
        src = src_dir / filename
        if not src.exists():
            pytest.skip("Required CSV inputs not found.")
        shutil.copy(src, dest_dir / filename)


def test_smoke_pipeline_missing_exogenous(tmp_path: Path) -> None:
    cfg = Config(info_dir=tmp_path, output_dir=tmp_path, html_output_path=tmp_path / "HenryFlows.html")
    _copy_required_inputs(Path(cfg.root_dir) / "INFO", tmp_path)
    result = run_pipeline(cfg)
    assert Path(result.html_path).exists()


def test_smoke_pipeline_with_exogenous(tmp_path: Path) -> None:
    cfg = Config(output_dir=tmp_path, html_output_path=tmp_path / "HenryFlows.html")
    if not cfg.flows_path.exists() or not cfg.prices_path.exists():
        pytest.skip("Required CSV inputs not found.")
    info_dir = Path(cfg.info_dir)
    exog_files = [
        info_dir / "CriterionLNGHist.csv",
        info_dir / "Fundy.csv",
        info_dir / "GridStatLoadHist.csv",
        info_dir / "PowerPrices.csv",
    ]
    if not any(path.exists() for path in exog_files):
        pytest.skip("Optional exogenous inputs not found.")
    result = run_pipeline(cfg)
    assert Path(result.html_path).exists()
