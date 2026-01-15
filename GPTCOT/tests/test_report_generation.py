from pathlib import Path

import pandas as pd

from gptcot.market_analysis import run_market_analysis


def test_report_generation_creates_assets(tmp_path):
    curve = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=365, freq="D"),
            "strip_12m": [2.5 + i * 0.001 for i in range(365)],
            "strip_12m_sma10": [2.5 + i * 0.001 for i in range(365)],
            "prompt_spread": [0.05 + 0.0001 * i for i in range(365)],
            "FWD_00": [2.5 + 0.001 * i for i in range(365)],
            "FWD_02": [2.6 + 0.001 * i for i in range(365)],
            "FWD_09": [2.7 + 0.001 * i for i in range(365)],
        }
    )
    cot = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="W"),
            "Total_MM_Net_pct_52": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Total_Prod_Net_pct_52": [0.6, 0.5, 0.4, 0.3, 0.2],
            "Total_Swap_Net_pct_52": [0.7, 0.6, 0.55, 0.5, 0.45],
            "Total_MM_Net_pct_156": [0.2, 0.25, 0.3, 0.35, 0.4],
            "Total_Prod_Net_pct_156": [0.7, 0.65, 0.6, 0.55, 0.5],
            "Total_Swap_Net_pct_156": [0.5, 0.55, 0.6, 0.65, 0.7],
        }
    )
    forward_returns = pd.DataFrame(
        {
            "cot_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-04-01", "2024-04-08"]),
            "horizon_date": pd.to_datetime(["2024-01-08", "2024-01-15", "2024-04-15", "2024-04-22"]),
            "horizon_days": [7, 7, 14, 14],
            "pct_change": [0.02, -0.03, 0.05, -0.01],
            "is_valid": [True, True, True, True],
            "Total_MM_Net_pct_52": [0.9, 0.1, 0.85, 0.2],
            "Total_Prod_Net_pct_52": [0.3, 0.8, 0.4, 0.9],
            "Total_Swap_Net_pct_52": [0.4, 0.6, 0.2, 0.7],
        }
    )

    overrides = {
        "curve": curve,
        "cot": cot,
        "forward_returns": forward_returns,
    }
    report_path = run_market_analysis(
        force=True,
        overrides=overrides,
        project_root=tmp_path,
        package_root=tmp_path,
        include_forward_curve=False,
    )

    assert report_path.exists()
    images = list(Path(tmp_path / "Scripts" / "MarketAnalysis_Report_Output").glob("*.png"))
    assert images, "Expected at least one chart image"

    html = report_path.read_text()
    assert "Forecasted Market Moves" in html
    assert "Predicted Return (% / Direction)" in html
    for horizon in (7, 14, 28, 30):
        assert f">{horizon}</td>" in html
    assert "Drivers &amp; rationale" in html
    assert "Insufficient signal to identify drivers." in html


def test_train_models_flag_creates_models(tmp_path, monkeypatch):
    # Prepare minimal INFO data
    info_dir = tmp_path / "INFO"
    info_dir.mkdir(parents=True, exist_ok=True)
    (info_dir / "EIAtotals.csv").write_text(
        "Date,Total,Total_5YAvg\n2023-12-29,3000,2900\n2024-01-05,3050,2920\n",
        encoding="utf-8",
    )
    fwd_rows = ["Date,FrontMonth_Label," + ",".join([f"FWD_{i:02d}" for i in range(12)])]
    fwd_rows.append("2024-01-01,Feb-2024," + ",".join(["2.5"] * 12))
    (info_dir / "HenryForwardCurve.csv").write_text("\n".join(fwd_rows), encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse overrides from the first test for speed
    curve = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=365, freq="D"),
            "strip_12m": [2.5 + i * 0.001 for i in range(365)],
            "strip_12m_sma10": [2.5 + i * 0.001 for i in range(365)],
            "prompt_spread": [0.05 + 0.0001 * i for i in range(365)],
            "prompt_spread_pct_52": [0.1 + 0.0005 * i for i in range(365)],
            "FWD_00": [2.5 + 0.001 * i for i in range(365)],
            "FWD_02": [2.6 + 0.001 * i for i in range(365)],
            "FWD_09": [2.7 + 0.001 * i for i in range(365)],
        }
    )
    cot = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="W"),
            "Total_MM_Net_pct_52": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Total_Prod_Net_pct_52": [0.6, 0.5, 0.4, 0.3, 0.2],
            "Total_Swap_Net_pct_52": [0.7, 0.6, 0.55, 0.5, 0.45],
        }
    )
    forward_returns = pd.DataFrame(
        {
            "cot_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-04-01", "2024-04-08"]),
            "horizon_date": pd.to_datetime(["2024-01-08", "2024-01-15", "2024-04-15", "2024-04-22"]),
            "horizon_days": [7, 7, 14, 14],
            "pct_change": [0.02, -0.03, 0.05, -0.01],
            "is_valid": [True, True, True, True],
            "Total_MM_Net_pct_52": [0.9, 0.1, 0.85, 0.2],
            "Total_Prod_Net_pct_52": [0.3, 0.8, 0.4, 0.9],
            "Total_Swap_Net_pct_52": [0.4, 0.6, 0.2, 0.7],
        }
    )
    storage = pd.DataFrame(
        {
            "date": pd.date_range("2023-12-01", periods=5, freq="W"),
            "Lower48": [3000, 3020, 3040, 3030, 3050],
            "avg_5y": [2950, 2955, 2960, 2965, 2970],
            "min_10y": [2800] * 5,
            "max_10y": [3200] * 5,
            "surplus_deficit_pct": [0.02] * 5,
        }
    )
    overrides = {"curve": curve, "cot": cot, "forward_returns": forward_returns, "storage": storage}

    report_path = run_market_analysis(
        force=True,
        overrides=overrides,
        project_root=tmp_path,
        package_root=tmp_path,
        include_forward_curve=False,
        train_models=True,
    )

    assert report_path.exists()
    for horizon in (7, 14):
        model_path = output_dir / f"price_forecast_model_{horizon}.pkl"
        assert model_path.exists(), f"Expected model file for {horizon}-day horizon"
    # New charts should be written
    assert (tmp_path / "Scripts" / "MarketAnalysis_Report_Output" / "front_month_history.png").exists()
    assert (tmp_path / "Scripts" / "MarketAnalysis_Report_Output" / "seasonal_contract_history.png").exists()
    for horizon in (28, 30):
        skipped_path = output_dir / f"price_forecast_model_{horizon}.pkl"
        assert not skipped_path.exists(), f"Did not expect model file for {horizon}-day horizon without data"
