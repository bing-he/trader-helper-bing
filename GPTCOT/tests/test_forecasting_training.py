from pathlib import Path

import pandas as pd

from gptcot.forecasting import load_price_model, train_and_save_models


def test_train_and_save_models_creates_file(tmp_path: Path):
    info_dir = tmp_path / "INFO"
    output_dir = tmp_path / "output"
    info_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cot_df = pd.DataFrame(
        {
            "cot_date": pd.to_datetime(["2023-12-01", "2023-12-08", "2024-01-05"]),
            "horizon_date": pd.to_datetime(["2023-12-02", "2023-12-09", "2024-01-06"]),
            "horizon_days": [1, 1, 1],
            "pct_change": [0.01, -0.02, 0.015],
            "strip_pct_change": [0.012, -0.018, 0.02],
            "Total_MM_Net_z_52": [1.0, 1.1, 1.2],
            "Total_Prod_Net_z_52": [0.5, 0.55, 0.6],
            "Total_Swap_Net_z_52": [0.25, 0.3, 0.35],
            "Total_MM_Net_pct_52": [0.2, 0.25, 0.3],
            "Total_Prod_Net_pct_52": [0.4, 0.35, 0.3],
            "Total_Swap_Net_pct_52": [0.5, 0.55, 0.6],
        }
    )
    cot_path = output_dir / "cot_forward_returns.csv"
    cot_df.to_csv(cot_path, index=False)

    eia_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-12-01", "2023-12-08", "2023-12-15"]),
            "Total": [3000, 3010, 3020],
            "Total_5YAvg": [2950, 2955, 2960],
        }
    )
    eia_df.to_csv(info_dir / "EIAtotals.csv", index=False)

    curve_cols = {f"FWD_{i:02d}": [2.5 + i * 0.01, 2.45 + i * 0.01, 2.55 + i * 0.01] for i in range(12)}
    curve_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-12-01", "2023-12-10", "2023-12-20"]),
            "FrontMonth_Label": ["Jan-2024", "Jan-2024", "Feb-2024"],
            **curve_cols,
        }
    )
    curve_df.to_csv(info_dir / "HenryForwardCurve.csv", index=False)

    train_and_save_models([1], info_dir, output_dir)

    strip_model_path = output_dir / "price_forecast_model_strip_1.pkl"
    spread_model_path = output_dir / "price_forecast_model_spread_change_1.pkl"
    assert strip_model_path.exists()
    loaded_strip = load_price_model(strip_model_path, target="strip")
    assert loaded_strip is not None
    strip_model, strip_features = loaded_strip
    assert hasattr(strip_model, "feature_names_in_")
    assert strip_features is not None
    assert spread_model_path.exists()
    loaded_spread = load_price_model(spread_model_path, target="spread_change")
    assert loaded_spread is not None
