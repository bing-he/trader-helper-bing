from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from gptcot.forecasting import predict_returns, save_model


def test_predict_returns_includes_importances(tmp_path: Path):
    info_dir = tmp_path / "INFO"
    output_dir = tmp_path / "output"
    info_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Required inputs for prepare_latest_features
    cot_df = pd.DataFrame(
        {
            "cot_date": pd.to_datetime(["2024-01-05"]),
            "horizon_days": [7],
            "pct_change": [0.01],
            "Total_MM_Net_z_52": [0.2],
            "Total_Prod_Net_z_52": [0.1],
            "Total_Swap_Net_z_52": [0.3],
            "Total_MM_Net_pct_52": [0.4],
            "Total_Prod_Net_pct_52": [0.5],
            "Total_Swap_Net_pct_52": [0.6],
        }
    )
    cot_df.to_csv(output_dir / "cot_forward_returns.csv", index=False)

    eia_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-04", "2024-01-11"]),
            "Total": [3000, 3010],
            "Total_5YAvg": [2900, 2910],
        }
    )
    eia_df.to_csv(info_dir / "EIAtotals.csv", index=False)

    curve_cols = {f"FWD_{i:02d}": [2.5 + 0.01 * i] for i in range(12)}
    curve_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-05"]),
            "FrontMonth_Label": ["Feb-2024"],
            **curve_cols,
        }
    )
    curve_df.to_csv(info_dir / "HenryForwardCurve.csv", index=False)

    # Train a tiny model with known feature order
    feature_names = ["Total_MM_Net_z_52", "Total_Prod_Net_z_52", "Total_Swap_Net_z_52"]
    X = np.array([[0.2, 0.1, 0.3], [0.1, 0.2, 0.4]])
    y = np.array([0.01, -0.02])
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X, y)
    save_model(model, feature_names, output_dir / "price_forecast_model_7.pkl")

    predictions = predict_returns(info_dir, [7])
    result = predictions.get(7)
    assert result is not None
    assert result["importances"] is not None
    assert len(result["importances"]) > 0
