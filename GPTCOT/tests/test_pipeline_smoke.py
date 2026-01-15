import importlib

import pandas as pd

from gptcot.config import PipelineConfig
from gptcot.features import compute_cot_features

pipeline_module = importlib.import_module("gptcot.pipeline.build_dataset")
build_dataset = pipeline_module.build_dataset


def test_pipeline_smoke(monkeypatch, tmp_path):
    cot_raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-09"]),
            "Total_OI": [100, 110],
            "Total_MM_Net": [10, 12],
            "Total_Prod_Net": [-5, -4],
            "Total_Swap_Net": [1, 2],
        }
    )
    cot_prepared = compute_cot_features(cot_raw, min_periods=1)

    forward_curve = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-09"]),
            "FrontMonth_Label": ["Feb-2024", "Mar-2024"],
        }
    )

    absolute_history = pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-09", "2024-01-10"]
            ),
            "2024-02-01": [2.0, 2.05, 2.1, 2.15],
            "2024-03-01": [2.6, 2.65, 2.8, 2.85],
        }
    )

    monkeypatch.setattr(
        pipeline_module, "_prepare_cot_dataframe", lambda config: cot_prepared.copy()
    )
    monkeypatch.setattr(
        pipeline_module, "_prepare_forward_curve", lambda config: forward_curve.copy()
    )
    monkeypatch.setattr(
        pipeline_module, "_prepare_absolute_history", lambda config: absolute_history.copy()
    )

    config = PipelineConfig.from_args(
        info_dir=tmp_path,
        output_dir=tmp_path,
        horizons=[7],
        max_price_snap_days=1,
        min_periods=1,
    )

    df = build_dataset(config)

    required_columns = [
        "cot_date",
        "horizon_days",
        "horizon_date",
        "fc_snap_date",
        "frontmonth_label",
        "target_contract_month",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "abs_change",
        "pct_change",
        "is_valid",
        "invalid_reason",
    ]
    for col in required_columns:
        assert col in df.columns

    assert df["is_valid"].any()
