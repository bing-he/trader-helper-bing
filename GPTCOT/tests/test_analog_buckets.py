import pandas as pd
import pytest

from gptcot.market_analysis import compute_analog_performance, compute_factor_buckets


def test_compute_factor_buckets_assigns_labels():
    df = pd.DataFrame(
        {
            "cot_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "horizon_days": [7, 7, 7, 7, 7],
            "pct_change": [0.02, -0.01, 0.03, 0.01, -0.02],
            "prompt_spread_pct_52": [0.05, 0.15, 0.55, 0.8, 0.9],
            "Total_MM_Net_pct_52": [0.9, 0.85, 0.2, 0.25, 0.3],
        }
    )

    bucketed, specs = compute_factor_buckets(df, ["prompt_spread_pct_52", "Total_MM_Net_pct_52"])

    assert "prompt_spread_pct_52_bucket" in bucketed.columns
    assert "Total_MM_Net_pct_52_bucket" in bucketed.columns
    assert set(specs.keys()) == {"prompt_spread_pct_52", "Total_MM_Net_pct_52"}
    prompt_buckets = set(bucketed["prompt_spread_pct_52_bucket"].dropna())
    mm_buckets = set(bucketed["Total_MM_Net_pct_52_bucket"].dropna())
    assert {"Low", "Mid", "High"} & prompt_buckets
    assert {"Low", "Mid", "High"} & mm_buckets


def test_compute_analog_performance_filters_and_scores():
    df = pd.DataFrame(
        {
            "cot_date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "horizon_days": [7] * 4 + [14] * 4,
            "pct_change": [0.02, 0.03, 0.01, -0.01, 0.05, 0.06, 0.02, -0.02],
            "prompt_spread_pct_52": [0.1, 0.12, 0.15, 0.18, 0.9, 0.92, 0.88, 0.86],
            "Total_MM_Net_pct_52": [0.9, 0.88, 0.91, 0.86, 0.2, 0.25, 0.22, 0.21],
        }
    )
    bucketed, _ = compute_factor_buckets(df, ["prompt_spread_pct_52", "Total_MM_Net_pct_52"])
    perf = compute_analog_performance(
        bucketed, ["prompt_spread_pct_52", "Total_MM_Net_pct_52"], horizons=[7, 14], min_count=2
    )

    assert set(perf["horizon"].unique()) == {7, 14}
    row7 = perf.loc[perf["horizon"] == 7].iloc[0]
    assert row7["count"] >= 2
    assert 0 <= row7["hit_rate"] <= 1
    row14 = perf.loc[perf["horizon"] == 14].iloc[0]
    assert row14["count"] >= 2
    assert 0 <= row14["hit_rate"] <= 1
