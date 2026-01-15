import pandas as pd
import pytest

from gptcot.market_analysis import detect_extreme_thresholds


def test_detect_extreme_threshold_selects_largest_gap():
    df = pd.DataFrame(
        {
            "horizon_days": [7] * 6,
            "pct_change": [0.5, 0.4, -0.4, -0.3, 0.1, -0.05],
            "Total_MM_Net_pct_52": [0.95, 0.92, 0.1, 0.12, 0.75, 0.25],
        }
    )

    results = detect_extreme_thresholds(df, horizons=[7], series_columns=["Total_MM_Net_pct_52"])
    assert results, "Expected a threshold result"

    best = results[0]
    assert best.percentile == 90
    assert best.favored_side == "high"
    assert best.high_count == 2
    assert best.low_count == 1
    assert best.delta_mean == pytest.approx(0.85)
