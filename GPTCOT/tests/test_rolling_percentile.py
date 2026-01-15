import pandas as pd

from gptcot.features import compute_rolling_percentile


def test_rolling_percentile_within_unit_interval():
    series = pd.Series(range(1, 11))
    result = compute_rolling_percentile(series, window=3, min_periods=2)
    non_null = result.dropna()
    assert ((non_null >= 0) & (non_null <= 1)).all()
