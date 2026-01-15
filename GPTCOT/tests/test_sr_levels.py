import pandas as pd

from gptcot.utils import identify_sr_levels


def test_identify_sr_levels_returns_ranked_levels():
    # Construct a series with repeated reversals around two clear levels.
    series = pd.Series([1, 2, 1.1, 2.1, 1.0, 5.0, 1.2, 5.1, 1.1, 4.9, 1.0, 2.0, 1.1, 2.05])

    levels = identify_sr_levels(series, n_levels=2)

    assert len(levels) >= 1
    # Scores should be sorted descending by significance.
    if len(levels) > 1:
        assert levels[0][1] >= levels[1][1]
        # Levels should be distinct
        assert abs(levels[0][0] - levels[1][0]) > 0.01


def test_identify_sr_levels_filters_outlier():
    series = pd.Series([1.0, 1.2, 1.1, 1.3, 1.2, 1.1, 10.0])  # extreme outlier at 10
    levels = identify_sr_levels(series, n_levels=3, outlier_threshold=1.5, max_multiple_of_current=2.0)
    assert levels
    assert all(level < 3.0 for level, _ in levels)  # outlier should be filtered out
