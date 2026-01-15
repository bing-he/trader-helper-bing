import pandas as pd

from gptcot.forward_curve import build_forward_curve_table_absolute, normalize_forward_curves


def _sample_history():
    return pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-05", "2023-12-29"]),
            "2024-02-01": [2.5, 2.4],
            "2024-03-01": [2.6, 2.5],
            "2024-04-01": [2.7, 2.6],
            "2024-05-01": [2.8, 2.7],
            "2024-06-01": [2.9, 2.8],
        }
    )


def test_build_forward_curve_table_respects_lookbacks():
    history = _sample_history()
    table = build_forward_curve_table_absolute(history, lookbacks=(0, 7), max_months=5)

    # Should select two distinct dates
    assert table.shape[1] == 2
    # Should include multiple forward months (prompt + carry)
    assert table.shape[0] >= 4
    assert table.columns.tolist() == ["2023-12-29", "2024-01-05"]


def test_normalize_forward_curves_scales_prompt_to_100():
    history = _sample_history()
    table = build_forward_curve_table_absolute(history, lookbacks=(0, 7), max_months=5)
    normalized = normalize_forward_curves(table)

    for col in normalized.columns:
        series = normalized[col]
        first_valid = series.dropna().iloc[0]
        assert first_valid == 100
        # Later months should preserve relative shape
        second_valid_idx = series.dropna().index[1]
        original = table[col].loc[second_valid_idx]
        anchor = table[col].dropna().iloc[0]
        assert series.loc[second_valid_idx] == (original / anchor) * 100
