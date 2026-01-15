import numpy as np
import pandas as pd

from gptcot.contracts import (
    INVALID_CONTRACT_COLUMN,
    INVALID_ENTRY_PRICE,
    INVALID_ENTRY_ZERO,
    INVALID_FORWARD_CURVE,
    resolve_contract,
    snap_prices,
)


def test_forward_curve_snap_backward():
    fc = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-05", "2024-01-08"]),
            "FrontMonth_Label": ["Feb-2024", "Mar-2024"],
        }
    )
    resolution, reason = resolve_contract(fc, pd.Timestamp("2024-01-07"))
    assert reason is None
    assert resolution.fc_snap_date == pd.Timestamp("2024-01-05")
    assert resolution.target_contract_month == "2024-02-01"


def test_snap_prices_forward_and_backward_search():
    prices = pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "2024-02-01": [np.nan, 3.0, np.nan, 3.5],
        }
    )
    entry_date, exit_date, entry_price, exit_price, reason = snap_prices(
        prices,
        "2024-02-01",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-05"),
        max_price_snap_days=2,
    )
    assert reason is None
    assert entry_date == pd.Timestamp("2024-01-03")
    assert entry_price == 3.0
    assert exit_date == pd.Timestamp("2024-01-05")
    assert exit_price == 3.5


def test_snap_prices_missing_entry_and_exit():
    prices = pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "2024-02-01": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    entry_date, exit_date, entry_price, exit_price, reason = snap_prices(
        prices,
        "2024-02-01",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-05"),
        max_price_snap_days=2,
    )
    assert reason == INVALID_ENTRY_PRICE
    assert entry_date == pd.Timestamp("2024-01-02")
    assert exit_date == pd.Timestamp("2024-01-05")
    assert entry_price is None
    assert exit_price is None


def test_entry_price_zero_flagged():
    prices = pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "2024-02-01": [0.0, 3.0],
        }
    )
    entry_date, exit_date, entry_price, exit_price, reason = snap_prices(
        prices,
        "2024-02-01",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        max_price_snap_days=1,
    )
    assert reason == INVALID_ENTRY_ZERO
    assert entry_date == pd.Timestamp("2024-01-02")
    assert entry_price == 0.0
    assert exit_price is None


def test_missing_forward_curve():
    fc = pd.DataFrame({"Date": pd.to_datetime([]), "FrontMonth_Label": []})
    resolution, reason = resolve_contract(fc, pd.Timestamp("2024-01-01"))
    assert resolution is None
    assert reason == INVALID_FORWARD_CURVE


def test_missing_contract_column():
    prices = pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "other": [1.0, 2.0],
        }
    )
    entry_date, exit_date, entry_price, exit_price, reason = snap_prices(
        prices,
        "2024-02-01",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        max_price_snap_days=1,
    )
    assert reason == INVALID_CONTRACT_COLUMN
    assert entry_date is None
    assert exit_date is None
