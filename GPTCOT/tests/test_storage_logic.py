from pathlib import Path

import pandas as pd

from gptcot.market_analysis import MarketAnalysis


def test_storage_logic_respects_year_lookback(tmp_path):
    info_dir = tmp_path / "INFO"
    info_dir.mkdir(parents=True, exist_ok=True)
    dates = [
        pd.to_datetime(f"{year}-W10-4", format="%G-W%V-%u").strftime("%Y-%m-%d")
        for year in (2015, 2016, 2017)
    ]
    data = pd.DataFrame({"Period": dates, "Lower48": [100, 110, 120]})
    data.to_csv(info_dir / "EIAtotals.csv", index=False)

    analysis = MarketAnalysis(force=True, project_root=tmp_path, package_root=tmp_path, overrides={})
    storage_df = analysis._compute_storage_dataset()  # pylint: disable=protected-access

    assert storage_df is not None
    storage_df = storage_df.reset_index(drop=True)
    # For 2016 entry, only 2015 contributes to 5y average
    assert storage_df.loc[1, "avg_5y"] == 100
    # For 2017 entry, 2015 and 2016 contribute (mean = 105)
    assert storage_df.loc[2, "avg_5y"] == 105
