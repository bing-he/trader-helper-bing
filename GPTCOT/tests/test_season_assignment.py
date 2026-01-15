import pandas as pd

from gptcot.market_analysis import assign_season


def test_assign_season_maps_months_correctly():
    dates = pd.to_datetime(
        ["2024-01-10", "2024-04-15", "2024-07-20", "2024-10-05", "2024-12-01"]
    )
    seasons = [assign_season(dt) for dt in dates]
    assert seasons == ["Winter", "Spring", "Summer", "Autumn", "Winter"]
