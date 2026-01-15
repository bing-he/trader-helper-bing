from datetime import datetime, timedelta

import pandas as pd

# --- Configuration ---
# Assume you're running this script on a given date (e.g., Monday/Tuesday/Wednesday)
# and want to see the next 5 EIA natural gas storage report dates and what week
# (Fri→Thu) each report covers.
today = datetime.now().date()

# --- Core logic ---
# EIA reports are released every Thursday at 10:30 AM ET.
# Each report represents storage change for the week ending the previous Friday→Thursday.
# For example:
#   Report Date: Thu Oct 17, 2024
#   Covers: Fri Oct 4 – Thu Oct 10, 2024
#   So, when forecasting early in the week (Mon–Wed Oct 14–16),
#   you're forecasting the change ending Thu Oct 10 (the *previous* week).


def next_thursdays(start_date, n=5):
    """Generate the next n Thursdays after a given start date."""
    days_ahead = (3 - start_date.weekday()) % 7  # 3 = Thursday
    first_thursday = start_date + timedelta(days=days_ahead)
    return [first_thursday + timedelta(weeks=i) for i in range(n)]


def eia_week_range(report_date):
    """Return the Fri→Thu week range covered by a given EIA report date."""
    end_of_week = report_date - timedelta(days=7)
    start_of_week = end_of_week - timedelta(days=6)
    # EIA aligns to Fri→Thu, so the 'week ending' is the Thursday before the report
    # For example, if report is Oct 17 → week ended Thu Oct 10
    # That week covered Fri Oct 4 – Thu Oct 10
    return (start_of_week + timedelta(days=1), end_of_week)  # Fri–Thu


# --- Compute ---
dates = []
for report_date in next_thursdays(today, 5):
    start, end = eia_week_range(report_date)
    # The "forecast window" is the Mon–Wed of the report week
    forecast_start = report_date - timedelta(days=3)
    forecast_end = report_date - timedelta(days=1)
    dates.append(
        {
            "Report Date (Thu)": report_date.strftime("%Y-%m-%d"),
            "Covers Fri–Thu": f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}",
            "Forecasting Mon–Wed": f"{forecast_start.strftime('%Y-%m-%d')} → {forecast_end.strftime('%Y-%m-%d')}",
        }
    )

# --- Display ---
df = pd.DataFrame(dates)
print(df.to_markdown(index=False))
