"""Plotly chart construction for LNG dashboard."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .features import TimeSeriesBundle
from .utils import drop_leap_days

LOGGER = logging.getLogger(__name__)

COLOR_SEQUENCE = px.colors.qualitative.D3


def _make_plot_date(series: pd.Series, current_year: int) -> pd.Series:
    return series.apply(lambda dt: datetime(current_year, dt.month, dt.day))


def _color_for_facility(facility: str) -> str:
    idx = abs(hash(facility)) % len(COLOR_SEQUENCE)
    return COLOR_SEQUENCE[idx]


def _add_prior_years(
    fig: go.Figure, df: pd.DataFrame, facility: str, current_year: int, lookback_years: int, color: str
) -> None:
    if df.empty:
        return
    df = df[df["facility"] == facility]
    df = drop_leap_days(df, "date")
    df["year"] = df["date"].dt.year
    min_year = current_year - lookback_years
    for year in sorted(df["year"].unique()):
        if year >= current_year or year < min_year:
            continue
        year_df = df[df["year"] == year].copy()
        year_df["plot_date"] = _make_plot_date(year_df["date"], current_year)
        fig.add_trace(
            go.Scatter(
                x=year_df["plot_date"],
                y=year_df["value"],
                mode="lines",
                name=f"{facility} {year}",
                line=dict(color=color, width=1.3),
                opacity=0.45,
                hovertemplate="%{x|%b %d}: %{y:.1f}",
                showlegend=True,
            )
        )


def build_series_chart(
    bundle: TimeSeriesBundle,
    facility: str,
    lookback_years: int,
    current_year: int,
    template: str,
) -> go.Figure:
    """Create a chart for a single facility or total."""
    fig = go.Figure()
    color = _color_for_facility(facility)

    actual_df = bundle.actual if facility != "Total" else bundle.total_actual
    forecast_df = bundle.forecast if facility != "Total" else bundle.total_forecast

    _add_prior_years(fig, actual_df, facility, current_year, lookback_years, color)

    # Current year actuals
    current_actual = actual_df[(actual_df["facility"] == facility) & (actual_df["date"].dt.year == current_year)]
    current_actual = drop_leap_days(current_actual, "date")
    if not current_actual.empty:
        current_actual = current_actual.copy()
        current_actual["plot_date"] = _make_plot_date(current_actual["date"], current_year)
        fig.add_trace(
            go.Scatter(
                x=current_actual["plot_date"],
                y=current_actual["value"],
                mode="lines",
                name=f"{facility} {current_year} actual",
                line=dict(color=color, width=2.4),
                hovertemplate="%{x|%b %d}: %{y:.1f}",
            )
        )

    # Forecast (stitched forward)
    if not forecast_df.empty:
        current_forecast = forecast_df[(forecast_df["facility"] == facility) & (forecast_df["date"].dt.year == current_year)]
        if not current_forecast.empty:
            current_forecast = drop_leap_days(current_forecast.copy(), "date")
            current_forecast["plot_date"] = _make_plot_date(current_forecast["date"], current_year)
            fig.add_trace(
                go.Scatter(
                    x=current_forecast["plot_date"],
                    y=current_forecast["value"],
                    mode="lines",
                    name=f"{facility} forecast",
                    line=dict(color=color, width=2.2, dash="dot"),
                    hovertemplate="%{x|%b %d}: %{y:.1f}",
                )
            )

    if bundle.forecast_start is not None:
        try:
            vline_date = datetime(current_year, bundle.forecast_start.month, bundle.forecast_start.day)
            fig.add_vline(x=vline_date, line_dash="dash", line_color="#222")
            fig.add_annotation(
                x=vline_date,
                y=1.02,
                yref="paper",
                showarrow=False,
                text="Forecast start",
                font=dict(color="#222", size=10),
                xanchor="left",
            )
        except ValueError:
            LOGGER.debug("Could not add forecast start line for %s", facility)

    fig.update_layout(
        template=template,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Day of Year",
        yaxis_title="MMcf/d",
    )
    return fig


def build_all_charts(
    bundle: TimeSeriesBundle, facilities: Iterable[str], lookback_years: int, current_year: int, template: str
) -> Dict[str, go.Figure]:
    """Build charts for total and selected facilities."""
    charts: Dict[str, go.Figure] = {}
    charts["Total"] = build_series_chart(bundle, "Total", lookback_years, current_year, template)
    for facility in facilities:
        charts[facility] = build_series_chart(bundle, facility, lookback_years, current_year, template)
    return charts
