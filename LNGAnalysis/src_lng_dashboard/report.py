"""HTML report assembly for the LNG dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.io as pio

from .config import DashboardConfig


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.0f}"
    return str(value)


def _fmt_delta(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:+,.0f}"


def _render_kpi_tiles(actual_metrics: Dict[str, Dict[str, object]]) -> str:
    total = actual_metrics.get("Total")
    if not total:
        return "<p>No KPI data available.</p>"
    deltas = total["deltas"]
    zscore = total.get("zscore_60d")
    tiles = [
        ("Latest LNG", f"{_fmt(total['latest_value'])} mmcf/d"),
        ("Δ1d / Δ5d / Δ10d", " / ".join([_fmt_delta(deltas.get(k)) for k in ("delta_1d", "delta_5d", "delta_10d")])),
        ("Z-score vs 60d", f"{zscore:+.2f}" if zscore is not None else "n/a"),
        ("Regime", total.get("regime", "Flat")),
    ]
    html = '<div class="kpi-grid">'
    for title, val in tiles:
        html += f'<div class="kpi"><div class="kpi-title">{title}</div><div class="kpi-value">{val}</div></div>'
    html += "</div>"
    return html


def _render_plot(fig, include_js: bool) -> str:
    return pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn" if include_js else False, config={"displayModeBar": False}
    )


def _render_contributions(contributions: List[Dict[str, object]]) -> str:
    if not contributions:
        return "<p>No forecast contributions available.</p>"
    rows = ""
    for row in contributions:
        change = row["expected_change"]
        change_str = _fmt_delta(change) if change is not None else "n/a"
        rows += f"<tr><td>{row['facility']}</td><td>{row['horizon_days']}d</td><td>{change_str}</td></tr>"
    return (
        "<table><thead><tr><th>Facility</th><th>Horizon</th><th>Expected Change</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _render_mapping_table(mapping: pd.DataFrame) -> str:
    if mapping.empty:
        return "<p>No facility mapping records found.</p>"
    rows = ""
    for _, row in mapping.sort_values(["canonical_name", "raw_name"]).iterrows():
        rows += f"<tr><td>{row['raw_name']}</td><td>{row['canonical_name']}</td><td>{row['source_file']}</td></tr>"
    return (
        "<table><thead><tr><th>Raw name</th><th>Canonical</th><th>Source</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _render_forecast_diagnostics(
    forecast_metrics: Dict[str, Dict[str, object]], facilities: List[str]
) -> str:
    rows = ""
    seen: List[str] = []
    ordered = ["Total"] + [f for f in facilities if f != "Total"]
    for facility in ordered:
        metrics = forecast_metrics.get(facility)
        if not metrics or facility in seen:
            continue
        seen.append(facility)
        hc = metrics["horizon_changes"]
        rows += (
            "<tr>"
            f"<td>{facility}</td>"
            f"<td>{metrics['forecast_start'].date()}</td>"
            f"<td>{_fmt(metrics['start_value'])}</td>"
            f"<td>{_fmt_delta(hc.get(7))}</td>"
            f"<td>{_fmt_delta(hc.get(14))}</td>"
            f"<td>{_fmt_delta(hc.get(21))}</td>"
            f"<td>{_fmt(metrics.get('max_next_14d'))}</td>"
            f"<td>{_fmt(metrics.get('min_next_14d'))}</td>"
            "</tr>"
        )
    if not rows:
        return "<p>No forecast diagnostics available.</p>"
    return (
        "<table><thead><tr><th>Facility</th><th>Forecast Start</th><th>Start</th>"
        "<th>Δ7d</th><th>Δ14d</th><th>Δ21d</th><th>Max next 14d</th><th>Min next 14d</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def build_html_report(
    run_ts: datetime,
    config: DashboardConfig,
    charts: Dict[str, object],
    actual_metrics: Dict[str, Dict[str, object]],
    forecast_metrics: Dict[str, Dict[str, object]],
    forecast_contributions: List[Dict[str, object]],
    selected_facilities: List[str],
    mapping: pd.DataFrame,
    warnings: List[str],
    narrative_text: str,
) -> str:
    """Assemble the final HTML report with responsive chart grid."""
    styles = """
    <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f7f7f7; color: #222; }
    header { background: #0b3954; color: #fff; padding: 16px 24px; }
    h1 { margin: 0; }
    main { padding: 16px 24px; }
    section { margin-bottom: 28px; background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    .kpi { background: #0b3954; color: #fff; padding: 12px; border-radius: 6px; }
    .kpi-title { font-size: 12px; text-transform: uppercase; opacity: 0.8; }
    .kpi-value { font-size: 20px; font-weight: bold; }
    .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
    .chart-card { background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { padding: 8px; border-bottom: 1px solid #ddd; text-align: left; }
    th { background: #f0f0f0; }
    .warnings { color: #b45309; }
    </style>
    """

    header = (
        f"<header><h1>LNG Feedgas Dashboard</h1>"
        f"<div>Run at {run_ts.strftime('%Y-%m-%d %H:%M UTC')} | Lookback years: {config.lookback_years} | Top facilities: {config.top_facilities}</div>"
        f"</header>"
    )

    warning_html = ""
    if warnings:
        warning_html = "<section><div class='warnings'><strong>Data warnings:</strong> " + "; ".join(warnings) + "</div></section>"

    body = "<main>"
    body += warning_html
    body += "<section><h2>KPI Tiles</h2>" + _render_kpi_tiles(actual_metrics) + "</section>"

    include_js = True
    total_fig = charts.get("Total")
    if total_fig is not None:
        body += "<section><h2>Total Feedgas</h2>" + _render_plot(total_fig, include_js) + "</section>"
        include_js = False

    facility_items = [(title, fig) for title, fig in charts.items() if title != "Total"]
    if facility_items:
        body += "<section><h2>Facility Feedgas</h2><div class=\"chart-grid\">"
        for title, fig in facility_items:
            body += "<div class=\"chart-card\">"
            body += f"<h3>{title}</h3>"
            body += _render_plot(fig, include_js)
            body += "</div>"
            include_js = False
        body += "</div></section>"

    body += "<section><h2>Expectations Going Forward</h2><pre>" + narrative_text + "</pre></section>"
    body += "<section><h2>Forecast Diagnostics</h2>" + _render_forecast_diagnostics(forecast_metrics, selected_facilities) + "</section>"
    body += "<section><h2>Forecast Contributors</h2>" + _render_contributions(forecast_contributions) + "</section>"
    body += "<section><h2>Appendix: Facility Mapping</h2>" + _render_mapping_table(mapping) + "</section>"
    body += "</main>"

    return "<html><head><meta charset='utf-8'>" + styles + "</head>" + header + body + "</html>"
