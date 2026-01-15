from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def render_trader_note(context: Dict, ai_commentary: Optional[str]) -> str:
    header_color = "#dc3545" if context["upstream_shock_count"] > 0 else "#0056b3"
    if ai_commentary:
        header_text = (
            "UPSTREAM SUPPLY SHOCK DETECTED"
            if context["upstream_shock_count"] > 0
            else f"Morning Flow Note ({context['ai_provider']})"
        )
        return (
            "<div style=\"font-family: sans-serif; background-color: #f0f7ff; "
            "padding: 20px; border-radius: 8px; border-left: 5px solid "
            f"{header_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);\">"
            f"<h3 style=\"margin-top: 0; color: {header_color};\">{header_text}</h3>"
            f"{ai_commentary}"
            f"<div style=\"margin-top: 12px; font-size: 0.8em; color: #666;\">"
            f"Model: MSI rank {context['msi_rank']:.1%}, Fragility {context['fragility']:.1f}, "
            f"Confidence {context['confidence']:.0%}"
            "</div></div>"
        )

    if context.get("slack_tightening"):
        transition_blurb = "Slack, but tightening fast - early warning."
    else:
        transition_blurb = (
            f"Transition alert: {context['transition_label']}."
            if context["transition_label"] != "None"
            else "No regime change today."
        )
    grid_blurb = (
        "Grid is locked up." if context["congested_pipes"] != "None" else "Reroute options exist."
    )
    header_text = "UPSTREAM SUPPLY SHOCK DETECTED" if context["upstream_shock_count"] > 0 else "Morning Flow Note"

    bullets = (
        "<ul>"
        f"<li><b>{context['regime']}</b> regime with MSI at {context['msi_rank']:.1%}. "
        f"{transition_blurb} {context['msi_dir']} stress keeps bias focused.</li>"
        f"<li><b>{context['top_risk_driver']}</b> is the bottleneck at {context['driver_util']*100:.1f}% util. "
        f"{grid_blurb} Fragility {context['fragility']:.1f} puts reroutes at risk.</li>"
        f"<li><b>{context['top_risk_hub']}</b> basis {context['hub_basis']:.3f} vs 30d avg {context['hub_basis_avg']:.3f}. "
        f"Confidence {context['confidence']:.0%}; failure if utilization eases or capacity rebounds.</li>"
        "</ul>"
    )

    return (
        "<div style=\"font-family: sans-serif; background-color: #f7f7f7; "
        "padding: 20px; border-radius: 8px; border-left: 5px solid "
        f"{header_color};\">"
        f"<h3 style=\"margin-top: 0; color: {header_color};\">{header_text}</h3>"
        f"{bullets}"
        "</div>"
    )


def render_cards(cards: List[Dict[str, str]]) -> str:
    if not cards:
        return ""
    html = ""
    for card in cards:
        callout_html = ""
        if card.get("callout"):
            callout_html = f"<div class=\"callout\">{card['callout']}</div>"
        html += (
            "<div class=\"card\">"
            f"<h3>{card['title']}</h3>"
            f"{card['content']}"
            f"<p class=\"caption\">{card['caption']}</p>"
            f"{callout_html}"
            "</div>"
        )
    return html


def render_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    rows = []
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, (float, int, np.floating, np.integer)):
                cells.append(float_fmt.format(val))
            else:
                cells.append(str(val))
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return (
        "<table class=\"simple-table\">"
        f"<tr>{headers}</tr>"
        + "".join(rows)
        + "</table>"
    )


def render_changes_section(
    system_rows: List[Dict[str, str]],
    hub_rows: List[Dict[str, str]],
    note: str,
    data_quality_html: str,
) -> str:
    system_html = "".join(
        "<tr>"
        f"<td>{row['metric']}</td>"
        f"<td>{row['today']}</td>"
        f"<td>{row['prev']}</td>"
        f"<td>{row['delta']}</td>"
        "</tr>"
        for row in system_rows
    )
    hub_html = "".join(
        "<tr>"
        f"<td>{row['hub']}</td>"
        f"<td>{row['vuln']}</td>"
        f"<td>{row['conf']}</td>"
        f"<td>{row['basis']}</td>"
        f"<td>{row['zscore']}</td>"
        "</tr>"
        for row in hub_rows
    )
    return (
        "<div class=\"change-panel\">"
        "<h2>What Changed Since Yesterday</h2>"
        "<div class=\"grid\">"
        "<div class=\"card\">"
        "<h3>System</h3>"
        "<table class=\"simple-table\">"
        "<tr><th>Metric</th><th>Today</th><th>Prev</th><th>Delta</th></tr>"
        f"{system_html}"
        "</table>"
        "</div>"
        "<div class=\"card\">"
        "<h3>Top Hubs</h3>"
        "<table class=\"simple-table\">"
        "<tr><th>Hub</th><th>Vuln</th><th>Conf</th><th>Basis</th><th>30d Z</th></tr>"
        f"{hub_html}"
        "</table>"
        "</div>"
        f"{data_quality_html}"
        "</div>"
        f"<p class=\"change-note\">{note}</p>"
        "</div>"
    )


def build_html_report(
    title: str,
    generated_on: str,
    top_banner_html: str,
    changes_section_html: str,
    gulf_coast_html: str,
    trader_note_html: str,
    evidence_cards: List[Dict[str, str]],
    evidence_note: str,
    pipe_table_html: str,
    quant_cards: List[Dict[str, str]],
    seasonal_cards: List[Dict[str, str]],
    regime_stats_html: str,
    walk_forward_html: Optional[str],
) -> str:
    evidence_cards_html = render_cards(evidence_cards)
    quant_cards_html = render_cards(quant_cards)
    seasonal_cards_html = render_cards(seasonal_cards)

    walk_forward_section = ""
    if walk_forward_html:
        walk_forward_section = (
            "<h3>Walk-Forward Diagnostics</h3>" + walk_forward_html
        )

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: "Segoe UI", Tahoma, sans-serif; margin: 0; padding: 20px; background: #fff; color: #222; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #222; }}
        h2 {{ color: #444; border-bottom: 2px solid #eee; padding-bottom: 8px; margin-top: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
        .card {{ background: #fafafa; padding: 16px; border-radius: 8px; border: 1px solid #e0e0e0; }}
        .caption {{ font-size: 0.85em; color: #555; margin-top: 8px; }}
        .simple-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        .simple-table th, .simple-table td {{ border-bottom: 1px solid #ddd; padding: 6px; text-align: center; }}
        .evidence-note {{ font-size: 0.9em; color: #666; margin-top: 8px; }}
        .change-panel {{ background: #f7fbff; border: 1px solid #d6e6f5; padding: 16px; border-radius: 8px; }}
        .change-note {{ font-size: 0.9em; color: #444; margin-top: 10px; }}
        .delta-up {{ color: #b00020; font-weight: 600; }}
        .delta-down {{ color: #1b7f3a; font-weight: 600; }}
        .delta-flat {{ color: #666; font-weight: 600; }}
        .callout {{ background: #fff8e1; border-left: 4px solid #f0b429; padding: 8px 10px; margin-top: 10px; font-size: 0.85em; }}
        .banner {{ background: #111827; color: #f9fafb; padding: 10px 14px; border-radius: 6px; font-size: 0.95em; }}
        .footer {{ margin-top: 40px; font-size: 0.8em; color: #777; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>Generated on: {generated_on}</p>
        {top_banner_html}
        <hr>

        {changes_section_html}

        {gulf_coast_html}

        <h2>1. Desk View (Trader Note)</h2>
        {trader_note_html}

        <h2>2. Evidence Locker (Verify the Call)</h2>
        <div class="grid">
            {evidence_cards_html}
        </div>
        <p class="evidence-note">{evidence_note}</p>
        <h3>Grid Lock Status: Can we reroute?</h3>
        {pipe_table_html}

        <h2>3. Quant Dashboard (The Math)</h2>
        <div class="grid">
            {quant_cards_html}
        </div>
        <h3>Regime Performance Snapshot</h3>
        {regime_stats_html}
        {walk_forward_section}

        <h2>4. Seasonal Overlays (Gas Year)</h2>
        <div class="grid">
            {seasonal_cards_html}
        </div>

        <div class="footer">Proprietary Model Output | TraderHelper</div>
    </div>
</body>
</html>
"""
