from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.stats import gas_year_key


def fig_to_base64(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getbuffer()).decode("ascii")


def make_price_regime_chart(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["Henry"], color="black", alpha=0.7, label="Henry")
    trans = df[df["Regime"] == "Transition"]
    bind = df[df["Regime"] == "Binding"]
    ax.scatter(trans.index, trans["Henry"], color="orange", s=10, label="Transition", zorder=2)
    ax.scatter(bind.index, bind["Henry"], color="red", s=12, label="Binding", zorder=3)
    ax.set_title("Henry Hub with Physical Regimes")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.axvline(df.index[-1], color="#111111", linestyle="--", linewidth=1, alpha=0.6)
    return fig_to_base64(fig)


def make_msi_chart(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["MSI"], color="#c0c0c0", alpha=0.4, linewidth=1)
    recent = df.tail(180)
    ax.plot(recent.index, recent["MSI"], color="purple", linewidth=2)
    ax.axhline(df["MSI"].quantile(0.85), color="red", linestyle="--", linewidth=1)
    ax.set_title("Marginal Stress Index (Tail-Aware)")
    ax.grid(True, alpha=0.2)
    ax.axvline(df.index[-1], color="#111111", linestyle="--", linewidth=1, alpha=0.6)
    return fig_to_base64(fig)


def make_fragility_chart(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(df.index, df["Sub_Failure_Score"], color="#c0c0c0", alpha=0.3)
    recent = df.tail(180)
    ax.plot(recent.index, recent["Sub_Failure_Score"], color="#444444", linewidth=2)
    ax.set_title("Substitution Failure (Fragility)")
    ax.grid(True, alpha=0.2)
    ax.axvline(df.index[-1], color="#111111", linestyle="--", linewidth=1, alpha=0.6)
    return fig_to_base64(fig)


def make_vulnerability_bar_chart(vuln: pd.DataFrame, top_n: int = 5) -> str:
    top = vuln.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(top["Hub"], top["Vulnerability_Score"], color="#b22222")
    ax.invert_yaxis()
    ax.set_title("Top Vulnerable Hubs")
    ax.set_xlabel("Score")
    return fig_to_base64(fig)


def make_basis_vs_util_chart(df: pd.DataFrame, hub: str, pipe: str, days: int = 90) -> str:
    fig, ax1 = plt.subplots(figsize=(10, 3))
    current_date = df.index[-1]
    start_date = current_date - pd.DateOffset(days=days)
    subset = df.loc[start_date:]

    basis_col = f"Basis_{hub}"
    util_col = f"Util_{pipe}"

    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{hub} Basis ($)", color="tab:blue")
    ax1.fill_between(subset.index, subset[basis_col], color="tab:blue", alpha=0.3)
    ax1.plot(subset.index, subset[basis_col], color="tab:blue", linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{pipe} Utilization", color="tab:red")
    ax2.plot(subset.index, subset[util_col], color="tab:red", linestyle="--", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1.1)

    ax1.set_title(f"Evidence: {hub} Basis vs {pipe} Utilization")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_fragility_basis_scatter(df: pd.DataFrame, hub: str, points: int = 180) -> str:
    basis_col = f"Target_Delta_Basis_{hub}"
    subset = df.dropna(subset=["Sub_Failure_Score", basis_col]).tail(points)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(subset["Sub_Failure_Score"], subset[basis_col], color="#1f77b4", alpha=0.6)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_title(f"Fragility vs Next-Day {hub} Basis Change")
    ax.set_xlabel("Fragility Score")
    ax.set_ylabel("Next-Day Basis Change ($)")
    ax.grid(True, alpha=0.2)
    return fig_to_base64(fig)


def make_transition_effect_chart(df: pd.DataFrame) -> str:
    transition = df["Transition_Any"].fillna(False)
    abs_ret = df["Target_Ret_Henry"].abs()
    trans_mean = abs_ret[transition].mean()
    base_mean = abs_ret[~transition].mean()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Transition Days", "Other Days"], [trans_mean, base_mean], color=["#d62728", "#7f7f7f"])
    ax.set_title("Transition Days vs Next-Day Vol")
    ax.set_ylabel("Abs Next-Day Return")
    return fig_to_base64(fig)


def make_rolling_corr_chart(
    series_x: pd.Series,
    series_y: pd.Series,
    title: str,
    window: int = 60,
) -> str:
    aligned = pd.concat([series_x, series_y], axis=1).dropna()
    if aligned.empty:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.set_title(title)
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.axis("off")
        return fig_to_base64(fig)
    roll_corr = aligned.iloc[:, 0].rolling(window=window, min_periods=max(10, window // 3)).corr(
        aligned.iloc[:, 1]
    )
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(roll_corr.index, roll_corr, color="#1f77b4", linewidth=2)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Rolling Corr")
    ax.grid(True, alpha=0.2)
    ax.axvline(roll_corr.index[-1], color="#111111", linestyle="--", linewidth=1, alpha=0.6)
    return fig_to_base64(fig)


def make_forecast_impulse_bar(
    impulses: dict,
    title: str = "Forecast Impulse vs Last 7 Days",
) -> str:
    labels = list(impulses.keys())
    values = [impulses[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#d62728" if val > 0 else "#2ca02c" for val in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Impulse")
    ax.grid(True, axis="y", alpha=0.2)
    return fig_to_base64(fig)


def make_seasonal_overlay_chart(
    df: pd.DataFrame,
    series: pd.Series,
    title: str,
    ylabel: str,
    years_back: int,
    gas_year_start_month: int,
) -> str:
    tmp = pd.DataFrame({"value": series}).dropna().copy()
    gas_years = []
    gas_days = []
    for date in tmp.index:
        gas_year, gas_day = gas_year_key(pd.Timestamp(date), gas_year_start_month)
        gas_years.append(gas_year)
        gas_days.append(gas_day)
    tmp["Gas_Year"] = gas_years
    tmp["Gas_Day"] = gas_days

    current_year = tmp["Gas_Year"].max()
    years = list(range(current_year - years_back + 1, current_year + 1))

    fig, ax = plt.subplots(figsize=(10, 3))
    for year in years:
        year_data = tmp[tmp["Gas_Year"] == year].sort_values("Gas_Day")
        if year_data.empty:
            continue
        lw = 2.5 if year == current_year else 1.0
        alpha = 0.9 if year == current_year else 0.5
        label = str(year) + (" (current)" if year == current_year else "")
        ax.plot(year_data["Gas_Day"], year_data["value"], linewidth=lw, alpha=alpha, label=label)

    ax.set_title(title)
    ax.set_xlabel("Gas Year Day (Apr=0)")
    ax.set_ylabel(ylabel)
    current_day = tmp[tmp["Gas_Year"] == current_year]["Gas_Day"].iloc[-1]
    ax.axvline(current_day, color="#111111", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.2)
    return fig_to_base64(fig)


def build_pipe_table_html(df: pd.DataFrame) -> str:
    last_row = df.iloc[-1]
    flow_cols = [c for c in df.columns if c.startswith("Flow_")]
    pipe_data = []
    for flow_col in flow_cols:
        pipe = flow_col.replace("Flow_", "")
        util = last_row.get(f"Util_{pipe}")
        flow = last_row.get(f"Flow_{pipe}")
        cap = last_row.get(f"Cap_{pipe}")
        anomaly = bool(last_row.get(f"Util_Anomaly_{pipe}", False))
        cap_valid = cap is not None and pd.notna(cap) and cap > 0
        status = "OPEN"
        color = "green"
        if not cap_valid:
            status = "DATA GAP"
            color = "gray"
        elif anomaly:
            status = "DATA ANOMALY"
            color = "purple"
        else:
            if util is not None and pd.notna(util) and util > 0.95:
                status = "CHOKED"
                color = "red"
            elif util is not None and pd.notna(util) and util > 0.85:
                status = "TIGHT"
                color = "orange"
        pipe_data.append(
            {
                "Pipeline": pipe,
                "Flow": f"{flow:,.0f}" if flow is not None and pd.notna(flow) else "n/a",
                "Cap": f"{cap:,.0f}" if cap is not None and pd.notna(cap) else "n/a",
                "Util": f"{util * 100:.1f}%" if util is not None and pd.notna(util) else "n/a",
                "Status": status,
                "_raw_util": util if util is not None and pd.notna(util) else -1.0,
                "_color": color,
            }
        )

    pipe_data.sort(key=lambda row: row["_raw_util"], reverse=True)

    html = (
        "<table style=\"width:100%; border-collapse: collapse; font-family: monospace; font-size: 0.9em;\">"
        "<tr style=\"background: #eee; text-align: left;\">"
        "<th style=\"padding: 8px;\">Pipeline</th>"
        "<th style=\"padding: 8px;\">Flow</th>"
        "<th style=\"padding: 8px;\">Cap</th>"
        "<th style=\"padding: 8px;\">Util</th>"
        "<th style=\"padding: 8px;\">Status</th>"
        "</tr>"
    )

    for row in pipe_data:
        html += (
            "<tr style=\"border-bottom: 1px solid #ddd;\">"
            f"<td style=\"padding: 8px;\">{row['Pipeline']}</td>"
            f"<td style=\"padding: 8px;\">{row['Flow']}</td>"
            f"<td style=\"padding: 8px;\">{row['Cap']}</td>"
            f"<td style=\"padding: 8px; font-weight: bold;\">{row['Util']}</td>"
            f"<td style=\"padding: 8px; color: {row['_color']}; font-weight: bold;\">{row['Status']}</td>"
            "</tr>"
        )

    html += "</table>"
    return html
