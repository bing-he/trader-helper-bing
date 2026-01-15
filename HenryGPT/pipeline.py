from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .ai.analyst import AIAnalyst
from .backtest import BacktestEngine
from .config import Config
from .data import DataManager
from .exogenous import ExogenousDataLoader
from .features import FeatureEngineer
from .reporting import charts
from .reporting.html import (
    build_html_report,
    render_changes_section,
    render_table,
    render_trader_note,
)
from .signals import SignalEngine
from .utils.logging import get_logger, setup_logging
from .utils.stats import gas_year_key, safe_corr, segment_correlation_stability


@dataclass
class PipelineResult:
    html_path: str
    last_date: pd.Timestamp


def _image_tag(base64_str: str) -> str:
    return f"<img src=\"data:image/png;base64,{base64_str}\" style=\"width:100%;\" />"


def _format_delta(
    delta: Optional[float],
    bad_when_positive: bool = True,
    fmt: str = "{:+.2f}",
    pct: bool = False,
) -> str:
    if delta is None or (isinstance(delta, (float, np.floating)) and np.isnan(delta)):
        return "n/a"
    if pct:
        delta = delta * 100
        fmt = "{:+.1f}%"
    if delta > 0:
        arrow = "&uarr;"
        bad = bad_when_positive
    elif delta < 0:
        arrow = "&darr;"
        bad = not bad_when_positive
    else:
        arrow = "&rarr;"
        bad = False
    cls = "delta-up" if bad else ("delta-down" if delta != 0 else "delta-flat")
    return f"<span class=\"{cls}\">{arrow} {fmt.format(delta)}</span>"


def _format_value(value: Optional[float], fmt: str = "{:.2f}", pct: bool = False) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "n/a"
    if pct:
        return f"{value * 100:.1f}%"
    return fmt.format(value)


def _row_value(row: pd.Series, col: str) -> Optional[float]:
    val = row.get(col)
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _calc_delta(today: Optional[float], prev: Optional[float]) -> Optional[float]:
    if today is None or prev is None:
        return None
    if not np.isfinite(today) or not np.isfinite(prev):
        return None
    return float(today - prev)


def _build_metric_table(rows: List[Dict[str, str]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{row['metric']}</td>"
        f"<td>{row['today']}</td>"
        f"<td>{row['prev']}</td>"
        f"<td>{row['delta']}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table class=\"simple-table\">"
        "<tr><th>Metric</th><th>Today</th><th>Prev</th><th>Delta</th></tr>"
        f"{body}"
        "</table>"
    )


def _evaluate_exog_evidence(
    series_x: pd.Series, series_y: pd.Series, cfg: Config
) -> Optional[Dict[str, float]]:
    aligned = pd.concat([series_x, series_y], axis=1).dropna()
    if aligned.shape[0] < cfg.evidence_min_sample:
        return None
    corr_val = safe_corr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    if not np.isfinite(corr_val) or abs(corr_val) < cfg.evidence_min_abs_corr:
        return None
    stability = segment_correlation_stability(
        aligned.iloc[:, 0],
        aligned.iloc[:, 1],
        segments=cfg.corr_stability_segments,
        min_abs_corr=cfg.evidence_min_abs_corr,
    )
    if stability < cfg.evidence_min_stability:
        return None

    roll = aligned.iloc[:, 0].rolling(window=30, min_periods=15)
    zscore = (aligned.iloc[:, 0] - roll.mean()) / roll.std()
    tight_mask = zscore > 1.0
    tight_sample = aligned.iloc[:, 1][tight_mask].dropna()
    base_sample = aligned.iloc[:, 1][~tight_mask].dropna()
    if tight_sample.shape[0] < cfg.evidence_min_tight_samples:
        return None
    if base_sample.shape[0] < max(cfg.evidence_min_sample // 2, 1):
        return None
    lift = float(tight_sample.mean() - base_sample.mean())
    if not np.isfinite(lift) or abs(lift) < cfg.evidence_min_lift:
        return None

    return {
        "corr": float(corr_val),
        "stability": float(stability),
        "sample": float(aligned.shape[0]),
        "tight_sample": float(tight_sample.shape[0]),
        "lift": float(lift),
    }


def _rolling_zscore(series: pd.Series, window: int = 30) -> Optional[float]:
    if series.empty:
        return None
    mean = series.rolling(window).mean().iloc[-1]
    std = series.rolling(window).std().iloc[-1]
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return None
    return (series.iloc[-1] - mean) / std


def _seasonal_zscore(series: pd.Series, gas_start_month: int, years_back: int) -> Optional[float]:
    if series.empty:
        return None
    current_date = series.index[-1]
    current_year, current_day = gas_year_key(pd.Timestamp(current_date), gas_start_month)
    values = []
    for date, val in series.dropna().items():
        year, day = gas_year_key(pd.Timestamp(date), gas_start_month)
        if year == current_year:
            continue
        if year < current_year - years_back + 1:
            continue
        if day == current_day:
            values.append(float(val))
    if len(values) < 3:
        return None
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    if std_val == 0:
        return None
    return (float(series.iloc[-1]) - mean_val) / std_val


def _seasonal_position_text(zscore: Optional[float]) -> str:
    if zscore is None or (isinstance(zscore, (float, np.floating)) and np.isnan(zscore)):
        return "Seasonal position: n/a."
    if zscore >= 1.0:
        return "Seasonal position: high vs history."
    if zscore <= -1.0:
        return "Seasonal position: low vs history."
    return "Seasonal position: near the seasonal norm."


def _format_callout(callout: Dict[str, str]) -> str:
    lines = [
        f"Importance: {callout.get('importance', 'n/a')}",
        f"Why: {callout.get('why', 'n/a')}",
        f"Important today?: {callout.get('important_today', 'n/a')}",
        f"Action: {callout.get('action', 'n/a')}",
    ]
    return "<b>Trader Take</b><br>" + "<br>".join(lines)


def _select_callout(
    panel_id: str,
    llm_callouts: Dict[str, Dict[str, str]],
    fallback_text: str,
) -> str:
    callout = llm_callouts.get(panel_id)
    if isinstance(callout, dict):
        return _format_callout(callout)
    lines = [line.strip() for line in fallback_text.splitlines() if line.strip()]
    return "<b>Trader Take</b><br>" + "<br>".join(lines[:4])


def _fallback_trader_take(panel: str, ctx: Dict, cfg: Config) -> str:
    transition_any = bool(ctx.get("transition_any"))
    msi_rank = ctx.get("msi_rank", 0)
    msi_rank_delta = ctx.get("msi_rank_delta", 0)
    fragility_rank = ctx.get("fragility_rank", 0)
    fragility_delta = ctx.get("fragility_delta", 0)
    binding_delta = ctx.get("binding_delta", 0)
    confidence = ctx.get("confidence", 0)
    vuln_delta = ctx.get("vuln_delta", 0)
    oos_corr = ctx.get("oos_corr")
    oos_hit = ctx.get("oos_hit")
    regime = ctx.get("regime")
    pct_transition = ctx.get("pct_transition", 0.85)

    importance = "Medium"
    important_today = False
    why = "Monitor risk conditions."
    flip = "if stress or transitions accelerate"
    action = "watch the tape"

    high_trigger = (
        transition_any
        or (msi_rank_delta >= cfg.high_delta_msi_pct)
        or (fragility_delta >= cfg.high_delta_fragility)
        or (binding_delta >= cfg.high_delta_binding)
        or ctx.get("upstream_shock", False)
    )
    slack_tightening = regime == "Slack" and high_trigger

    if panel == "Henry Price + Regimes":
        if transition_any or regime in ("Transition", "Binding") or slack_tightening:
            importance = "High"
            important_today = True
            why = "Regime shifts set directional bias and volatility."
            action = "size with the regime"
            flip = "if regime flips back to Slack"
            if slack_tightening:
                why = "Slack, but tightening fast - early warning."
                flip = "if stress reverses"
        elif msi_rank < 0.30 and not high_trigger:
            importance = "Low"
            why = "Slack regime reduces squeeze risk."
            action = "keep size light"
            flip = "if MSI percentile lifts above 60%"
    elif panel == "Marginal Stress Index":
        if msi_rank >= pct_transition or high_trigger or slack_tightening:
            importance = "High"
            important_today = True
            why = "MSI flags binding risk across the grid."
            action = "tighten risk limits"
            flip = "if MSI percentile falls back below 60%"
        elif msi_rank < 0.30 and not high_trigger:
            importance = "Low"
            why = "Stress is muted across pipes."
            action = "avoid overreacting to noise"
            flip = "if MSI percentile jumps"
    elif panel == "Substitution Fragility":
        if fragility_rank >= 0.85 or high_trigger:
            importance = "High"
            important_today = True
            why = "Fragility signals limited reroute capacity."
            action = "protect against basis blowouts"
            flip = "if fragility cools materially"
        elif fragility_rank < 0.40 and not high_trigger:
            importance = "Low"
            why = "Reroute capacity remains healthy."
            action = "stay flexible"
            flip = "if fragility spikes"
    elif panel == "Hub Vulnerability Ranking":
        if confidence >= 0.6 and (vuln_delta > 0 or high_trigger):
            importance = "High"
            important_today = True
            why = "Drivers are stable and tightening for key hubs."
            action = f"watch {ctx.get('top_hubs', 'top hubs')}"
            flip = "if confidence fades"
        elif confidence < 0.3 and not high_trigger:
            importance = "Low"
            why = "Signal is noisy today."
            action = "treat as watchlist only"
            flip = "if correlations stabilize"
    elif panel == "Walk-Forward Diagnostics":
        if oos_corr is None:
            importance = "Low"
            why = "Not enough data for OOS validation."
            action = "use as context only"
            flip = "if samples grow"
        elif oos_corr < 0.1 or (oos_hit is not None and oos_hit < 0.5):
            importance = "Low"
            why = "OOS stats are weak; situational awareness only."
            action = "do not anchor on model"
            flip = "if OOS stats improve"
        else:
            importance = "Medium"
            important_today = True
            why = "OOS diagnostics support regime context."
            action = "respect regime-based vol bias"
            flip = "if OOS stats deteriorate"
    else:
        if transition_any or msi_rank >= pct_transition:
            importance = "Medium"
            important_today = True
            why = "Seasonal positioning adds context to stress."
            action = "compare to prior years"
            flip = "if current track reverts to median"
        elif msi_rank < 0.30:
            importance = "Low"
            why = "Seasonal read is calm versus history."
            action = "avoid overfitting seasonals"
            flip = "if stress builds"

    if slack_tightening and importance == "High":
        why = "Slack, but tightening fast - early warning."

    today_line = "Yes" if important_today else "No"
    return "\n".join(
        [
            f"Importance: {importance}",
            f"Why: {why}",
            f"Important today?: {today_line} - {flip}",
            f"Action: {action}",
        ]
    )


def _fallback_gulf_pulse_take(
    lng_delta: Optional[float],
    load_delta: Optional[float],
    lmp_z: Optional[float],
    lmp_spike: bool,
    fundy_movers: List[Dict[str, Optional[float]]],
    pressure_regime: str,
    pressure_confidence: float,
) -> str:
    parts = []
    if lng_delta is not None:
        direction = "higher" if lng_delta > 0 else ("lower" if lng_delta < 0 else "flat")
        parts.append(f"LNG Gulf feedgas is {direction} on the day.")
    if load_delta is not None:
        direction = "up" if load_delta > 0 else ("down" if load_delta < 0 else "flat")
        parts.append(f"ERCOT load is {direction}, keeping power-driven demand in focus.")
    if lmp_z is not None:
        stress_line = f"Power stress z-score {lmp_z:+.2f}"
        if lmp_spike:
            stress_line += " with spike risk flagged"
        parts.append(stress_line + ".")
    if fundy_movers:
        top = fundy_movers[0]
        parts.append(f"Fundy mover: {top['item']} leads the SouthCentral shift.")
    if pressure_regime != "Neutral":
        parts.append(
            f"Gulf Coast pulse reads {pressure_regime} (confidence {pressure_confidence:.0%})."
        )
    return " ".join(parts) if parts else "Insufficient Gulf Coast inputs for a clean read."


def _fallback_outlook_take(
    pressure_regime: str,
    outlook_confidence: float,
    reliability_label: str,
) -> str:
    if pressure_regime == "Tightening Risk":
        base = "Near-term pressure risk tilts higher; treat as risk, not price."
    elif pressure_regime == "Loose":
        base = "Near-term pressure risk looks softer; stay alert for reversals."
    else:
        base = "Near-term pressure risk is balanced; wait for confirmation."
    if reliability_label == "LOW RELIABILITY":
        base += " LOW RELIABILITY due to weak walk-forward support."
    base += f" Confidence {outlook_confidence:.0%}."
    return base


def run_pipeline(config: Optional[Config] = None) -> PipelineResult:
    setup_logging()
    logger = get_logger(__name__)
    cfg = config or Config()

    logger.info("--- Starting Henry Hub Pipeline ---")
    dm = DataManager(cfg)
    dm.load_data()
    bundle = dm.process_vectors()

    exog_bundle = ExogenousDataLoader(cfg.info_dir).load()
    fe = FeatureEngineer(bundle.data, bundle.active_pipes, cfg, exogenous=exog_bundle)
    df = fe.run()
    exog_meta = fe.feature_meta

    sig = SignalEngine(df, bundle.active_pipes, cfg)
    df = sig.add_regime_transitions()
    pressure_summary = sig.add_gulf_pressure_signals()
    df = sig.df
    vuln = sig.compute_vulnerability_scores()

    bt = BacktestEngine(df, cfg)
    regime_stats = bt.evaluate_regimes()
    walk_forward = bt.walk_forward_vol()

    ai = AIAnalyst(cfg)

    last_row = df.iloc[-1]
    prev_idx = df.index[-2] if len(df) > 1 else df.index[-1]
    prev_row = df.loc[prev_idx]
    prev_label = str(prev_idx.date())
    if (last_row.name - prev_idx).days > 1:
        prev_label = f"{prev_label} (prev avail)"

    msi_delta = last_row["MSI"] - prev_row["MSI"]
    msi_rank_delta = last_row["MSI_Pct_Rank"] - prev_row["MSI_Pct_Rank"]
    fragility_delta = last_row["Sub_Failure_Score"] - prev_row["Sub_Failure_Score"]
    regime = last_row["Regime"]
    prev_regime = prev_row.get("Regime", "n/a")
    transition_label = last_row.get("Transition_Label", "None")
    pressure_regime = last_row.get("gulf_tightening_pressure_regime", "Neutral")
    pressure_confidence = float(last_row.get("gulf_tightening_pressure_confidence", 0.0) or 0.0)
    pressure_score = last_row.get("gulf_tightening_pressure_score")

    top_risk = vuln.iloc[0] if not vuln.empty else None
    driver_pipe = top_risk["Primary_Driver_Pipe"] if top_risk is not None else "None"
    risk_hub = top_risk["Hub"] if top_risk is not None else "None"
    confidence = float(top_risk["Confidence"]) if top_risk is not None else 0.0

    driver_util = float(last_row.get(f"Util_{driver_pipe}", 0))
    driver_flow = float(last_row.get(f"Flow_{driver_pipe}", 0))
    driver_cap = float(last_row.get(f"Cap_{driver_pipe}", 0))

    hub_basis = float(last_row.get(f"Basis_{risk_hub}", 0))
    hub_basis_avg = (
        df.get(f"Basis_{risk_hub}", pd.Series(dtype=float)).rolling(30).mean().iloc[-1]
        if risk_hub != "None"
        else 0.0
    )

    util_cols = [
        c
        for c in df.columns
        if c.startswith("Util_") and "Clipped" not in c and "Anomaly" not in c
    ]
    congested_list = []
    for col in util_cols:
        pipe_name = col.replace("Util_", "")
        if pipe_name != driver_pipe and last_row[col] > 0.90:
            congested_list.append(f"{pipe_name} ({last_row[col]*100:.0f}%)")
    congested_str = ", ".join(congested_list) if congested_list else "None"

    binding_count = 0
    binding_prev = 0
    for pipe in bundle.active_pipes:
        util_val = last_row.get(f"Util_{pipe}")
        anomaly_val = last_row.get(f"Util_Anomaly_{pipe}", False)
        if util_val is not None and pd.notna(util_val) and util_val > 0.90 and not anomaly_val:
            binding_count += 1
        util_prev = prev_row.get(f"Util_{pipe}")
        anomaly_prev = prev_row.get(f"Util_Anomaly_{pipe}", False)
        if util_prev is not None and pd.notna(util_prev) and util_prev > 0.90 and not anomaly_prev:
            binding_prev += 1
    binding_delta = binding_count - binding_prev

    def _select_bottleneck(row: pd.Series) -> Dict[str, float]:
        best_pipe = "None"
        best_score = -np.inf
        best_util = 0.0
        for pipe in bundle.active_pipes:
            util_val = row.get(f"Util_{pipe}")
            cap_val = row.get(f"Cap_{pipe}")
            if (
                pd.isna(util_val)
                or pd.isna(cap_val)
                or cap_val <= 0
                or util_val > cfg.util_anomaly_threshold
            ):
                continue
            score = util_val * cap_val if pd.notna(cap_val) and cap_val > 0 else util_val
            if score > best_score:
                best_score = score
                best_pipe = pipe
                best_util = float(util_val)
        return {"pipe": best_pipe, "util": best_util}

    bottleneck_today = _select_bottleneck(last_row)
    bottleneck_prev_util = float(prev_row.get(f"Util_{bottleneck_today['pipe']}", np.nan))

    prev_df = df.loc[:prev_idx].copy()
    vuln_prev = SignalEngine(prev_df, bundle.active_pipes, cfg).compute_vulnerability_scores()
    vuln_prev_map = {row["Hub"]: row for _, row in vuln_prev.iterrows()}

    high_trigger = (
        bool(last_row.get("Transition_Any", False))
        or (msi_rank_delta >= cfg.high_delta_msi_pct)
        or (fragility_delta >= cfg.high_delta_fragility)
        or (binding_delta >= cfg.high_delta_binding)
        or bool(last_row.get("Upstream_Shock_Count", 0) > 0)
    )
    slack_tightening = regime == "Slack" and high_trigger

    context = {
        "date": last_row.name.date(),
        "regime": regime,
        "transition_label": transition_label,
        "transition_any": bool(last_row.get("Transition_Any", False)),
        "msi_rank": last_row["MSI_Pct_Rank"],
        "msi_rank_delta": msi_rank_delta,
        "msi_delta": msi_delta,
        "msi_dir": "RISING" if msi_delta > 0 else "FALLING",
        "fragility": last_row["Sub_Failure_Score"],
        "fragility_delta": fragility_delta,
        "vol_expect": regime_stats.loc[regime, "Ann_Vol"] * 100
        if regime in regime_stats.index
        else 0.0,
        "win_rate": regime_stats.loc[regime, "Win_Rate"] * 100
        if regime in regime_stats.index
        else 0.0,
        "top_risk_hub": risk_hub,
        "top_risk_driver": driver_pipe,
        "driver_flow": driver_flow,
        "driver_cap": driver_cap,
        "driver_util": driver_util,
        "hub_basis": hub_basis,
        "hub_basis_avg": hub_basis_avg,
        "congested_pipes": congested_str,
        "confidence": confidence,
        "upstream_shock_count": last_row.get("Upstream_Shock_Count", 0),
        "upstream_shock_pipe": last_row.get("Upstream_Shock_Pipe", "None"),
        "ai_provider": ai.provider.title() if ai.provider else "AI",
        "slack_tightening": slack_tightening,
        "pressure_regime": pressure_regime,
        "pressure_confidence": pressure_confidence,
        "pressure_score": pressure_score,
    }

    fragility_rank = float(df["Sub_Failure_Score"].rank(pct=True).iloc[-1])
    fragility_rank_prev = float(prev_df["Sub_Failure_Score"].rank(pct=True).iloc[-1])

    top_hubs_list = [row["Hub"] for _, row in vuln.head(3).iterrows()]
    top_hub = top_hubs_list[0] if top_hubs_list else "None"
    top_prev = vuln_prev_map.get(top_hub)
    top_vuln_delta = (
        float(vuln.iloc[0]["Vulnerability_Score"]) - float(top_prev["Vulnerability_Score"])
        if top_prev is not None and not vuln.empty
        else 0.0
    )

    anomaly_cols = [c for c in df.columns if c.startswith("Util_Anomaly_")]
    anomaly_pipe_days = int(df[anomaly_cols].sum().sum()) if anomaly_cols else 0
    anomaly_days = int(df[anomaly_cols].any(axis=1).sum()) if anomaly_cols else 0

    system_rows = [
        {
            "metric": "Date",
            "today": str(last_row.name.date()),
            "prev": prev_label,
            "delta": "",
        },
        {
            "metric": "Regime",
            "today": regime,
            "prev": prev_regime,
            "delta": transition_label,
        },
        {
            "metric": "MSI",
            "today": _format_value(float(last_row["MSI"])),
            "prev": _format_value(float(prev_row["MSI"])),
            "delta": _format_delta(float(msi_delta), bad_when_positive=True),
        },
        {
            "metric": "MSI Percentile",
            "today": _format_value(float(last_row["MSI_Pct_Rank"]), pct=True),
            "prev": _format_value(float(prev_row["MSI_Pct_Rank"]), pct=True),
            "delta": _format_delta(float(msi_rank_delta), bad_when_positive=True, pct=True),
        },
        {
            "metric": "Fragility",
            "today": _format_value(float(last_row["Sub_Failure_Score"])),
            "prev": _format_value(float(prev_row["Sub_Failure_Score"])),
            "delta": _format_delta(float(fragility_delta), bad_when_positive=True),
        },
        {
            "metric": "Binding Count (>90%)",
            "today": str(binding_count),
            "prev": str(binding_prev),
            "delta": _format_delta(float(binding_delta), bad_when_positive=True, fmt="{:+.0f}"),
        },
        {
            "metric": "Top Bottleneck",
            "today": f"{bottleneck_today['pipe']} ({_format_value(bottleneck_today['util'], pct=True)})",
            "prev": _format_value(bottleneck_prev_util, pct=True),
            "delta": _format_delta(
                bottleneck_today["util"] - bottleneck_prev_util,
                bad_when_positive=True,
                pct=True,
            ),
        },
        {
            "metric": "Upstream Shock",
            "today": (
                f"Yes ({last_row.get('Upstream_Shock_Pipe', 'None')})"
                if last_row.get("Upstream_Shock_Count", 0) > 0
                else "No"
            ),
            "prev": (
                f"Yes ({prev_row.get('Upstream_Shock_Pipe', 'None')})"
                if prev_row.get("Upstream_Shock_Count", 0) > 0
                else "No"
            ),
            "delta": "",
        },
    ]

    missing_pipes_text = ", ".join(
        f"{pipe} ({count})" for pipe, count in bundle.quality.top_missing_pipes
    )
    data_quality_html = (
        "<div class=\"card\">"
        "<h3>Data Quality</h3>"
        "<ul>"
        f"<li>Excluded pipe-days (cap missing/invalid): {bundle.quality.excluded_pipe_days}</li>"
        f"<li>Top missing caps: {missing_pipes_text if missing_pipes_text else 'n/a'}</li>"
        f"<li>Util > {cfg.util_anomaly_threshold:.1f} days: {anomaly_days} (pipe-days {anomaly_pipe_days})</li>"
        "</ul>"
        "<p class=\"caption\">Pipes with DATA GAP are excluded from MSI and binding count.</p>"
        "</div>"
    )

    hub_rows = []
    llm_hubs = []
    for _, row in vuln.head(3).iterrows():
        hub = row["Hub"]
        prev_row_hub = vuln_prev_map.get(hub)
        prev_score = float(prev_row_hub["Vulnerability_Score"]) if prev_row_hub is not None else None
        prev_conf = float(prev_row_hub["Confidence"]) if prev_row_hub is not None else None

        basis_col = f"Basis_{hub}"
        basis_today = float(last_row.get(basis_col, np.nan))
        basis_prev = float(prev_row.get(basis_col, np.nan))
        basis_delta = basis_today - basis_prev if np.isfinite(basis_today) and np.isfinite(basis_prev) else None
        basis_z = _rolling_zscore(df.get(basis_col, pd.Series(dtype=float)))

        vuln_cell = f"{row['Vulnerability_Score']:.2f} {_format_delta(row['Vulnerability_Score'] - prev_score if prev_score is not None else None)}"
        conf_cell = (
            f"{row['Confidence']:.0%} "
            f"{_format_delta(row['Confidence'] - prev_conf if prev_conf is not None else None, bad_when_positive=False, pct=True)}"
        )
        basis_cell = f"{basis_today:.3f} {_format_delta(basis_delta)}" if np.isfinite(basis_today) else "n/a"
        z_cell = f"{basis_z:.2f}" if basis_z is not None else "n/a"

        hub_rows.append(
            {
                "hub": hub,
                "vuln": vuln_cell,
                "conf": conf_cell,
                "basis": basis_cell,
                "zscore": z_cell,
            }
        )
        llm_hubs.append(
            {
                "hub": hub,
                "vulnerability": float(row["Vulnerability_Score"]),
                "vulnerability_delta": float(row["Vulnerability_Score"] - prev_score) if prev_score is not None else None,
                "confidence": float(row["Confidence"]),
                "confidence_delta": float(row["Confidence"] - prev_conf) if prev_conf is not None else None,
                "basis": basis_today if np.isfinite(basis_today) else None,
                "basis_delta": basis_delta,
                "basis_z": basis_z,
            }
        )

    tighten_score = sum(
        [
            float(msi_delta) > 0,
            float(binding_delta) > 0,
            float(fragility_delta) > 0,
        ]
    )
    system_trend = "tightened" if tighten_score >= 2 else ("loosened" if tighten_score == 0 else "mixed")
    avg_vuln_delta = (
        np.mean(
            [
                (float(row["Vulnerability_Score"]) - float(vuln_prev_map.get(row["Hub"], {}).get("Vulnerability_Score", row["Vulnerability_Score"])))
                for _, row in vuln.head(3).iterrows()
            ]
        )
        if not vuln.empty
        else 0.0
    )
    risk_trend = "rising" if avg_vuln_delta > 0 else ("falling" if avg_vuln_delta < 0 else "flat")
    watch_hubs = ", ".join(top_hubs_list) if top_hubs_list else "no hubs"
    change_note = f"Net-net: system {system_trend}; risk {risk_trend}; watch {watch_hubs}."

    changes_section_html = render_changes_section(system_rows, hub_rows, change_note, data_quality_html)

    if last_row.get("Upstream_Shock_Count", 0) > 0:
        top_banner_text = f"Upstream shock - treat capacity stale; watch {watch_hubs}."
    elif slack_tightening:
        top_banner_text = f"Slack but tightening fast - watch {watch_hubs}."
    elif regime == "Binding":
        top_banner_text = f"Binding regime - protect basis exposure; watch {watch_hubs}."
    elif regime == "Transition":
        top_banner_text = f"Transition regime - volatility rising; watch {watch_hubs}."
    else:
        top_banner_text = f"Slack regime - monitor for shifts; watch {watch_hubs}."

    if (
        pressure_confidence >= cfg.gulf_pressure_confidence_threshold
        and pressure_regime != "Neutral"
    ):
        top_banner_text = f"{top_banner_text} Gulf Coast pulse: {pressure_regime}."

    top_banner_html = f"<div class=\"banner\">Today's setup: {top_banner_text}</div>"

    oos_corr = None
    oos_hit = None
    oos_samples = None
    if walk_forward is not None and not walk_forward.empty:
        oos_corr = float(walk_forward["OOS_Corr"].iloc[0])
        oos_hit = float(walk_forward["OOS_Hit_Rate"].iloc[0])
        oos_samples = int(walk_forward["Samples"].iloc[0])

    exog_coverage = exog_meta.get("exog_coverage", {}) if isinstance(exog_meta, dict) else {}
    total_days = len(df.index)

    lng_today = _row_value(last_row, "lng_gulf_total")
    lng_prev = _row_value(prev_row, "lng_gulf_total")
    lng_delta = _calc_delta(lng_today, lng_prev)
    lng_impulse = _row_value(last_row, "lng_gulf_forecast_7d_avg_minus_last7d")

    load_today = _row_value(last_row, "ercot_load_total")
    load_prev = _row_value(prev_row, "ercot_load_total")
    load_delta = _calc_delta(load_today, load_prev)
    load_impulse = _row_value(last_row, "ercot_load_forecast_7d_avg_minus_last7d")

    lmp_today = _row_value(last_row, "ercot_max_lmp")
    lmp_prev = _row_value(prev_row, "ercot_max_lmp")
    lmp_delta = _calc_delta(lmp_today, lmp_prev)
    lmp_z = _row_value(last_row, "ercot_lmp_7d_z")
    lmp_spike = bool(last_row.get("ercot_lmp_spike_flag", False))

    fundy_items = exog_meta.get("fundy_items", []) if isinstance(exog_meta, dict) else []
    fundy_movers = []
    for item in fundy_items:
        col = f"SouthCentral__{item}"
        today_val = _row_value(last_row, col)
        prev_val = _row_value(prev_row, col)
        delta_val = _calc_delta(today_val, prev_val)
        impulse_val = _row_value(last_row, f"{col}_forecast_7d_avg_minus_last7d")
        move_score = None
        if impulse_val is not None:
            move_score = abs(impulse_val)
        elif delta_val is not None:
            move_score = abs(delta_val)
        if move_score is None:
            continue
        fundy_movers.append(
            {
                "item": item,
                "delta": delta_val,
                "impulse": impulse_val,
                "score": move_score,
            }
        )
    fundy_movers = sorted(fundy_movers, key=lambda x: x["score"], reverse=True)[:3]
    fundy_proxy = _row_value(last_row, "southcentral_tightness_proxy")

    exog_evidence_specs = []
    lng_metrics = None
    if "lng_gulf_total" in df.columns and "Ret_Henry" in df.columns:
        lng_metrics = _evaluate_exog_evidence(df["lng_gulf_total"], df["Ret_Henry"], cfg)
        if lng_metrics:
            exog_evidence_specs.append(
                {
                    "panel_id": "evidence_lng_corr",
                    "title": "Henry Returns vs LNG Gulf Total",
                    "series_x": df["lng_gulf_total"],
                    "series_y": df["Ret_Henry"],
                    "chart_title": "Henry Returns vs LNG Gulf Total (Rolling Corr)",
                    "caption": (
                        f"Corr {lng_metrics['corr']:.2f}, stability {lng_metrics['stability']:.2f}, "
                        f"lift {lng_metrics['lift']:.3f}, tight n={int(lng_metrics['tight_sample'])}"
                    ),
                    "score": abs(lng_metrics["corr"]) * lng_metrics["stability"],
                    "context": lng_metrics,
                }
            )

    load_metrics = None
    if "ercot_load_total" in df.columns and "Ret_Henry" in df.columns:
        load_metrics = _evaluate_exog_evidence(df["ercot_load_total"], df["Ret_Henry"], cfg)
    if load_metrics:
        exog_evidence_specs.append(
            {
                "panel_id": "evidence_ercot_load_corr",
                "title": "Henry Returns vs ERCOT Load",
                "series_x": df["ercot_load_total"],
                "series_y": df["Ret_Henry"],
                "chart_title": "Henry Returns vs ERCOT Load (Rolling Corr)",
                "caption": (
                    f"Corr {load_metrics['corr']:.2f}, stability {load_metrics['stability']:.2f}, "
                    f"lift {load_metrics['lift']:.3f}, tight n={int(load_metrics['tight_sample'])}"
                ),
                "score": abs(load_metrics["corr"]) * load_metrics["stability"],
                "context": load_metrics,
            }
        )
    elif "ercot_lmp_7d_z" in df.columns and "Ret_Henry" in df.columns:
        lmp_metrics = _evaluate_exog_evidence(df["ercot_lmp_7d_z"], df["Ret_Henry"], cfg)
        if lmp_metrics:
            exog_evidence_specs.append(
                {
                    "panel_id": "evidence_ercot_lmp_corr",
                    "title": "Henry Returns vs ERCOT LMP Stress",
                    "series_x": df["ercot_lmp_7d_z"],
                    "series_y": df["Ret_Henry"],
                    "chart_title": "Henry Returns vs ERCOT LMP Stress (Rolling Corr)",
                    "caption": (
                        f"Corr {lmp_metrics['corr']:.2f}, stability {lmp_metrics['stability']:.2f}, "
                        f"lift {lmp_metrics['lift']:.3f}, tight n={int(lmp_metrics['tight_sample'])}"
                    ),
                    "score": abs(lmp_metrics["corr"]) * lmp_metrics["stability"],
                    "context": lmp_metrics,
                }
            )

    impulse_ready = (
        lng_impulse is not None
        and load_impulse is not None
        and np.isfinite(lng_impulse)
        and np.isfinite(load_impulse)
        and (
            abs(lng_impulse) >= cfg.evidence_min_lift
            or abs(load_impulse) >= cfg.evidence_min_lift
        )
        and pressure_confidence >= cfg.gulf_pressure_confidence_threshold
        and (lng_metrics is not None or load_metrics is not None)
    )
    if impulse_ready:
        exog_evidence_specs.append(
            {
                "panel_id": "evidence_forecast_impulse",
                "title": "Forecast Impulse vs Last 7 Days",
                "impulses": {
                    "LNG Gulf": float(lng_impulse),
                    "ERCOT Load": float(load_impulse),
                },
                "caption": "Forecast 7d average minus last 7d actuals.",
                "score": 0.25,
                "context": {
                    "lng_impulse": float(lng_impulse),
                    "ercot_load_impulse": float(load_impulse),
                },
            }
        )

    panel_contexts = {
        "henry_regime": {
            "regime": regime,
            "transition": transition_label,
            "msi_pct": round(float(last_row["MSI_Pct_Rank"]), 4),
            "msi_pct_delta": round(float(msi_rank_delta), 4),
            "slack_tightening": slack_tightening,
        },
        "msi": {
            "msi": round(float(last_row["MSI"]), 4),
            "msi_pct": round(float(last_row["MSI_Pct_Rank"]), 4),
            "msi_pct_delta": round(float(msi_rank_delta), 4),
        },
        "fragility": {
            "fragility": round(float(last_row["Sub_Failure_Score"]), 2),
            "fragility_delta": round(float(fragility_delta), 2),
            "binding_delta": int(binding_delta),
        },
        "vulnerability": {
            "top_hubs": watch_hubs,
            "confidence": round(float(confidence), 2),
            "vuln_delta": round(float(top_vuln_delta), 4),
        },
        "walk_forward": {
            "oos_corr": oos_corr,
            "oos_hit": oos_hit,
            "samples": oos_samples,
        },
        "seasonal_henry": {
            "seasonal_zscore": _seasonal_zscore(df["Henry"], cfg.gas_year_start_month, cfg.seasonal_years),
        },
        "seasonal_msi": {
            "seasonal_zscore": _seasonal_zscore(df["MSI"], cfg.gas_year_start_month, cfg.seasonal_years),
        },
    }

    for spec in exog_evidence_specs:
        panel_contexts[spec["panel_id"]] = {
            **spec.get("context", {}),
            "panel_title": spec.get("title", spec["panel_id"]),
        }

    for hub in top_hubs_list:
        basis_col = f"Basis_{hub}"
        if basis_col in df.columns:
            panel_contexts[f"seasonal_basis_{hub}"] = {
                "seasonal_zscore": _seasonal_zscore(df[basis_col], cfg.gas_year_start_month, cfg.seasonal_years),
            }

    llm_payload = {
        "today": str(last_row.name.date()),
        "prev_date": prev_label,
        "regime": regime,
        "prev_regime": prev_regime,
        "transition_label": transition_label,
        "msi": round(float(last_row["MSI"]), 4),
        "msi_pct": round(float(last_row["MSI_Pct_Rank"]), 4),
        "msi_delta": round(float(msi_delta), 4),
        "msi_pct_delta": round(float(msi_rank_delta), 4),
        "fragility": round(float(last_row["Sub_Failure_Score"]), 2),
        "fragility_delta": round(float(fragility_delta), 2),
        "binding_count": binding_count,
        "binding_delta": binding_delta,
        "top_bottleneck": bottleneck_today["pipe"],
        "top_bottleneck_util": round(float(bottleneck_today["util"]), 3),
        "top_bottleneck_util_delta": round(float(bottleneck_today["util"] - bottleneck_prev_util), 3),
        "upstream_shock": bool(last_row.get("Upstream_Shock_Count", 0) > 0),
        "top_hubs": llm_hubs,
        "data_quality": {
            "excluded_pipe_days": bundle.quality.excluded_pipe_days,
            "top_missing_pipes": bundle.quality.top_missing_pipes,
            "util_anomaly_days": anomaly_days,
            "util_anomaly_pipe_days": anomaly_pipe_days,
        },
        "exog_coverage": exog_coverage,
        "gulf_pulse": {
            "lng_total": lng_today,
            "lng_delta": lng_delta,
            "lng_forecast_impulse": lng_impulse,
            "ercot_load_total": load_today,
            "ercot_load_delta": load_delta,
            "ercot_load_forecast_impulse": load_impulse,
            "ercot_lmp_z": lmp_z,
            "ercot_lmp_spike": lmp_spike,
            "fundy_movers": fundy_movers,
            "pressure_regime": pressure_regime,
            "pressure_confidence": pressure_confidence,
            "pressure_score": pressure_score,
        },
        "pressure_outlook": {
            "regime": pressure_regime,
            "confidence": pressure_confidence,
            "oos_corr": oos_corr,
            "oos_hit": oos_hit,
        },
        "oos": {
            "corr": oos_corr,
            "hit_rate": oos_hit,
            "samples": oos_samples,
        },
        "panel_ids": list(panel_contexts.keys()),
        "panel_contexts": panel_contexts,
        "banner": top_banner_text,
        "importance_rules": {
            "high_msi_pct_delta": cfg.high_delta_msi_pct,
            "high_fragility_delta": cfg.high_delta_fragility,
            "high_binding_delta": cfg.high_delta_binding,
        },
    }

    batched_output = None
    if ai.provider and cfg.enable_llm:
        batched_output = ai.generate_batched_trader_output(
            {"payload_json": json.dumps(llm_payload, default=str)}
        )

    llm_callouts = {}
    desk_note_html = None
    what_changed_summary = None
    gulf_pulse_take = None
    outlook_take = None
    if isinstance(batched_output, dict):
        desk_note_html = batched_output.get("desk_note_html")
        what_changed_summary = batched_output.get("what_changed_summary")
        gulf_pulse_take = batched_output.get("gulf_coast_pulse_take")
        outlook_take = batched_output.get("outlook_take")
        panel_callouts = batched_output.get("panel_callouts", {})
        if isinstance(panel_callouts, dict):
            llm_callouts = panel_callouts

    if desk_note_html:
        trader_note_html = render_trader_note(context, desk_note_html)
    else:
        trader_note_html = render_trader_note(context, None)
    if what_changed_summary:
        change_note = what_changed_summary
        changes_section_html = render_changes_section(system_rows, hub_rows, change_note, data_quality_html)

    coverage_lines = []
    coverage_map = {
        "lng": "LNG Gulf",
        "fundy": "Fundy SouthCentral",
        "ercot_load": "ERCOT Load",
        "ercot_power": "ERCOT Power",
    }
    for key, label in coverage_map.items():
        cov = exog_coverage.get(key)
        if not cov:
            continue
        hist_days = cov.get("hist_days", 0)
        forecast_days = cov.get("forecast_days", 0)
        missing_days = cov.get("missing_days", 0)
        missing_pct = (missing_days / total_days) if total_days else 0.0
        coverage_lines.append(
            f"{label}: {hist_days}d (fcst {forecast_days}d, missing {missing_days}d/{missing_pct:.0%})"
        )
    coverage_note = (
        "Data coverage for exogenous inputs: " + "; ".join(coverage_lines)
        if coverage_lines
        else "Data coverage for exogenous inputs: n/a."
    )

    coverage_scores = []
    component_keys = {
        "lng": pressure_summary.get("lng_z"),
        "ercot_load": pressure_summary.get("load_z"),
        "ercot_power": pressure_summary.get("lmp_z"),
        "fundy": pressure_summary.get("fundy_z"),
    }
    for key, value in component_keys.items():
        if value is None:
            continue
        cov = exog_coverage.get(key)
        if not cov:
            continue
        if total_days:
            coverage_scores.append(cov.get("hist_days", 0) / total_days)
    coverage_score = min(coverage_scores) if coverage_scores else 0.0
    data_confidence = pressure_confidence * (coverage_score if coverage_score > 0 else 1.0)

    corr_score = 0.0
    hit_score = 1.0
    if oos_corr is not None:
        corr_score = max(0.0, min(1.0, oos_corr / 0.3))
    if oos_hit is not None:
        hit_score = max(0.0, min(1.0, oos_hit / 0.6))
    wf_confidence = min(corr_score, hit_score) if corr_score > 0 else 0.0

    outlook_confidence = min(data_confidence, wf_confidence) if wf_confidence > 0 else 0.0
    reliability_label = (
        "LOW RELIABILITY"
        if (
            oos_corr is None
            or oos_corr < cfg.gulf_pressure_oos_corr_min
            or (oos_hit is not None and oos_hit < cfg.gulf_pressure_oos_hit_min)
        )
        else "OK"
    )

    if gulf_pulse_take:
        gulf_pulse_text = gulf_pulse_take
    else:
        gulf_pulse_text = _fallback_gulf_pulse_take(
            lng_delta,
            load_delta,
            lmp_z,
            lmp_spike,
            fundy_movers,
            pressure_regime,
            pressure_confidence,
        )
        if reliability_label == "LOW RELIABILITY":
            gulf_pulse_text += " Walk-forward support is weak; treat as context."

    if outlook_take:
        outlook_text = outlook_take
    else:
        outlook_text = _fallback_outlook_take(pressure_regime, outlook_confidence, reliability_label)

    lng_rows = [
        {
            "metric": "LNG Gulf Total",
            "today": _format_value(lng_today, fmt="{:.1f}"),
            "prev": _format_value(lng_prev, fmt="{:.1f}"),
            "delta": _format_delta(lng_delta, bad_when_positive=True, fmt="{:+.1f}"),
        }
    ]
    lng_card = (
        "<div class=\"card\">"
        "<h3>LNG Gulf Feedgas</h3>"
        f"{_build_metric_table(lng_rows)}"
        f"<p class=\"caption\">Forecast impulse: {_format_value(lng_impulse, fmt='{:+.1f}')}</p>"
        "</div>"
    )

    load_rows = [
        {
            "metric": "ERCOT Load",
            "today": _format_value(load_today, fmt="{:.0f}"),
            "prev": _format_value(load_prev, fmt="{:.0f}"),
            "delta": _format_delta(load_delta, bad_when_positive=True, fmt="{:+.0f}"),
        },
        {
            "metric": "ERCOT Max LMP",
            "today": _format_value(lmp_today, fmt="{:.1f}"),
            "prev": _format_value(lmp_prev, fmt="{:.1f}"),
            "delta": _format_delta(lmp_delta, bad_when_positive=True, fmt="{:+.1f}"),
        },
    ]
    load_caption = (
        f"Forecast impulse: {_format_value(load_impulse, fmt='{:+.1f}')}; "
        f"LMP z: {_format_value(lmp_z, fmt='{:+.2f}')}; "
        f"Spike: {'Yes' if lmp_spike else 'No'}."
    )
    load_card = (
        "<div class=\"card\">"
        "<h3>ERCOT Load + Power Stress</h3>"
        f"{_build_metric_table(load_rows)}"
        f"<p class=\"caption\">{load_caption}</p>"
        "</div>"
    )

    fundy_lines = []
    for mover in fundy_movers:
        delta_text = _format_value(mover.get("delta"), fmt="{:+.2f}")
        impulse_val = mover.get("impulse")
        if impulse_val is not None and np.isfinite(impulse_val):
            impulse_text = _format_value(impulse_val, fmt="{:+.2f}")
            fundy_lines.append(
                f"{mover['item']}: Delta {delta_text}, impulse {impulse_text}"
            )
        else:
            fundy_lines.append(f"{mover['item']}: Delta {delta_text}")
    fundy_html = (
        "<ul>" + "".join(f"<li>{line}</li>" for line in fundy_lines) + "</ul>"
        if fundy_lines
        else "<p>Insufficient Fundy coverage.</p>"
    )
    fundy_caption = (
        f"Tightness proxy: {_format_value(fundy_proxy, fmt='{:+.2f}')}"
        if fundy_proxy is not None
        else "Tightness proxy: n/a."
    )
    fundy_card = (
        "<div class=\"card\">"
        "<h3>SouthCentral Fundy</h3>"
        f"{fundy_html}"
        f"<p class=\"caption\">{fundy_caption}</p>"
        "</div>"
    )

    gulf_coast_html = (
        "<h2>Gulf Coast Pulse</h2>"
        "<div class=\"grid\">"
        f"{lng_card}{load_card}{fundy_card}"
        "</div>"
        f"<p class=\"caption\">{coverage_note}</p>"
        f"<p>{gulf_pulse_text}</p>"
        "<div class=\"card\">"
        "<h3>Pressure / Risk Outlook (next 1-3 days)</h3>"
        f"<p>{outlook_text}</p>"
        f"<p class=\"caption\">Reliability: {reliability_label}. Confidence {outlook_confidence:.0%}. "
        "Not a price forecast.</p>"
        "</div>"
    )

    evidence_cards = []
    evidence_note = (
        "Charts shown only when effect size, stability, lift, and sample thresholds are met."
    )
    evidence_reasons = []

    if top_risk is not None:
        if driver_pipe != "None" and risk_hub != "None" and top_risk.get("Evidence_Eligible", False):
            img = charts.make_basis_vs_util_chart(df, risk_hub, driver_pipe)
            evidence_cards.append(
                {
                    "title": f"{risk_hub} Basis vs {driver_pipe} Util",
                    "content": _image_tag(img),
                    "caption": (
                        f"Corr {top_risk['Primary_Driver_Corr']:.2f}, "
                        f"stability {top_risk['Primary_Driver_Stability']:.2f}, "
                        f"lift {top_risk['Primary_Driver_Lift']:.3f}, "
                        f"tight n={int(top_risk['Primary_Driver_Tight_Sample'])}"
                    ),
                    "score": float(top_risk.get("Confidence", 0)),
                }
            )
        else:
            if top_risk.get("Primary_Driver_Sample", 0) < cfg.evidence_min_sample:
                evidence_reasons.append("Insufficient sample size")
            if abs(top_risk.get("Primary_Driver_Corr", 0)) < cfg.evidence_min_abs_corr:
                evidence_reasons.append("Low correlation")
            if top_risk.get("Primary_Driver_Stability", 0) < cfg.evidence_min_stability:
                evidence_reasons.append("Low stability")
            if top_risk.get("Primary_Driver_Tight_Sample", 0) < cfg.evidence_min_tight_samples:
                evidence_reasons.append("Insufficient tight samples")
            lift_val = top_risk.get("Primary_Driver_Lift")
            if lift_val is None or not np.isfinite(lift_val) or abs(lift_val) < cfg.evidence_min_lift:
                evidence_reasons.append("Lift too small")

    if risk_hub != "None" and f"Target_Delta_Basis_{risk_hub}" in df.columns:
        frag_series = df["Sub_Failure_Score"]
        basis_delta = df[f"Target_Delta_Basis_{risk_hub}"]
        frag_sample = int(pd.concat([frag_series, basis_delta], axis=1).dropna().shape[0])
        frag_corr = safe_corr(frag_series, basis_delta)
        frag_stability = segment_correlation_stability(
            frag_series,
            basis_delta,
            segments=cfg.corr_stability_segments,
            min_abs_corr=cfg.evidence_min_abs_corr,
        )
        if (
            frag_sample >= cfg.evidence_min_sample
            and abs(frag_corr) >= cfg.evidence_min_abs_corr
            and frag_stability >= cfg.evidence_min_stability
        ):
            img = charts.make_fragility_basis_scatter(df, risk_hub)
            evidence_cards.append(
                {
                    "title": f"Fragility vs {risk_hub} Next-Day Basis",
                    "content": _image_tag(img),
                    "caption": (
                        f"Corr {frag_corr:.2f}, stability {frag_stability:.2f}, "
                        f"sample {frag_sample}"
                    ),
                    "score": 0.4,
                }
            )

    transition = df["Transition_Any"].fillna(False).astype(float)
    abs_ret = df["Target_Ret_Henry"].abs()
    trans_sample = int(transition.sum())
    trans_corr = safe_corr(transition, abs_ret)
    trans_stability = segment_correlation_stability(
        transition,
        abs_ret,
        segments=cfg.corr_stability_segments,
        min_abs_corr=cfg.evidence_min_abs_corr,
    )
    if (
        trans_sample >= cfg.evidence_min_transition_samples
        and abs(trans_corr) >= cfg.evidence_min_abs_corr
        and trans_stability >= cfg.evidence_min_stability
    ):
        img = charts.make_transition_effect_chart(df)
        evidence_cards.append(
            {
                "title": "Transition Days vs Next-Day Vol",
                "content": _image_tag(img),
                "caption": (
                    f"Corr {trans_corr:.2f}, stability {trans_stability:.2f}, "
                    f"transitions {trans_sample}"
                ),
                "score": 0.3,
            }
        )

    for spec in exog_evidence_specs:
        if "impulses" in spec:
            img = charts.make_forecast_impulse_bar(spec["impulses"])
            fallback = "\n".join(
                [
                    "Importance: Medium",
                    "Why: Forecast impulse is active for LNG and ERCOT load.",
                    "Important today?: Yes if impulse persists",
                    "Action: watch Gulf Coast demand pressure",
                ]
            )
        else:
            img = charts.make_rolling_corr_chart(
                spec["series_x"],
                spec["series_y"],
                spec.get("chart_title", spec["title"]),
            )
            ctx = spec.get("context", {})
            fallback = "\n".join(
                [
                    "Importance: Medium",
                    f"Why: Corr {ctx.get('corr', 0):.2f}, stability {ctx.get('stability', 0):.2f}.",
                    "Important today?: Yes if correlation holds",
                    "Action: size risk with Gulf Coast pressure",
                ]
            )
        evidence_cards.append(
            {
                "title": spec["title"],
                "content": _image_tag(img),
                "caption": spec.get("caption", ""),
                "score": float(spec.get("score", 0)),
                "callout": _select_callout(spec["panel_id"], llm_callouts, fallback),
            }
        )

    if not evidence_cards:
        if evidence_reasons:
            evidence_note = "No charts: " + "; ".join(sorted(set(evidence_reasons))) + "."
        else:
            evidence_note = "No statistically stable drivers met thresholds today."

    evidence_cards = sorted(evidence_cards, key=lambda x: x.get("score", 0), reverse=True)[
        : cfg.evidence_max_charts
    ]

    pipe_table_html = charts.build_pipe_table_html(df)

    callout_base = {
        "regime": regime,
        "transition_any": bool(last_row.get("Transition_Any", False)),
        "transition_label": transition_label,
        "msi_rank": float(last_row["MSI_Pct_Rank"]),
        "msi_rank_delta": float(msi_rank_delta),
        "fragility_rank": fragility_rank,
        "fragility_rank_prev": fragility_rank_prev,
        "fragility_delta": float(fragility_delta),
        "binding_delta": float(binding_delta),
        "pct_transition": cfg.pct_transition,
        "confidence": float(confidence),
        "vuln_delta": float(top_vuln_delta),
        "top_hubs": ", ".join(top_hubs_list) if top_hubs_list else "n/a",
        "oos_corr": oos_corr,
        "oos_hit": oos_hit,
        "upstream_shock": bool(last_row.get("Upstream_Shock_Count", 0) > 0),
    }

    quant_cards = [
        {
            "title": "Henry Price + Regimes",
            "content": _image_tag(charts.make_price_regime_chart(df)),
            "caption": (
                "What: Price colored by regime. Why: shows binding periods. "
                "How: align risk with physical stress. Failure: regime mislabels data gaps."
            ),
            "callout": _select_callout(
                "henry_regime",
                llm_callouts,
                _fallback_trader_take("Henry Price + Regimes", callout_base, cfg),
            ),
        },
        {
            "title": "Marginal Stress Index",
            "content": _image_tag(charts.make_msi_chart(df)),
            "caption": (
                "What: tail-weighted utilization stress. Why: catches binding risk. "
                "How: watch percentile shifts. Failure: stale capacity estimates."
            ),
            "callout": _select_callout(
                "msi",
                llm_callouts,
                _fallback_trader_take("Marginal Stress Index", callout_base, cfg),
            ),
        },
        {
            "title": "Substitution Fragility",
            "content": _image_tag(charts.make_fragility_chart(df)),
            "caption": (
                "What: spare vs flow shock. Why: flags reroute risk. "
                "How: rising fragility = grid lock. Failure: sudden flow anomalies."
            ),
            "callout": _select_callout(
                "fragility",
                llm_callouts,
                _fallback_trader_take("Substitution Fragility", callout_base, cfg),
            ),
        },
        {
            "title": "Hub Vulnerability Ranking",
            "content": _image_tag(charts.make_vulnerability_bar_chart(vuln)),
            "caption": (
                "What: capacity-weighted hub risk. Why: prioritizes basis exposure. "
                "How: focus on top names. Failure: correlations break in new regimes."
            ),
            "callout": _select_callout(
                "vulnerability",
                llm_callouts,
                _fallback_trader_take("Hub Vulnerability Ranking", callout_base, cfg),
            ),
        },
    ]

    if walk_forward is not None and not walk_forward.empty:
        walk_forward_table = render_table(walk_forward, float_fmt="{:.3f}")
        quant_cards.append(
            {
                "title": "Walk-Forward Diagnostics",
                "content": walk_forward_table,
                "caption": (
                    "What: rolling OOS regime-vol check. Why: stress test signal stability. "
                    "How: treat as context. Failure: regime shifts or structural breaks."
                ),
                "callout": _select_callout(
                    "walk_forward",
                    llm_callouts,
                    _fallback_trader_take("Walk-Forward Diagnostics", callout_base, cfg),
                ),
            }
        )

    seasonal_cards = []
    henry_seasonal_z = _seasonal_zscore(df["Henry"], cfg.gas_year_start_month, cfg.seasonal_years)
    henry_ctx = {**callout_base, "seasonal_zscore": henry_seasonal_z}
    seasonal_cards.append(
        {
            "title": "Henry Seasonal Overlay",
            "content": _image_tag(
                charts.make_seasonal_overlay_chart(
                    df,
                    df["Henry"],
                    "Henry Hub Seasonal (Gas Year)",
                    "Price ($)",
                    cfg.seasonal_years,
                    cfg.gas_year_start_month,
                )
            ),
            "caption": (
                "What: gas-year alignment. Why: seasonality context. "
                f"How: compare to prior years. {_seasonal_position_text(henry_seasonal_z)}"
            ),
            "callout": _select_callout(
                "seasonal_henry",
                llm_callouts,
                _fallback_trader_take("Seasonal Overlay", henry_ctx, cfg),
            ),
        }
    )
    msi_seasonal_z = _seasonal_zscore(df["MSI"], cfg.gas_year_start_month, cfg.seasonal_years)
    msi_ctx = {**callout_base, "seasonal_zscore": msi_seasonal_z}
    seasonal_cards.append(
        {
            "title": "MSI Seasonal Overlay",
            "content": _image_tag(
                charts.make_seasonal_overlay_chart(
                    df,
                    df["MSI"],
                    "MSI Seasonal (Gas Year)",
                    "MSI",
                    cfg.seasonal_years,
                    cfg.gas_year_start_month,
                )
            ),
            "caption": (
                "What: stress seasonality. Why: detect unusual tightness. "
                f"How: compare to recent years. {_seasonal_position_text(msi_seasonal_z)}"
            ),
            "callout": _select_callout(
                "seasonal_msi",
                llm_callouts,
                _fallback_trader_take("Seasonal Overlay", msi_ctx, cfg),
            ),
        }
    )

    top_hubs = [row["Hub"] for _, row in vuln.head(cfg.seasonal_hub_count).iterrows()]
    for hub in top_hubs:
        basis_col = f"Basis_{hub}"
        if basis_col not in df.columns:
            continue
        basis_z = _rolling_zscore(df[basis_col])
        hub_ctx = {**callout_base, "seasonal_zscore": basis_z}
        seasonal_cards.append(
            {
                "title": f"{hub} Basis Seasonal",
                "content": _image_tag(
                    charts.make_seasonal_overlay_chart(
                        df,
                        df[basis_col],
                        f"{hub} Basis Seasonal (Gas Year)",
                        "Basis ($)",
                        cfg.seasonal_years,
                        cfg.gas_year_start_month,
                    )
                ),
                "caption": (
                    "What: basis seasonality. Why: spot dislocation risk. "
                    f"How: track against history. {_seasonal_position_text(basis_z)}"
                ),
                "callout": _select_callout(
                    f"seasonal_basis_{hub}",
                    llm_callouts,
                    _fallback_trader_take("Seasonal Overlay", hub_ctx, cfg),
                ),
            }
        )

    regime_stats_html = render_table(regime_stats.reset_index(), float_fmt="{:.4f}")
    walk_forward_html = None

    html_content = build_html_report(
        title="Henry Hub Quantitative Pressure Dashboard",
        generated_on=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        top_banner_html=top_banner_html,
        changes_section_html=changes_section_html,
        gulf_coast_html=gulf_coast_html,
        trader_note_html=trader_note_html,
        evidence_cards=evidence_cards,
        evidence_note=evidence_note,
        pipe_table_html=pipe_table_html,
        quant_cards=quant_cards,
        seasonal_cards=seasonal_cards,
        regime_stats_html=regime_stats_html,
        walk_forward_html=walk_forward_html,
    )

    cfg.html_output_path.write_text(html_content, encoding="utf-8")
    logger.info("HTML dashboard written to %s", cfg.html_output_path)

    top3 = ""
    if not vuln.empty:
        top3 = ", ".join(
            f"{row['Hub']} ({row['Confidence']*100:.0f}%)"
            for _, row in vuln.head(3).iterrows()
        )
    upstream_flag = "Yes" if context["upstream_shock_count"] > 0 else "No"

    logger.info(
        "Run summary | date=%s | regime=%s | transition=%s | top_hubs=%s | upstream_shock=%s",
        last_row.name.date(),
        regime,
        transition_label,
        top3 if top3 else "None",
        upstream_flag,
    )
    logger.info(
        "Gulf pulse | lng_impulse=%s | ercot_load_impulse=%s | ercot_lmp_z=%s | southcentral_proxy=%s | regime=%s | confidence=%.2f",
        _format_value(lng_impulse, fmt="{:+.2f}"),
        _format_value(load_impulse, fmt="{:+.2f}"),
        _format_value(lmp_z, fmt="{:+.2f}"),
        _format_value(fundy_proxy, fmt="{:+.2f}"),
        pressure_regime,
        pressure_confidence,
    )

    logger.info("--- Pipeline Complete ---")
    return PipelineResult(html_path=str(cfg.html_output_path), last_date=last_row.name)
