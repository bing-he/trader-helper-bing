"""Narrative generation for the LNG dashboard.

This module reads LLM credentials from the environment, preferring Google Gemini
and falling back to OpenAI. The Gemini API key is accepted from either
`GEMINI_KEY` or `GeminiKey` to match existing .env files. If no LLM succeeds,
a deterministic narrative is produced.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .config import DashboardConfig

LOGGER = logging.getLogger(__name__)
_GEMINI_MODELS_LOGGED = False
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"


def _fmt_delta(value: Optional[float]) -> str:
    return f"{value:+.0f}" if value is not None else "n/a"


def _format_metric_block(label: str, metric: Dict[str, object]) -> str:
    deltas = metric["deltas"]
    zscore = metric.get("zscore_60d")
    shock = metric.get("shock", {})
    parts = [
        f"{label}: {metric['latest_value']:.0f} mmcf/d",
        f"Δ1d {_fmt_delta(deltas.get('delta_1d'))} | Δ5d {_fmt_delta(deltas.get('delta_5d'))} | Δ10d {_fmt_delta(deltas.get('delta_10d'))}",
    ]
    if zscore is not None:
        parts.append(f"Z60d {zscore:+.2f}")
    parts.append(f"Regime {metric.get('regime', 'Flat')}")
    if shock.get("is_shock"):
        parts.append("Shock vs history")
    return "; ".join(parts)


def _build_prompt(
    actual_metrics: Dict[str, Dict[str, object]],
    forecast_contributions: List[Dict[str, object]],
    warnings: List[str],
    config: DashboardConfig,
) -> str:
    lines = ["You are an LNG feedgas risk analyst writing a concise, neutral desk update."]
    if warnings:
        lines.append(f"Data caveats: {', '.join(warnings)}")
    if "Total" in actual_metrics:
        lines.append("Total LNG:")
        lines.append(_format_metric_block("Total", actual_metrics["Total"]))
    lines.append("Facilities snapshot:")
    for facility, metric in actual_metrics.items():
        if facility == "Total":
            continue
        lines.append(_format_metric_block(facility, metric))

    lines.append("Forecast signals:")
    for contrib in forecast_contributions:
        change = contrib["expected_change"]
        if change is None:
            continue
        lines.append(
            f"{contrib['facility']}: {change:+.0f} mmcf/d over next {contrib['horizon_days']}d vs start"
        )

    lines.append(
        "Write three brief sections: (1) What changed today, (2) Expectations next 7-14 days, "
        "including drivers and risk flags, (3) What would invalidate the view."
    )
    lines.append("Tone: concise, factual, risk-aware, no hype.")
    return "\n".join(lines)


def _log_available_gemini_models(genai) -> None:
    """Log available Gemini models once to aid debugging."""
    global _GEMINI_MODELS_LOGGED
    if _GEMINI_MODELS_LOGGED:
        return
    try:
        models = [m.name for m in genai.list_models()]  # type: ignore[attr-defined]
        LOGGER.info("Available Gemini models: %s", ", ".join(models))
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOGGER.debug("Unable to list Gemini models: %s", exc)
    finally:
        _GEMINI_MODELS_LOGGED = True


def _call_gemini(api_key: str, prompt: str, model_name: str = GEMINI_MODEL_NAME) -> Optional[str]:
    """Call Gemini and return the narrative text, or None on failure."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("Gemini call failed (model=%s): %s", model_name, exc)
        try:
            import google.generativeai as genai  # type: ignore

            _log_available_gemini_models(genai)
        except Exception:
            LOGGER.debug("Skipping Gemini model listing after failure.")
        return None


def _call_openai(api_key: str, project: Optional[str], prompt: str) -> Optional[str]:
    """Call OpenAI chat completion and return text, or None on failure."""
    try:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key) if not project else OpenAI(api_key=api_key, project=project)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise natural gas trader assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return response.choices[0].message.content
        except ImportError:
            import openai  # type: ignore

            openai.api_key = api_key
            if project:
                openai.project = project
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise natural gas trader assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return response["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("OpenAI call failed%s: %s", f" (project={project})" if project else "", exc)
        return None


def _deterministic_narrative(
    actual_metrics: Dict[str, Dict[str, object]], forecast_contributions: List[Dict[str, object]]
) -> str:
    lines = ["Expectations Going Forward (deterministic fallback)."]
    total = actual_metrics.get("Total")
    if total:
        deltas = total["deltas"]
        lines.append(
            f"Total LNG last value {total['latest_value']:.0f} mmcf/d "
            f"(Δ1d {_fmt_delta(deltas.get('delta_1d'))}, Δ5d {_fmt_delta(deltas.get('delta_5d'))})."
        )
        z = total.get("zscore_60d")
        if z is not None:
            lines.append(f"Versus 60d mean, z-score {z:+.2f}; regime {total.get('regime', 'Flat')}.")
    if forecast_contributions:
        described = [
            f"{row['facility']}: {_fmt_delta(row['expected_change'])} mmcf/d next {row['horizon_days']}d"
            for row in forecast_contributions
        ]
        if described:
            lines.append("Forecast focus: " + "; ".join(described))
    lines.append("Watchpoints: facility-level deltas and any shocks flagged in metrics.")
    return "\n".join(lines)


def generate_narrative(
    actual_metrics: Dict[str, Dict[str, object]],
    forecast_contributions: List[Dict[str, object]],
    warnings: List[str],
    config: DashboardConfig,
    env: Dict[str, Optional[str]],
    use_llm: bool,
) -> str:
    """Generate the Expectations Going Forward narrative using Gemini or OpenAI, with deterministic fallback.

    Environment handling:
    - Gemini key is read from either ``GEMINI_KEY`` or ``GeminiKey``.
    - OpenAI uses ``OPENAI_API_KEY`` and optionally ``OPENAI_PROJECT`` when it resembles a valid project id.
    - When keys are missing or calls fail, the deterministic narrative is returned.
    """
    prompt = _build_prompt(actual_metrics, forecast_contributions, warnings, config)
    if not use_llm:
        LOGGER.info("LLM disabled via CLI; using deterministic narrative.")
        return _deterministic_narrative(actual_metrics, forecast_contributions)

    gemini_key = env.get("GEMINI_KEY") or env.get("GeminiKey")
    openai_key = env.get("OPENAI_API_KEY")
    openai_project_raw = env.get("OPENAI_PROJECT")
    openai_project = (
        openai_project_raw
        if openai_project_raw and openai_project_raw.startswith("proj_") and len(openai_project_raw) > 6
        else None
    )
    if openai_project_raw and not openai_project:
        LOGGER.warning("OPENAI_PROJECT present but not in expected format; ignoring project parameter.")

    if not gemini_key and not openai_key:
        LOGGER.warning("No LLM keys found; falling back to deterministic narrative.")
        return _deterministic_narrative(actual_metrics, forecast_contributions)

    if gemini_key:
        narrative = _call_gemini(gemini_key, prompt)
        if narrative:
            return narrative

    if openai_key:
        if openai_project:
            narrative = _call_openai(openai_key, openai_project, prompt)
            if not narrative:
                LOGGER.warning("Retrying OpenAI without project after project-based attempt.")
                narrative = _call_openai(openai_key, None, prompt)
        else:
            narrative = _call_openai(openai_key, None, prompt)
        if narrative:
            return narrative

    LOGGER.info("Using deterministic narrative fallback after LLM attempts.")
    return _deterministic_narrative(actual_metrics, forecast_contributions)
