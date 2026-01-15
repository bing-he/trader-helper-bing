from __future__ import annotations

from typing import Dict, Optional

import json
import re

from ..config import Config
from ..utils.logging import get_logger


class AIAnalyst:
    """Uses LLMs to generate trader commentary when keys are available."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.logger = get_logger(__name__)
        self.client = None
        self.provider: Optional[str] = None
        self.gemini_model = None
        self.gemini_model_name = None
        self.gemini_fallbacks = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

        if self.cfg.gemini_api_key and self.cfg.enable_llm:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.cfg.gemini_api_key)
                self.gemini_model_name = self._select_gemini_model(genai)
                self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                self.provider = "gemini"
                source = self.cfg.gemini_key_source or "unknown"
                self.logger.info("AI Analyst initialized (Gemini via %s).", source)
                return
            except Exception as exc:
                self.logger.warning("Gemini init failed: %s", exc)

        if self.cfg.openai_api_key and self.cfg.enable_llm:
            try:
                from openai import OpenAI

                self.client = OpenAI(
                    api_key=self.cfg.openai_api_key,
                    project=self.cfg.openai_project,
                )
                self.provider = "openai"
                self.logger.info("AI Analyst initialized (OpenAI).")
            except Exception as exc:
                self.logger.warning("OpenAI init failed: %s", exc)

        if not self.provider:
            self.logger.info("AI Analyst skipped (no keys or disabled).")

    def generate_prompt(self, context: Dict) -> str:
        upstream_warning = ""
        if context["upstream_shock_count"] > 0:
            upstream_warning = (
                "CRITICAL ALERT: UPSTREAM SUPPLY SHOCK DETECTED.\n"
                f"- Pipeline: {context['upstream_shock_pipe']}\n"
                "- Signal: Large Flow Drop + STABLE Capacity + PRICE RISING.\n"
                "- Meaning: This is NOT demand destruction. This is an upstream supply cut.\n"
                "- Action: Treat reported capacity as stale. Bias bullish.\n"
            )

        return (
            "Act as a Head Gas Trader explaining the morning setup to the desk.\n\n"
            f"{upstream_warning}\n"
            f"CURRENT SETUP ({context['date']}):\n\n"
            "1. PHYSICAL REALITY:\n"
            f"   - Regime: {context['regime']} ({context['transition_label']}).\n"
            f"   - System stress percentile: {context['msi_rank']:.1%}.\n"
            f"   - Top bottleneck: {context['top_risk_driver']} (util {context['driver_util']*100:.1f}%).\n\n"
            "2. SUBSTITUTION RISK:\n"
            f"   - Fragility score: {context['fragility']:.1f}.\n"
            f"   - Congested alternatives: {context['congested_pipes']}.\n\n"
            "3. PRICE IMPACT:\n"
            f"   - Watch Hub: {context['top_risk_hub']} (basis {context['hub_basis']:.3f}).\n\n"
            "YOUR OUTPUT:\n"
            "Give a 3-bullet morning note in trader speak.\n"
            "- Bullet 1: Flow state (tight vs loose), mention pipes.\n"
            "- Bullet 2: Bottleneck risk and reroute ability.\n"
            "- Bullet 3: The trade (volatility and win rate).\n\n"
            "Use HTML <b> tags for emphasis. Be concise. No fluff."
        )

    def generate_trader_take_prompt(self, panel_name: str, context: Dict) -> str:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in context.items())
        return (
            "You are a senior gas trader. Translate the panel into trader-speak.\n"
            f"Panel: {panel_name}\n"
            "Context:\n"
            f"{context_lines}\n\n"
            "Return exactly 4 lines, max 60 words total:\n"
            "Importance: High/Medium/Low\n"
            "Why: one sentence\n"
            "Important today?: Yes/No + the condition that flips it\n"
            "Action: one short action\n"
        )

    def generate_batched_prompt(self, context: Dict) -> str:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in context.items())
        return (
            "You are a senior gas trader. Produce strict JSON only.\n"
            "Return JSON with keys: desk_note_html, what_changed_summary, gulf_coast_pulse_take, "
            "outlook_take, panel_callouts, risk_flags.\n"
            "desk_note_html: HTML <ul><li> with 3 bullets in trader-speak using <b> tags.\n"
            "what_changed_summary: 1-2 concise lines.\n"
            "gulf_coast_pulse_take: 1 paragraph trader translation of the Gulf Coast Pulse cards.\n"
            "outlook_take: 1 paragraph for Pressure / Risk Outlook, explicitly NOT a price forecast.\n"
            "panel_callouts: object keyed by panel_id with fields importance, why, important_today, action.\n"
            "risk_flags: list of short strings.\n"
            "Constraints: keep each callout under ~60 words.\n"
            "Use panel_ids from payload_json; include all panel_ids with callouts.\n"
            "Context:\n"
            f"{context_lines}\n"
        )

    def _select_gemini_model(self, genai_module) -> str:
        try:
            models = list(genai_module.list_models())
            candidates = [
                m.name
                for m in models
                if "generateContent" in getattr(m, "supported_generation_methods", [])
            ]
            if candidates:
                return candidates[0]
        except Exception as exc:
            self.logger.debug("Gemini model listing failed: %s", exc)
        return self.gemini_fallbacks[0]

    def fetch_analysis(self, context: Dict) -> Optional[str]:
        prompt = self.generate_prompt(context)
        return self._call_model(prompt)

    def fetch_trader_take(self, panel_name: str, context: Dict) -> Optional[str]:
        prompt = self.generate_trader_take_prompt(panel_name, context)
        return self._call_model(prompt)

    def generate_batched_trader_output(self, context: Dict) -> Optional[Dict]:
        prompt = self.generate_batched_prompt(context)
        self.logger.info("Requesting batched trader output (%s).", self.provider or "disabled")
        response = self._call_model(prompt)
        if not response:
            return None
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    def _call_model(self, prompt: str) -> Optional[str]:
        if not self.provider:
            return None
        try:
            if self.provider == "gemini":
                return self.gemini_model.generate_content(prompt).text
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a veteran gas trader."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                )
                return response.choices[0].message.content
        except Exception as exc:
            if self.provider == "gemini":
                for model_name in self.gemini_fallbacks:
                    if model_name == self.gemini_model_name:
                        continue
                    try:
                        import google.generativeai as genai

                        self.gemini_model = genai.GenerativeModel(model_name)
                        self.gemini_model_name = model_name
                        return self.gemini_model.generate_content(prompt).text
                    except Exception:
                        continue
            self.logger.warning("AI generation failed (%s): %s", self.provider, exc)
        return None
