# LNG Feedgas Dashboard

Production-grade dashboard for Gulf Coast LNG feedgas actuals and forecasts. Builds daily HTML with facility-level overlays, stitched forecasts, diagnostics, and a trader-readable narrative.

## Quick start

```bash
cd C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r LNGAnalysis/requirements.txt
python -m LNGAnalysis.lng_dashboard --lookback-years 5 --top-facilities 8 --use-llm true
# or
python LNGAnalysis/lng_dashboard.py --lookback-years 5 --top-facilities 8 --use-llm true
```

## Inputs
- INFO directory (default: repo_root/INFO) with `CriterionLNGHist.csv` and `CriterionLNGForecast.csv`
- `.env` in `Scripts/` containing `GeminiKey` (or `GEMINI_KEY`), `OPENAI_API_KEY`, and optionally `OPENAI_PROJECT`

Use `--info-dir` to override data location and `--out-dir` to change where HTML is written.

## Outputs
- `out/lng_dashboard/YYYY-MM-DD/lng_dashboard.html`
- `out/lng_dashboard/latest/lng_dashboard.html`

## CLI flags
- `--info-dir`: path to INFO folder (default auto-detected)
- `--out-dir`: output root (default `Scripts/out`)
- `--lookback-years`: prior-year overlays (default 5)
- `--top-facilities`: number of facility charts (default 5)
- `--use-llm`: `true`/`false` to enable LLM narrative (Gemini preferred, OpenAI fallback)

## Features
- Canonical facility normalization with warning for unknowns
- Actual/forecast stitching with day-of-year overlays and forecast start marker
- Level, delta, z-score, regime, shock detection, and forecast diagnostics
- Gemini narrative first (accepts `GeminiKey` or `GEMINI_KEY`), OpenAI fallback (ignores invalid `OPENAI_PROJECT`), deterministic text if LLM unavailable
- Responsive charts: Total feedgas first, facility charts in a three-across grid
