#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Screening Report (Hard-coded I/O)
---------------------------------------------
Reads:   ./output/Combined_Wide_Data.csv   (index is a calendar date; numeric columns are features)
Writes:  ./output/corr_report/
  - correlation_summary.csv         # per-feature metrics and flags
  - low_corr_features.csv           # features flagged as "lonely" by mean|ρ| threshold
  - keep_features_after_screening.txt
  - drop_features_after_screening.txt
  - correlation_histogram.png
  - low_corr_heatmap.png            # subset heatmap for interpretability
  - corr_matrix.npy                 # optional: full matrix for advanced users
    - data_lightness_crosstab.csv     # four-way diagnostic flag counts
  - readme.txt                      # human-readable summary

Design:
- Fast, conservative screening. Do not remove columns based only on feature–feature correlation inside the modeling pipeline.
- Accurate Spearman with pairwise NaN handling if SciPy is available; otherwise a fast, documented approximation.
- Pre-filter by missingness and near-constant variance to reduce width and noise.
- All thresholds are intentionally conservative and documented below.

Trade-offs:
- Fallback mode fills NaN ranks by column mean rank to enable fast BLAS correlation. This introduces small bias; acceptable for screening and always reported.
- Heatmaps are limited to a manageable subset (≤50 columns) for readability. The full matrix is stored separately (npy) if needed.

"""

import hashlib
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# ------------------------------ hard-coded I/O ------------------------------

ROOT = (
    os.path.abspath(os.path.dirname(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
IN_FEATURES = os.path.join(ROOT, "output", "Combined_Wide_Data.csv")
OUTDIR = os.path.join(ROOT, "output", "corr_report")

# ------------------------------ tunables (conservative) ------------------------------

MAX_ROWS_FOR_CORR = 2000  # cap rows for correlation to control runtime; seed is fixed
MISSINGNESS_DROP_THRESHOLD = 0.70  # drop columns with >70% NaN
VARIANCE_EPS = 1e-12  # drop near-constant columns by nanvar
LOW_CORR_THRESHOLD = 0.05  # "lonely" if mean absolute Spearman < 0.05
LOW_HEATMAP_MAX = 50  # at most this many columns in the low-corr heatmap
SAVE_FULL_MATRIX = True  # save full matrix as .npy for offline analysis
HIGH_MISSINGNESS_FLAG_THRESHOLD = 0.50  # flag as high missingness when >50% NaN
EFFECTIVE_OBS_FRACTION_THRESHOLD = 0.30  # flag low overlap when mean pair fraction <30%

# ------------------------------ utils ------------------------------


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha1_series(x: pd.Series) -> str:
    # hash ignoring index; treat NaN consistently
    # convert to bytes via numpy; use float64 for stable bit patterns
    arr = x.to_numpy(dtype=np.float64, copy=False)
    # replace NaNs with a sentinel that won't collide with real numbers
    arr = np.where(np.isnan(arr), np.float64(1.23456789012345e308), arr)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] <= 1:
        return df
    hashes = df.apply(_sha1_series, axis=0)
    _, unique_idx = np.unique(hashes.to_numpy(), return_index=True)
    kept_cols = df.columns.sort_values()[list(sorted(unique_idx))]
    return df[kept_cols]


def drop_by_missingness_and_variance(
    df: pd.DataFrame, miss_thresh: np.float64, var_eps: np.float64
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stats = pd.DataFrame(index=df.columns)
    stats["non_null"] = df.notna().sum(axis=0)
    stats["frac_missing"] = 1.0 - (stats["non_null"] / len(df))
    stats["nanvar"] = df.var(axis=0, ddof=1, numeric_only=True)
    keep_mask = (stats["frac_missing"] <= miss_thresh) & (stats["nanvar"] > var_eps)
    return df.loc[:, keep_mask.values], stats


def _spearman_corr_scipy(df: pd.DataFrame) -> pd.DataFrame:
    # Accurate Spearman with pairwise NaN omission in C
    from scipy.stats import spearmanr

    r, _ = spearmanr(df, axis=0, nan_policy="omit")
    # spearmanr returns ndarray (n+nvars)×(n+nvars) if input is 2D rows; we passed (nrows, ncols) so r is (ncols, ncols)
    return pd.DataFrame(r, index=df.columns, columns=df.columns)


def _spearman_corr_fallback(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # Approximate Spearman: rank each column, fill NaN ranks by column mean rank (documented bias), Pearson on ranks
    ranked = df.rank(method="average")
    # column-mean rank imputation for NaNs
    col_means = ranked.mean(axis=0)
    ranked = ranked.fillna(col_means)
    # float32 to reduce memory/compute
    X = ranked.to_numpy(dtype=np.float32, copy=False)
    # center to improve numerical stability
    X -= X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0.0] = 1.0
    X /= std
    # Pearson correlation on standardized ranks
    corr = (X.T @ X) / (X.shape[0] - 1)
    # clip numerical noise
    corr = np.clip(corr, -1.0, 1.0, out=corr)
    return pd.DataFrame(corr, index=df.columns, columns=df.columns)


def compute_spearman(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    # Try SciPy; fallback is documented
    try:
        corr = _spearman_corr_scipy(df)
        mode = "scipy_pairwise_omit"
    except Exception:
        rng = np.random.default_rng(42)
        corr = _spearman_corr_fallback(df, rng)
        mode = "fallback_rank_pearson_mean_fill"
    return corr, mode


def mean_abs_corr(corr: pd.DataFrame) -> pd.Series:
    a = corr.abs()
    # exclude self-correlation (diag=1). Adjust mean accordingly.
    n = a.shape[0]
    if n <= 1:
        return pd.Series(0.0, index=a.index)
    # subtract 1 on diag, then divide by (n-1)
    s = (a.sum(axis=1) - 1.0) / (n - 1)
    return s


def plot_hist(mean_abs: pd.Series, threshold: np.float64, dest: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(mean_abs.values, bins=50)
    plt.axvline(threshold, linestyle="--")
    plt.title("Distribution of mean |Spearman| per feature")
    plt.xlabel("mean |ρ|")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(dest, dpi=160)
    plt.close()


def plot_heatmap(corr_subset: pd.DataFrame, dest: str) -> None:
    # Basic imshow for speed and zero extra deps
    k = corr_subset.shape[0]
    figsize = (min(12, 0.2 * k + 4), min(12, 0.2 * k + 4))
    plt.figure(figsize=figsize)
    plt.imshow(corr_subset.values, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Correlation heatmap (subset)")
    # Tick management for readability
    step = max(1, k // 20)
    ticks = np.arange(0, k, step)
    labels = corr_subset.index.tolist()[::step]
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.savefig(dest, dpi=160)
    plt.close()


# ------------------------------ main ------------------------------


def main() -> None:
    t0 = time.perf_counter()
    ensure_outdir(OUTDIR)

    # Load
    if not os.path.exists(IN_FEATURES):
        raise FileNotFoundError(f"Missing input: {IN_FEATURES}")
    raw = pd.read_csv(IN_FEATURES, index_col=0, parse_dates=True)

    # Tabulate feature columns up front for quick inspection (non-blocking)
    column_names = sorted(map(str, raw.columns))
    rows = [(idx + 1, name) for idx, name in enumerate(column_names)]
    print(tabulate(rows, headers=["#", "column_name"], tablefmt="github"))
    num = raw.select_dtypes(include=[np.number])

    # Pre-filters
    # 1) duplicates
    num_dedup = drop_duplicates(num)
    # 2) missingness + variance
    num_filt, basic_stats = drop_by_missingness_and_variance(
        num_dedup, MISSINGNESS_DROP_THRESHOLD, VARIANCE_EPS
    )

    # Effective sample size: mean pairwise non-NaN overlaps per feature
    if num_filt.shape[1] > 0:
        mask = num_filt.notna().to_numpy(dtype=np.int64, copy=False)
        non_null_counts = mask.sum(axis=0)
        if num_filt.shape[1] > 1:
            overlap_counts = mask.T @ mask
            np.fill_diagonal(overlap_counts, non_null_counts)
            effective_obs_values = (overlap_counts.sum(axis=1) - non_null_counts) / (
                num_filt.shape[1] - 1
            )
        else:
            effective_obs_values = non_null_counts.astype(np.float64)
        effective_obs = pd.Series(effective_obs_values, index=num_filt.columns)
    else:
        effective_obs = pd.Series(dtype=np.float64)

    # Row cap for correlation only (screening), not for reporting of missingness/variance
    rng = np.random.default_rng(42)
    if len(num_filt) > MAX_ROWS_FOR_CORR:
        sampled_idx = rng.choice(
            num_filt.index.to_numpy(), size=MAX_ROWS_FOR_CORR, replace=False
        )
        work = num_filt.loc[np.sort(sampled_idx)]
        row_cap_note = f"rows_capped_to_{MAX_ROWS_FOR_CORR}"
    else:
        work = num_filt
        row_cap_note = "no_row_cap"

    # Correlation
    corr, mode = compute_spearman(work)

    # Metrics
    mac = mean_abs_corr(corr)
    low = mac[mac < LOW_CORR_THRESHOLD].sort_values()
    n_total = num.shape[1]
    n_after_dup = num_dedup.shape[1]
    n_after_prefilt = num_filt.shape[1]
    n_low = len(low)

    # Effective observation metrics (reindex to correlation columns)
    effective_obs = effective_obs.reindex(mac.index)
    total_rows = max(len(num), 1)
    effective_obs_frac = effective_obs / total_rows

    # Diagnostic flags
    frac_missing = basic_stats.reindex(mac.index)["frac_missing"].to_numpy()
    nanvar = basic_stats.reindex(mac.index)["nanvar"].to_numpy()
    flag_high_missing = frac_missing > HIGH_MISSINGNESS_FLAG_THRESHOLD
    flag_low_variance = nanvar <= VARIANCE_EPS
    flag_low_corr = mac.values < LOW_CORR_THRESHOLD
    flag_low_effective = (
        effective_obs_frac.to_numpy() < EFFECTIVE_OBS_FRACTION_THRESHOLD
    )

    # Persist artifacts
    if SAVE_FULL_MATRIX and corr.shape[0] <= 1200:  # guard disk/memory
        np.save(os.path.join(OUTDIR, "corr_matrix.npy"), corr.to_numpy())

    # Summary table
    summary = pd.DataFrame(
        {
            "feature": mac.index,
            "mean_abs_spearman": mac.values,
            "non_null": basic_stats.reindex(mac.index)["non_null"].to_numpy(),
            "frac_missing": basic_stats.reindex(mac.index)["frac_missing"].to_numpy(),
            "nanvar": basic_stats.reindex(mac.index)["nanvar"].to_numpy(),
            "effective_obs": effective_obs.to_numpy(),
            "effective_obs_frac": effective_obs_frac.to_numpy(),
            "flag_high_missing": flag_high_missing,
            "flag_low_variance": flag_low_variance,
            "flag_low_corr": flag_low_corr,
            "flag_low_effective_overlap": flag_low_effective,
        }
    )
    summary.to_csv(os.path.join(OUTDIR, "correlation_summary.csv"), index=False)

    # Crosstab of diagnostic flags for data lightness awareness
    if not summary.empty:
        crosstab = (
            summary[
                [
                    "flag_high_missing",
                    "flag_low_variance",
                    "flag_low_corr",
                    "flag_low_effective_overlap",
                ]
            ]
            .value_counts()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
        crosstab.to_csv(
            os.path.join(OUTDIR, "data_lightness_crosstab.csv"), index=False
        )

    low.to_frame(name="mean_abs_spearman").to_csv(
        os.path.join(OUTDIR, "low_corr_features.csv")
    )

    # Keep/Drop lists for screening-only use outside the modeling pipeline
    keep_list = summary.loc[~summary["flag_low_corr"], "feature"].tolist()
    drop_list = summary.loc[summary["flag_low_corr"], "feature"].tolist()
    with open(
        os.path.join(OUTDIR, "keep_features_after_screening.txt"), "w", encoding="utf-8"
    ) as fh:
        fh.write("\n".join(keep_list))
    with open(
        os.path.join(OUTDIR, "drop_features_after_screening.txt"), "w", encoding="utf-8"
    ) as fh:
        fh.write("\n".join(drop_list))

    # Plots
    plot_hist(
        mac, LOW_CORR_THRESHOLD, os.path.join(OUTDIR, "correlation_histogram.png")
    )

    if n_low > 0:
        subset_cols = low.index[:LOW_HEATMAP_MAX]
        corr_subset = corr.loc[subset_cols, subset_cols]
        plot_heatmap(corr_subset, os.path.join(OUTDIR, "low_corr_heatmap.png"))

    # Human-readable readme
    t1 = time.perf_counter()
    lines = []
    lines.append("Correlation Screening Report")
    lines.append("===========================")
    lines.append(f"input_file                : {IN_FEATURES}")
    lines.append(f"output_dir                : {OUTDIR}")
    lines.append(f"rows_total                : {len(num)}")
    lines.append(f"columns_total             : {n_total}")
    lines.append(f"after_duplicates          : {n_after_dup}")
    lines.append(f"after_missing+variance    : {n_after_prefilt}")
    lines.append(f"corr_compute_mode         : {mode}")
    lines.append(f"row_cap_mode              : {row_cap_note}")
    lines.append(f"low_corr_threshold        : {LOW_CORR_THRESHOLD}")
    lines.append(f"low_corr_count            : {n_low}")
    lines.append(
        f"low_effective_overlap_thr  : {EFFECTIVE_OBS_FRACTION_THRESHOLD} (fraction of rows)"
    )
    if not summary.empty:
        n_low_effective = int(summary["flag_low_effective_overlap"].sum())
        n_high_missing = int(summary["flag_high_missing"].sum())
        lines.append(f"low_effective_overlap_cnt : {n_low_effective}")
        lines.append(f"high_missing_flag_cnt     : {n_high_missing}")
        lines.append(
            "data_lightness_crosstab   : data_lightness_crosstab.csv (counts by flag combination)"
        )
    if n_low > 0:
        lines.append("examples_low_corr         : " + ", ".join(low.index[:10]))
    lines.append(f"runtime_seconds           : {t1 - t0:0.2f}")
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append(
        "- Pre-filters removed duplicates, very sparse, and near-constant columns."
    )
    lines.append(
        "- Spearman correlation was computed with accurate pairwise NaN omission if SciPy was available;"
    )
    lines.append(
        "  otherwise a fast approximation used column-mean rank imputation before Pearson on ranks."
    )
    lines.append(
        "- Use the keep/drop lists only as a first-pass screen. Retain target-aware selection in the modeling pipeline."
    )
    with open(os.path.join(OUTDIR, "readme.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
