"""
Fast_scm_power_helpers.py

Minimum Detectable Effect (MDE) analysis for Synthetic Experimental Design,
following the permutation inference procedure of Abadie and Zhao (2021) as
implemented in Vives-i-Bastida (2022).

This version uses Monte Carlo sampling for the null distribution and includes
a Pareto-ranking heuristic to prioritize historical fit and early-period power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import comb
from typing import Dict, List, Literal, Optional
from .structure import SEDCandidate

# =========================================================
# TEST STATISTIC
# =========================================================

def test_statistic(e: np.ndarray, method: str = "mean_abs") -> float:
    """Aggregate residual vector into a scalar test statistic."""
    if method == "mean_abs":
        return float(np.mean(np.abs(e)))
    elif method == "mean":
        return float(np.mean(e))
    elif method == "rms":
        return float(np.sqrt(np.mean(e ** 2)))
    else:
        raise ValueError(f"Unknown statistic method: '{method}'")

# =========================================================
# NULL DISTRIBUTION (Monte Carlo)
# =========================================================

def compute_null_distribution(
    full_series: np.ndarray,
    n_post: int,
    n_sims: int = 100000,
    seed: Optional[int] = None
) -> np.ndarray:
    """Monte Carlo permutation null distribution of S(e)."""
    rng = np.random.default_rng(seed)
    n_total = len(full_series)
    null_stats = np.zeros(n_sims)
    
    for i in range(n_sims):
        idx = rng.choice(n_total, size=n_post, replace=False)
        null_stats[i] = test_statistic(full_series[idx])
        
    null_stats.sort()
    return null_stats

def critical_value_from_null(null_stats: np.ndarray, alpha: float) -> float:
    """One-sided upper critical value c*."""
    return float(np.quantile(null_stats, 1 - alpha))

# =========================================================
# NOISE IMPUTATION
# =========================================================

PostImputation = Literal["mean", "max", "double_max"]

def impute_noise_level(residuals_B: np.ndarray, method: PostImputation) -> float:
    """Impute post-period noise scalar from blank residuals."""
    abs_resid = np.abs(residuals_B)
    if method == "mean":
        return float(np.mean(abs_resid))
    elif method == "max":
        return float(np.max(abs_resid))
    elif method == "double_max":
        return float(2.0 * np.max(abs_resid))
    else:
        raise ValueError(f"Unknown post_imputation: '{method}'")

# =========================================================
# ANALYTICAL MDE (The Solver)
# =========================================================

def _analytical_mde(
    residuals_B: np.ndarray,
    synth_treated: np.ndarray,
    n_post: int,
    alpha: float,
    noise_level: float,
    statistic: str,
    n_sims: int = 100000,
    seed: Optional[int] = None
) -> Dict:
    """Computes MDE analytically using the inverse of the test statistic."""
    n_B = len(residuals_B)
    baseline = float(np.mean(synth_treated[-n_post:]))
    if abs(baseline) < 1e-8:
        baseline = 1e-8

    n_perms_exact = comb(n_B + n_post, n_post)
    p_value_lb = 1.0 / n_perms_exact

    post_null = np.full(n_post, noise_level)
    full_series_null = np.concatenate([residuals_B, post_null])
    
    null_stats = compute_null_distribution(full_series_null, n_post, n_sims, seed)
    c_star = critical_value_from_null(null_stats, alpha)

    # tau* = (c* - noise) / baseline
    mde_tau = max(0.0, (c_star - noise_level) / baseline)
    
    return {
        "mde_tau": float(mde_tau),
        "mde_pct": float(100.0 * mde_tau),
        "baseline": baseline,
        "critical_stat": float(c_star),
        "n_perms_exact": n_perms_exact,
        "n_sims": n_sims,
        "p_value_lb": p_value_lb,
    }

# =========================================================
# PUBLIC INTERFACE & BATCH
# =========================================================

def compute_detectability_curve(
    candidate: SEDCandidate,
    n_post_grid: List[int],
    alpha: float = 0.05,
    n_sims: int = 100000,
    post_imputation: PostImputation = "mean",
    statistic: str = "mean_abs",
    seed: Optional[int] = None
) -> Dict:
    """Computes MDE for a range of post-period lengths."""
    residuals_B = np.asarray(candidate.predictions.residuals_B)
    synth_treated = np.asarray(candidate.predictions.synthetic_treated)
    n_B = len(residuals_B)
    noise_level = impute_noise_level(residuals_B, post_imputation)
    
    curve, details = {}, {}
    for n_post in n_post_grid:
        feasibility = check_inference_feasibility(n_B, n_post, alpha)
        if not feasibility["feasible"]:
            curve[n_post] = np.nan
            continue

        res = _analytical_mde(residuals_B, synth_treated, n_post, alpha, 
                               noise_level, statistic, n_sims, seed)
        curve[n_post] = res["mde_tau"]
        details[n_post] = res

    return {
        "curve": curve,
        "details": details,
        "n_post_10pct": next((k for k, v in curve.items() if v <= 0.10), np.nan),
        "n_post_5pct": next((k for k, v in curve.items() if v <= 0.05), np.nan),
        "noise_level": noise_level,
    }

def run_mde_analysis(
    candidates: List[SEDCandidate],
    n_post_grid: Optional[List[int]] = None,
    alpha: float = 0.05,
    n_sims: int = 100000,
    post_imputation: PostImputation = "mean",
    statistic: str = "mean_abs",
    seed: Optional[int] = None
) -> List[SEDCandidate]:
    if n_post_grid is None:
        n_post_grid = list(range(2, 9))
    for cand in candidates:
        cand.mde_results = compute_detectability_curve(
            cand, n_post_grid, alpha, n_sims, post_imputation, statistic, seed
        )
    return candidates

def check_inference_feasibility(n_B: int, n_post: int, alpha: float = 0.05) -> Dict:
    n_perms = comb(n_B + n_post, n_post)
    p_lb = 1.0 / n_perms
    return {"n_perms": n_perms, "p_value_lb": p_lb, "feasible": p_lb < alpha}

# =========================================================
# RECOMMENDATION & TABLES
# =========================================================

def mde_summary_table(candidates: List[SEDCandidate]) -> pd.DataFrame:
    """Summary of MDE percentages for weeks 2-8."""
    rows = []
    for cand in candidates:
        r = cand.mde_results or {}
        details = r.get("details", {})
        row = {
            "tuple_id": cand.identification.tuple_id,
            "nmse_B": cand.losses.nmse_B,
            "noise_level": r.get("noise_level", np.nan),
        }
        for week in range(2, 9):
            row[f"mde_{week}w"] = details.get(week, {}).get("mde_pct", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("nmse_B").reset_index(drop=True) if not df.empty else df

def rank_candidates(candidates: List[SEDCandidate], w_bias: float = 0.5) -> pd.DataFrame:
    """
    Ranks candidates by balancing historical fit and early detectability (wks 2-4).
    Uses Min-Max normalization to identify the Pareto-optimal recommendations.
    """
    df = mde_summary_table(candidates)
    if df.empty: return df

    # Heuristic: Average MDE in the early window (weeks 2, 3, 4)
    early_cols = [c for c in ['mde_2w', 'mde_3w', 'mde_4w'] if c in df.columns]
    df['early_mde_avg'] = df[early_cols].mean(axis=1)

    # Min-Max Normalization (0 = best in pool, 1 = worst in pool)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    df['norm_bias'] = normalize(df['nmse_B'])
    df['norm_power'] = normalize(df['early_mde_avg'])

    # Weighted Score (Lower is better)
    w_power = 1.0 - w_bias
    df['sed_score'] = (w_bias * df['norm_bias']) + (w_power * df['norm_power'])

    return df.sort_values('sed_score').reset_index(drop=True)
