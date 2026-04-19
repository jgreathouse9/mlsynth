"""
Fast_scm_power_helpers.py

Minimum Detectable Effect (MDE) analysis for Synthetic Experimental Design,
following the permutation inference procedure of Abadie and Zhao (2021) as
implemented in Vives-i-Bastida (2022).

Key design choices aligned with the paper:
  - Test statistic : S(e) = mean(|e_t|)  [mean absolute residual, eq. in Section 2]
  - Null distribution: exact permutation over all C(n_B + n_post, n_post) combinations
    of the pooled blank + post residuals  [Section 2, permutation p-value]
  - MDE: smallest tau s.t. the permutation p-value <= alpha
  - Imputation of unobserved post noise: mean / max / 2*max of blank residuals,
    matching Table 3 of Vives-i-Bastida (2022)
  - tau expressed as a fraction of the synthetic treated baseline (scale-invariant)

NOT used here (deviations from original code corrected):
  - Bootstrap resampling for the null  (was: rng.choice comparison)
  - Sigma-normalized mean statistic    (was: mean(x) / sigma)
  - alpha unused in power decision     (was: T_alt > T_null coin-flip)
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .structure import SEDCandidate


# =========================================================
# TEST STATISTIC  (Vives-i-Bastida 2022, Section 2)
# =========================================================

def test_statistic(e: np.ndarray) -> float:
    """
    Mean absolute residual statistic.

        S(e) = (1 / |T|) * sum_t |e_t|

    Parameters
    ----------
    e : np.ndarray, shape (n,)
        Residual vector (effects or placebo effects).

    Returns
    -------
    float
        Mean absolute value of e.

    Notes
    -----
    Matches S(e) in Vives-i-Bastida (2022) Section 2 and Abadie & Zhao (2021).
    """
    return float(np.mean(np.abs(e)))


# =========================================================
# PERMUTATION P-VALUE
# =========================================================

def permutation_pvalue(
    full_series: np.ndarray,
    n_post: int,
    obs_stat: float,
) -> float:
    """
    Exact permutation p-value over all C(n_total, n_post) index combinations.

    The null hypothesis is that the post-period residuals are exchangeable with
    the blank-period residuals.  Under H0, every assignment of n_post slots out
    of the pooled series is equally likely.

    Parameters
    ----------
    full_series : np.ndarray, shape (n_B + n_post,)
        Pooled residuals: blank period residuals concatenated with post-period
        residuals (possibly with imputed noise under the alternative).
    n_post : int
        Number of post-treatment periods.
    obs_stat : float
        Observed test statistic S(e_post) computed on the true post slots
        (i.e. the last n_post elements of full_series).

    Returns
    -------
    float
        Permutation p-value: fraction of permutations with S >= obs_stat.

    Notes
    -----
    - p-value is bounded below by 1 / C(n_B + n_post, n_post).
    - With n_B=4, n_post=3 this gives 1/C(7,3) = 1/35 ≈ 0.0286, safely below 0.05.
    - With n_B=3, n_post=3 this gives 1/C(6,3) = 1/20 = 0.05, at the boundary.
    """
    n_total = len(full_series)
    all_combos = combinations(range(n_total), n_post)

    count_geq = 0
    n_perms = 0

    for combo in all_combos:
        s = test_statistic(full_series[list(combo)])
        if s >= obs_stat:
            count_geq += 1
        n_perms += 1

    return count_geq / n_perms


# =========================================================
# CORE MDE
# =========================================================

PostImputation = Literal["mean", "max", "double_max"]


def compute_mde_for_candidate(
    candidate: SEDCandidate,
    n_post: int,
    tau_grid: np.ndarray,
    alpha: float = 0.05,
    power_target: float = 0.8,
    post_imputation: PostImputation = "mean",
) -> Dict:
    """
    Compute the Minimum Detectable Effect (MDE) for a single SED candidate
    using the exact permutation test of Abadie & Zhao (2021).

    The procedure mirrors Vives-i-Bastida (2022) Table 3:

    1. Treat blank-period effect residuals û_t as the noise pool.
    2. For each tau on the grid, construct an augmented post-period series
       as (imputed_noise + tau * baseline) for t in post.
    3. Pool blank residuals with augmented post residuals.
    4. Compute the permutation p-value over all C(n_B + n_post, n_post)
       index combinations.
    5. Power = 1 if p-value <= alpha, else 0.
    6. MDE = smallest tau achieving power >= power_target.

    Parameters
    ----------
    candidate : SEDCandidate
        A fitted SED candidate with predictions.residuals_B and
        predictions.synthetic_treated populated.
    n_post : int
        Number of post-treatment periods (T - T0).
    tau_grid : np.ndarray
        Grid of effect sizes expressed as fractions of the synthetic treated
        baseline mean (scale-invariant).  E.g. 0.10 = 10% effect.
    alpha : float, optional
        Significance level for the permutation test (default 0.05).
    power_target : float, optional
        Desired power level for MDE determination (default 0.80).
    post_imputation : {'mean', 'max', 'double_max'}, optional
        How to impute unobserved post-period noise from blank residuals:
          - 'mean'       : noise = mean(|û_t|) for t in B  (conservative)
          - 'max'        : noise = max(|û_t|)  for t in B
          - 'double_max' : noise = 2 * max(|û_t|)          (pessimistic)
        Matches the three columns of Table 3 in Vives-i-Bastida (2022).

    Returns
    -------
    dict with keys:
        mde_tau        : float  – MDE as fraction of baseline
        mde_abs        : float  – MDE in outcome units (tau * baseline)
        mde_pct        : float  – MDE as percentage of baseline
        sigma          : float  – std of blank residuals (noise floor)
        baseline       : float  – mean synthetic treated over post period
        power_curve    : dict   – {tau: power} for all tau in tau_grid
        n_perms        : int    – number of permutations C(n_B+n_post, n_post)
        p_value_lb     : float  – lower bound on permutation p-value (1/n_perms)
        alpha          : float
        power_target   : float
        post_imputation: str

    Notes
    -----
    - The permutation lower bound 1/n_perms must be < alpha to allow rejection.
      With n_B=4, n_post=3: 1/C(7,3) ≈ 0.029 < 0.05  ✓
      With n_B=3, n_post=3: 1/C(6,3) = 0.05  (boundary — inference is fragile)
      With n_B=2, n_post=3: 1/C(5,3) ≈ 0.10 > 0.05  ✗  (cannot reject at 5%)
    - tau_grid values should be positive (one-sided test for positive effects).
      For two-sided use, pass abs(tau) values and interpret accordingly.
    """
    residuals_B = np.asarray(candidate.predictions.residuals_B)
    n_B = len(residuals_B)

    # ---- noise floor diagnostics ----
    sigma = float(np.std(residuals_B, ddof=1)) or 1e-8

    # ---- baseline: mean synthetic treated over post window ----
    synth_treated_full = np.asarray(candidate.predictions.synthetic_treated)
    synth_post = synth_treated_full[-n_post:]
    baseline = float(np.mean(synth_post))
    if abs(baseline) < 1e-8:
        baseline = 1e-8  # guard against zero baseline

    # ---- imputed post noise scalar (one value repeated n_post times) ----
    abs_resid = np.abs(residuals_B)
    if post_imputation == "mean":
        noise_level = float(np.mean(abs_resid))
    elif post_imputation == "max":
        noise_level = float(np.max(abs_resid))
    elif post_imputation == "double_max":
        noise_level = float(2.0 * np.max(abs_resid))
    else:
        raise ValueError(f"Unknown post_imputation: '{post_imputation}'. "
                         "Choose 'mean', 'max', or 'double_max'.")

    # ---- combinatorial size ----
    n_perms = comb(n_B + n_post, n_post)
    p_value_lb = 1.0 / n_perms

    power_curve: Dict[float, float] = {}

    for tau in tau_grid:
        effect_abs = float(tau) * baseline

        # Post-period residuals under H_alt: imputed noise + constant effect
        post_residuals = np.full(n_post, noise_level + effect_abs)

        # Pool: [blank residuals | post residuals under alt]
        full_series = np.concatenate([residuals_B, post_residuals])

        # Observed stat: S on the true post slots (last n_post)
        obs_stat = test_statistic(full_series[n_B:])

        p_val = permutation_pvalue(full_series, n_post, obs_stat)

        power_curve[float(tau)] = 1.0 if p_val <= alpha else 0.0

    # ---- MDE: smallest tau with power >= power_target ----
    mde_tau = min(
        (t for t, p in power_curve.items() if p >= power_target),
        default=np.nan
    )
    mde_abs = mde_tau * baseline if np.isfinite(mde_tau) else np.nan
    mde_pct = 100.0 * mde_tau if np.isfinite(mde_tau) else np.nan

    return {
        "mde_tau":         float(mde_tau) if np.isfinite(mde_tau) else np.nan,
        "mde_abs":         float(mde_abs) if np.isfinite(mde_abs) else np.nan,
        "mde_pct":         float(mde_pct) if np.isfinite(mde_pct) else np.nan,
        "sigma":           sigma,
        "baseline":        baseline,
        "power_curve":     power_curve,
        "n_perms":         n_perms,
        "p_value_lb":      p_value_lb,
        "alpha":           alpha,
        "power_target":    power_target,
        "post_imputation": post_imputation,
    }


# =========================================================
# BATCH
# =========================================================

def run_mde_analysis(
    candidates: List[SEDCandidate],
    n_post: int = 3,
    tau_grid: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    power_target: float = 0.8,
    post_imputation: PostImputation = "mean",
) -> List[SEDCandidate]:
    """
    Run MDE analysis for all candidates in-place, attaching results to
    candidate.mde_results.

    Parameters
    ----------
    candidates : list of SEDCandidate
    n_post : int
        Number of post-treatment periods.
    tau_grid : np.ndarray, optional
        Effect size grid (fractions of baseline).  Defaults to a grid that
        is dense in the range [0.01, 0.50] where detectable effects typically
        fall for moderate n_B and n_post.
    alpha : float
        Significance level (default 0.05).
    power_target : float
        Power threshold for MDE (default 0.80).
    post_imputation : {'mean', 'max', 'double_max'}
        Noise imputation strategy for unobserved post periods (default 'mean').

    Returns
    -------
    list of SEDCandidate
        Same list, with mde_results populated on each candidate.

    Notes
    -----
    Pass post_imputation='max' or 'double_max' to replicate the more
    conservative columns of Table 3 in Vives-i-Bastida (2022).
    """
    if tau_grid is None:
        tau_grid = np.concatenate([
            np.linspace(0.01,  0.10, 30),   # fine grid where MDE often lands
            np.linspace(0.10,  0.50, 40),   # coarser above 10%
            np.linspace(0.50,  1.50, 20),   # tail for high-variance outcomes
        ])

    for cand in candidates:
        cand.mde_results = compute_mde_for_candidate(
            candidate=cand,
            n_post=n_post,
            tau_grid=tau_grid,
            alpha=alpha,
            power_target=power_target,
            post_imputation=post_imputation,
        )

    return candidates


# =========================================================
# SUMMARY TABLE
# =========================================================

def mde_summary_table(candidates: List[SEDCandidate]) -> pd.DataFrame:
    """
    Build a summary DataFrame of MDE results across all candidates.

    Columns
    -------
    tuple_id, loss_E, nmse_E, nmse_B,
    sigma, baseline, n_perms, p_value_lb,
    mde_tau, mde_abs, mde_pct,
    post_imputation
    """
    rows = []

    for cand in candidates:
        res = cand.mde_results or {}

        rows.append({
            "tuple_id":        cand.identification.tuple_id,
            "loss_E":          cand.losses.loss_E,
            "nmse_E":          cand.losses.nmse_E,
            "nmse_B":          cand.losses.nmse_B,
            "sigma":           res.get("sigma",           np.nan),
            "baseline":        res.get("baseline",        np.nan),
            "n_perms":         res.get("n_perms",         np.nan),
            "p_value_lb":      res.get("p_value_lb",      np.nan),
            "mde_tau":         res.get("mde_tau",         np.nan),
            "mde_abs":         res.get("mde_abs",         np.nan),
            "mde_pct":         res.get("mde_pct",         np.nan),
            "post_imputation": res.get("post_imputation", np.nan),
        })

    return pd.DataFrame(rows)


# =========================================================
# POWER CURVE EXTRACTION
# =========================================================

def extract_power_curves(candidates: List[SEDCandidate]) -> Dict[str, Dict]:
    """
    Extract power curves from all candidates.

    Returns
    -------
    dict
        {tuple_id: {tau: power, ...}}
    """
    return {
        cand.identification.tuple_id: cand.mde_results.get("power_curve", {})
        for cand in candidates
    }


# =========================================================
# FEASIBILITY CHECK
# =========================================================

def check_inference_feasibility(n_B: int, n_post: int, alpha: float = 0.05) -> Dict:
    """
    Check whether the permutation test can achieve the target significance level
    given the blank period and post-treatment period lengths.

    Parameters
    ----------
    n_B : int
        Number of blank period observations.
    n_post : int
        Number of post-treatment periods.
    alpha : float
        Target significance level (default 0.05).

    Returns
    -------
    dict with keys:
        n_perms      : int   – total permutations C(n_B + n_post, n_post)
        p_value_lb   : float – minimum achievable p-value (1 / n_perms)
        feasible     : bool  – True if p_value_lb < alpha
        warning      : str   – human-readable guidance

    Examples
    --------
    >>> check_inference_feasibility(n_B=4, n_post=3)
    # n_perms=35, p_value_lb≈0.029, feasible=True

    >>> check_inference_feasibility(n_B=3, n_post=3)
    # n_perms=20, p_value_lb=0.05, feasible=False (boundary)

    >>> check_inference_feasibility(n_B=2, n_post=3)
    # n_perms=10, p_value_lb=0.10, feasible=False
    """
    n_perms = comb(n_B + n_post, n_post)
    p_lb = 1.0 / n_perms
    feasible = p_lb < alpha

    if feasible:
        warning = (
            f"Inference feasible: p-value lower bound {p_lb:.4f} < alpha={alpha}. "
            f"Total permutations: {n_perms}."
        )
    else:
        needed_B = 1
        while comb(needed_B + n_post, n_post) < int(1 / alpha) + 1:
            needed_B += 1
        warning = (
            f"Inference NOT feasible: p-value lower bound {p_lb:.4f} >= alpha={alpha}. "
            f"Need at least n_B={needed_B} blank periods for n_post={n_post} "
            f"to achieve alpha={alpha}."
        )

    return {
        "n_perms":    n_perms,
        "p_value_lb": p_lb,
        "feasible":   feasible,
        "warning":    warning,
    }