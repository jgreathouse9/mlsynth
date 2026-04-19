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
    """
    Aggregate a residual vector into a scalar test statistic.

    Parameters
    ----------
    e : np.ndarray, shape (T,)
        Residual vector (e.g., treatment effects or placebo residuals).
    method : {"mean_abs", "mean", "rms"}, default="mean_abs"
        Aggregation method:
        - "mean_abs": mean absolute deviation
        - "mean": signed mean
        - "rms": root mean square

    Returns
    -------
    stat : float
        Scalar test statistic.

    Raises
    ------
    ValueError
        If an unknown method is provided.

    Notes
    -----
    - This statistic defines the rejection region for permutation inference.
    - "mean_abs" is robust to sign and commonly used in SCM placebo tests.
    """
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
    n_sims: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Estimate the permutation null distribution of the test statistic via Monte Carlo.

    Parameters
    ----------
    full_series : np.ndarray, shape (T_total,)
        Combined series of residuals under the null (pre + imputed post).
    n_post : int
        Number of time points assigned to the pseudo post-treatment period.
    n_sims : int, default=1000
        Number of Monte Carlo permutations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    null_stats : np.ndarray, shape (n_sims,)
        Sorted array of simulated test statistics.

    Notes
    -----
    - Each simulation randomly selects `n_post` indices without replacement.
    - This approximates the permutation distribution used in SCM inference.
    - Sorting enables efficient quantile lookup for critical values.
    """
    rng = np.random.default_rng(seed)
    n_total = len(full_series)
    null_stats = np.zeros(n_sims)
    
    for i in range(n_sims):
        idx = rng.choice(n_total, size=n_post, replace=False)
        null_stats[i] = test_statistic(full_series[idx])
        
    null_stats.sort()
    return null_stats

def critical_value_from_null(null_stats: np.ndarray, alpha: float) -> float:
    """
    Compute the upper critical value from a null distribution.

    Parameters
    ----------
    null_stats : np.ndarray
        Sorted null distribution of the test statistic.
    alpha : float
        Significance level.

    Returns
    -------
    c_star : float
        (1 - alpha) quantile of the null distribution.

    Notes
    -----
    - Defines the rejection threshold for a one-sided test.
    """
    return float(np.quantile(null_stats, 1 - alpha))

# =========================================================
# NOISE IMPUTATION
# =========================================================

PostImputation = Literal["mean", "max", "double_max"]

def impute_noise_level(residuals_B: np.ndarray, method: PostImputation) -> float:
    """
    Estimate post-treatment noise level from pre-treatment residuals.

    Parameters
    ----------
    residuals_B : np.ndarray
        Residuals from the baseline (pre-treatment validation) period.
    method : {"mean", "max", "double_max"}
        Method for estimating noise magnitude.

    Returns
    -------
    noise_level : float
        Scalar noise estimate used to construct null post-period values.

    Notes
    -----
    - "mean": average absolute residual (typical noise level)
    - "max": worst-case residual
    - "double_max": conservative upper bound
    - This value is used to simulate post-treatment noise under the null.
    """
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
    """
    Compute the minimum detectable effect (MDE) via inversion of a permutation test.

    Parameters
    ----------
    residuals_B : np.ndarray, shape (T_B,)
        Residuals from the baseline (pre-treatment) period.
    synth_treated : np.ndarray, shape (T,)
        Synthetic treated time series.
    n_post : int
        Length of the post-treatment period.
    alpha : float
        Significance level for the test.
    noise_level : float
        Imputed noise level for post-treatment residuals under the null.
    statistic : str
        Test statistic method (passed to `test_statistic`).
    n_sims : int, default=100000
        Number of Monte Carlo samples for null distribution.
    seed : int, optional
        Random seed.

    Returns
    -------
    result : dict
        Dictionary containing:
        - mde_tau : float
            Minimum detectable effect (fraction of baseline).
        - mde_pct : float
            MDE as percentage.
        - baseline : float
            Mean treated level over post period.
        - critical_stat : float
            Critical value c* from null distribution.
        - n_perms_exact : int
            Total number of exact permutations.
        - n_sims : int
            Number of Monte Carlo draws used.
        - p_value_lb : float
            Minimum attainable p-value.

    Notes
    -----
    - Implements analytical inversion:
        tau* = (c* - noise_level) / baseline
    - Assumes additive constant treatment effect.
    - Uses Monte Carlo approximation to permutation distribution.
    - Baseline is stabilized to avoid division by zero.
    """
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
    """
    Compute MDE as a function of post-treatment duration.

    Parameters
    ----------
    candidate : SEDCandidate
        Evaluated candidate with residuals and predictions.
    n_post_grid : list of int
        Grid of post-period lengths to evaluate.
    alpha : float, default=0.05
        Significance level.
    n_sims : int, default=100000
        Number of Monte Carlo samples.
    post_imputation : {"mean", "max", "double_max"}, default="mean"
        Method for imputing post-period noise.
    statistic : str, default="mean_abs"
        Test statistic method.
    seed : int, optional
        Random seed.

    Returns
    -------
    results : dict
        Contains:
        - "curve": dict mapping n_post → MDE (tau)
        - "details": dict of full MDE outputs per n_post
        - "n_post_10pct": smallest n_post achieving MDE ≤ 10%
        - "n_post_5pct": smallest n_post achieving MDE ≤ 5%
        - "noise_level": imputed noise level

    Notes
    -----
    - Skips infeasible configurations (insufficient permutation resolution).
    - Provides a “power curve” over experiment duration.
    """
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

    """
    Compute detectability curves for a list of candidates.

    Parameters
    ----------
    candidates : list of SEDCandidate
        Candidates to evaluate.
    n_post_grid : list of int, optional
        Grid of post-period lengths. Defaults to range(2, 9).
    alpha : float, default=0.05
        Significance level.
    n_sims : int, default=100000
        Number of Monte Carlo simulations.
    post_imputation : str
        Noise imputation method.
    statistic : str
        Test statistic method.
    seed : int, optional
        Random seed.

    Returns
    -------
    candidates : list of SEDCandidate
        Same objects with `mde_results` populated.

    Notes
    -----
    - Mutates candidate objects in place.
    """

    
    if n_post_grid is None:
        n_post_grid = list(range(2, 9))
    for cand in candidates:
        cand.mde_results = compute_detectability_curve(
            cand, n_post_grid, alpha, n_sims, post_imputation, statistic, seed
        )
    return candidates




def check_inference_feasibility(n_B: int, n_post: int, alpha: float = 0.05) -> Dict:
    """
    Check whether permutation inference is feasible at a given significance level.

    This function evaluates whether the number of possible permutations in the
    observed pre/post structure is large enough to support hypothesis testing
    at level alpha.

    Parameters
    ----------
    n_B : int
        Number of baseline (pre-treatment) observations.
    n_post : int
        Length of post-treatment period.
    alpha : float, default=0.05
        Significance level for inference feasibility.

    Returns
    -------
    result : dict
        Dictionary containing:
        - n_perms : int
            Total number of valid permutations of (n_B + n_post choose n_post).
        - p_value_lb : float
            Minimum achievable p-value under exact permutation enumeration.
        - feasible : bool
            Whether inference is feasible at level alpha.

    Notes
    -----
    - If p_value_lb >= alpha, no statistically valid rejection region exists
      at the specified significance level.
    """
    n_perms = comb(n_B + n_post, n_post)
    p_lb = 1.0 / n_perms
    return {"n_perms": n_perms, "p_value_lb": p_lb, "feasible": p_lb < alpha}



# =========================================================
# RECOMMENDATION & TABLES
# =========================================================



def mde_summary_table(candidates: List[SEDCandidate]) -> pd.DataFrame:
    """
    Construct a summary table of Minimum Detectable Effect (MDE) results
    across all evaluated synthetic experiment designs.

    Parameters
    ----------
    candidates : list of SEDCandidate
        List of evaluated candidate designs containing MDE results.

    Returns
    -------
    df : pd.DataFrame
        Summary table with columns:
        - tuple_id : identifier for the treated unit configuration
        - nmse_B : baseline fit quality (lower is better)
        - noise_level : estimated noise scale from residuals
        - mde_{k}w : MDE percentage for post-period horizon k (k = 2,...,8)

    Notes
    -----
    - Candidates are sorted by nmse_B (best pre-treatment fit first).
    - Missing MDE values are filled with NaN.
    - This table is used as the input for Pareto ranking and SED scoring.
    """
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
    Rank candidate experimental designs by balancing pre-treatment fit
    and early-period statistical power.

    This function implements a heuristic Pareto-style ranking over evaluated
    Synthetic Experimental Design (SED) candidates by combining:

        (i)  Historical fit quality (NMSE_B)
        (ii) Early-period detectability (MDE over weeks 2–4)

    The goal is to identify designs that achieve a favorable tradeoff between
    low pre-treatment bias and high statistical power.

    Parameters
    ----------
    candidates : list of SEDCandidate
        Evaluated experimental designs produced by the full LEXSCM pipeline.
        Each candidate must contain:
            - losses.nmse_B
            - mde_results (from power analysis stage)

    w_bias : float, default=0.5
        Weight assigned to historical fit (NMSE_B).
        The remaining weight (1 - w_bias) is assigned to power (MDE).

    Returns
    -------
    df : pd.DataFrame
        Ranked table containing:

        - nmse_B
            Pre-treatment fit quality (lower is better)
        - early_mde_avg
            Average MDE over weeks 2–4 (lower implies higher power)
        - norm_bias
            Min-max normalized NMSE_B
        - norm_power
            Min-max normalized early MDE
        - sed_score
            Combined ranking score (lower is better)

    Notes
    -----
    - Uses min-max normalization within the candidate pool.
    - Early-period power is defined as mean MDE over weeks 2, 3, and 4.
    - The SED score is a convex combination:

            sed_score = w_bias * norm_bias + (1 - w_bias) * norm_power

    - This ranking is heuristic and does not guarantee Pareto optimality,
      but performs well empirically for experimental design selection.
    """
    df = mde_summary_table(candidates)
    if df.empty:
        return df

    # Heuristic: average MDE in early post-treatment window (weeks 2–4)
    early_cols = [c for c in ['mde_2w', 'mde_3w', 'mde_4w'] if c in df.columns]
    df['early_mde_avg'] = df[early_cols].mean(axis=1)

    # Min-max normalization helper
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    df['norm_bias'] = normalize(df['nmse_B'])
    df['norm_power'] = normalize(df['early_mde_avg'])

    # Combined SED score (lower is better)
    w_power = 1.0 - w_bias
    df['sed_score'] = (w_bias * df['norm_bias']) + (w_power * df['norm_power'])

    return df.sort_values('sed_score').reset_index(drop=True)
