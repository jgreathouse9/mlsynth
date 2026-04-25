from __future__ import annotations

import numpy as np
import pandas as pd
from math import comb
from typing import Dict, List, Literal, Optional, Tuple
from .structure import SEDCandidate

# =========================================================
# 1. TEST STATISTIC DEFINITIONS
# =========================================================

def test_statistic(e: np.ndarray, method: str = "mean_abs") -> float:
    """
    Compute a scalar test statistic from a vector of residuals.

    In the SED framework, this represents the 'magnitude of the gap'
    between the treated unit and its synthetic counterfactual.

    Parameters
    ----------
    e : np.ndarray
        Vector of residuals (Actual - Synthetic).
    method : {"mean_abs", "mean", "rms"}
        - "mean_abs": Average of absolute gaps. Standard for placebo tests.
        - "mean": Signed average. Risk of positive/negative gaps canceling out.
        - "rms": Root Mean Square. Penalizes large outliers more heavily.

    Returns
    -------
    float
        The calculated gap magnitude.
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
# 2. PERMUTATION-BASED NULL DISTRIBUTION
# =========================================================

def compute_null_distribution(
        full_series: np.ndarray,
        n_post: int,
        n_sims: int = 100000,
        seed: Optional[int] = None
) -> np.ndarray:
    """
    Estimate the distribution of the gap statistic under the 'Sharp Null'.

    The Sharp Null assumes the treatment has zero effect. We simulate this
    by randomly 'shuffling' which time periods are considered 'post-treatment'
    and measuring the resulting gap.

    Parameters
    ----------
    full_series : np.ndarray
        The combined history of residuals (Training + Validation).
    n_post : int
        The number of periods to be evaluated (the 'experiment duration').
    n_sims : int
        Number of Monte Carlo permutations. Higher = smoother p-values.

    Returns
    -------
    np.ndarray
        Sorted distribution of 'placebo' test statistics.
    """
    rng = np.random.default_rng(seed)
    n_total = len(full_series)
    null_stats = np.zeros(n_sims)

    # Randomly sample n_post periods from the history to build the null
    for i in range(n_sims):
        idx = rng.choice(n_total, size=n_post, replace=False)
        null_stats[i] = test_statistic(full_series[idx])

    null_stats.sort()
    return null_stats


# =========================================================
# 3. NOISE IMPUTATION (PRE-TREATMENT TO POST-TREATMENT)
# =========================================================

PostImputation = Literal["mean", "max", "double_max"]

def impute_noise_level(residuals_B: np.ndarray, method: PostImputation = "mean") -> float:
    """
    Predict the magnitude of future 'noise' based on validation period errors.

    If your model fits perfectly in the blank period (the 'one-line' fit),
    the noise level will be extremely low, leading to highly sensitive MDEs.

    Parameters
    ----------
    residuals_B : np.ndarray
        Errors from the Out-of-Sample (OOS) validation period.
    method :
        - "mean": Expected future noise equals average historical error.
        - "max": Conservative. Future noise equals the single worst historical day.
    """
    abs_resid = np.abs(residuals_B)
    if method == "mean":
        return float(np.mean(abs_resid))
    elif method == "max":
        return float(np.max(abs_resid))
    elif method == "double_max":
        return float(2.0 * np.max(abs_resid))
    else:
        raise ValueError(f"Unknown post_imputation method: '{method}'")


# =========================================================
# 4. MINIMUM DETECTABLE EFFECT (MDE) CALCULATION
# =========================================================

def _analytical_mde(
        residuals_B: np.ndarray,
        synth_treated: np.ndarray,
        n_post: int,
        alpha: float = 0.05,
        noise_level: float = None,
        statistic: str = "mean_abs",
        n_sims: int = 100000,
        seed: Optional[int] = None
) -> Dict:
    """
    Inverts the permutation test to solve for the detectable signal.

    Mathematically, it finds the lift (tau) such that:
    P(Placebo_Gap > Observed_Gap + tau | Null) < alpha.
    """
    if noise_level is None:
        noise_level = impute_noise_level(residuals_B, "mean")

    n_B = len(residuals_B)
    # The average value of the unit in the post-treatment window (the denominator for %)
    baseline = float(np.mean(synth_treated[-n_post:])) if n_post > 0 else 1.0
    baseline = max(baseline, 1e-8) 

    # FEASIBILITY: Can we even run a 5% test? (Calculates exact permutation count)
    n_perms_exact = comb(n_B + n_post, n_post)
    p_value_lb = 1.0 / n_perms_exact

    if p_value_lb >= alpha:
        return {"feasible": False, "p_value_lb": p_value_lb, "mde_pct": np.nan}

    # Simulate the Null Distribution
    post_null = np.full(n_post, noise_level)
    full_series_null = np.concatenate([residuals_B, post_null])
    null_stats = compute_null_distribution(full_series_null, n_post, n_sims, seed)
    
    # Calculate Critical Value (the 1-alpha quantile)
    c_star = float(np.quantile(null_stats, 1 - alpha))

    # SOLVE FOR TAU: 
    # tau = (Critical_Statistic - Background_Noise) / Baseline_Level
    mde_tau = max(0.0, (c_star - noise_level) / baseline)

    return {
        "mde_tau": float(mde_tau),
        "mde_pct": float(100.0 * mde_tau),
        "baseline": baseline,
        "critical_stat": c_star,
        "feasible": True
    }


# =========================================================
# 5. SELECTION ENGINE: VALIDITY -> POWER
# =========================================================

def select_best_tuple(
        candidates: List[SEDCandidate],
        relative_delta: float = 1.5,
        target_mde_horizon: str = "early_mde_avg"
) -> Tuple[SEDCandidate, pd.DataFrame]:
    """
    Selects the optimal design using a Validity-First rule.

    Algorithm:
    1. Find the best OOS fit (lowest NMSE_B).
    2. Keep all candidates within 'relative_delta' (e.g., 50% worse fit than best).
    3. Among those valid candidates, choose the one with the lowest MDE (highest power).
    """
    df = mde_summary_table(candidates).copy()
    if df.empty: raise ValueError("No candidates found.")

    # 1. Establish the 'Validity Threshold'
    best_nmse = df['nmse_B'].min()
    threshold = best_nmse * relative_delta

    # 2. Filter for validity
    passing = df[df['nmse_B'] <= threshold].copy()

    # 3. Optimize for power
    if target_mde_horizon == "early_mde_avg":
        early_cols = [f"mde_{w}w" for w in range(2, 5) if f"mde_{w}w" in passing.columns]
        passing['early_mde_avg'] = passing[early_cols].mean(axis=1)

    passing = passing.sort_values(target_mde_horizon).reset_index(drop=True)

    winner_id = passing.iloc[0]['tuple_id']
    winner = next(c for c in candidates if c.identification.tuple_id == winner_id)

    print(f"Selection Result:")
    print(f" > Winner: {winner_id}")
    print(f" > Fit (NMSE_B): {passing.iloc[0]['nmse_B']:.5f}")
    print(f" > MDE: {passing.iloc[0][target_mde_horizon]:.2f}%")

    return winner, passing
