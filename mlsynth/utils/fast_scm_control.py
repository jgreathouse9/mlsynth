from typing import List, Dict, Tuple
import numpy as np
from .fast_scm_control_helpers import solve_control_qp, compute_effect_series, compute_nmse
from scipy.stats import norm



import numpy as np
from typing import Dict, Tuple

def compute_mde_and_pvalue(
    effects: np.ndarray,
    B_idx: np.ndarray,
    post_idx: np.ndarray,
    n_permutations: int = 500,
    alpha: float = 0.05,
    target_power: float = 0.80,
    statistic: str = "mean"
) -> Dict:
    """Two-sided permutation test + MDE for one treated tuple (null simulation safe)."""
    post_effects = effects[post_idx]
    
    if statistic == "mean":
        obs_stat = float(np.mean(post_effects))
    elif statistic == "rms":
        obs_stat = float(np.sqrt(np.mean(post_effects ** 2)))
    else:
        raise ValueError("statistic must be 'mean' or 'rms'")

    # Placebo distribution from blank period residuals
    blank_residuals = effects[B_idx].copy()
    n_post = len(post_idx)

    placebo_stats = np.zeros(n_permutations)
    rng = np.random.default_rng()   # reproducible and faster

    for i in range(n_permutations):
        # Random sign flips → symmetric under null
        signs = rng.choice([-1.0, 1.0], size=len(blank_residuals))
        placebo_blank = blank_residuals * signs

        # Resample to match post-period length
        placebo_post = rng.choice(placebo_blank, size=n_post, replace=True)

        if statistic == "mean":
            placebo_stats[i] = np.mean(placebo_post)
        else:
            placebo_stats[i] = np.sqrt(np.mean(placebo_post ** 2))

    placebo_sd = float(np.std(placebo_stats, ddof=1))

    # === TWO-SIDED p-value (correct for null) ===
    p_value = float((np.sum(np.abs(placebo_stats) >= np.abs(obs_stat)) + 1) 
                    / (n_permutations + 1))

    # === MDE: smallest |effect| detectable with target_power at alpha ===
    z_alpha = norm.ppf(1 - alpha / 2)      # two-sided
    z_beta = norm.ppf(target_power)
    mde = (z_alpha + z_beta) * placebo_sd

    return {
        "p_value": p_value,
        "mde": float(mde),
        "observed_stat": obs_stat,
        "placebo_sd": placebo_sd,
        "n_permutations": n_permutations,
        "statistic": statistic
    }


def evaluate_candidates(
    candidates: List[Tuple[float, List[int], np.ndarray]],
    X: np.ndarray,
    X_E: np.ndarray,
    Y: np.ndarray,
    f: np.ndarray,
    B_idx: np.ndarray,
    post_idx: np.ndarray,
    lambda_penalty: float
) -> List[Dict]:
    """
    Evaluate candidate treated unit sets using synthetic control methods.

    Parameters
    ----------
    candidates : list of tuples
        Each tuple contains (loss, treated_indices, treated_weights):
            - loss : float
                Pre-computed loss on estimation period (used for ranking candidates)
            - treated_indices : list of int
                Indices of treated units in this candidate set
            - treated_weights : np.ndarray, shape (len(treated_indices),)
                Weights for treated units
    X : np.ndarray, shape (T, N)
        Full data matrix of all units over all time periods.
    X_E : np.ndarray, shape (len(E_idx), N)
        Standardized estimation-period matrix for synthetic control fitting.
    Y : np.ndarray, shape (T, J)
        Observed outcomes for J outcome variables.
    f : np.ndarray, shape (J,)
        Weights for averaging outcomes to form population-level target series.
    B_idx : np.ndarray
        Indices of the backcast / blank period (pre-treatment validation).
    post_idx : np.ndarray
        Indices of the post-treatment period.
    lambda_penalty : float
        Regularization weight for control-unit optimization.

    Returns
    -------
    results : list of dict
        Each dictionary contains information for one candidate treated set:
            - 'treated_idx' : list of int
                Indices of treated units
            - 'treated_weights' : np.ndarray
                Fitted weights for treated units
            - 'control_weights' : np.ndarray
                Fitted synthetic control weights for non-treated units
            - 'loss_E' : float
                Candidate loss from estimation period
            - 'nmse_B' : float
                Normalized mean squared error on blank (validation) period
            - 'effects' : np.ndarray, shape (T,)
                Treatment effect series (treated - synthetic control)
            - 'conformal_lower' : float
                Lower bound of conformal interval based on blank period
            - 'conformal_upper' : float
                Upper bound of conformal interval based on blank period
            - 'conformal_width' : float
                Width of the conformal interval
            - 'conformal_coverage' : float
                Proportion of post-period effects contained in conformal interval

    Notes
    -----
    - Computes synthetic control weights for each candidate treated set.
    - Measures goodness-of-fit on backcast period (nmse_B).
    - Computes post-treatment effects for inference.
    - Provides simple conformal interval estimates for post-treatment effects using blank period residuals.
    - Designed to be called after a candidate set selection (e.g., branch-and-bound top-K).
    """

    results = []

    # FIXED: Target = population average over the Y outcome units ONLY
    J = Y.shape[1]                                   # number of outcome variables
    target = X[:, :J] @ f[:J]                        # use only first J weights

    for loss, idx, w in candidates:
        # Build treated synthetic unit on estimation period
        treated_vec = X_E[:, idx] @ w

        # Solve for control weights
        v = solve_control_qp(X_E, treated_vec, idx, lambda_penalty)
        if v is None:
            continue

        # Fit on blank period
        nmse_B = compute_nmse(X, w, target, B_idx, treated_idx=idx)

        # Treatment effects
        effects = X[:, idx] @ w - X @ v

        mde_info = compute_mde_and_pvalue(
            effects, 
            B_idx, 
            post_idx, 
            n_permutations=500,   # pass from main function
            statistic="mean"                 # or "rms" — I recommend "mean"
        )

        # Conformal inference using blank periods
        residuals = effects[B_idx]
        q = np.quantile(np.abs(residuals), 0.95)

        lower = -q
        upper = q
        width = upper - lower

        coverage = np.mean(
            (effects[post_idx] >= lower) &
            (effects[post_idx] <= upper)
        )

        results.append({
            "treated_idx": idx.copy(),
            "treated_weights": w.copy(),
            "control_weights": v.copy(),
            "synthetic_control": X @ v,
            "synthetic_treated": X[:, idx] @ w,
            "loss_E": float(loss),
            "nmse_B": float(nmse_B),
            "effects": effects,
            "conformal_lower": lower,
            "conformal_upper": upper,
            "conformal_width": width,
            "conformal_coverage": coverage,
            **mde_info
        })

    return results