import numpy as np
from typing import Optional

from .structure import SEDCandidate

def compute_post_inference(candidate, post_idx, alpha: float = 0.05, n_perms: int = 10000, seed: int = 42):
    """
    Compute permutation-based p-values for post-intervention treatment effects.

    This function implements the inferential framework described in Vives-i-Bastida (2022),
    where the null hypothesis (no treatment effect) is tested by comparing observed
    post-treatment gaps to the distribution of gaps during the 'Blank' (backcast) period.

    Parameters
    ----------
    candidate : SEDCandidate
        The candidate experimental design containing predicted effects and residuals.
    post_idx : np.ndarray
        Array of indices corresponding to the post-intervention time periods.
    alpha : float, default=0.05
        Significance level for the hypothesis test.
    n_perms : int, default=10000
        Number of permutations used to approximate the null distribution.
    seed : int, default=42
        Seed for the local random number generator to ensure reproducibility.

    Returns
    -------
    candidate : SEDCandidate
        The updated candidate object with `inference.p_value` and `inference.ate` populated.

    Notes
    -----
    - The test statistic is the average absolute value of the residuals.
    - Residuals from the Blank period (B) and the Post-treatment period are pooled
      under the assumption of exchangeability under the null.
    - A local RNG is used to ensure that parallel execution or multiple calls
      produce deterministic results.
    """

    post_idx = np.asarray(post_idx, dtype=int)

    # Early exit safety for empty post-period
    if post_idx.size == 0:
        candidate.inference.p_value = None
        candidate.inference.ate = 0.0
        return candidate


    # Use a local RNG for thread-safety and reproducibility
    rng = np.random.default_rng(seed)

    e_B = candidate.predictions.residuals_B
    e_post = candidate.predictions.effects[post_idx]

    # The pool represents all non-estimation residuals available for permutation
    pool = np.concatenate([e_B, e_post])
    n_post = len(e_post)
    observed_stat = np.mean(np.abs(e_post))

    # Permutation Loop: How often does random noise look as large as our 'effect'?
    perm_stats = []
    for _ in range(n_perms):
        # replace=False ensures we are truly permuting (shuffling) the indices
        perm_sample = rng.choice(pool, size=n_post, replace=False)
        perm_stats.append(np.mean(np.abs(perm_sample)))

    p_val = np.mean(np.array(perm_stats) >= observed_stat)

    candidate.inference.p_value = p_val
    candidate.inference.ate = np.mean(e_post)

    return candidate

def compute_moving_block_conformal_ci(
    candidate: SEDCandidate,
    post_idx: np.ndarray,
    alpha: float = 0.05,
    block_size: Optional[int] = None,
    seed: int = 42
) -> SEDCandidate:
    """
    Moving Block Conformal Confidence Interval for the Average Treatment Effect (ATE).

    This function implements a time-series-aware conformal inference procedure
    using moving (overlapping) blocks of pre-treatment residuals to construct
    a nonparametric confidence interval for the post-treatment average effect.

    The method accounts for temporal dependence (autocorrelation, seasonality)
    by using block-level conformity scores rather than i.i.d. assumptions.

    Parameters
    ----------
    candidate : SEDCandidate
        Candidate object containing:
        - `predictions.effects` (full time series treatment effects)
        - `predictions.residuals_B` (pre-treatment / blank period residuals)

    post_idx : np.ndarray
        Integer indices corresponding to post-treatment time periods in the
        full effects vector.

    alpha : float, default=0.05
        Significance level. The resulting interval aims for (1 - alpha) coverage.

    block_size : int, optional
        Size of moving blocks used to compute conformity scores.
        If None, defaults to max(3, sqrt(T_post)).

    seed : int, default=42
        Random seed for reproducibility (reserved for future stochastic extensions).

    Returns
    -------
    SEDCandidate
        Updated candidate with:
        - `inference.ci_lower`
        - `inference.ci_upper`

    Notes
    -----
    - Uses absolute mean block residuals as conformity scores.
    - Includes both standard and circular moving blocks.
    - If no valid conformal solutions exist, returns a wide fallback interval.
    - Fully deterministic given fixed inputs.

    Edge Cases
    ----------
    - If `post_idx` is empty:
        Returns CI = (NaN, NaN)
    - If pre-period residuals are too short:
        fallback interval is used
    """

    # =========================================================
    # Defensive indexing
    # =========================================================
    post_idx = np.asarray(post_idx, dtype=int)

    rng = np.random.default_rng(seed)

    e_B = np.asarray(candidate.predictions.residuals_B).flatten()
    e_post = np.asarray(candidate.predictions.effects)[post_idx].flatten()

    n_B = len(e_B)
    n_post = len(e_post)

    # =========================================================
    # Edge case: no post-treatment data
    # =========================================================
    if n_post == 0:
        candidate.inference.ci_lower = np.nan
        candidate.inference.ci_upper = np.nan
        return candidate

    observed_ate = float(np.mean(e_post))

    # =========================================================
    # Block size heuristic
    # =========================================================
    if block_size is None:
        block_size = max(3, int(np.sqrt(n_post)))

    # =========================================================
    # Construct conformity scores (moving blocks)
    # =========================================================
    conformity_scores = []

    # Standard moving blocks
    for i in range(max(0, n_B - block_size + 1)):
        block = e_B[i:i + block_size]
        conformity_scores.append(np.mean(np.abs(block)))

    # Circular blocks for edge coverage
    for i in range(max(0, n_B - block_size + 1)):
        tail = e_B[i:]
        head_len = max(0, block_size - len(tail))
        block = np.concatenate([tail, e_B[:head_len]])

        if len(block) == block_size:
            conformity_scores.append(np.mean(np.abs(block)))

    conformity_scores = np.asarray(conformity_scores)

    # =========================================================
    # Grid search around observed ATE
    # =========================================================
    std_err_proxy = np.std(e_B) / np.sqrt(max(n_post, 1)) if n_B > 0 else 1.0
    grid_width = 6 * std_err_proxy

    grid = np.linspace(
        observed_ate - grid_width,
        observed_ate + grid_width,
        200
    )

    accepted_thetas = []

    for theta in grid:
        adjusted_post = e_post - theta
        post_score = np.mean(np.abs(adjusted_post))

        p_val = np.mean(conformity_scores >= post_score)

        if p_val > alpha:
            accepted_thetas.append(theta)

    # =========================================================
    # Confidence interval construction
    # =========================================================
    if len(accepted_thetas) > 0:
        candidate.inference.ci_lower = float(np.min(accepted_thetas))
        candidate.inference.ci_upper = float(np.max(accepted_thetas))
    else:
        fallback = 4 * std_err_proxy
        candidate.inference.ci_lower = observed_ate - fallback
        candidate.inference.ci_upper = observed_ate + fallback

    return candidate
