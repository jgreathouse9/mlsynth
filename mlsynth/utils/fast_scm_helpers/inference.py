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
    Compute Moving Block Conformal Confidence Intervals for the Average Treatment Effect (ATE).

    This is a time-series-aware conformal inference method that uses overlapping (moving) 
    blocks to better capture temporal dependence, seasonality, and autocorrelation in sales data.

    It is generally preferred over non-overlapping block conformal when working with 
    marketing time series data.

    Parameters
    ----------
    candidate : SEDCandidate
        Candidate containing `predictions.effects` and `predictions.residuals_B`.
    post_idx : np.ndarray
        Indices of the post-treatment periods in the full timeline.
    alpha : float, default=0.05
        Significance level (target coverage = 1 - alpha).
    block_size : int, optional
        Size of each moving block. If None, defaults to roughly sqrt(T_post).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    candidate : SEDCandidate
        Updated candidate with `inference.ci_lower` and `inference.ci_upper` populated.
    """
    rng = np.random.default_rng(seed)

    e_B = np.asarray(candidate.predictions.residuals_B).flatten()
    e_post = np.asarray(candidate.predictions.effects[post_idx]).flatten()

    n_B = len(e_B)
    n_post = len(e_post)

    if n_post == 0:
        candidate.inference.ci_lower = np.nan
        candidate.inference.ci_upper = np.nan
        return candidate

    observed_ate = float(np.mean(e_post))

    # Set block size (common heuristic for time series)
    if block_size is None:
        block_size = max(3, int(np.sqrt(n_post)))

    # Create conformity scores using moving blocks from blank period
    conformity_scores = []

    # Moving blocks from blank period (pre-treatment residuals)
    for i in range(n_B - block_size + 1):
        block = e_B[i : i + block_size]
        conformity_scores.append(np.mean(np.abs(block)))

    # Also add circular blocks for better coverage at edges (optional but recommended)
    for i in range(n_B - block_size + 1):
        block = np.concatenate([e_B[i:], e_B[:max(0, block_size - (n_B - i))]])
        if len(block) == block_size:
            conformity_scores.append(np.mean(np.abs(block)))

    conformity_scores = np.array(conformity_scores)

    # For each possible theta, compute conformity score of adjusted post-period
    # We use a grid search around the observed ATE
    std_err_proxy = np.std(e_B) / np.sqrt(n_post) if n_B > 0 else 1.0
    grid_width = 6 * std_err_proxy
    grid = np.linspace(observed_ate - grid_width, observed_ate + grid_width, 200)

    accepted_thetas = []

    for theta in grid:
        adjusted_post = e_post - theta
        post_score = np.mean(np.abs(adjusted_post))

        # Count how many blank-period block scores are >= this post score
        p_val = np.mean(conformity_scores >= post_score)

        if p_val > alpha:
            accepted_thetas.append(theta)

    if accepted_thetas:
        candidate.inference.ci_lower = float(min(accepted_thetas))
        candidate.inference.ci_upper = float(max(accepted_thetas))
    else:
        # Fallback: very wide interval if nothing is accepted
        candidate.inference.ci_lower = observed_ate - 4 * std_err_proxy
        candidate.inference.ci_upper = observed_ate + 4 * std_err_proxy

    return candidate
