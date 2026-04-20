import numpy as np


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


def compute_conformal_ci(candidate, post_idx, alpha: float = 0.05, n_perms: int = 1000, seed: int = 42):
    """
    Compute Block Conformal Confidence Intervals for the Average Treatment Effect (ATE).

    This function inverts the permutation test to find the set of ATE values (thetas) 
    that are statistically consistent with the observed data. This provides a 
    non-parametric confidence interval that does not rely on asymptotic normality 
    assumptions.

    Parameters
    ----------
    candidate : SEDCandidate
        The candidate experimental design containing predicted effects and residuals.
    post_idx : np.ndarray
        Array of indices corresponding to the post-intervention time periods.
    alpha : float, default=0.05
        The significance level (the resulting CI has coverage 1 - alpha).
    n_perms : int, default=1000
        Number of permutations per point in the grid search.
    seed : int, default=42
        Seed for the local random number generator.

    Returns
    -------
    candidate : SEDCandidate
        The updated candidate object with `inference.ci_lower` and `inference.ci_upper` populated.

    Notes
    -----
    - The method performs a grid search over potential ATE values. 
    - For each theta, it 'de-treats' the post-period residuals (e_post - theta) and 
      checks if the resulting series is indistinguishable from the Blank period noise.
    - The grid is centered on the observed ATE and extends 4 standard errors in 
      both directions based on Blank period variance.
    """
    rng = np.random.default_rng(seed)
    e_B = candidate.predictions.residuals_B
    e_post = candidate.predictions.effects[post_idx]
    n_post = len(e_post)

    # Define a search grid around the observed ATE
    observed_ate = np.mean(e_post)
    # Estimate standard error using Blank period noise as a proxy for idiosyncratic variance
    std_err = np.std(e_B) / np.sqrt(n_post) if len(e_B) > 1 else np.abs(observed_ate)

    grid = np.linspace(observed_ate - 4 * std_err, observed_ate + 4 * std_err, 100)

    accepted_thetas = []

    for theta in grid:
        # If theta is the 'true' effect, then (e_post - theta) should be null noise
        adjusted_post = e_post - theta
        pool = np.concatenate([e_B, adjusted_post])

        obs_stat = np.mean(np.abs(adjusted_post))

        # Perform permutation test on the adjusted residuals
        perm_samples = [
            np.mean(np.abs(rng.choice(pool, size=n_post, replace=False)))
            for _ in range(n_perms)
        ]
        p_val = np.mean(np.array(perm_samples) >= obs_stat)

        # If p-value > alpha, we cannot reject the hypothesis that theta is the true ATE
        if p_val > alpha:
            accepted_thetas.append(theta)

    if accepted_thetas:
        candidate.inference.ci_lower = min(accepted_thetas)
        candidate.inference.ci_upper = max(accepted_thetas)
    else:
        # Fallback if no values are accepted (highly noisy or poor fit)
        candidate.inference.ci_lower = np.nan
        candidate.inference.ci_upper = np.nan

    return candidate
