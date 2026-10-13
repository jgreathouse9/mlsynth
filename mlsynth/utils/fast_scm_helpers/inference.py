import warnings

import numpy as np
from typing import Optional
from .structure import SEDCandidate


def compute_moving_block_conformal_ci(
        candidate: SEDCandidate,
        post_idx: np.ndarray,
        alpha: float = 0.05,
        block_size: Optional[int] = None,
        seed: int = 42
) -> SEDCandidate:
    """
    Computes Conformal Confidence Intervals for absolute ATE, percentage lift,
    and total lift using moving block permutations.
    """
    post_idx = np.asarray(post_idx, dtype=int)

    # residuals_B = pre-treatment residuals (used for conformity scores)
    e_B = np.asarray(candidate.predictions.residuals_B).flatten()
    # effects = observed_post - synthetic_post
    e_post = np.asarray(candidate.predictions.effects)[post_idx].flatten()

    # We need the observed post-treatment values to establish the baseline
    y_obs_post = np.asarray(candidate.predictions.synthetic_treated)[post_idx].flatten()

    n_B = len(e_B)
    n_post = len(e_post)

    # =========================================================
    # EDGE CASE
    # =========================================================
    if n_post == 0:
        candidate.inference.ci_lower = np.nan
        candidate.inference.ci_upper = np.nan
        candidate.inference.p_value = np.nan
        candidate.inference.lift_pct = np.nan
        candidate.inference.total_lift = np.nan
        candidate.inference.conformal_empty = None
        return candidate

    observed_ate = float(np.mean(e_post))

    # The Counterfactual Baseline: Y_synthetic = Y_observed - Effect
    # We use this as the denominator for percentage lift
    y_syn_post = y_obs_post - e_post
    baseline_mean = float(np.mean(y_syn_post))

    # =========================================================
    # BLOCK SIZE
    # =========================================================
    if block_size is None:
        block_size = max(3, int(np.sqrt(n_post)))

    # =========================================================
    # MOVING BLOCK CONFORMITY SCORES
    # =========================================================
    conformity_scores = []

    # Standard blocks
    for i in range(max(0, n_B - block_size + 1)):
        block = e_B[i:i + block_size]
        conformity_scores.append(np.mean(np.abs(block)))

    # Circular blocks to handle boundary effects
    for i in range(max(0, n_B - block_size + 1)):
        tail = e_B[i:]
        head_len = max(0, block_size - len(tail))
        block = np.concatenate([tail, e_B[:head_len]])
        if len(block) == block_size:
            conformity_scores.append(np.mean(np.abs(block)))

    conformity_scores = np.asarray(conformity_scores)

    # =========================================================
    # CONFORMAL P-VALUE (Null Hypothesis: Theta = 0)
    # =========================================================
    observed_score = np.mean(np.abs(e_post))
    conformal_p = np.mean(conformity_scores >= observed_score)
    candidate.inference.p_value = float(conformal_p)

    # =========================================================
    # GRID SEARCH FOR ATE CI
    # =========================================================
    # Charge the search range on the information the post window actually holds.
    # Serially correlated periods carry less than their count, so dividing by
    # sqrt(n_post) understates the standard error and can leave the accepted set
    # touching the grid edge -- an interval truncated by the search rather than
    # determined by the data.
    if n_B > 2:
        centred = e_B - e_B.mean()
        lag = float(np.corrcoef(centred[1:], centred[:-1])[0, 1])
        rho = float(np.clip(lag, -0.99, 0.99)) if np.isfinite(lag) else 0.0
    else:
        rho = 0.0
    eff_post = min(max(n_post * (1.0 - rho) / (1.0 + rho), 1.0), float(n_post))
    std_err_proxy = np.std(e_B) / np.sqrt(eff_post) if n_B > 0 else 1.0
    grid_width = 6 * std_err_proxy
    grid = np.linspace(observed_ate - grid_width, observed_ate + grid_width, 200)
    candidate.inference.conformal_grid_span = float(2 * grid_width)

    accepted_thetas = []
    for theta in grid:
        adjusted_post = e_post - theta
        post_score = np.mean(np.abs(adjusted_post))
        if np.mean(conformity_scores >= post_score) > alpha:
            accepted_thetas.append(theta)

    # =========================================================
    # CONVERT TO METRICS (Absolute, Percentage, Total)
    # =========================================================
    # The gap is measured on the weighted synthetic treated unit, whose weights
    # sum to one, so the cumulative over the window is the gap times the number
    # of periods. There is no per-unit multiplier: summing across treated
    # markets is a different quantity and needs their individual gaps.
    multiplier = n_post

    if len(accepted_thetas) > 0:
        candidate.inference.conformal_empty = False
        # Absolute ATET
        candidate.inference.ci_lower = float(np.min(accepted_thetas))
        candidate.inference.ci_upper = float(np.max(accepted_thetas))

        # Percentage Lift: (ATET / Baseline) * 100
        if abs(baseline_mean) > 1e-9:
            candidate.inference.lift_pct = (observed_ate / baseline_mean) * 100
            candidate.inference.lift_pct_lower = (candidate.inference.ci_lower / baseline_mean) * 100
            candidate.inference.lift_pct_upper = (candidate.inference.ci_upper / baseline_mean) * 100

        # Total incremental lift over the window for the synthetic treated unit.
        candidate.inference.total_lift = observed_ate * multiplier
        candidate.inference.total_lift_lower = candidate.inference.ci_lower * multiplier
        candidate.inference.total_lift_upper = candidate.inference.ci_upper * multiplier

    else:
        # No constant per-period effect is consistent with the data. That is a
        # result -- the sharp null the interval inverts is rejected at every
        # candidate value -- and substituting a normal-looking interval would
        # return something outside the conformal guarantee that a caller cannot
        # distinguish from one inside it.
        candidate.inference.conformal_empty = True
        candidate.inference.ci_lower = np.nan
        candidate.inference.ci_upper = np.nan
        candidate.inference.total_lift = observed_ate * multiplier
        candidate.inference.total_lift_lower = np.nan
        candidate.inference.total_lift_upper = np.nan
        warnings.warn(
            f"no constant per-period effect in the searched range was consistent "
            f"with the data at alpha={alpha:g}, so the conformal set is empty and "
            f"the interval is NaN; inference.conformal_empty records this.",
            UserWarning, stacklevel=2)


    # =========================================================
    # POINTWISE INTERVALS
    # =========================================================
    q = np.quantile(conformity_scores, 1 - alpha)
    candidate.inference.pointwise_lower = e_post - q
    candidate.inference.pointwise_upper = e_post + q

    return candidate
