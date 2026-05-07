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

    post_idx = np.asarray(post_idx, dtype=int)

    e_B = np.asarray(candidate.predictions.residuals_B).flatten()
    e_post = np.asarray(candidate.predictions.effects)[post_idx].flatten()

    n_B = len(e_B)
    n_post = len(e_post)

    # =========================================================
    # EDGE CASE
    # =========================================================
    if n_post == 0:

        candidate.inference.ci_lower = np.nan
        candidate.inference.ci_upper = np.nan

        candidate.inference.p_value = np.nan

        candidate.inference.pointwise_lower = np.array([])
        candidate.inference.pointwise_upper = np.array([])

        return candidate

    observed_ate = float(np.mean(e_post))

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

        conformity_scores.append(
            np.mean(np.abs(block))
        )

    # Circular blocks
    for i in range(max(0, n_B - block_size + 1)):

        tail = e_B[i:]
        head_len = max(0, block_size - len(tail))

        block = np.concatenate([
            tail,
            e_B[:head_len]
        ])

        if len(block) == block_size:

            conformity_scores.append(
                np.mean(np.abs(block))
            )

    conformity_scores = np.asarray(conformity_scores)

    # =========================================================
    # CONFORMAL P-VALUE FOR NULL THETA = 0
    # =========================================================
    observed_score = np.mean(np.abs(e_post))

    conformal_p = np.mean(
        conformity_scores >= observed_score
    )

    candidate.inference.p_value = float(conformal_p)

    # =========================================================
    # GRID SEARCH FOR ATE CI
    # =========================================================
    std_err_proxy = (
        np.std(e_B) / np.sqrt(max(n_post, 1))
        if n_B > 0 else 1.0
    )

    grid_width = 6 * std_err_proxy

    grid = np.linspace(
        observed_ate - grid_width,
        observed_ate + grid_width,
        200
    )

    accepted_thetas = []

    for theta in grid:

        adjusted_post = e_post - theta

        post_score = np.mean(
            np.abs(adjusted_post)
        )

        p_val = np.mean(
            conformity_scores >= post_score
        )

        if p_val > alpha:
            accepted_thetas.append(theta)

    # =========================================================
    # ATE CONFIDENCE INTERVAL
    # =========================================================
    if len(accepted_thetas) > 0:

        candidate.inference.ci_lower = float(
            np.min(accepted_thetas)
        )

        candidate.inference.ci_upper = float(
            np.max(accepted_thetas)
        )

    else:

        fallback = 4 * std_err_proxy

        candidate.inference.ci_lower = (
            observed_ate - fallback
        )

        candidate.inference.ci_upper = (
            observed_ate + fallback
        )

    # =========================================================
    # POINTWISE POST-TREATMENT INTERVALS
    # =========================================================
    q = np.quantile(
        conformity_scores,
        1 - alpha
    )

    pointwise_lower = e_post - q
    pointwise_upper = e_post + q

    candidate.inference.pointwise_lower = pointwise_lower
    candidate.inference.pointwise_upper = pointwise_upper

    return candidate
