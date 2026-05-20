"""Moving-block conformal inference for the SPCD post-period effect.

Adapts the moving-block conformal procedure from
``mlsynth.utils.fast_scm_helpers.inference.compute_moving_block_conformal_ci``
to the SPCD setting, where the test statistic is the post-period
synthetic-gap mean and the calibration set is the out-of-sample
residual vector ``r_B`` from the holdout window.

The procedure is exchangeability-based: confidence coverage holds in
finite samples under the assumption that overlapping blocks of ``r_B``
have the same distribution as overlapping blocks of the post-period
gap under the null. This is a stronger assumption than IID noise and
weaker than perfect H0; in practice it is the standard for synthetic
control conformal inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SPCDConformalResult:
    """Container for the moving-block conformal output.

    Attributes
    ----------
    att : float
        Observed ATT (mean of the post-period synthetic gap).
    ci_lower, ci_upper : float
        Conformal CI for the ATT at level ``1 - alpha``.
    p_value : float
        Conformal p-value vs. ``H_0: tau = 0``.
    pointwise_lower, pointwise_upper : np.ndarray
        Period-by-period conformal bands of length equal to the post
        horizon.
    block_size : int
        Moving-block size used.
    alpha : float
        Two-sided significance level.
    method : str
        Identifier string. Always ``"moving_block_conformal"``.
    """

    att: float
    ci_lower: float
    ci_upper: float
    p_value: float
    pointwise_lower: np.ndarray
    pointwise_upper: np.ndarray
    block_size: int
    alpha: float
    method: str = "moving_block_conformal"


def _moving_block_scores(
    residuals: np.ndarray, block_size: int
) -> np.ndarray:
    """Return mean-absolute conformity scores over standard + circular blocks.

    Mirrors the procedure in
    ``fast_scm_helpers.inference.compute_moving_block_conformal_ci`` so
    inference behavior is consistent across estimators.
    """

    n = len(residuals)
    scores = []
    # Standard blocks
    for i in range(max(0, n - block_size + 1)):
        scores.append(float(np.mean(np.abs(residuals[i : i + block_size]))))
    # Circular blocks
    for i in range(max(0, n - block_size + 1)):
        tail = residuals[i:]
        head_len = max(0, block_size - len(tail))
        block = np.concatenate([tail, residuals[:head_len]])
        if len(block) == block_size:
            scores.append(float(np.mean(np.abs(block))))
    return np.asarray(scores)


def compute_conformal_ci(
    residuals_B: np.ndarray,
    post_gap: np.ndarray,
    alpha: float = 0.05,
    block_size: Optional[int] = None,
    grid_size: int = 200,
) -> SPCDConformalResult:
    """Compute a moving-block conformal CI for the SPCD post-period ATT.

    Parameters
    ----------
    residuals_B : np.ndarray
        Length-``n_B`` out-of-sample residuals from the holdout window
        (``Y_B @ contrast_weights``).
    post_gap : np.ndarray
        Length-``S`` post-period synthetic gap
        (``Y_post @ contrast_weights``).
    alpha : float
        Two-sided significance level. Coverage is ``1 - alpha``.
    block_size : int, optional
        Moving-block size. Defaults to ``max(3, floor(sqrt(S)))``.
    grid_size : int
        Number of grid points to search for the CI inversion.

    Returns
    -------
    SPCDConformalResult
    """

    residuals_B = np.asarray(residuals_B, dtype=float).ravel()
    post_gap = np.asarray(post_gap, dtype=float).ravel()

    n_B = len(residuals_B)
    n_post = len(post_gap)

    if block_size is None:
        block_size = max(3, int(np.sqrt(max(n_post, 1))))
    block_size = max(1, min(block_size, n_B))

    observed_att = float(np.mean(post_gap))

    scores = _moving_block_scores(residuals_B, block_size)

    # Conformal p-value vs H_0: tau = 0
    observed_score = float(np.mean(np.abs(post_gap)))
    p_value = float(np.mean(scores >= observed_score)) if len(scores) > 0 else 1.0

    # Grid search for the CI by inversion: include theta if the
    # adjusted-residual score is in-distribution relative to scores.
    if len(scores) > 0:
        std_err_proxy = float(np.std(residuals_B)) / np.sqrt(max(n_post, 1))
        grid_width = 6.0 * (std_err_proxy if std_err_proxy > 0 else 1.0)
        grid = np.linspace(observed_att - grid_width, observed_att + grid_width, grid_size)

        accepted = []
        for theta in grid:
            adj_score = float(np.mean(np.abs(post_gap - theta)))
            if np.mean(scores >= adj_score) > alpha:
                accepted.append(theta)

        if accepted:
            ci_lower = float(min(accepted))
            ci_upper = float(max(accepted))
        else:
            fallback = 4.0 * (std_err_proxy if std_err_proxy > 0 else 1.0)
            ci_lower = observed_att - fallback
            ci_upper = observed_att + fallback

        q = float(np.quantile(scores, 1 - alpha))
    else:
        ci_lower = observed_att
        ci_upper = observed_att
        q = 0.0

    pointwise_lower = post_gap - q
    pointwise_upper = post_gap + q

    return SPCDConformalResult(
        att=observed_att,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        pointwise_lower=pointwise_lower,
        pointwise_upper=pointwise_upper,
        block_size=int(block_size),
        alpha=float(alpha),
    )
