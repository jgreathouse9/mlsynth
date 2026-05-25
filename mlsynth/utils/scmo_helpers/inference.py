"""Conformal inference for SCMO (Chernozhukov-Wuethrich-Zhu, multi-outcome form).

This is the single inference procedure used for *every* weighting scheme, after
Sun, Ben-Michael & Feller (2025), Online Appendix A, which adapts the conformal
test of Chernozhukov, Wuethrich & Zhu (2021) to multiple outcomes.

For a constant-effect null ``H0: tau = tau0`` the synthetic-control weights do
not depend on the post-period outcome (the matching matrix ``Z`` is built from
pre-period information), so the adjusted residual is simply the gap shifted by
``tau0`` in the post-period. The per-period test statistic is

    S_q(u_t) = ( (1/sqrt(K)) * sum_k |u_tk|^q )^{1/q},   default q = 1,

and the conformal p-value ranks the post-treatment statistic against the
distribution of pre-treatment (moving-block) statistics. Inverting the test
over a grid of ``tau0`` yields a confidence interval for the ATT. With a single
predicted outcome (``K = 1``) the statistic reduces to ``|gap|``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def conformal_inference(
    y: np.ndarray, counterfactual: np.ndarray, T0: int,
    alpha: float = 0.1, q: float = 1.0, n_grid: int = 600,
) -> Tuple[float, float, Tuple[float, float]]:
    """CWZ conformal ATT, p-value, and confidence interval from a gap series.

    Tests the *average* post-treatment effect (the scalar case of Sun-Ben-Michael-
    Feller Online Appendix A / Chernozhukov-Wuethrich-Zhu 2021): the test statistic
    is the post-period mean gap, and the reference distribution is the set of
    pre-treatment moving-block means of the same length. Inverting the test over
    ``tau0`` yields the ATT confidence interval. (For a single predicted outcome
    the per-period ``S_q`` reduces to ``|gap|``; ``q`` only bites with ``K > 1``
    outcomes.)

    Parameters
    ----------
    y, counterfactual : np.ndarray
        Observed treated outcome and estimated counterfactual, shape ``(T,)``.
    T0 : int
        Number of pre-treatment periods.
    alpha : float
        Miscoverage rate (e.g. 0.1 -> 90% interval).
    q : float
        Norm exponent (kept for the multi-outcome generalization; inert for K=1).
    n_grid : int
        Resolution of the test-inversion grid for the confidence interval.

    Returns
    -------
    att : float
        Mean post-treatment gap.
    p_value : float
        Conformal p-value for the sharp null of no average effect.
    ci : tuple of float
        ``(lower, upper)`` confidence interval for the ATT.
    """
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    pre, post = gap[:T0], gap[T0:]
    L = post.shape[0]
    L = L if T0 >= L else max(1, T0 // 2)
    n_blocks = T0 - L + 1
    block_means = np.array([pre[s:s + L].mean() for s in range(n_blocks)])   # signed
    ref = np.abs(block_means)                                                # |avg gap| under no effect

    att = float(np.mean(post))

    def p_of(tau0: float) -> float:
        return (1.0 + np.sum(ref >= abs(att - tau0))) / (n_blocks + 1.0)

    p_value = p_of(0.0)

    spread = abs(att) + 4.0 * (ref.max() + np.std(pre)) + 1e-9
    grid = np.linspace(att - spread, att + spread, n_grid)
    keep = np.array([p_of(t) >= alpha for t in grid])
    ci = (float(grid[keep].min()), float(grid[keep].max())) if keep.any() else (float("nan"), float("nan"))
    return att, p_value, ci
