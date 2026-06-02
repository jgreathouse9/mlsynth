"""Proximal (two-stage / GMM) debiasing with excluded donors.

O'Riordan & Gilligan-Lee (2025), Section 3.3, equation (5). When the kept
donors are *noisy* (imperfect) proxies of the latents, a synthetic control
fit directly on them suffers errors-in-variables (attenuation) bias even with
a perfect valid-donor selection (the paper's Figure 4). The donors **excluded**
by the spillover screen -- though invalid for *building* the counterfactual --
satisfy the proxy condition on pre-intervention data, so they can serve as
proximal control variables to debias the weights.

The paper notes (page 9) that the joint model of equation (5) "effectively
combines [a] two-stage process into a single model", and that this is the
standard proximal / instrumental-variables estimator: regress the kept donors
:math:`X` on the excluded donors :math:`Z` to form :math:`\\hat X`, then regress
the target :math:`y` on :math:`\\hat X`. Crucially, **only pre-intervention data
from the excluded donors is used**, so their post-intervention spillover
dynamics never enter the estimate. This module implements that two-stage
estimator in closed form -- no probabilistic-programming dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ProximalDebiasFit:
    """Result of the two-stage proximal debiasing."""

    weights: np.ndarray            # debiased donor weights (kept donors), length k
    intercept: float               # stage-2 intercept (alpha)
    counterfactual: np.ndarray     # debiased counterfactual, length T
    att: float                     # debiased ATT (mean post-period gap)
    n_instruments: int             # number of excluded donors used as proxies


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares with an intercept column already prepended to ``X``."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def proximal_debias(
    y: np.ndarray,
    X_kept: np.ndarray,
    Z_excluded: np.ndarray,
    T0: int,
) -> ProximalDebiasFit:
    """Two-stage proximal debiasing of the SC weights (equation 5).

    Parameters
    ----------
    y : np.ndarray
        Treated-unit outcome, length ``T``.
    X_kept : np.ndarray
        Kept (valid) donor matrix used to build the SC, shape ``(T, k)``.
    Z_excluded : np.ndarray
        Excluded donor matrix used as proximal controls, shape ``(T, m)``.
        Only the pre-intervention rows enter the estimate.
    T0 : int
        Number of pre-intervention periods.

    Returns
    -------
    ProximalDebiasFit

    Notes
    -----
    Stage 1 regresses each kept donor on the excluded donors over the
    pre-intervention window to form the proximal projection
    :math:`\\hat X = Z\\,(Z_{\\text{pre}}^+ X_{\\text{pre}})`. Stage 2 regresses
    the treated unit on :math:`\\hat X` over the pre-window to obtain the
    debiased weights, and the counterfactual is the kept donors evaluated at
    those weights. With fewer excluded donors than kept donors the projection is
    rank-deficient and debiasing is skipped (the kept-donor fit is returned).
    """
    T, k = X_kept.shape
    m = Z_excluded.shape[1]
    post = np.arange(T) >= T0

    Zc_pre = np.column_stack([np.ones(T0), Z_excluded[:T0]])
    Zc_all = np.column_stack([np.ones(T), Z_excluded])

    if m < k or m == 0:
        # Not enough instruments to identify the k weights; fall back to OLS-on-X.
        Xc_pre = np.column_stack([np.ones(T0), X_kept[:T0]])
        coef = _ols(Xc_pre, y[:T0])
        alpha, w = float(coef[0]), coef[1:]
        cf = alpha + X_kept @ w
        return ProximalDebiasFit(weights=w, intercept=alpha, counterfactual=cf,
                                 att=float(np.mean((y - cf)[post])),
                                 n_instruments=m)

    # Stage 1: project each kept donor onto the excluded-donor span (pre-period
    # fit, applied over all periods).
    Pi = _ols(Zc_pre, X_kept[:T0])             # (m+1, k)
    Xhat = Zc_all @ Pi                          # (T, k)

    # Stage 2: regress the treated unit on the projected donors (pre-period).
    Xhat_pre = np.column_stack([np.ones(T0), Xhat[:T0]])
    coef = _ols(Xhat_pre, y[:T0])
    alpha, w = float(coef[0]), coef[1:]

    # Structural counterfactual uses the (valid) kept donors with debiased weights.
    cf = alpha + X_kept @ w
    return ProximalDebiasFit(weights=w, intercept=alpha, counterfactual=cf,
                             att=float(np.mean((y - cf)[post])),
                             n_instruments=m)
