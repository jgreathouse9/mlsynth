"""The SRC weight computation (Zhu 2023, Algorithm 1).

Two steps, both cheap and deterministic:

1. Unit matching (paper eq. 3) -- for each donor ``j`` a *univariate* OLS
   ``theta_j = (x_j' y) / (x_j' x_j)`` rescales that one donor's matching column
   to the treated unit. The matched control is ``theta_j x_j``.
2. Synthesis (paper eq. 4, 8-9) -- box-``[0, 1]`` weights ``w`` combine the
   matched controls by minimising the Mallows/Cp unbiased-risk criterion
   ``C(w) = ||y - E w||^2 + 2 sigma^2 sum(w)`` over the box, where column ``j``
   of ``E`` is the matched control ``theta_j x_j`` and ``sigma^2`` is the
   full-model residual variance. The combined donor coefficient is
   ``theta_j * w_j`` (which may be negative -> controlled extrapolation).

The QP is solved exactly by :func:`~mlsynth.utils.src_helpers.solver.solve_box_qp`.
The Cp penalty on ``sum(w)`` is what identifies ``w`` -- SRC needs no predictor
(``V``) search to pin down the weights, which is why the estimator is
deterministic. An optional ``V`` (diagonal predictor weighting) is threaded
through for the covariate-augmented variant.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .solver import solve_box_qp


class SRCWeights(NamedTuple):
    """Output of the SRC weight computation on a matching matrix."""

    theta: np.ndarray      # (J,) per-donor univariate OLS coefficients
    w: np.ndarray          # (J,) box-[0, 1] synthesis weights
    combined: np.ndarray   # (J,) theta * w -- the donor coefficients
    bias: float            # intercept (recentring constant)
    sigma2: float          # plug-in full-model residual variance


def src_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 1e-3,
    V: Optional[np.ndarray] = None,
) -> SRCWeights:
    """Compute SRC donor coefficients on a matching matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n, J)
        Matching matrix: ``n`` matching rows (pre-treatment outcomes, optionally
        stacked with standardized covariate rows) by ``J`` donors.
    y : np.ndarray, shape (n,)
        Treated unit's matching vector.
    ridge : float
        Tikhonov stabiliser added to the Cp Gram so the box QP is strictly
        convex (the reference's ``0.001 I``). Not part of the paper's criterion;
        set small.
    V : np.ndarray, shape (n,), optional
        Non-negative predictor weights (Abadie's ``V``). ``None`` uses equal
        weights -- the deterministic Algorithm 1 / Algorithm 3 default.

    Returns
    -------
    SRCWeights
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, J = X.shape
    if V is None:
        y0, X0 = y, X
    else:
        s = np.sqrt(np.clip(np.asarray(V, dtype=float).ravel(), 0.0, None))
        y0, X0 = s * y, s[:, None] * X

    yc = y0 - y0.mean()
    Xc = X0 - X0.mean(axis=0, keepdims=True)

    # Plug-in sigma^2 from the full-model residual (paper eq. 8). Guard the dof
    # when donors are not fewer than matching rows (J >= n): use a min-norm fit
    # and floor the degrees of freedom at 1.
    beta_full, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    dof = max(n - J, 1)
    sigma2 = float(np.sum((yc - Xc @ beta_full) ** 2) / dof)

    col_ss = np.einsum("ij,ij->j", Xc, Xc)
    col_ss = np.where(col_ss > 0, col_ss, 1.0)          # a constant donor -> theta 0
    theta = (Xc.T @ yc) / col_ss
    E = Xc * theta                                       # matched controls, column-wise

    D = E.T @ E + ridge * np.eye(J)
    d = E.T @ yc - sigma2                                # trace(H_j) = 1 for rank-1 hats
    w = solve_box_qp(D, d)

    combined = theta * w
    bias = float(y0.mean() - (X0 @ combined).mean())
    return SRCWeights(theta=theta, w=w, combined=combined, bias=bias, sigma2=sigma2)


def counterfactual(Y_full: np.ndarray, weights: SRCWeights) -> np.ndarray:
    """Full-path SRC counterfactual ``Y_hat = bias + Y_full @ (theta * w)``.

    ``Y_full`` is the donor outcomes over every period (``(T, J)``); the combined
    coefficients are applied to the raw (unweighted) donor outcomes.
    """
    return weights.bias + np.asarray(Y_full, dtype=float) @ weights.combined
