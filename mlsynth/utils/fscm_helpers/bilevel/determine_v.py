"""Canonicalisation of the predictor weights ``V`` (MSCMT ``determine_v``).

The bilevel SCM optimum pins down the donor weights ``W`` and the *fit*, but
the predictor weights ``V`` are generically **not** identified: a whole
polytope of ``V`` reproduces the same ``W`` (Becker & Kloessner 2018). Two
optimiser runs that find the same ``W`` can therefore report very different
``V`` -- the source of the "different answer for the same input" instability.

The R package MSCMT resolves this with its ``determine_v`` step, which -- given
the optimal ``W`` -- selects a *canonical* ``V`` by solving small linear
programs over the optimality (KKT) polytope of the lower-level problem. This
module ports the most important of those, ``min.loss.w``:

    among all ``V`` for which ``W`` is the inner optimum, choose the one
    minimising the predictor loss ``sum_k V_k (X1 - X0 W)_k^2``.

Because every ``V`` in the polytope yields the *same* ``W`` (and hence the same
counterfactual), canonicalisation changes only the *reported* predictor
weights, never the estimate -- but it makes those weights reproducible.

References
----------
Becker, M., & Kloessner, S. (2018). Fast and reliable computation of
generalized synthetic controls. Econometrics and Statistics, 5, 1-19.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linprog

from .mscmt import _inner_weights
from .simplex import mspe
from .structure import BilevelProblem


def kkt_matrix(prob: BilevelProblem, W: np.ndarray) -> np.ndarray:
    """Lower-level KKT matrix ``M`` (Becker & Kloessner 2018, ``determine_v``).

    With predictor *discrepancies* ``X = X1 - X0`` (``K x J``, column ``j`` is
    treated-minus-donor ``j``) and synthetic gap ``g = X1 - X0 W``,

        M = (I_J - 1_J W^T) X^T diag(g),   shape (J, K).

    For a candidate predictor weight vector ``v`` (length ``K``), ``M @ v`` is
    the projected gradient of the ``v``-weighted predictor loss w.r.t. each
    donor weight. ``W`` is the inner optimum for ``v`` iff

        (M @ v)_j == 0   for donors with W_j > 0   (stationarity), and
        (M @ v)_j >= 0   for donors with W_j == 0   (KKT inequality).
    """
    W = np.asarray(W, dtype=float).ravel()
    X = prob.X1[:, None] - prob.X0            # (K, J): treated - donor
    g = prob.X1 - prob.X0 @ W                  # (K,): synthetic predictor gap
    J = prob.n_donors
    proj = np.eye(J) - np.ones((J, J)) * W[None, :]   # I_J - 1 W^T
    return proj @ (X.T * g[None, :])           # (J,K): X^T diag(g)


def min_loss_w_v(
    prob: BilevelProblem,
    W: np.ndarray,
    *,
    lb: float = 1e-8,
) -> np.ndarray:
    """Canonical ``min.loss.w`` predictor weights for the optimal ``W``.

    Solves the linear-fractional program

        min_v  (g^2 . v) / (sum v)
        s.t.   (M v)_j == 0   (W_j > 0),   (M v)_j >= 0   (W_j == 0),
               lb <= v_k <= 1

    via the Charnes-Cooper transform ``y = t v, t = 1 / sum(v)``, which makes
    it a plain LP in ``(y, t)``. Returns ``v`` normalised to ``max(v) = 1``
    (MSCMT's reporting convention), or an all-NaN vector if the LP is
    infeasible / degenerate.
    """
    W = np.asarray(W, dtype=float).ravel()
    K = prob.n_predictors
    M = kkt_matrix(prob, W)
    g = prob.X1 - prob.X0 @ W
    obj = g ** 2                                # per-predictor squared gap
    active = W > 0

    # Fix the scale with ``sum(v) = 1``; under that normalisation the linear
    # objective ``obj . v`` equals R's scale-free fractional ``obj . v / sum(v)``
    # exactly, with no free scale for the LP to exploit. Stationarity for active
    # donors is a tolerance band (``W`` comes from an approximate inner solve,
    # so ``M v`` is near-zero, not exactly zero); the band is small relative to
    # the row scale, so the optimiser's stationary ``V`` is feasible but
    # KKT-violating corners are not.
    band = 1e-6 * np.maximum(np.sum(np.abs(M), axis=1), 1e-12)

    A_eq = np.ones((1, K)); b_eq = np.array([1.0])             # sum(v) = 1
    A_ub_rows, b_ub = [], []
    for j in np.where(active)[0]:               # |M_j v| <= band_j
        A_ub_rows.append(M[j]); b_ub.append(band[j])
        A_ub_rows.append(-M[j]); b_ub.append(band[j])
    for j in np.where(~active)[0]:              # (M v)_j >= -band_j
        A_ub_rows.append(-M[j]); b_ub.append(band[j])
    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub = np.asarray(b_ub) if A_ub_rows else None

    res = linprog(
        obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=[(0.0, 1.0)] * K, method="highs",
    )
    if not res.success:
        return np.full(K, np.nan)
    v = res.x
    m = v.max()
    if m <= 0:
        return np.full(K, np.nan)
    return np.clip(v / m, lb, 1.0)             # report normalised to max(v) = 1


def check_v(
    prob: BilevelProblem,
    v: np.ndarray,
    W: np.ndarray,
    *,
    lb: float = 1e-8,
    tol_lp: float = 1e-3,
    tol_loss: float = 1e-2,
) -> bool:
    """Verify ``v`` reproduces ``W`` as the inner optimum (MSCMT ``check_v``).

    Checks the KKT residuals of ``M @ v`` (stationarity for active donors,
    non-negativity for inactive donors), the box ``lb <= v/max(v) <= 1``, and
    that re-solving the inner problem at ``v`` does not worsen the outcome loss.

    The default tolerances (``tol_lp`` on the relative KKT residual, ``tol_loss``
    on the relative outcome-loss increase) are looser than MSCMT's R defaults
    because the outer DE search and the big-M simplex inner solve give a
    *fuzzier* optimum than R's exact WNNLS -- so the KKT polytope that defines
    "``v`` reproduces ``W``" is only approximate. They accept a ``v`` that
    recovers ``W`` to within ~1% of the outcome loss.
    """
    v = np.asarray(v, dtype=float).ravel()
    if np.any(~np.isfinite(v)):
        return False
    mx = v.max()
    if mx <= 0 or (v / mx).min() < lb - 1e-15 or mx > 1.0 + 1e-9:
        return False

    W = np.asarray(W, dtype=float).ravel()
    M = kkt_matrix(prob, W)
    lhs = M @ v
    active = W > 0
    eqmax = float(np.max(np.abs(lhs[active]))) if active.any() else 0.0
    ineqmax = float(np.max(np.maximum(0.0, -lhs[~active]))) if (~active).any() else 0.0
    # KKT residuals scale with v; normalise by the gradient magnitude.
    scale = max(1.0, float(np.max(np.abs(lhs))))
    if max(eqmax, ineqmax) / scale > tol_lp:
        return False

    W_new = _inner_weights(prob, v)
    loss_old = mspe(prob.y1_pre, prob.Y0_pre, W)
    loss_new = mspe(prob.y1_pre, prob.Y0_pre, W_new)
    loss_d = (loss_new - loss_old) / loss_old if loss_old > 0 else loss_new - loss_old
    return loss_d <= tol_loss


def canonical_v(
    prob: BilevelProblem,
    W: np.ndarray,
    *,
    lb: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """Return ``(v, ok)`` -- the ``min.loss.w`` canonical ``V`` and whether it
    passed :func:`check_v`. ``v`` is normalised to the simplex (``sum v = 1``)
    for reporting; ``ok`` is ``False`` when the LP failed or the result does not
    certify, in which case the caller should fall back to the optimiser's ``V``.
    """
    v = min_loss_w_v(prob, W, lb=lb)
    ok = check_v(prob, v, W, lb=lb)
    if not ok:
        return v, False
    s = v.sum()
    return (v / s if s > 0 else v), True
