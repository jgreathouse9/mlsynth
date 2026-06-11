"""Canonicalisation of the predictor weights ``V`` (MSCMT ``determine_v``).

The bilevel SCM optimum pins down the donor weights ``W`` and the *fit*, but
the predictor weights ``V`` are generically **not** identified: a whole
polytope of ``V`` reproduces the same ``W`` (Becker & Kloessner 2018). Two
optimiser runs that find the same ``W`` can therefore report very different
``V`` -- the source of the "different answer for the same input" instability.

The R package MSCMT resolves this with its ``determine_v`` step, which -- given
the optimal ``W`` -- selects a *canonical* ``V`` by solving small linear
programs over the optimality (KKT) polytope of the lower-level problem. This
module ports both canonical choices:

* ``min.loss.w`` -- among all ``V`` for which ``W`` is the inner optimum, the
  one minimising the predictor loss ``sum_k V_k (X1 - X0 W)_k^2`` (tends to a
  sparse corner of the polytope).
* ``max.order`` -- the *leximin* ``V``: maximise the smallest predictor weight,
  then the next, ... (the most balanced point), with a PUFAS uniqueness
  certificate as the stopping rule.

Reporting both gives a free identification diagnostic: if they agree, ``V`` is
well identified; if they diverge, the predictor weights are fragile.

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


def _kkt_band(prob: BilevelProblem, W: np.ndarray):
    """KKT polytope as ``A_ub v <= b_ub`` (stationarity band for active donors,
    non-negativity band for inactive ones). Returns ``(M, A_ub, b_ub)``.
    """
    W = np.asarray(W, dtype=float).ravel()
    M = kkt_matrix(prob, W)
    active = W > 0
    band = 1e-6 * np.maximum(np.sum(np.abs(M), axis=1), 1e-12)
    rows, b = [], []
    for j in np.where(active)[0]:               # |M_j v| <= band_j
        rows.append(M[j]); b.append(band[j])
        rows.append(-M[j]); b.append(band[j])
    for j in np.where(~active)[0]:              # (M v)_j >= -band_j
        rows.append(-M[j]); b.append(band[j])
    A = np.vstack(rows) if rows else None
    return M, A, (np.asarray(b) if rows else None)


def _unique_optimum(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    *, n_probe: int = 3, tol: float = 1e-7, seed: int = 0) -> bool:
    """PUFAS-style certificate: is ``min c.x`` over the polytope a *unique*
    optimum?  Pin the optimal face ``{x : c.x <= z* + eps}`` and check that it
    collapses to a point by maximising/minimising several random directions over
    it -- if every probe has zero spread, the optimum is unique (Appa 2002).
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return False
    z = res.fun
    n = len(res.x)
    eps = 1e-9 * (1.0 + abs(z))
    # Restrict to the optimal face: c.x <= z + eps  (c.x >= z holds at the optimum).
    A_face = np.vstack([A_ub, c]) if A_ub is not None else c[None, :]
    b_face = np.concatenate([b_ub, [z + eps]]) if b_ub is not None else np.array([z + eps])
    rng = np.random.default_rng(seed)
    for _ in range(n_probe):
        d = rng.normal(size=n)
        hi = linprog(-d, A_ub=A_face, b_ub=b_face, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
        lo = linprog(d, A_ub=A_face, b_ub=b_face, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
        if not (hi.success and lo.success):
            return False
        if (-hi.fun) - lo.fun > tol * (1.0 + abs(z)):
            return False
    return True


def max_order_v(
    prob: BilevelProblem,
    W: np.ndarray,
    *,
    lb: float = 1e-8,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, bool]:
    """Canonical ``max.order`` (leximin) predictor weights for the optimal ``W``.

    Among all ``V`` that reproduce ``W`` (the KKT polytope), pick the
    *lexicographically maximal* one: maximise the smallest predictor weight,
    then the next-smallest, and so on. Unlike ``min.loss.w`` (which tends to a
    sparse corner), this gives the most *balanced* ``V`` -- it refuses to zero a
    predictor unless the data force it.

    Implemented as iterated max-min LPs (Becker & Kloessner 2018, ``single_v``):
    each round maximises the minimum free weight, then *fixes* the weights that
    cannot exceed that minimum, and recurses. A PUFAS uniqueness certificate
    (:func:`_unique_optimum`) provides the early stop. Returns ``(v, unique)``
    with ``v`` normalised to ``max(v) = 1`` (or all-NaN on failure).
    """
    W = np.asarray(W, dtype=float).ravel()
    K = prob.n_predictors
    _, A_kkt, b_kkt = _kkt_band(prob, W)
    fixed: dict = {}
    last = None
    unique = False

    for _ in range(K + 1):
        free = [k for k in range(K) if k not in fixed]
        if not free:
            unique = True
            break
        # max-min LP over the free weights: variables [v (K), t]; maximise t
        # s.t. v_k >= t for free k, v_k == fixed[k], KKT band, 0 <= v <= 1.
        rows, bub = [], []
        if A_kkt is not None:
            for r, bb in zip(A_kkt, b_kkt):
                rows.append(np.concatenate([r, [0.0]])); bub.append(bb)
        for k in free:                          # t - v_k <= 0  ->  v_k >= t
            row = np.zeros(K + 1); row[k] = -1.0; row[K] = 1.0
            rows.append(row); bub.append(0.0)
        A_eq = [np.concatenate([np.eye(K)[k], [0.0]]) for k in fixed]
        b_eq = [fixed[k] for k in fixed]
        c = np.zeros(K + 1); c[K] = -1.0        # maximise t
        bounds = [(0.0, 1.0)] * K + [(0.0, None)]
        A_ub = np.vstack(rows); b_ub = np.asarray(bub)
        A_eqv = np.vstack(A_eq) if A_eq else None
        b_eqv = np.asarray(b_eq) if A_eq else None
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eqv, b_eq=b_eqv,
                      bounds=bounds, method="highs")
        if not res.success:
            return np.full(K, np.nan), False
        tstar = res.x[K]; vstar = res.x[:K]; last = vstar

        if _unique_optimum(c, A_ub, b_ub, A_eqv, b_eqv, bounds):
            for k in free:
                fixed[k] = vstar[k]
            unique = True
            break

        # Fix every free weight that cannot exceed tstar (the binding ones).
        forced = []
        for k in free:
            r2, b2 = [], []
            if A_kkt is not None:
                r2 = [row for row in A_kkt]; b2 = list(b_kkt)
            for j in free:                      # keep other free weights >= tstar
                if j == k:
                    continue
                row = np.zeros(K); row[j] = -1.0
                r2.append(row); b2.append(-tstar)
            Aeq2 = [np.eye(K)[kk] for kk in fixed]
            beq2 = [fixed[kk] for kk in fixed]
            c2 = np.zeros(K); c2[k] = -1.0      # maximise v_k
            rr = linprog(c2, A_ub=np.vstack(r2) if r2 else None,
                         b_ub=np.asarray(b2) if r2 else None,
                         A_eq=np.vstack(Aeq2) if Aeq2 else None,
                         b_eq=np.asarray(beq2) if Aeq2 else None,
                         bounds=[(0.0, 1.0)] * K, method="highs")
            vmax_k = (-rr.fun) if rr.success else tstar
            if vmax_k <= tstar + tol:
                forced.append(k)
        if not forced:
            forced = [k for k in free if vstar[k] <= tstar + tol] or list(free)
        for k in forced:
            fixed[k] = tstar

    v = np.array([fixed.get(k, last[k] if last is not None else np.nan)
                  for k in range(K)])
    m = v.max()
    if not np.isfinite(m) or m <= 0:
        return np.full(K, np.nan), False
    return np.clip(v / m, lb, 1.0), unique


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
    method: str = "min.loss.w",
    lb: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """Return ``(v, ok)`` -- a canonical ``V`` for the optimal ``W`` and whether
    it passed :func:`check_v`.

    ``method`` selects the canonicalisation:

    * ``"min.loss.w"`` -- the predictor-loss-minimising ``V`` (sparse / corner).
    * ``"max.order"``  -- the leximin ``V`` (balanced; see :func:`max_order_v`).

    ``v`` is normalised to the simplex (``sum v = 1``) for reporting; ``ok`` is
    ``False`` when the LP failed or the result does not certify, in which case
    the caller should fall back to the optimiser's ``V``.
    """
    if method == "max.order":
        v, _unique = max_order_v(prob, W, lb=lb)
    elif method in ("min.loss.w", "min.loss"):
        v = min_loss_w_v(prob, W, lb=lb)
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'min.loss.w' or 'max.order'."
        )
    ok = check_v(prob, v, W, lb=lb)
    if not ok:
        return v, False
    s = v.sum()
    return (v / s if s > 0 else v), True


def canonical_v_diagnostics(
    prob: BilevelProblem,
    W: np.ndarray,
    *,
    lb: float = 1e-8,
) -> dict:
    """Both canonical predictor weights plus an identification diagnostic.

    Returns a dict with the simplex-normalised ``min.loss.w`` and ``max.order``
    weights, their certification flags, and ``agreement`` -- the max absolute
    difference between them. Small ``agreement`` means ``V`` is well identified;
    a large value means the predictor weights are fragile (the two canonical
    choices disagree), which is itself a useful warning.
    """
    v_min, ok_min = canonical_v(prob, W, method="min.loss.w", lb=lb)
    v_max, ok_max = canonical_v(prob, W, method="max.order", lb=lb)
    if np.all(np.isfinite(v_min)) and np.all(np.isfinite(v_max)):
        agreement = float(np.max(np.abs(v_min - v_max)))
    else:
        agreement = float("nan")
    return {
        "min.loss.w": v_min, "min.loss.w_ok": ok_min,
        "max.order": v_max, "max.order_ok": ok_max,
        "agreement": agreement,
    }
