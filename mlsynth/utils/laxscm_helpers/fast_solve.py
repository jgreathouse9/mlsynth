"""DPP-parametrized, Gram-based fast solves for the RESCM penalized program.

cvxpy re-canonicalizes a fresh ``Problem`` on every ``.solve()``; in the
CV / lambda / Monte-Carlo loops that *fixed* compile cost dominates the actual
solve. This module builds each problem **shape** once with ``cp.Parameter`` for
the (square-rooted) Gram and the penalty coefficients, then re-solves by
assigning ``.value`` -- cvxpy reuses the cached canonicalization (DPP).

Two ideas combine:

* **Gram reformulation.** ``||X w - y||^2 = ||G^{1/2} w||^2 - 2 h'w + const`` with
  ``G = X'X`` and ``h = X'y``. The solve depends on ``X`` only through the
  ``J x J`` Gram, so it is **independent of T0** and identical across CV folds of
  different lengths. ``G^{1/2}`` is the symmetric PSD square root (via ``eigh``),
  which exists even when ``G`` is singular (the ``J >= T0`` regime), unlike a
  Cholesky factor.
* **DPP parameters.** The Gram root ``G^{1/2}``, the linear term ``h`` and the
  penalty coefficients enter as ``cp.Parameter`` so a single compiled problem
  serves the whole grid. ``lambda * alpha`` is split into independent
  coefficient parameters (``param * param`` is not DPP).

An intercept is handled by augmenting ``X`` with a constant column (so the level
lives in the last weight and is excluded from the penalty/constraints).

This reproduces :func:`mlsynth.utils.laxscm_helpers.optutils.Opt2.SCopt`'s
penalized branch and is guarded cell-by-cell by the LINF / RESCM
cross-validation benchmarks.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - cvxpy is a hard dependency in practice
    cp = None


# Cache of compiled parametrized problems, keyed by structure (never by data).
_CACHE: Dict[tuple, "_PenalizedProblem"] = {}


def _gram_root(X: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root ``S`` of ``G = X'X`` (so ``S @ S == G``).

    Uses ``eigh`` and clips tiny negative eigenvalues to zero, so it is robust
    when ``G`` is singular (``J >= T0``) where a Cholesky factor would not exist.
    """
    G = X.T @ X
    vals, vecs = np.linalg.eigh(G)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


class _PenalizedProblem:
    """A compiled, DPP-parametrized penalized SCM problem of a fixed shape."""

    def __init__(self, n: int, J: int, kind: str, constraint_type: str):
        from .opthelpers import OptHelpers

        self.J = J
        w = cp.Variable(n)
        wd = w[:J]                       # donor weights (intercept, if any, is w[J])
        self.w = w
        self.S = cp.Parameter((n, n))    # Gram square root of the augmented X
        self.h = cp.Parameter(n)         # X_aug' y
        self.c1 = cp.Parameter(nonneg=True)   # L1 coefficient (lambda * alpha)
        self.c2 = cp.Parameter(nonneg=True)   # second-norm coefficient

        loss = cp.sum_squares(self.S @ w) - 2.0 * (self.h @ w)
        if kind == "l2sq":               # ridge: lambda * ||w||_2^2
            penalty = self.c2 * cp.sum_squares(wd)
        elif kind == "mix2":             # c1 ||w||_1 + c2 ||w||_2
            penalty = self.c1 * cp.norm(wd, 1) + self.c2 * cp.norm(wd, 2)
        elif kind == "mixinf":           # c1 ||w||_1 + c2 ||w||_inf
            penalty = self.c1 * cp.norm(wd, 1) + self.c2 * cp.norm(wd, "inf")
        else:  # pragma: no cover - guarded by _kind_of
            raise ValueError(f"unknown penalty kind {kind!r}")

        constraints = OptHelpers.build_constraints(wd, constraint_type=constraint_type)
        self.problem = cp.Problem(cp.Minimize(loss + penalty), constraints)


def _kind_of(second_norm: str, alpha: float) -> str:
    """Map (second_norm, alpha) to the penalty template SCopt would build."""
    if second_norm == "L1_INF":
        return "mixinf"
    # L1_L2: SCopt short-circuits alpha == 0 to a *squared* L2 (ridge) penalty.
    return "l2sq" if alpha == 0.0 else "mix2"


def solve_penalized(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lam: float,
    alpha: float,
    second_norm: str = "L1_L2",
    constraint_type: str = "simplex",
    fit_intercept: bool = False,
    solver: str = "CLARABEL",
) -> Tuple[np.ndarray, float]:
    """Solve one penalized SCM program, reusing a cached compiled problem.

    Returns ``(donor_weights (J,), intercept)``. Mirrors ``Opt2.SCopt``'s
    penalized branch but skips re-canonicalization across calls of equal shape.
    """
    if cp is None:  # pragma: no cover
        raise RuntimeError("cvxpy is required for solve_penalized")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    T0, J = X.shape
    Xa = np.hstack([X, np.ones((T0, 1))]) if fit_intercept else X
    n = Xa.shape[1]

    kind = _kind_of(second_norm, alpha)
    key = (n, J, kind, constraint_type)
    prob = _CACHE.get(key)
    if prob is None:
        prob = _PenalizedProblem(n, J, kind, constraint_type)
        _CACHE[key] = prob

    prob.S.value = _gram_root(Xa)
    prob.h.value = Xa.T @ y
    if kind == "l2sq":
        prob.c1.value = 0.0
        prob.c2.value = float(lam)
    else:
        prob.c1.value = float(lam * alpha)
        prob.c2.value = float(lam * (1.0 - alpha))

    prob.problem.solve(solver=solver, verbose=False)
    wv = prob.w.value
    if wv is None:  # pragma: no cover - solver failure
        return np.zeros(J, dtype=float), 0.0
    w_full = np.asarray(wv, dtype=float).ravel()
    b0 = float(w_full[J]) if fit_intercept else 0.0
    return w_full[:J], b0


# Constraint types whose QP form OSQP can express directly.
_OSQP_CONSTRAINTS = {"simplex", "affine", "nonneg", "unconstrained"}


def _osqp_plan(second_norm: str, alpha: float):
    """Resolve (ridge, c_inf, c_l1, has_inf, has_l1) for the OSQP QP, or raise.

    OSQP is a pure QP solver: squared-L2 (ridge) folds into ``P``; L1 and
    L-infinity penalties use the box + auxiliary-variable trick. The *non-squared*
    L2 of elastic net (0 < alpha < 1, ``L1_L2``) is a second-order cone, which
    OSQP cannot express -- the caller falls back to cvxpy for that.
    """
    if second_norm == "L1_INF":
        return 0.0, float(1.0 - alpha), float(alpha), True, alpha > 0.0
    if second_norm == "L1_L2":
        if alpha == 0.0:
            return 1.0, 0.0, 0.0, False, False        # ridge (squared L2) -> P
        if alpha == 1.0:
            return 0.0, 0.0, 1.0, False, True          # lasso (pure L1) -> u
    raise NotImplementedError(f"OSQP cannot express second_norm={second_norm!r}, alpha={alpha}")


def solve_penalized_osqp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lam: float,
    alpha: float,
    second_norm: str = "L1_L2",
    constraint_type: str = "simplex",
    fit_intercept: bool = False,
) -> Tuple[np.ndarray, float]:
    """Native warm-startable OSQP solve of the penalized SCM QP (no cvxpy).

    Mirrors ``Opt2.SCopt``'s penalized branch for the QP-expressible families
    (SC, ridge, lasso, LINF, L1LINF) via the Gram and the box+auxiliary
    reformulation; raises ``NotImplementedError`` for SOC cases (elastic-net's
    non-squared L2) so the caller can fall back. Returns ``(weights, intercept)``.
    """
    import osqp
    import scipy.sparse as sp

    if constraint_type not in _OSQP_CONSTRAINTS:
        raise NotImplementedError(f"OSQP path does not handle constraint_type={constraint_type!r}")
    ridge_coef, c_inf, c_l1, has_inf, has_l1 = _osqp_plan(second_norm, alpha)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    T0, J = X.shape
    Xa = np.hstack([X, np.ones((T0, 1))]) if fit_intercept else X
    n = Xa.shape[1]
    G = Xa.T @ Xa
    h = Xa.T @ y

    t_idx = n if has_inf else None
    u0 = n + (1 if has_inf else 0)
    N = n + (1 if has_inf else 0) + (J if has_l1 else 0)
    INF = np.inf

    P = np.zeros((N, N))
    P[:n, :n] = 2.0 * G
    if ridge_coef:
        P[np.arange(J), np.arange(J)] += 2.0 * lam * ridge_coef
    q = np.zeros(N)
    q[:n] = -2.0 * h
    if has_inf:
        q[t_idx] = lam * c_inf
    if has_l1:
        q[u0:u0 + J] = lam * c_l1

    rows, lo, hi = [], [], []

    def _row(*entries):
        r = np.zeros(N)
        for idx, val in entries:
            r[idx] = val
        return r

    if constraint_type in ("simplex", "affine"):           # sum_j w_j = 1
        r = np.zeros(N); r[:J] = 1.0
        rows.append(r); lo.append(1.0); hi.append(1.0)
    if constraint_type in ("simplex", "nonneg"):           # w_j >= 0
        for d in range(J):
            rows.append(_row((d, 1.0))); lo.append(0.0); hi.append(INF)
    if has_inf:                                            # |w_j| <= t
        for d in range(J):
            rows.append(_row((d, 1.0), (t_idx, -1.0))); lo.append(-INF); hi.append(0.0)
            rows.append(_row((d, -1.0), (t_idx, -1.0))); lo.append(-INF); hi.append(0.0)
    if has_l1:                                             # |w_j| <= u_j
        for d in range(J):
            rows.append(_row((d, 1.0), (u0 + d, -1.0))); lo.append(-INF); hi.append(0.0)
            rows.append(_row((d, -1.0), (u0 + d, -1.0))); lo.append(-INF); hi.append(0.0)

    if rows:
        A = sp.csc_matrix(np.vstack(rows)); l = np.array(lo); u = np.array(hi)
    else:  # OSQP needs at least one (here trivially unbounded) constraint row
        A = sp.csc_matrix((1, N)); l = np.array([-INF]); u = np.array([INF])

    m = osqp.OSQP()
    m.setup(P=sp.csc_matrix(P), q=q, A=A, l=l, u=u, verbose=False,
            eps_abs=1e-9, eps_rel=1e-9, max_iter=40000, polish=True)
    res = m.solve()
    x = getattr(res, "x", None)
    if x is None or np.any(np.isnan(x)):  # pragma: no cover - solver failure
        return np.zeros(J, dtype=float), 0.0
    x = np.asarray(x, dtype=float)
    b0 = float(x[J]) if fit_intercept else 0.0
    return x[:J], b0


# ---------------------------------------------------------------------------
# Relaxed branch (SCM-relaxation): solve over the balance polytope.
#
# All three relaxations minimise a divergence D(w) over
#   w in simplex,  ||(h - G w)/T0 + gamma * 1||_inf <= tau   (gamma a free scalar)
# where G = X'X, h = X'y. With the Gram, the balance L-infinity ball is just 2J
# linear inequalities in (w, gamma):  for each j,
#   -tau <= (h_j - (G w)_j)/T0 + gamma <= tau.
# L2 (D = ||w||^2) is then a QP -> OSQP; entropy / EL are max-entropy / EL over
# the same polytope (smooth-dual targets, handled separately).
# ---------------------------------------------------------------------------

def _balance_rows(G: np.ndarray, h: np.ndarray, T0: int, tau: float, n_extra: int):
    """Linear rows/bounds for ``||(h - G w)/T0 + gamma||_inf <= tau``.

    Variable layout is ``[w (J), gamma (1), <n_extra aux>]``; gamma is column J.
    Returns ``(rows, lo, hi)`` with one pair of bounds folded into ``[lo, hi]``
    per donor (the two-sided inequality).
    """
    J = G.shape[1]
    N = J + 1 + n_extra
    rows, lo, hi = [], [], []
    for j in range(J):
        r = np.zeros(N)
        r[:J] = -G[j, :] / T0           # -(G w)_j / T0
        r[J] = 1.0                       # + gamma
        rows.append(r)
        lo.append(-tau - h[j] / T0)      # -tau <= ... - h_j/T0  (move constant)
        hi.append(tau - h[j] / T0)
    return rows, lo, hi


def solve_relaxed_l2_osqp(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray | None:
    """Native OSQP solve of the L2 SCM-relaxation (``min ||w||^2`` over the
    simplex and the relaxed-balance polytope). Returns donor weights ``(J,)`` or
    ``None`` on failure. Mirrors ``Opt2.SCopt(objective_type='relaxed',
    relaxation_type='l2', constraint_type='simplex')``.
    """
    import osqp
    import scipy.sparse as sp

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    T0, J = X.shape
    G = X.T @ X
    h = X.T @ y
    N = J + 1                                   # [w (J), gamma]

    P = np.zeros((N, N))
    P[np.arange(J), np.arange(J)] = 2.0          # (1/2) x'Px = ||w||^2
    q = np.zeros(N)

    rows, lo, hi = [], [], []
    r = np.zeros(N); r[:J] = 1.0                 # sum_j w_j = 1
    rows.append(r); lo.append(1.0); hi.append(1.0)
    for d in range(J):                           # w_j >= 0
        rr = np.zeros(N); rr[d] = 1.0
        rows.append(rr); lo.append(0.0); hi.append(np.inf)
    br, blo, bhi = _balance_rows(G, h, T0, tau, n_extra=0)
    rows += br; lo += blo; hi += bhi

    A = sp.csc_matrix(np.vstack(rows))
    m = osqp.OSQP()
    try:
        m.setup(P=sp.csc_matrix(P), q=q, A=A, l=np.array(lo), u=np.array(hi),
                verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=40000, polish=True)
        res = m.solve()
    except Exception:  # pragma: no cover - solver setup/solve failure
        return None
    x = getattr(res, "x", None)
    if x is None or np.any(np.isnan(x)):
        return None
    w = np.asarray(x[:J], dtype=float)
    if np.allclose(w, 0):
        return None
    return w


# Cache of compiled DPP relaxed problems, keyed by (J, relaxation_type).
_RELAX_CACHE: Dict[tuple, "_RelaxedProblem"] = {}


class _RelaxedProblem:
    """A compiled, DPP-parametrized SCM-relaxation problem of fixed shape.

    The Gram sufficient statistics enter as parameters ``Gn = X'X / T0`` and
    ``hn = X'y / T0`` (so the solve is T0-independent and reused across CV folds),
    ``tau`` is a parameter, and a single compiled problem serves the whole tau
    grid. The divergence objective carries no parameters, so DPP holds.
    """

    def __init__(self, J: int, relaxation_type: str):
        w = cp.Variable(J)
        gam = cp.Variable()
        self.w = w
        self.Gn = cp.Parameter((J, J))
        self.hn = cp.Parameter(J)
        self.tau = cp.Parameter(nonneg=True)

        residual = self.hn - self.Gn @ w            # (h - G w) / T0
        constraints = [cp.sum(w) == 1, w >= 0,
                       cp.norm(residual + gam, "inf") <= self.tau]
        if relaxation_type == "entropy":
            obj = cp.sum(-cp.entr(w))               # sum w_i log w_i
        elif relaxation_type == "el":
            obj = cp.sum(-cp.log(w))                # -sum log w_i
        elif relaxation_type == "l2":
            obj = cp.sum_squares(w)
        else:  # pragma: no cover
            raise ValueError(f"unknown relaxation_type {relaxation_type!r}")
        self.problem = cp.Problem(cp.Minimize(obj), constraints)


def solve_relaxed_dpp(
    X: np.ndarray, y: np.ndarray, tau: float, relaxation_type: str = "entropy",
    solver: str = "CLARABEL",
) -> np.ndarray | None:
    """DPP/Gram relaxed solve reusing a cached compiled problem across the tau
    grid. Mirrors ``Opt2.SCopt(objective_type='relaxed', ...)``; returns donor
    weights ``(J,)`` or ``None`` on failure.
    """
    if cp is None:  # pragma: no cover
        raise RuntimeError("cvxpy is required for solve_relaxed_dpp")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    T0, J = X.shape

    key = (J, relaxation_type)
    prob = _RELAX_CACHE.get(key)
    if prob is None:
        prob = _RelaxedProblem(J, relaxation_type)
        _RELAX_CACHE[key] = prob

    prob.Gn.value = (X.T @ X) / T0
    prob.hn.value = (X.T @ y) / T0
    prob.tau.value = float(tau)
    try:
        prob.problem.solve(solver=solver, verbose=False)
    except Exception:  # pragma: no cover - solver failure
        return None
    wv = prob.w.value
    if wv is None:
        return None
    w = np.asarray(wv, dtype=float).ravel()
    if np.allclose(w, 0):
        return None
    return w
