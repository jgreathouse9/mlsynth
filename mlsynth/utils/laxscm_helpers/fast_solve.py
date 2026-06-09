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
