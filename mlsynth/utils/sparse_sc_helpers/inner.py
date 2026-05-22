"""Inner W-weight QP for SparseSC.

Given V-weights ``v`` (a vector of predictor importance), the donor
weights ``w`` solve the canonical SCM simplex QP

    min over w >= 0, sum(w) = 1:
        w' X0' diag(v) X0 w  -  2 X1' diag(v) X0 w.

The inner QP is called O(grid x outer_iters x P) times during a
SparseSC fit, so its per-call cost dominates the total wall time. We
solve it by calling Clarabel directly (skipping CVXPY's
canonicalization layer); CVXPY adds ~10-50 ms of parsing overhead per
call on a 39-donor problem, while the underlying solve is microseconds.

A tiny ridge is added to the quadratic term for numerical stability:
when the augmented Vives k~40 specification is used the donor design
matrix can be rank-deficient (more predictors than donors), which makes
H = X0' diag(v) X0 numerically singular under any QP solver. The ridge
is scaled by the trace of H so it is invariant to the units of v and X0.

The ``solver`` argument is retained for API compatibility but no longer
selects between back-ends; Clarabel is used unconditionally.
"""

from __future__ import annotations

from typing import Any, Optional

import clarabel
import numpy as np
import scipy.sparse as sp


# Module-level static Clarabel objects so we don't re-allocate on every
# call. The constraint matrix A and right-hand side b only depend on the
# donor count N; the cones too. We rebuild them whenever N changes. Two
# settings profiles are kept: a tight default and a loose fallback for
# ill-conditioned regions of v-space that the outer L-BFGS-B may
# explore on rank-deficient designs (Vives's augmented k>=N spec).
_CACHE: dict[int, tuple[sp.csc_matrix, np.ndarray, list,
                         clarabel.DefaultSettings, clarabel.DefaultSettings]] = {}


def _build_settings(tol: float, max_iter: int) -> clarabel.DefaultSettings:
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.tol_gap_abs = tol
    s.tol_gap_rel = tol
    s.tol_feas = tol
    s.max_iter = max_iter
    return s


def _get_constraint_skeleton(N: int):
    """Return ``(A, b, cones, settings_tight, settings_loose)``.

    Constraints:
      * ``w >= 0`` modelled as ``-w + s = 0``, ``s in R_+^N``.
      * ``sum(w) = 1`` modelled as ``1' w + s = 1``, ``s in {0}``.
    """
    cached = _CACHE.get(N)
    if cached is not None:
        return cached
    A_ineq = -sp.eye(N, format="csr")
    A_eq = sp.csr_matrix(np.ones((1, N)))
    A = sp.vstack([A_ineq, A_eq]).tocsc()
    b = np.concatenate([np.zeros(N), [1.0]])
    cones = [clarabel.NonnegativeConeT(N), clarabel.ZeroConeT(1)]
    settings_tight = _build_settings(tol=1e-8, max_iter=500)
    settings_loose = _build_settings(tol=1e-5, max_iter=2000)
    _CACHE[N] = (A, b, cones, settings_tight, settings_loose)
    return _CACHE[N]


def _uniform_w(N: int) -> np.ndarray:
    """Uniform simplex point ``1/N``, used as the last-resort fallback."""
    return np.full(N, 1.0 / N)


def solve_w(
    v: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    solver: Any = None,  # noqa: ARG001 — kept for API compatibility
) -> np.ndarray:
    """Return the donor-weight vector ``w`` on the simplex.

    Parameters
    ----------
    v : np.ndarray
        Length-``P`` V-weight vector.
    X1 : np.ndarray
        Length-``P`` treated predictor vector.
    X0 : np.ndarray
        ``(P, N)`` donor predictor matrix.
    solver : Any, optional
        Unused. Retained for backwards compatibility with the previous
        CVXPY-based interface.
    """
    v = np.asarray(v, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    N = X0.shape[1]

    # H = X0' diag(v) X0 without materialising diag(v).
    vX0 = X0 * v[:, None]                # (P, N)
    H = X0.T @ vX0                       # (N, N), symmetric PSD
    # Symmetrise to kill numerical asymmetry from the matmul.
    H = 0.5 * (H + H.T)
    # Trace-scaled ridge for robustness under rank deficiency.
    diag_mean = np.trace(H) / max(N, 1)
    if diag_mean <= 0:
        diag_mean = 1.0
    H = H + 1e-10 * diag_mean * np.eye(N)

    # Linear term q = -2 X0' diag(v) X1.
    q = -2.0 * (X0.T @ (v * X1))

    # Clarabel objective is 0.5 w' P w + q' w; we want min w' H w + q' w,
    # i.e. set P = 2 H.
    P_mat = sp.csc_matrix(2.0 * H)

    A, b, cones, settings_tight, settings_loose = _get_constraint_skeleton(N)

    accepted = {"Solved", "AlmostSolved"}

    # First try at the tight tolerance.
    sol = clarabel.DefaultSolver(P_mat, q, A, b, cones, settings_tight).solve()
    if str(sol.status) in accepted:
        return np.clip(np.asarray(sol.x, dtype=float), 0.0, None)

    # If the inner problem is degenerate (often on rank-deficient
    # predictor matrices where P > N) Clarabel can return
    # InsufficientProgress at tight tolerance. Retry with a stronger
    # trace-scaled ridge and looser tolerances before giving up.
    H_strong = H + 1e-6 * diag_mean * np.eye(N)
    P_strong = sp.csc_matrix(2.0 * H_strong)
    sol = clarabel.DefaultSolver(P_strong, q, A, b, cones, settings_loose).solve()
    if str(sol.status) in accepted:
        return np.clip(np.asarray(sol.x, dtype=float), 0.0, None)

    # Last resort: a uniform-weight feasible point. The outer L-BFGS-B
    # will see this as a poor objective and back off; using a uniform
    # fallback is preferable to raising because raising aborts the
    # entire lambda sweep on a single bad v exploration step.
    return _uniform_w(N)
