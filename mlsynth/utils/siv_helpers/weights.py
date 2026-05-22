"""Per-unit synthetic-control weight solver for SIV.

For each unit ``i`` we solve the inner SC problem

    min_w  ||D_i - D_{-i}' w||_2^2

subject to either the standard SCM simplex ``w >= 0, sum(w) = 1``
(the default per the paper's empirical applications) or the
``l1``-ball ``||w||_1 <= C`` relaxation introduced in section 3.

We solve the simplex variant by calling Clarabel directly (same
pattern as the SparseSC inner QP) to avoid CVXPY canonicalisation
overhead. The ``l1``-ball variant uses an unconstrained Lagrangian
form solved by NNLS-style projected gradient is more complex than
the simplex; we fall back to CVXPY for that path because (a) it is
rare in practice and (b) the projected gradient code is brittle.
"""

from __future__ import annotations

from typing import Tuple

import clarabel
import numpy as np
import scipy.sparse as sp

from ...exceptions import MlsynthEstimationError
from .structures import SIVInputs, SIVWeights


_CACHE: dict[int, tuple] = {}


def _build_simplex_skeleton(N: int):
    """Return cached ``(A, b, cones, settings)`` for a length-N simplex QP."""

    cached = _CACHE.get(N)
    if cached is not None:
        return cached
    A_ineq = -sp.eye(N, format="csr")
    A_eq = sp.csr_matrix(np.ones((1, N)))
    A = sp.vstack([A_ineq, A_eq]).tocsc()
    b = np.concatenate([np.zeros(N), [1.0]])
    cones = [clarabel.NonnegativeConeT(N), clarabel.ZeroConeT(1)]
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.tol_gap_abs = 1e-9
    s.tol_gap_rel = 1e-9
    s.tol_feas = 1e-9
    s.max_iter = 500
    _CACHE[N] = (A, b, cones, s)
    return _CACHE[N]


def _solve_simplex_sc(D_i: np.ndarray, D_others: np.ndarray) -> np.ndarray:
    """Solve the simplex SC QP for one unit.

    Parameters
    ----------
    D_i : np.ndarray
        Length-``p`` predictor vector for the focal unit.
    D_others : np.ndarray
        Shape ``(J - 1, p)`` predictor matrix for the donors.

    Returns
    -------
    np.ndarray
        Length-``(J - 1)`` simplex-constrained weight vector.
    """

    if D_others.shape[0] == 0:
        return np.asarray([], dtype=float)
    # Inner QP: min 0.5 w' (2 D D') w + (-2 D_others D_i)' w
    H = D_others @ D_others.T
    H = 0.5 * (H + H.T)
    diag_mean = np.trace(H) / max(H.shape[0], 1)
    if diag_mean <= 0:
        diag_mean = 1.0
    H = H + 1e-10 * diag_mean * np.eye(H.shape[0])
    q = -2.0 * (D_others @ D_i)
    P_mat = sp.csc_matrix(2.0 * H)

    N = D_others.shape[0]
    A, b, cones, settings = _build_simplex_skeleton(N)
    sol = clarabel.DefaultSolver(P_mat, q, A, b, cones, settings).solve()
    status = str(sol.status)
    if status not in {"Solved", "AlmostSolved"}:
        # Last-resort uniform fallback so a single bad-unit doesn't abort
        # the whole panel SC.
        return np.full(N, 1.0 / N)
    return np.clip(np.asarray(sol.x, dtype=float), 0.0, None)


def _solve_l1ball_sc(D_i: np.ndarray, D_others: np.ndarray, C: float) -> np.ndarray:
    """Solve the L1-ball relaxation of the SC QP for one unit.

    Falls back to CVXPY (the L1 constraint with a free sign is awkward
    to express directly in Clarabel; CVXPY is acceptable here because
    the L1-ball variant is rare in practice).
    """

    if D_others.shape[0] == 0:
        return np.asarray([], dtype=float)

    import cvxpy as cp

    N = D_others.shape[0]
    w = cp.Variable(N)
    obj = cp.sum_squares(D_i - D_others.T @ w)
    constraints = [cp.norm1(w) <= C]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        # Try OSQP as a fallback
        prob.solve(solver=cp.OSQP)
    if w.value is None:
        raise MlsynthEstimationError(
            "SIV inner SC QP (L1-ball variant) failed."
        )
    return np.asarray(w.value, dtype=float)


def fit_synthetic_controls(
    design: np.ndarray,
    constraint: str = "simplex",
    l1_C: float = 1.0,
) -> np.ndarray:
    """Fit per-unit synthetic-control weights on the supplied design.

    Parameters
    ----------
    design : np.ndarray
        ``(J, p)`` predictor matrix. Row ``i`` is the focal unit; the
        other ``J - 1`` rows are donor predictors.
    constraint : {"simplex", "l1_ball"}
        Simplex constraint is the canonical SCM choice; the L1-ball is
        the regularised relaxation of Doudchenko & Imbens (2016) used
        in the paper's theoretical analysis.
    l1_C : float
        L1-ball radius (used only when ``constraint == "l1_ball"``).

    Returns
    -------
    np.ndarray
        ``(J, J)`` weight matrix. ``W[i, i] = 0``; the remaining ``J -
        1`` columns of row ``i`` are the simplex / L1-ball weights for
        unit ``i``.
    """

    if constraint not in {"simplex", "l1_ball"}:
        raise MlsynthEstimationError(
            f"Unknown weight constraint {constraint!r}; expected "
            f"'simplex' or 'l1_ball'."
        )

    J = design.shape[0]
    W = np.zeros((J, J), dtype=float)

    for i in range(J):
        D_i = design[i]
        donor_mask = np.ones(J, dtype=bool)
        donor_mask[i] = False
        D_others = design[donor_mask]
        if constraint == "simplex":
            w_donors = _solve_simplex_sc(D_i, D_others)
        else:
            w_donors = _solve_l1ball_sc(D_i, D_others, l1_C)
        W[i, donor_mask] = w_donors

    return W


def assemble_weights(
    inputs: SIVInputs,
    W: np.ndarray,
    constraint: str,
) -> SIVWeights:
    """Form synthetic and debiased series from a weight matrix."""

    Y_sc = W @ inputs.Y
    R_sc = W @ inputs.R
    Z_sc = W @ inputs.Z
    return SIVWeights(
        W=W,
        Y_sc=Y_sc, R_sc=R_sc, Z_sc=Z_sc,
        Y_tilde=inputs.Y - Y_sc,
        R_tilde=inputs.R - R_sc,
        Z_tilde=inputs.Z - Z_sc,
        constraint=constraint,
    )
