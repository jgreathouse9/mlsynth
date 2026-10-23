"""Simplex-constrained least squares with linear equalities and inequalities, solved exactly.

    min ||z - Z b||^2   over   b >= 0,  1'b = 1,  A b = c,  G b >= h

The equalities are eliminated in a null-space basis. What remains is a least-squares
problem with inequality constraints, which Lawson and Hanson (1974, Solving Least
Squares Problems, ch. 23) reduce to a least-distance problem solved by one
nonnegative least-squares call. SciPy's ``nnls`` is compiled, so a solve on a face of
a few donors takes 0.2 to 0.25 ms on Proposition 99.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls


def _ldp(G, h):
    """min ||x|| subject to G x >= h, by one NNLS. None when infeasible."""
    p, n = G.shape
    E = np.vstack([G.T, h[None, :]])
    f = np.zeros(n + 1)
    f[-1] = 1.0
    u, _ = nnls(E, f, maxiter=50 * max(p, 1))
    r = E @ u - f
    if abs(r[-1]) < 1e-12 or np.linalg.norm(r) < 1e-12:
        return None
    return -r[:n] / r[-1]


def _lsi(E, f, G, h):
    """min ||E y - f|| subject to G y >= h. None when infeasible.

    The reduction needs E of full column rank. When it is not (fewer validation periods
    than free weights on a face), a ridge of 1e-12 times the scale of E is appended,
    which picks the least-norm minimiser and moves the loss by O(1e-24).
    """
    if E.shape[0] < E.shape[1] or np.linalg.matrix_rank(E) < E.shape[1]:
        eps = 1e-12 * max(1.0, float(np.abs(E).max()))
        E = np.vstack([E, eps * np.eye(E.shape[1])])
        f = np.concatenate([f, np.zeros(E.shape[1])])
    Q, R = np.linalg.qr(E)
    fq = Q.T @ f
    GRi = solve_triangular(R, G.T, trans="T").T          # G R^{-1}
    x = _ldp(GRi, h - GRi @ fq)
    if x is None:
        return None
    return solve_triangular(R, x + fq)


def simplex_ls(z, Z, Aeq=None, beq=None, Gin=None, hin=None):
    """Return (loss, b) for the program in the module docstring, or (None, None) if infeasible."""
    m = Z.shape[1]
    A = np.ones((1, m)) if Aeq is None or len(Aeq) == 0 else np.vstack([np.ones((1, m)), Aeq])
    c = np.ones(1) if beq is None or len(beq) == 0 else np.concatenate([[1.0], beq])
    b0, *_ = np.linalg.lstsq(A, c, rcond=None)
    if np.abs(A @ b0 - c).max() > 1e-9 * max(1.0, np.abs(c).max()):
        return None, None
    _, sv, Vt = np.linalg.svd(A)
    rank = int((sv > 1e-10 * max(sv[0], 1.0)).sum())
    N = Vt[rank:].T                                       # null-space basis, m x d
    if N.shape[1] == 0:
        if b0.min() < -1e-10:
            return None, None
        if Gin is not None and len(Gin) and (Gin @ b0 - hin).min() < -1e-9:
            return None, None
        b = np.clip(b0, 0.0, None)
    else:
        G, h = N, -b0
        if Gin is not None and len(Gin):
            G = np.vstack([G, Gin @ N])
            h = np.concatenate([h, hin - Gin @ b0])
        y = _lsi(Z @ N, z - Z @ b0, G, h)
        if y is None:
            return None, None
        b = b0 + N @ y
        if b.min() < -1e-7:
            return None, None
        b = np.clip(b, 0.0, None)
    res = z - Z @ b
    return float(res @ res), b


def sign_piece_ls(X0, X1, Z1, Z0, cols, s):
    """Best validation fit on face `cols` within the piece {s_k r_k(w) >= 0 for s_k != 0}.

    r(w) = X1 - X0 w is the predictor residual; s_k = 0 leaves coordinate k free.
    """
    act = np.flatnonzero(s != 0)
    Gin = -(s[act, None] * X0[np.ix_(act, cols)])
    hin = -(s[act] * X1[act])
    return simplex_ls(Z1, Z0[:, cols], Gin=Gin, hin=hin)
