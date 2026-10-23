"""Simplex-constrained least squares with linear equalities and inequalities, solved exactly.

    min ||z - Z b||^2   over   b >= 0,  1'b = 1,  A b = c,  G b >= h

The equalities are eliminated in a null-space basis. What remains is a least-squares
problem with inequality constraints, which Lawson and Hanson (1974, Solving Least
Squares Problems, ch. 23) reduce to a least-distance problem solved by one
nonnegative least-squares call. SciPy's ``nnls`` is compiled, so a solve on a face of
a few donors takes 0.2 to 0.26 ms on Proposition 99.

Every point returned is checked against the constraints it was solved under. A point
that fails the check, or a reduction asked to work outside its domain, raises
ValueError; "infeasible" is returned only when the constraints themselves have no
verified solution.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

COND_MAX = 1e4      # above this the back-transform through R^{-1} leaves fewer digits than VERIFY asks for
VERIFY = 1e-7       # a returned point may miss a row by this much, relative to the size of the row's terms


def _satisfies(G, h, x, tol=VERIFY):
    """G x >= h, each row to tol times the size of its terms."""
    slack = G @ x - h
    return bool(np.all(slack >= -tol * (np.linalg.norm(G, axis=1) * np.linalg.norm(x) + np.abs(h))))


def _ldp_candidate(G, h):
    """Lawson and Hanson's point -r_{1:n} / r_{n+1} from one NNLS, or None when r is not a solution's.

    At an NNLS solution of [G'; h'] u ~ e_{n+1} the residual satisfies r_{n+1} = -||r||^2,
    and G x >= h has no solution exactly when r = 0. An infeasible system with margin
    delta leaves a rounding residual of about eps / delta, which clears any absolute floor
    once delta is small; its r_{n+1} is of the order of ||r||, not -||r||^2.
    """
    p, n = G.shape
    E = np.vstack([G.T, h[None, :]])
    f = np.zeros(n + 1)
    f[-1] = 1.0
    u, _ = nnls(E, f, maxiter=50 * max(p, 1))
    r = E @ u - f
    rr = float(r @ r)
    if not (r[-1] < 0.0 and abs(r[-1] + rr) <= 1e-2 * rr):
        return None
    return -r[:n] / r[-1]


def _ldp(G, h):
    """min ||x|| subject to G x >= h, by one NNLS. None when no solution is found and verified."""
    x = _ldp_candidate(G, h)
    return x if x is not None and _satisfies(G, h, x) else None


def _lsi(E, f, G, h):
    """min ||E y - f|| subject to G y >= h. None when the reduced problem has no verified solution.

    The reduction divides by R from E = QR, so it needs E of full column rank, and its
    answer passes the 1e-7 check only while cond(E) stays below about 1e4: on random
    nearly collinear faces every piece is exact below 1e4 and 9 per cent fail between
    1e4 and 1e5. A face with more free weights than validation periods, or with donors
    whose validation paths are nearly collinear, is refused. Proposition 99, West Germany
    and the seeded designs in results/ stay below 600.
    """
    sv = np.linalg.svd(E, compute_uv=False)
    if E.shape[0] < E.shape[1] or not sv[-1] > sv[0] / COND_MAX:
        cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        raise ValueError(
            f"the face's validation block is rank-deficient or ill-conditioned ({E.shape[0]} periods, "
            f"{E.shape[1]} free weights, condition number {cond:.1e}); the exact least-squares "
            "reduction needs full column rank")
    Q, R = np.linalg.qr(E)
    fq = Q.T @ f
    GRi = solve_triangular(R, G.T, trans="T").T          # G R^{-1}
    x = _ldp(GRi, h - GRi @ fq)
    return None if x is None else solve_triangular(R, x + fq)


def simplex_ls(z, Z, Aeq=None, beq=None, Gin=None, hin=None):
    """Return (loss, b) for the program in the module docstring, or (None, None) if infeasible.

    Raises ValueError when the face is outside the reduction's domain (see ``_lsi``) or the
    reduction's answer fails its check, so that a numerical failure is never read as an
    empty piece.
    """
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
            if _ldp(G, h) is not None:                    # the constraints alone, without R^{-1}
                raise ValueError("the least-squares reduction found no solution on a piece whose "
                                 "constraints have one; the reduction failed")
            return None, None
        b = b0 + N @ y
        if b.min() < -VERIFY or (Gin is not None and len(Gin) and not _satisfies(Gin, hin, b)):
            raise ValueError("the least-squares reduction returned a point that breaks its constraints")
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
