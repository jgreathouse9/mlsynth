"""Simplex-constrained least squares with linear inequalities, solved exactly.

    min ||z - Z b||^2   over   b >= 0,  1'b = 1,  G b >= h

The equality 1'b = 1 is eliminated in a null-space basis. What remains is a least-squares
problem with inequality constraints, which Lawson and Hanson (1974, Solving Least
Squares Problems, ch. 23) reduce to a least-distance problem solved by one
nonnegative least-squares call. SciPy's ``nnls`` is compiled, so a solve on a face of
a few donors takes 0.09 to 0.16 ms on Proposition 99.

Every point returned is checked against the constraints it was solved under, and every
face fit is certified optimal by the backward error of its KKT conditions on the face's
own data (``kkt_backward_error``). A point that fails either check raises
MlsynthEstimationError, and a reduction asked to work outside its domain (a
rank-deficient face) raises MlsynthDataError; "infeasible" is returned only when the
constraints themselves have no verified solution.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import linprog, lsq_linear, nnls

from ....exceptions import MlsynthDataError, MlsynthEstimationError

VERIFY = 1e-7       # a returned point may miss a row by this much, relative to the size of the row's terms
CERTIFY = 1e-12     # a face fit's KKT conditions may miss by this much, relative to the face's data


def _satisfies(G, h, x, tol=VERIFY):
    """G x >= h, each row to tol times the size of its terms."""
    slack = G @ x - h
    return bool(np.all(slack >= -tol * (np.linalg.norm(G, axis=1) * np.linalg.norm(x) + np.abs(h))))


def _nnls_optimal(E, f, u, tol=1e-9):
    """Whether u minimises ||E u - f|| over u >= 0: the dual E'(f - E u) is <= 0, and 0 where u > 0.

    Both are checked to tol times the size of the terms; on 56,073 solves from the
    applications and the seeded designs the largest violation is 6.1e-13 of that size.
    """
    w = E.T @ (f - E @ u)
    scale = tol * max(1.0, float(np.abs(E).max())) * (1.0 + float(np.linalg.norm(E @ u)))
    return bool(u.min(initial=0.0) >= 0.0 and w.max(initial=0.0) <= scale
                and np.abs(w[u > 0]).max(initial=0.0) <= scale)


def _ldp_nnls(G, h):
    """One NNLS on [G'; h'] u ~ e_{n+1}: the multipliers u >= 0 and the residual r.

    SciPy's nnls can return a point that is no minimiser, with a success status and a
    residual norm of 0 (scipy 1.18.1 on the degenerate system in
    mlsynth/tests/fixtures/exact/nnls_incident.json), so its answer is checked against the optimality
    conditions and solved again by bounded-variable least squares when it fails them.
    """
    p, n = G.shape
    E = np.vstack([G.T, h[None, :]])
    f = np.zeros(n + 1)
    f[-1] = 1.0
    u, _ = nnls(E, f, maxiter=50 * max(p, 1))
    if not _nnls_optimal(E, f, u):
        u = np.maximum(lsq_linear(E, f, bounds=(0.0, np.inf), method="bvls", tol=1e-14).x, 0.0)
    return u, E @ u - f


def _point_from_residual(r):
    """Lawson and Hanson's point -r_{1:n} / r_{n+1}, or None when r is not a solution's residual.

    At an NNLS solution of [G'; h'] u ~ e_{n+1} the residual satisfies r_{n+1} = -||r||^2,
    and G x >= h has no solution exactly when r = 0. An infeasible system with margin
    delta leaves a rounding residual of about eps / delta, which clears any absolute floor
    once delta is small; its r_{n+1} is of the order of ||r||, not -||r||^2.
    """
    rr = float(r @ r)
    if not (r[-1] < 0.0 and abs(r[-1] + rr) <= 1e-2 * rr):
        return None
    return -r[:-1] / r[-1]


def _ldp_candidate(G, h):
    """Lawson and Hanson's point from one NNLS, or None (see ``_point_from_residual``)."""
    return _point_from_residual(_ldp_nnls(G, h)[1])


def _ldp(G, h):
    """min ||x|| subject to G x >= h, by one NNLS. None when no solution is found and verified."""
    x = _ldp_candidate(G, h)
    return x if x is not None and _satisfies(G, h, x) else None


def _certifies_infeasible(G, h, u, tol=1e-9):
    """Whether u proves that G x >= h has no solution: u >= 0, G'u = 0 and h'u > 0 (Farkas).

    G'u is checked to tol times the size of its terms, and h'u must be near the 1 that
    the NNLS aims at when the rows are infeasible.
    """
    scale = float((np.abs(G).T @ u).max(initial=0.0))
    return bool(u.min(initial=0.0) >= 0.0 and h @ u > 0.5
                and np.abs(G.T @ u).max(initial=0.0) <= tol * scale)


def ldp_verdict(G, h):
    """Whether G x >= h has a solution, with the evidence: (True, x), (False, u) or (None, None).

    One NNLS decides both ways. x comes back only when it satisfies the rows, u only when
    it is a Farkas certificate; when neither checks, the caller decides by another method.
    """
    u, r = _ldp_nnls(G, h)
    x = _point_from_residual(r)
    if x is not None and _satisfies(G, h, x):
        return True, x
    if _certifies_infeasible(G, h, u):
        return False, u
    return None, None


def _constraints_feasible(G, h):
    """Whether G y >= h has a solution: from an evidence-carrying verdict, or HiGHS when it has none."""
    verdict, _ = ldp_verdict(G, h)
    if verdict is None:
        res = linprog(np.zeros(G.shape[1]), A_ub=-G, b_ub=-h, bounds=(None, None), method="highs")
        return res.status == 0
    return verdict


def _factor(Ga):
    """G_A's SVD and numerical rank, for the active-set solve and its multipliers."""
    U, S, Vt = np.linalg.svd(Ga, full_matrices=True)
    return U, S, Vt, int(np.sum(S > max(Ga.shape) * np.finfo(float).eps * S[0]))


def kkt_backward_error(E, f, G, h, y, s1=None, factored=None):
    """How far y is from the KKT conditions of min ||E y - f|| subject to G y >= h, relative to the data.

    With r = E y - f and g = E'r (E'E is never formed), the rows within VERIFY of their
    bound are taken as active and the multipliers fit g = G_A' lam by least squares. The
    error is the larger of the stationarity residual ||g - G_A' lam|| and the most negative
    multiplier, divided by s_1 (s_1 ||y|| + ||f||), the size of g's terms. It reads as a
    backward error, so it does not grow with the condition number: on 4,192 face fits from
    the Basque specification it was at most 1.3e-14 up to a condition number of 2.6e4,
    and an answer moved 1e-6 along a direction free of its active rows scored at least
    1.6e-10 (agents_solvers.md, rule 2). ``s1``, E's largest singular value, is computed
    when not given; ``factored``, the active rows with their ``_factor``, likewise.
    """
    s1 = float(np.linalg.norm(E, 2)) if s1 is None else float(s1)
    g = E.T @ (E @ y - f)
    act, fac = factored if factored is not None else (_active_rows(G, h, y), None)
    if act.size:
        U, S, Vt, r = fac if fac is not None else _factor(G[act])
        lam = U[:, :r] @ ((Vt[:r] @ g) / S[:r])            # the least squares solution of G_A' lam = g
        stat, neg = float(np.linalg.norm(g - G[act].T @ lam)), float(max(0.0, -lam.min()))
    else:
        stat, neg = float(np.linalg.norm(g)), 0.0
    size = s1 * (s1 * float(np.linalg.norm(y)) + float(np.linalg.norm(f)))
    return max(stat, neg) / size if size > 0.0 else max(stat, neg)


def _active_rows(G, h, y):
    """The rows of G y >= h that y meets to VERIFY times the size of the row's terms."""
    return np.flatnonzero(G @ y - h <= VERIFY * (np.linalg.norm(G, axis=1) * np.linalg.norm(y) + np.abs(h)))


def _polish(E, f, G, h, y):
    """The least squares point on y's active rows: min ||E y - f|| subject to G_A y = h_A.

    Once the active set is known the problem has equality constraints only (Björck,
    Numerical Methods in Matrix Computations, p. 371). It is solved on the face's own
    data in the null space of G_A, from orthogonal factorizations and with no inverse
    of E formed (p. 372), so the active rows hold to rounding and not to the digits the
    least distance reduction kept.

    Returns the point with the active rows and their ``_factor``, which certify it.
    """
    act = _active_rows(G, h, y)
    if not act.size:
        return np.linalg.lstsq(E, f, rcond=None)[0], (act, None)
    U, S, Vt, r = fac = _factor(G[act])
    yp = Vt[:r].T @ ((U[:, :r].T @ h[act]) / S[:r])
    Q2 = Vt[r:].T
    if Q2.shape[1]:
        yp = yp + Q2 @ np.linalg.lstsq(E @ Q2, f - E @ yp, rcond=None)[0]
    return yp, (act, fac)


def _lsi(E, f, G, h):
    """min ||E y - f|| subject to G y >= h. None when the reduced problem has no verified solution.

    The reduction divides by R = S V' from the thin SVD E = U S V', so it needs E of full
    column rank, decided at max(L, n) eps times s_1 (agents_solvers.md, rule 6). A face
    with more free weights than validation periods, or with two donors whose validation
    paths agree to rounding, is refused. A full-rank face is solved whatever its
    condition number. The reduction's point names the active rows, the fit is solved on
    them directly (``_polish``) and certified with ``kkt_backward_error``, and a point
    that breaks a row or misses ``CERTIFY`` raises. On the Basque specification the faces reach a condition
    number of 2.6e4, and against a 50-digit reference their weights were right to
    9.1e-9 and their losses to 1.2e-11; the weights' error grows like cond(E)^2 eps,
    the condition of the problem itself (Björck, Numerical Methods in Matrix
    Computations, Theorem 2.2.6), and the loss's does not.

    The least-distance point x is read off a residual of size 1 / (1 + ||x||^2), so it keeps
    about 16 - 2 log10 ||x|| digits, and ||x|| is in the outcomes' units: West German GDP
    refitted over 1960-1990 gives ||x|| near 1.5e4 and eight digits lost
    (fixtures/exact/scale_incident.json). Every point of the simplex lies within 1 of its
    centre, so ||R y|| <= s_1 and the least-distance point of a feasible piece has
    ||x|| <= c = s_1 + ||U'f||; the step is solved in units of c, where ||x|| <= 1.
    """
    U, sv, Vt = np.linalg.svd(E, full_matrices=False)
    cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
    if E.shape[0] < E.shape[1] or not sv[-1] > max(E.shape) * np.finfo(float).eps * sv[0]:
        raise MlsynthDataError(
            f"the face's validation block is rank-deficient ({E.shape[0]} periods, {E.shape[1]} free "
            f"weights, condition number {cond:.1e}); the exact least-squares reduction needs full "
            "column rank")
    fq = U.T @ f
    c = sv[0] + float(np.linalg.norm(fq))
    GRi = c * (G @ Vt.T) / sv                            # G R^{-1} = G V S^{-1}, in units of c
    x = _ldp(GRi, h - GRi @ fq / c)
    if x is None:
        return None
    y, factored = _polish(E, f, G, h, Vt.T @ ((c * x + fq) / sv))
    err = kkt_backward_error(E, f, G, h, y, s1=sv[0], factored=factored) if _satisfies(G, h, y) else np.inf
    if not err <= CERTIFY:
        raise MlsynthEstimationError(
            f"the least-squares reduction's answer on a face with condition number {cond:.1e} misses "
            f"its optimality conditions by {err:.1e} of the data's size, above the certificate's "
            f"{CERTIFY:.0e}")
    return y


@lru_cache(maxsize=None)
def _simplex_basis(m):
    """The simplex's centre 1/m and an orthonormal basis of the directions 1'd = 0, read-only."""
    b0 = np.full(m, 1.0 / m)
    N = np.linalg.svd(np.ones((1, m)))[2][1:].T
    b0.setflags(write=False)
    N.setflags(write=False)
    return b0, N


def simplex_ls(z, Z, Gin=None, hin=None):
    """Return (loss, b) for the program in the module docstring, or (None, None) if infeasible.

    Raises MlsynthDataError when the face is outside the reduction's domain (see ``_lsi``)
    and MlsynthEstimationError when the reduction's answer fails its constraint check or
    its optimality certificate, so that a numerical failure is never read as an empty
    piece.
    """
    b0, N = _simplex_basis(Z.shape[1])
    if N.shape[1] == 0:                                   # one donor: the simplex is its vertex
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
            if _constraints_feasible(G, h):               # the constraints alone, without R^{-1}
                raise MlsynthEstimationError("the least-squares reduction found no solution on a piece whose "
                                 "constraints have one; the reduction failed")
            return None, None
        b = b0 + N @ y
        if b.min() < -VERIFY or (Gin is not None and len(Gin) and not _satisfies(Gin, hin, b)):
            raise MlsynthEstimationError("the least-squares reduction returned a point that breaks its constraints")
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
