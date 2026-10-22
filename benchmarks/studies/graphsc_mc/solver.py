"""Eq. (2) over ``w`` in ``R^J`` by an exact active-set search, certified.

Graph-SC's signed objective is a lasso on a stacked design,

    f(w) = 1/2 ||M w - b||^2 + lambda_1 ||w||_1,
    M = [X0; sqrt(lambda_g) R; sqrt(lambda_sum) 1'],  b = [x1; 0; sqrt(lambda_sum)],

with ``R'R = L``. The search runs on the Gram form ``H = M'M``, ``c = M'b``: it
adds the zero coefficient with the largest violation of the optimality
condition, solves ``H_AA w_A = c_A - lambda_1 s_A`` on the signed support, and
line-searches from the current point to that solution over the points where a
coefficient changes sign, keeping the best (Lee, Battle, Raina & Ng 2007,
"feature-sign search").

Termination. A step that lowers ``f`` cannot return to a sign pattern already
left, and there are finitely many patterns. When no candidate on the segment
lowers ``f``, the search keeps the coordinates whose sign the reduced solve
confirms, and that move carries no descent guarantee. So the guarantee is the
cap, ``max_iter = 20 J + 100``, and its status; optimality is the certificate's
claim, not the pivot rule's (NW pp. 477-478 on cycling in active-set QP).

Accuracy. The Gram form squares the condition number of ``M`` (NW pp. 251-253),
so it is used to find the support only. The returned point is re-solved on that
support by QR of ``M_A`` and certified by the proximal-gradient KKT residual on
the caller's own data; the certificate, not a condition estimate, measures what
the conditioning cost. A capped search, a singular reduced system and a point
that fails the certificate run the interior-point solve of the replication
oracle, and the better certified of the two points is kept. A point no solve
certifies is returned with a :class:`CertificateWarning`.

Tolerances (``agents_solvers.md`` rules 3 and 4), from a measured spread over
170 problems with data levels from 1e-3 to 1e3, signed weights outside the
donor hull, and draws from all four simulation families:

- an optimality condition holds when its residual is within ``64 eps`` of the
  size of its terms, ``|c_i| + (|H||w|)_i + lambda_1``; the worst measured was
  ``1.9 eps``;
- the returned point is certified at a KKT residual of ``1e-12``; the worst
  measured was ``3.2e-16``. The interior-point solve is tolerance-limited, so a
  point it alone certifies is accepted at the oracle's ``1e-7``.

Warm starts are what make cross-fitting cheap: each fold drops one block of
pre-periods, so the full-sample solution is close to every fold's.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..graphsc_replicate import oracle as G

EPS = np.finfo(float).eps
ZERO = 64 * EPS
KKT_TOL = 1e-12
FALLBACK_TOL = 1e-7


class CertificateWarning(RuntimeWarning):
    """A weight vector that no solve certified, returned anyway so the run can count it."""


def gram(x1, X0, L, lamg, lamsum):
    """``(H, c)`` of the stacked design, without forming it."""
    J = X0.shape[1]
    return X0.T @ X0 + lamg * L + lamsum * np.ones((J, J)), X0.T @ x1 + lamsum * np.ones(J)


def stacked(x1, X0, L, lamg, lamsum):
    """``(M, b)`` with ``1/2 ||M w - b||^2`` equal to eq. (2)'s smooth part up to a constant."""
    ev, U = np.linalg.eigh(L)
    R = np.sqrt(np.clip(ev, 0.0, None))[:, None] * U.T             # R'R = L
    J = X0.shape[1]
    M = np.vstack([X0, np.sqrt(lamg) * R, np.sqrt(lamsum) * np.ones((1, J))])
    return M, np.r_[x1, np.zeros(J), np.sqrt(lamsum)]


def active_set(H, c, lam, w0=None, max_iter=None):
    """``(w, steps, status)`` minimising ``1/2 w'Hw - c'w + lam ||w||_1``.

    ``status`` is ``"optimal"`` when every coefficient meets its optimality
    condition to ``64 eps`` of the size of its terms, ``"cap"`` when
    ``max_iter`` steps ran out first, and ``"singular"`` when a reduced system
    ``H_AA`` could not be solved.
    """
    J = len(c)
    max_iter = 20 * J + 100 if max_iter is None else int(max_iter)
    absH = np.abs(H)
    f = lambda x: 0.5 * x @ H @ x - c @ x + lam * np.abs(x).sum()
    w = np.zeros(J) if w0 is None else np.asarray(w0, dtype=float).copy()
    for step in range(max_iter):
        g = c - H @ w
        tol = ZERO * (np.abs(c) + absH @ np.abs(w) + lam)
        zero = w == 0                       # this loop stores exact zeros; the test reads them back
        act = ~zero
        settled = np.all(np.abs(g[act] - lam * np.sign(w[act])) <= tol[act]) if act.any() else True
        viol = np.where(zero, np.abs(g) - lam - tol, -np.inf)
        signs = np.sign(w)
        if settled:
            if viol.max() <= 0.0:
                return w, step, "optimal"
            i = int(np.argmax(viol))
            signs[i] = 1.0 if g[i] > 0 else -1.0
        A = np.flatnonzero(signs)
        try:
            target = np.zeros(J)
            target[A] = np.linalg.solve(H[np.ix_(A, A)], c[A] - lam * signs[A])
        except np.linalg.LinAlgError:
            return w, step + 1, "singular"
        d = target - w
        with np.errstate(divide="ignore", invalid="ignore"):
            crossings = -w[A] / d[A]
        best, best_f = w, f(w)
        for t in [1.0] + [float(x) for x in crossings if 0.0 < x < 1.0]:
            x = w + t * d
            if t < 1.0:
                # The coordinate that crossed is zero to the rounding of its two terms.
                x[np.abs(x) <= ZERO * (np.abs(w) + np.abs(t * d))] = 0.0
            fx = f(x)
            if fx < best_f:
                best, best_f = x, fx
        w = best if best is not w else np.where(np.sign(target) == signs, target, 0.0)
    return w, max_iter, "cap"


def _polish(w, M, b, lam):
    """Re-solve ``w`` on its signed support by QR of ``M_A``; ``None`` if a sign moves or ``R`` is singular."""
    A = np.flatnonzero(w)
    if A.size == 0:
        return w
    s = np.sign(w[A])
    Q, Rq = np.linalg.qr(M[:, A])
    try:
        z = np.linalg.solve(Rq.T, lam * s)                           # R' z = lam s
        wA = np.linalg.solve(Rq, Q.T @ b - z)                       # R w_A = Q'b - z
    except np.linalg.LinAlgError:
        return None
    if not np.array_equal(np.sign(wA), s):
        return None
    out = np.zeros_like(w)
    out[A] = wA
    return out


def solve_signed(x1, X0, L, lam1, lamg, lamsum, w0=None, max_iter=None):
    """``(w, info)``: eq. (2) over ``R^J``.

    ``info`` holds ``status`` (the search's), ``steps``, ``polished`` (whether
    the QR re-solve was kept), ``fallback`` (whether the interior-point solve
    ran), and ``kkt``, the residual of the point returned.
    """
    x1, X0, L = (np.asarray(a, dtype=float) for a in (x1, X0, L))
    H, c = gram(x1, X0, L, lamg, lamsum)
    w, steps, status = active_set(H, c, lam1, w0=w0, max_iter=max_iter)
    info = {"status": status, "steps": steps, "polished": False, "fallback": False}
    if status == "optimal":
        M, b = stacked(x1, X0, L, lamg, lamsum)
        p = _polish(w, M, b, lam1)
        if p is not None:
            w, info["polished"] = p, True
    kkt = G.kkt_residual(w, x1, X0, L, lam1, lamg, lamsum, signed=True)
    if status != "optimal" or not kkt <= KKT_TOL:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # re-certified below
            v = G.solve_signed(x1, X0, L, lam1, lamg, lamsum)
        kv = G.kkt_residual(v, x1, X0, L, lam1, lamg, lamsum, signed=True)
        info["fallback"] = True
        if not kkt <= kv:                                            # the better certified point
            w, kkt = v, kv
    info["kkt"] = kkt
    if not kkt <= FALLBACK_TOL:
        warnings.warn(CertificateWarning(
            f"eq. (2), signed: no solve certified the weights (KKT residual {kkt:.2e})"), stacklevel=2)
    return w, info
