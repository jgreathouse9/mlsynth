"""The study's own linear-programming solvers, in NumPy.

Two programmes, both L1 losses with linear constraints on the coefficients:

* ``rank_lad``: ``min sum_{i<j} h_i h_j |e_i - e_j|``, ``e = y - X beta``,
  subject to ``G beta >= g`` and ``E beta = f``. With no constraints this is a
  weighted Wilcoxon fit (Section 2.3); with ``beta`` on the simplex it is the
  model-weight programme of Section 2.4 (``simplex_rank_lad``).
* ``lad``: ``min sum_t c_t |e_t|`` under the same constraints; on the simplex it
  is the weights-on-levels ablation (``simplex_lad``).

Both walk vertex to vertex and stop at a vertex that a duality-gap certificate
accepts to the relative tolerance ``TOL``.

``rank_lad`` works on periods, never on pairs (see ``solver_pairfree``). A vertex
is a labelling of the periods into tie groups plus a set ``A`` of active
inequalities, with ``sum(|G| - 1) + |A| + rows(E) = p`` equations. At a vertex
the subgradient condition reads ``X'(a + psi) + G_A' nu + E' lambda = 0``, with
``a`` the rank scores of the untied pairs and ``psi`` the net flows the tied pairs
carry; the multipliers come from one ``p x p`` solve with the transposed vertex
matrix. The vertex is optimal when every tie group passes the prefix-cut test
and ``nu >= 0``. Otherwise the solver either splits a group along its most
violated prefix (directional derivative ``h(U) h(G \\ U) - psi(U)``) or releases
the active inequality with the most negative ``nu`` (directional derivative
``nu_k``), and steps to the first residual crossing where the slope turns
non-negative, or to the first inequality it meets, whichever comes first.

``lad`` is the same walk with the data rows in place of tie groups: a vertex is
a set of data rows with zero residual plus active inequalities, and the
multiplier of a basic data row must lie in ``[-c_t, c_t]``. It is the
Barrodale-Roberts step for the constrained problem.

Both pivot on ``y`` plus ``solver_pairfree.perturbation(y)`` and certify the
final vertex on ``y`` itself by the duality gap: the dual found on the
perturbed data is feasible for the original, since the dual constraints do not
involve ``y``. The LP texts read for the study (Pan 2023; Vanderbei 2020;
Bazaraa, Jarvis and Sherali 2010; Cottle and Thapa 2017; Luenberger and Ye
2021; Schrijver 1986; Dantzig and Thapa 2003) give anti-cycling rules for the
primal simplex only (lexicographic, Bland, perturbation); none gives one for a
bounded dual simplex, which is why the certificate, not the pivot rule, carries
the claim of optimality. A vertex that fails it raises ``CertificateWarning``.

Numerics follow ``agents/agents_solvers.md``, as in ``solver_pairfree``: every
tolerance is ``TOL`` times the size of what it compares, with no absolute floor,
so the answer does not depend on the units of ``y``, ``X`` or the weights
(Nocedal and Wright 2006, pp. 26-27); ``rank_lad`` removes the column means of
``y`` and ``X`` first, and ``simplex_lad`` a common level, since neither changes
its programme (p. 615); a residual or a slack below the rounding error of the
terms it is computed from counts as zero (Russinoff 2022, Lemma 6.45), and an
inactive inequality violated by more than ``TOL`` is an error. The pivot rule
compares violations in their own units divided by their own scale.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from .solver_pairfree import (TOL, CertificateWarning, _centre, _check_weights, _slope_tol,
                              certificate, crash, pairwise_objective, perturbation, rank_scores)

_EPS = np.finfo(float).eps


def _constraints(G, g, E, f, p):
    G = np.zeros((0, p)) if G is None else np.atleast_2d(np.asarray(G, dtype=float))
    g = np.zeros(0) if g is None else np.atleast_1d(np.asarray(g, dtype=float))
    E = np.zeros((0, p)) if E is None else np.atleast_2d(np.asarray(E, dtype=float))
    f = np.zeros(0) if f is None else np.atleast_1d(np.asarray(f, dtype=float))
    return G, g, E, f


def _ties(labels):
    i = np.flatnonzero(labels != np.arange(len(labels)))
    return i, labels[i]


def _first_crossing(e, u, ld, h, Fp):
    """The first residual crossing at which the slope ``Fp`` turns non-negative."""
    n = len(e)
    app = (e[:, None] < e[None, :]) & (u[:, None] > u[None, :]) & (ld[:, None] != ld[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(app, (e[None, :] - e[:, None]) / (u[:, None] - u[None, :]), np.inf).ravel()
    inc = np.where(app, 2 * h[:, None] * h[None, :] * (u[:, None] - u[None, :]), 0.0).ravel()
    o = np.argsort(tt, kind="stable")
    slope = Fp + np.cumsum(inc[o])
    ok = slope >= -_slope_tol(h, u)
    if not ok.any() or not np.isfinite(tt[o[np.argmax(ok)]]):
        return np.inf, None
    flat = o[np.argmax(ok)]
    return tt[flat], (flat // n, flat % n)


def _first_hit(G, g, beta, v, inactive):
    """The first inactive inequality the step ``beta + t v`` meets.

    ``beta`` comes from a ``p x p`` solve, so its entries carry an error of
    about ``eps max|beta|``; a slack within 64 times that, row by row, is zero,
    and a degenerate step then has ``t = 0`` exactly. A slack below ``-TOL``
    times the same scale means the iterate left the feasible set: an error.
    """
    if not inactive:
        return np.inf, None
    idx = np.asarray(inactive)
    Gi = G[idx]
    Gv = Gi @ v
    slack = Gi @ beta - g[idx]
    mag = np.abs(Gi).sum(axis=1) * np.abs(beta).max(initial=0.0) + np.abs(g[idx])
    if np.any(slack < -TOL * mag):
        k = int(np.argmin(slack / np.where(mag > 0, mag, 1.0)))
        raise RuntimeError(f"inactive inequality {int(idx[k])} is violated by {-slack[k]:.3g}: "
                           "the iterate left the feasible set")
    slack = np.where(slack <= 64 * _EPS * mag, 0.0, slack)
    down = Gv < -64 * _EPS * (np.abs(Gi) @ np.abs(v))
    if not down.any():
        return np.inf, None
    t = np.where(down, slack / np.where(down, -Gv, 1.0), np.inf)
    k = int(np.argmin(t))
    return t[k], int(idx[k])


def rank_lad(y, X, h, G=None, g=None, E=None, f=None, labels: Optional[np.ndarray] = None,
             active: Optional[Sequence[int]] = None, max_pivots: int = 5000) -> np.ndarray:
    """``argmin sum_{i<j} h_i h_j |e_i - e_j|`` subject to ``G beta >= g``, ``E beta = f``.

    ``labels`` and ``active`` give a starting vertex; without constraints the
    default is ``solver_pairfree.crash``. Returns ``beta``; a final vertex that
    fails the certificate raises ``CertificateWarning``.
    """
    y, X, h = (np.asarray(a, dtype=float) for a in (y, X, h))
    _check_weights(h)
    n, p = X.shape
    G, g, E, f = _constraints(G, g, E, f, p)
    y, X = _centre(y, X)            # the objective reads differences; constraints act on beta
    if labels is None:
        if len(g) or len(f):
            raise ValueError("a starting vertex is needed when there are constraints")
        labels, active = crash(y[None], X[None], h[None])[0], []
    yp = y + perturbation(y)
    labels, active = _rank_lad_pivot(yp, X, h, G, g, E, f,
                                     np.asarray(labels).copy(), list(active or []), max_pivots)
    beta = _rank_vertex(y, X, G, g, E, f, labels, active)
    if not _rank_certified(y, X, h, G, g, E, f, beta, active,
                           *_rank_dual(yp, X, h, G, g, E, f, labels, active)):
        warnings.warn(CertificateWarning("rank_lad: the final vertex failed the duality-gap certificate"),
                      stacklevel=2)
    return beta


def _rank_certified(y, X, h, G, g, E, f, beta, active, phi, nu, lam) -> bool:
    """``rank_lad``'s certificate for ``beta`` with the dual ``(phi, nu, lambda)``.

    ``phi`` is dual feasible when it passes Hoffman's cut test on all the
    periods; the gap is tested against ``TOL h(.)^2`` times the largest
    ``|y_i| + |x_i|'|beta|``, the size of the terms each residual comes from.
    """
    H2 = h.sum() ** 2
    feasible = (certificate(h[None], np.zeros((1, len(y)), int), phi[None])[0][0] <= TOL * H2 / 4
                and abs(phi.sum()) <= TOL * H2 / 4)
    mag = float(np.max(np.abs(y) + np.abs(X) @ np.abs(beta)))
    F = float(pairwise_objective(y[None], X[None], h[None], beta[None])[0])
    return bool(feasible and _gap_ok(F, phi, y - X @ beta, X, G, g, E, f, beta, active, nu, lam,
                                     TOL * H2 * mag))


def _rank_dual(y, X, h, G, g, E, f, labels, active):
    """``(phi, nu, lambda)`` at a vertex: net flows and constraint multipliers."""
    n = len(y)
    beta = _rank_vertex(y, X, G, g, E, f, labels, active)
    e = (y - X @ beta)[labels]
    a = rank_scores(h[None], labels[None], e[None])[0]
    ti, tj = _ties(labels)
    M = np.vstack([X[ti] - X[tj], G[active], E])
    z = np.linalg.solve(M.T, -(X.T @ a))
    psi = np.zeros(n)
    np.add.at(psi, ti, z[:len(ti)])
    np.add.at(psi, tj, -z[:len(ti)])
    return a + psi, z[len(ti):len(ti) + len(active)], z[len(ti) + len(active):]


def _gap_ok(F, d, e, X, G, g, E, f, beta, active, nu, lam, gap_tol):
    """Weak duality: ``F(b) >= d'(y - X b) >= d'y + nu'g_A + lambda'f`` for every feasible ``b``.

    Valid when ``d`` is dual feasible, ``nu >= 0`` and ``X'd + G_A'nu + E'lambda = 0``;
    a zero gap at a feasible ``beta`` proves it optimal. With stationarity the
    bound equals ``d'(y - X beta) - nu'(G_A beta - g_A) - lambda'(E beta - f)``.
    Stationarity and ``nu >= 0`` are tested against ``TOL`` times the largest
    sum of the magnitudes of the terms in ``X'd + G_A'nu + E'lambda``, and the
    gap against ``gap_tol``, which the caller builds from the size of its
    objective's terms. Feasibility of row ``k`` is tested against ``TOL`` times
    ``sum|G_k| max|beta| + |g_k|``: ``beta`` comes from a ``p x p`` solve whose
    error is about ``eps max|beta|`` in every entry, so a weight held at zero
    reads as ``-1e-17``, which a test scaled by that weight's own size rejects.
    """
    GA = G[active]
    stat = X.T @ d + GA.T @ nu + E.T @ lam
    s_scale = (np.abs(X).T @ np.abs(d) + np.abs(GA).T @ np.abs(nu)
               + np.abs(E).T @ np.abs(lam)).max(initial=0.0)
    bmax = np.abs(beta).max(initial=0.0)
    feasible = (np.all(G @ beta - g >= -TOL * (np.abs(G).sum(axis=1) * bmax + np.abs(g)))
                and np.all(np.abs(E @ beta - f) <= TOL * (np.abs(E).sum(axis=1) * bmax + np.abs(f))))
    nu_ok = not len(nu) or (nu * np.abs(GA).max(axis=1)).min() >= -TOL * s_scale
    lower = float(d @ e) - float(nu @ (GA @ beta - g[active])) - float(lam @ (E @ beta - f))
    return bool(feasible and nu_ok and np.abs(stat).max(initial=0.0) <= TOL * s_scale
                and F - lower <= gap_tol)


def _rank_vertex(y, X, G, g, E, f, labels, active):
    ti, tj = _ties(labels)
    M = np.vstack([X[ti] - X[tj], G[active], E])
    return np.linalg.solve(M, np.r_[y[ti] - y[tj], g[active], f])


def _rank_lad_pivot(y, X, h, G, g, E, f, labels, active, max_pivots):
    """The pivot loop of ``rank_lad`` on whatever ``y`` it is given; returns the final vertex."""
    n, p = X.shape
    cut_scale = h.sum() ** 2 / 4
    for _ in range(max_pivots):
        ti, tj = _ties(labels)
        M = np.vstack([X[ti] - X[tj], G[active], E])
        if M.shape[0] != p:
            raise ValueError(f"vertex has {M.shape[0]} equations, expected {p}")
        beta = np.linalg.solve(M, np.r_[y[ti] - y[tj], g[active], f])
        e = (y - X @ beta)[labels]
        a = rank_scores(h[None], labels[None], e[None])[0]
        grad = X.T @ a
        z = np.linalg.solve(M.T, -grad)
        theta, nu = z[:len(ti)], z[len(ti):len(ti) + len(active)]
        psi = np.zeros(n)
        np.add.at(psi, ti, theta)
        np.add.at(psi, tj, -theta)
        viol, grp, U = (v[0] for v in certificate(h[None], labels[None], psi[None]))
        # each violation divided by its own scale: h(G)^2 / 4 for a cut, max|X'a| for nu
        nu_scale = np.abs(grad).max(initial=0.0)
        nu_rel = nu / nu_scale if nu_scale > 0 else np.zeros_like(nu)
        k_nu = int(np.argmin(nu_rel)) if len(nu) else -1
        cut_bad = viol > TOL * cut_scale
        nu_bad = k_nu >= 0 and nu_rel[k_nu] < -TOL
        if not (cut_bad or nu_bad):
            return labels, active
        if cut_bad and (not nu_bad or viol / cut_scale >= -nu_rel[k_nu]):
            ld = labels.copy()
            W = (labels == grp) & ~U
            rU, rW = np.flatnonzero(U)[0], np.flatnonzero(W)[0]
            ld[U], ld[W] = rU, rW
            ti2, tj2 = _ties(ld)
            rest = active
            Md = np.vstack([X[ti2] - X[tj2], G[rest], E, X[rU] - X[rW]])
            rhs = np.zeros(p)
            rhs[-1] = -1.0                                           # U rises against G \ U
        else:
            ld = labels
            kk = active[k_nu]
            rest = [k for k in active if k != kk]
            Md = np.vstack([X[ti] - X[tj], G[rest], E, G[kk]])
            rhs = np.zeros(p)
            rhs[-1] = 1.0                                            # into the feasible side
        v = np.linalg.solve(Md, rhs)
        u = (-(X @ v))[ld]
        Fp = float(rank_scores(h[None], ld[None], e[None], u[None])[0] @ u)
        t_cross, pair = _first_crossing(e, u, ld, h, Fp)
        t_hit, k_hit = _first_hit(G, g, beta, v, [k for k in range(len(g)) if k not in rest])
        if not (np.isfinite(t_cross) or np.isfinite(t_hit)):  # pragma: no cover - bounded below
            raise RuntimeError("unbounded step")
        if t_hit <= t_cross:
            labels, active = ld, rest + [k_hit]
        else:
            la, lb = ld[pair[0]], ld[pair[1]]
            labels = np.where((ld == la) | (ld == lb), min(la, lb), ld)
            active = rest
    raise RuntimeError("pivot limit")                               # pragma: no cover


def simplex_rank_lad(y, P, h) -> np.ndarray:
    """Section 2.4: ``argmin_w sum_{t<t'} h_t h_t' |(y_t - y_t') - (P_t - P_t') w|`` on the simplex.

    Starts from the best single model, a vertex of the simplex.
    """
    y, P, h = (np.asarray(a, dtype=float) for a in (y, P, h))
    n, S = P.shape
    if S == 1:
        return np.ones(1)
    E_res = y[:, None] - P
    F = [float(rank_scores(h[None], np.arange(n)[None], E_res[:, s][None])[0] @ E_res[:, s])
         for s in range(S)]
    s0 = int(np.argmin(F))
    w = rank_lad(y, P, h, G=np.eye(S), g=np.zeros(S), E=np.ones((1, S)), f=[1.0],
                 labels=np.arange(n), active=[k for k in range(S) if k != s0])
    # The certificate held w >= -TOL |w| and sum(w) = 1 to TOL, or warned; this
    # removes only that rounding.
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def lad(y, X, c, G=None, g=None, E=None, f=None, basis: Sequence[int] = (),
        active: Sequence[int] = (), max_pivots: int = 5000) -> np.ndarray:
    """``argmin sum_t c_t |y_t - x_t' beta|`` subject to ``G beta >= g``, ``E beta = f``.

    ``basis`` (data rows with zero residual) and ``active`` give the starting
    vertex. Returns ``beta``; a final vertex that fails the certificate raises
    ``CertificateWarning``.
    """
    y, X, c = (np.asarray(a, dtype=float) for a in (y, X, c))
    _check_weights(c)
    n, p = X.shape
    G, g, E, f = _constraints(G, g, E, f, p)
    yp = y + perturbation(y)
    B, active = _lad_pivot(yp, X, c, G, g, E, f, list(basis), list(active), max_pivots)
    M = np.vstack([X[B], G[active], E])
    beta = np.linalg.solve(M, np.r_[y[B], g[active], f])
    if not _lad_certified(y, X, c, G, g, E, f, beta, active, *_lad_dual(yp, X, c, G, g, E, f, B, active)):
        warnings.warn(CertificateWarning("lad: the final vertex failed the duality-gap certificate"),
                      stacklevel=2)
    return beta


def _lad_dual(y, X, c, G, g, E, f, B, active):
    """``(d, nu, lambda)`` at the vertex ``(B, active)`` of the programme on ``y``.

    ``d`` is ``c`` times the sign of each nonbasic residual; the basic rows'
    entries and the constraint multipliers come from one solve with the
    transposed vertex matrix.
    """
    n = len(y)
    M = np.vstack([X[B], G[active], E])
    b = np.linalg.solve(M, np.r_[y[B], g[active], f])
    nb = np.ones(n, bool)
    nb[B] = False
    d = c * np.sign(y - X @ b) * nb
    z = np.linalg.solve(M.T, -(X.T @ d))
    d[B] = z[:len(B)]
    return d, z[len(B):len(B) + len(active)], z[len(B) + len(active):]


def _lad_certified(y, X, c, G, g, E, f, beta, active, d, nu, lam) -> bool:
    """``lad``'s certificate for ``beta`` with the dual ``(d, nu, lambda)``.

    ``d`` is dual feasible when ``|d| <= c`` to ``TOL max(c)``; the gap is tested
    against ``TOL sum(c)`` times the largest ``|y_t| + |x_t|'|beta|``.
    """
    feasible = np.all(np.abs(d) <= c + TOL * c.max())
    F = float(np.sum(c * np.abs(y - X @ beta)))
    mag = float(np.max(np.abs(y) + np.abs(X) @ np.abs(beta)))
    return bool(feasible and _gap_ok(F, d, y - X @ beta, X, G, g, E, f, beta, active, nu, lam,
                                     TOL * c.sum() * mag))


def _lad_pivot(y, X, c, G, g, E, f, B, active, max_pivots):
    """The pivot loop of ``lad`` on whatever ``y`` it is given; returns the final vertex."""
    n, p = X.shape
    for _ in range(max_pivots):
        M = np.vstack([X[B], G[active], E])
        if M.shape[0] != p:
            raise ValueError(f"vertex has {M.shape[0]} equations, expected {p}")
        beta = np.linalg.solve(M, np.r_[y[B], g[active], f])
        e = y - X @ beta
        e[B] = 0.0
        # a residual below the rounding of its terms is a tie
        tie = np.abs(e) <= 64 * _EPS * (np.abs(y) + np.abs(X) @ np.abs(beta))
        nb = np.ones(n, bool)
        nb[B] = False
        a = c * np.sign(e) * (nb & ~tie)
        grad = X.T @ a
        z = np.linalg.solve(M.T, -grad)
        d, nu = z[:len(B)], z[len(B):len(B) + len(active)]
        # each rate divided by its own scale: c for a basic row, max|X'a| for nu
        nu_scale = np.abs(grad).max(initial=0.0)
        rates = np.r_[(c[B] - np.abs(d)) / c.max(),
                      nu / nu_scale if nu_scale > 0 else np.zeros_like(nu)]
        if not len(rates) or rates.min() >= -TOL:
            return B, active
        k = int(np.argmin(rates))
        if k < len(B):
            row = B[k]
            B2, rest = [r for r in B if r != row], active
            Md = np.vstack([X[B2], G[rest], E, X[row]])
            rhs = np.zeros(p)
            rhs[-1] = -np.sign(d[k])
        else:
            kk = active[k - len(B)]
            B2, rest = B, [r for r in active if r != kk]
            Md = np.vstack([X[B2], G[rest], E, G[kk]])
            rhs = np.zeros(p)
            rhs[-1] = 1.0
        v = np.linalg.solve(Md, rhs)
        xv = X @ v
        zero = tie.copy()
        zero[B2] = False
        Fp = float(np.sum(np.where(zero, c * np.abs(xv), -c * np.sign(e) * xv)))
        tol = TOL * float(np.sum(c * np.abs(xv)))            # the size of the slope's terms
        moving = np.abs(xv) > 64 * _EPS * (np.abs(X) @ np.abs(v))
        cand = ~zero & (np.sign(e) == np.sign(xv)) & moving
        cand[B2] = False
        idx = np.flatnonzero(cand)
        t_cross, enter = np.inf, None
        if idx.size:
            tt = e[idx] / xv[idx]
            o = np.argsort(tt, kind="stable")
            slope = Fp + np.cumsum(2 * c[idx][o] * np.abs(xv[idx][o]))
            ok = slope >= -tol
            if ok.any():
                q = int(np.argmax(ok))
                t_cross, enter = tt[o[q]], int(idx[o[q]])
        t_hit, k_hit = _first_hit(G, g, beta, v, [r for r in range(len(g)) if r not in rest])
        if Fp >= -tol and not (np.isfinite(t_hit) and t_hit == 0.0):
            raise RuntimeError("degenerate vertex: the chosen edge does not descend")
        if not (np.isfinite(t_cross) or np.isfinite(t_hit)):  # pragma: no cover - bounded below
            raise RuntimeError("unbounded step")
        if t_hit <= t_cross:
            B, active = B2, rest + [k_hit]
        else:
            B, active = B2 + [enter], rest
    raise RuntimeError("pivot limit")                               # pragma: no cover


def simplex_lad(y, P, c) -> np.ndarray:
    """The weights-on-levels ablation: ``argmin_w sum_t c_t |y_t - P_t w|`` on the simplex.

    On the simplex ``(y - m) - (P - m) w = y - P w``, so a common level ``m`` is
    removed first.
    """
    y, P, c = (np.asarray(a, dtype=float) for a in (y, P, c))
    S = P.shape[1]
    if S == 1:
        return np.ones(1)
    m = y.mean()
    y, P = y - m, P - m
    s0 = int(np.argmin([np.sum(c * np.abs(y - P[:, s])) for s in range(S)]))
    w = lad(y, P, c, G=np.eye(S), g=np.zeros(S), E=np.ones((1, S)), f=[1.0],
            active=[k for k in range(S) if k != s0])
    w = np.clip(w, 0.0, None)
    return w / w.sum()
