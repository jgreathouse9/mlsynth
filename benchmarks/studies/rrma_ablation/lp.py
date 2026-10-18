"""The study's own linear-programming solvers, in NumPy.

Two programmes, both L1 losses with linear constraints on the coefficients:

* ``rank_lad``: ``min sum_{i<j} h_i h_j |e_i - e_j|``, ``e = y - X beta``,
  subject to ``G beta >= g`` and ``E beta = f``. With no constraints this is a
  weighted Wilcoxon fit (Section 2.3); with ``beta`` on the simplex it is the
  model-weight programme of Section 2.4 (``simplex_rank_lad``).
* ``lad``: ``min sum_t c_t |e_t|`` under the same constraints; on the simplex it
  is the weights-on-levels ablation (``simplex_lad``).

Both walk vertex to vertex and stop at an exact optimality certificate.

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
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .solver_pairfree import TOL, certificate, crash, perturbation, rank_scores

#: Problems whose final vertex failed the certificate on the unperturbed data.
UNCERTIFIED = {"rank_lad": 0, "lad": 0}


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
    ok = slope >= -TOL * max(1.0, abs(Fp))
    if not ok.any() or not np.isfinite(tt[o[np.argmax(ok)]]):
        return np.inf, None
    flat = o[np.argmax(ok)]
    return tt[flat], (flat // n, flat % n)


def _first_hit(G, g, beta, v, inactive):
    """The first inactive inequality the step ``beta + t v`` meets."""
    if not inactive:
        return np.inf, None
    idx = np.asarray(inactive)
    Gv = G[idx] @ v
    slack = np.maximum(G[idx] @ beta - g[idx], 0.0)
    down = Gv < -1e-14 * max(1.0, np.abs(v).max())
    if not down.any():
        return np.inf, None
    t = np.where(down, slack / np.where(down, -Gv, 1.0), np.inf)
    k = int(np.argmin(t))
    return t[k], int(idx[k])


def rank_lad(y, X, h, G=None, g=None, E=None, f=None, labels: Optional[np.ndarray] = None,
             active: Optional[Sequence[int]] = None, max_pivots: int = 5000) -> np.ndarray:
    """``argmin sum_{i<j} h_i h_j |e_i - e_j|`` subject to ``G beta >= g``, ``E beta = f``.

    ``labels`` and ``active`` give a starting vertex; without constraints the
    default is ``solver_pairfree.crash``. Returns ``beta``.
    """
    y, X, h = (np.asarray(a, dtype=float) for a in (y, X, h))
    n, p = X.shape
    G, g, E, f = _constraints(G, g, E, f, p)
    if labels is None:
        if len(g) or len(f):
            raise ValueError("a starting vertex is needed when there are constraints")
        labels, active = crash(y[None], X[None], h[None])[0], []
    yp = y + perturbation(y)
    labels, active = _rank_lad_pivot(yp, X, h, G, g, E, f,
                                     np.asarray(labels).copy(), list(active or []), max_pivots)
    beta = _rank_vertex(y, X, G, g, E, f, labels, active)
    phi, nu, lam = _rank_dual(yp, X, h, G, g, E, f, labels, active)
    if not _gap_ok(_pair_F(y, X, h, beta), phi, y - X @ beta, X, G, g, E, f, beta, active, nu, lam):
        UNCERTIFIED["rank_lad"] += 1
    return beta


def _pair_F(y, X, h, beta):
    e = y - X @ beta
    return float(rank_scores(h[None], np.arange(len(y))[None], e[None])[0] @ e)


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


def _gap_ok(F, d, e, X, G, g, E, f, beta, active, nu, lam):
    """Weak duality: ``F(beta) >= d'(y - X b) >= d'y + nu'g_A + lambda'f`` for every feasible ``b``.

    Valid when ``d`` is dual feasible, ``nu >= 0`` and ``X'd + G_A'nu + E'lambda = 0``;
    a zero gap at a feasible ``beta`` proves it optimal. ``d'(y - X beta)`` with
    the active rows holding equals that bound, so the gap is computed from it.
    """
    F_scale = max(1.0, abs(F))
    stat = X.T @ d + G[active].T @ nu + E.T @ lam
    s_scale = max(1.0, np.abs(X).max(initial=0.0) * np.abs(d).sum())
    feasible = (not len(g)) or np.all(G @ beta - g >= -1e-9 * max(1.0, np.abs(g).max(initial=0.0)))
    lower = float(d @ e) + float(nu @ (G[active] @ beta - g[active]))
    return (feasible and (not len(nu) or nu.min() >= -1e-9 * s_scale)
            and np.abs(stat).max(initial=0.0) <= 1e-9 * s_scale
            and F - lower <= 1e-9 * F_scale)


def _rank_vertex(y, X, G, g, E, f, labels, active):
    ti, tj = _ties(labels)
    M = np.vstack([X[ti] - X[tj], G[active], E])
    return np.linalg.solve(M, np.r_[y[ti] - y[tj], g[active], f])


def _rank_lad_pivot(y, X, h, G, g, E, f, labels, active, max_pivots):
    """The pivot loop of ``rank_lad`` on whatever ``y`` it is given; returns the final vertex."""
    n, p = X.shape
    tol_cut = TOL * max(1.0, h.sum() ** 2 / 4)
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
        tol_nu = TOL * max(1.0, np.abs(grad).max(initial=0.0))
        k_nu = int(np.argmin(nu)) if len(nu) else -1
        cut_bad = viol > tol_cut
        nu_bad = k_nu >= 0 and nu[k_nu] < -tol_nu
        if not (cut_bad or nu_bad):
            return labels, active
        if cut_bad and (not nu_bad or -viol <= nu[k_nu]):
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
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def lad(y, X, c, G=None, g=None, E=None, f=None, basis: Sequence[int] = (),
        active: Sequence[int] = (), max_pivots: int = 5000) -> np.ndarray:
    """``argmin sum_t c_t |y_t - x_t' beta|`` subject to ``G beta >= g``, ``E beta = f``.

    ``basis`` (data rows with zero residual) and ``active`` give the starting
    vertex. Returns ``beta``.
    """
    y, X, c = (np.asarray(a, dtype=float) for a in (y, X, c))
    n, p = X.shape
    G, g, E, f = _constraints(G, g, E, f, p)
    yp = y + perturbation(y)
    B, active = _lad_pivot(yp, X, c, G, g, E, f, list(basis), list(active), max_pivots)
    M = np.vstack([X[B], G[active], E])
    beta = np.linalg.solve(M, np.r_[y[B], g[active], f])
    # the dual at the final vertex of the perturbed problem
    bp = np.linalg.solve(M, np.r_[yp[B], g[active], f])
    ep = yp - X @ bp
    nb = np.ones(n, bool)
    nb[B] = False
    d = c * np.sign(ep) * nb
    z = np.linalg.solve(M.T, -(X.T @ d))
    d[B] = z[:len(B)]
    nu, lam = z[len(B):len(B) + len(active)], z[len(B) + len(active):]
    dual_feasible = np.all(np.abs(d) <= c * (1 + 1e-9) + 1e-12)
    F = float(np.sum(c * np.abs(y - X @ beta)))
    if not (dual_feasible and _gap_ok(F, d, y - X @ beta, X, G, g, E, f, beta, active, nu, lam)):
        UNCERTIFIED["lad"] += 1
    return beta


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
        nb = np.ones(n, bool)
        nb[B] = False
        a = c * np.sign(e) * nb
        grad = X.T @ a
        z = np.linalg.solve(M.T, -grad)
        d, nu = z[:len(B)], z[len(B):len(B) + len(active)]
        tol = TOL * max(1.0, np.abs(grad).max(initial=0.0), c.max(initial=0.0))
        rates = np.r_[c[B] - np.abs(d), nu]
        if not len(rates) or rates.min() >= -tol:
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
        zero = (e == 0.0)
        zero[B2] = False
        Fp = float(np.sum(np.where(zero, c * np.abs(xv), -c * np.sign(e) * xv)))
        cand = ~zero & (np.sign(e) == np.sign(xv)) & (xv != 0)
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
        if t_hit <= t_cross:
            B, active = B2, rest + [k_hit]
        else:
            B, active = B2 + [enter], rest
    raise RuntimeError("pivot limit")                               # pragma: no cover


def simplex_lad(y, P, c) -> np.ndarray:
    """The weights-on-levels ablation: ``argmin_w sum_t c_t |y_t - P_t w|`` on the simplex."""
    y, P, c = (np.asarray(a, dtype=float) for a in (y, P, c))
    S = P.shape[1]
    if S == 1:
        return np.ones(1)
    s0 = int(np.argmin([np.sum(c * np.abs(y - P[:, s])) for s in range(S)]))
    w = lad(y, P, c, G=np.eye(S), g=np.zeros(S), E=np.ones((1, S)), f=[1.0],
            active=[k for k in range(S) if k != s0])
    w = np.clip(w, 0.0, None)
    return w / w.sum()
