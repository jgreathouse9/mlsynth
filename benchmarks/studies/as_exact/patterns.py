"""Allowed sign patterns S(sigma) of a face, and attainability tests, mostly without LPs.

A point w on face sigma (its donor support) is in the closure of the attainable set
under positive diagonal predictor weights exactly when some nonzero c in the normal
cone N(sigma) has c_k r_k >= 0 for every k, with r = X1 - X0 w (paper, Lemma A.1 and
Proposition A.1). The attainable part of a face is therefore a union of polyhedral
pieces {w : s_k r_k(w) >= 0}, one per pattern s in S(sigma), and S(sigma) depends
only on the cone:

  facet (|sigma| = K):      N is the ray of the facet normal, one pattern;
  ridge (|sigma| = K - 1):  N = cone{a1, a2}; along c(t) = cos t a1 + sin t a2 each
                            coordinate changes sign at most once, so sorting those
                            angles lists at most K + 1 patterns;
  smaller faces:            N = {c : E c = 0, (X_v - X_a)'c <= 0 for v in link(sigma)}.
                            The facets of N(sigma) are the normal cones of the faces
                            sigma + {v}, so the link vertices are the only walls.
                            ``branch_patterns`` fixes one sign at a time. A sub-cube of
                            sign vectors is dropped when the cone misses all of it, and
                            closed when some nonzero c in N has zeros on the unfixed
                            coordinates: that c lies in every orthant of the sub-cube,
                            so one pattern with 0 on those coordinates covers their
                            pieces. Each test is one least-distance problem whose verdict
                            carries a point or a Farkas certificate (lsq.ldp_verdict; a
                            Phase I by least squares, Dantzig and Thapa, Linear
                            Programming 2, 2003, sec. 6.7), and HiGHS decides the rare
                            test where neither checks.

On a face of m donors the residual arrangement {r_k(w) = 0} has dimension m - 1 and
the cone arrangement has dimension K - m, so one of them is small. Edges are handled
in the residual arrangement: at most K + 1 intervals, one attainability test each.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from . import lsq

FREE = 2                                   # a sign branch_patterns has not fixed yet


def facet_patterns(n, tol=1e-12):
    a = n / np.abs(n).max()
    s = np.where(np.abs(a) <= tol, 0, np.sign(a)).astype(int)
    return [s]


def ridge_patterns(a1, a2, tol=1e-12):
    """S(R) for a ridge with normal cone cone{a1, a2}; 0 marks a free coordinate."""
    a1 = a1 / np.abs(a1).max()
    a2 = a2 / np.abs(a2).max()
    cross = np.flatnonzero((a1 * a2 < 0) & (np.abs(a1) > tol) & (np.abs(a2) > tol))
    bps = np.sort(np.arctan2(-a1[cross], a2[cross]) % np.pi)
    bps = bps[(bps > 0) & (bps < np.pi / 2)]
    edges = np.concatenate([[0.0], bps, [np.pi / 2]])
    thetas = list(0.5 * (edges[:-1] + edges[1:])) + [0.0, np.pi / 2]
    out, seen = [], set()
    for th in thetas:
        c = np.cos(th) * a1 + np.sin(th) * a2
        s = np.where(np.abs(c) <= tol, 0, np.sign(c)).astype(int)
        if tuple(s) not in seen and np.any(s != 0):
            seen.add(tuple(s))
            out.append(s)
    return out


def _equalities(E, W, g, zero, cache):
    """A point c0 and a null-space basis N of {E c = 0, g'c = 1, c_zero = 0}, with W N, W c0 and
    the consistency gap; cached per zero set, which sibling tests and every free test share."""
    key = zero.tobytes()
    if key not in cache:
        K = E.shape[1]
        Aeq = np.vstack([E, g[None, :], np.eye(K)[zero]])
        beq = np.r_[np.zeros(len(E)), 1.0, np.zeros(len(zero))]
        U, sv, Vt = np.linalg.svd(Aeq)
        rank = int((sv > 1e-10 * sv[0]).sum())
        c0 = Vt[:rank].T @ ((U[:, :rank].T @ beq) / sv[:rank])
        N = Vt[rank:].T
        cache[key] = (c0, N, W @ N, W @ c0, float(np.abs(Aeq @ c0 - beq).max()))
    return cache[key]


def _cone_test(E, W, g, assign, stats, cache):
    """Is there c with E c = 0, W c <= 0, g'c = 1 and the signs `assign` asks for? (feasible, c).

    assign_k is +1 or -1 for a sign, 0 for c_k = 0 and FREE for no condition. The
    equalities are eliminated in a null-space basis and the rest is one least-distance
    problem whose verdict carries its evidence; HiGHS decides when neither a point nor a
    certificate checks, or when the equalities are too close to inconsistent to say.
    """
    K = len(assign)
    stats["pattern_tests"] = stats.get("pattern_tests", 0) + 1
    c0, N, WN, Wc0, gap = _equalities(E, W, g, np.flatnonzero(assign == 0), cache)
    if gap > 1e-6:                                       # the equalities alone have no solution
        return False, None
    if gap <= 1e-9:
        signed = np.flatnonzero((assign == 1) | (assign == -1))
        if N.shape[1] == 0:                              # one point: check it where it lives
            Gc = np.vstack([-W, assign[signed, None] * np.eye(K)[signed]])
            ok = lsq._satisfies(Gc, np.zeros(len(Gc)), c0)
            return ok, (c0 if ok else None)
        G = np.vstack([-WN, assign[signed, None] * N[signed]])
        h = np.r_[Wc0, -assign[signed] * c0[signed]]
        verdict, evidence = lsq.ldp_verdict(G, h)
        if verdict is True:
            return True, c0 + N @ evidence
        if verdict is False:
            return False, None
    stats["pattern_lp_checks"] = stats.get("pattern_lp_checks", 0) + 1
    bounds = [(0, None) if s == 1 else (None, 0) if s == -1 else (0, 0) if s == 0 else (None, None) for s in assign]
    res = linprog(np.zeros(K), A_ub=W, b_ub=np.zeros(len(W)), A_eq=np.vstack([E, g[None, :]]),
                  b_eq=np.r_[np.zeros(len(E)), 1.0], bounds=bounds, method="highs")
    return (True, res.x) if res.status == 0 else (False, None)


def branch_patterns(X0, sigma, link, stats=None):
    """S(sigma) by fixing one sign at a time (module docstring); 0 marks a free sign.

    The completions of the returned patterns are exactly the sign vectors whose closed
    orthant meets N(sigma) away from the origin. ``stats`` counts the least-distance
    tests and the ones HiGHS decided.
    """
    if not link:
        raise ValueError("a proper face has a nonempty link; pass the vertices of its star outside the face")
    stats = {} if stats is None else stats
    K = X0.shape[0]
    cols = sorted(sigma)
    a = cols[0]
    E = (X0[:, cols[1:]] - X0[:, [a]]).T                      # E c = 0
    W = (X0[:, sorted(link)] - X0[:, [a]]).T                  # W c <= 0
    g = -W.sum(axis=0)                                        # g'c > 0 on N minus the origin
    g /= np.linalg.norm(g)
    out, cache = [], {}
    stack = [(np.full(K, FREE), None)]
    while stack:
        assign, c = stack.pop()
        closed = np.where(assign == FREE, 0, assign)
        if np.any(closed != 0):
            ok, _ = _cone_test(E, W, g, closed, stats, cache)
            if ok:                                            # one witness covers the sub-cube
                out.append(closed.astype(int))
                continue
        if c is None:
            ok, c = _cone_test(E, W, g, assign, stats, cache)
            if not ok:                                        # the cone misses the sub-cube
                continue
        free = np.flatnonzero(assign == FREE)
        if free.size == 0 or not np.any(c[free]):            # c itself witnesses the closed sub-cube
            out.append(closed.astype(int))
            continue
        k = free[np.argmax(np.abs(c[free]))]
        s = 1 if c[k] > 0 else -1
        same, other = assign.copy(), assign.copy()
        same[k], other[k] = s, -s
        stack.append((other, None))
        stack.append((same, c))                               # c already has sign s at k
    return out


def cone_compatible(X0, sigma, sgn, zero):
    """Is there c != 0 in N(sigma) with sgn_k c_k >= 0 (k not in zero) and c_k free on zero?

    N(sigma) is written with every donor as a wall, so no star is needed. Two stages:
    a c with a nonzero coordinate off the zero set (normalised there), else a c supported
    on the zero set, each sign of it in turn.
    """
    K, J = X0.shape
    cols = sorted(sigma)
    a = cols[0]
    E = [X0[:, j] - X0[:, a] for j in cols[1:]]
    W = [X0[:, i] - X0[:, a] for i in range(J) if i not in sigma]
    nz = np.flatnonzero(~zero)
    zk = np.flatnonzero(zero)

    def feasible(extra, norm, bounds):
        A_ub = np.array(W + extra) if (W or extra) else None
        res = linprog(np.zeros(K), A_ub=A_ub, b_ub=np.zeros(len(A_ub)) if A_ub is not None else None,
                      A_eq=np.array(E + [norm]), b_eq=np.concatenate([np.zeros(len(E)), [1.0]]),
                      bounds=bounds, method="highs")
        return res.status == 0

    rows = []
    for k in nz:
        row = np.zeros(K)
        row[k] = -sgn[k]
        rows.append(row)
    if len(nz):
        norm = np.zeros(K)
        norm[nz] = sgn[nz]
        if feasible(rows, norm, [(None, None)] * K):
            return True
    bounds = [(0.0, 0.0)] * K
    for k in zk:
        bounds[k] = (None, None)
    for sz in itertools.product((-1.0, 1.0), repeat=len(zk)):
        norm = np.zeros(K)
        norm[zk] = sz
        extra = []
        for k, s in zip(zk, sz):
            row = np.zeros(K)
            row[k] = -s
            extra.append(row)
        if feasible(extra, norm, bounds):
            return True
    return False


def edge_min(X0, X1, Z1, Z0, sigma, U, tol=1e-12):
    """Exact attainable minimum on an edge, in the residual arrangement.

    Returns ((loss, t), n_lp) with t the weight on the larger donor index; loss is inf
    when no point of the edge beats U.
    """
    i, j = sorted(sigma)
    g = X1 - X0[:, i]                          # r(t) = g - t h
    h = X0[:, j] - X0[:, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = np.where(np.abs(h) > tol, g / h, np.nan)
    bps = np.unique(roots[(roots > 1e-12) & (roots < 1 - 1e-12)])
    knots = np.concatenate([[0.0], bps, [1.0]])
    a, b = Z1 - Z0[:, i], Z0[:, j] - Z0[:, i]   # loss(t) = ||a - t b||^2
    bb = float(b @ b)
    t_free = (a @ b) / bb if bb > 0 else 0.0
    scale = max(1.0, np.abs(X1).max())
    best = (np.inf, None)
    n_lp = 0
    for lo, hi in zip(knots[:-1], knots[1:]):  # open intervals carry one sign vector
        t = float(np.clip(t_free, lo, hi))
        val = float(np.sum((a - t * b) ** 2))
        if val >= min(U, best[0]):
            continue
        r = g - 0.5 * (lo + hi) * h
        n_lp += 1
        if cone_compatible(X0, sigma, np.sign(r), np.abs(r) <= 1e-12 * scale):
            best = (val, t)
    for t in bps:                               # isolated points where a residual vanishes
        val = float(np.sum((a - t * b) ** 2))
        if val >= min(U, best[0]):
            continue
        r = g - t * h
        n_lp += 1
        if cone_compatible(X0, sigma, np.sign(r), np.abs(r) <= 1e-9 * scale):
            best = (val, float(t))
    return best, n_lp
