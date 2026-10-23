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
                            sigma + {v}, so the link vertices are the only walls. The
                            extreme rays of N cut by the coordinate hyperplanes come from
                            (d-1)-subsets of {walls, coordinate planes} (the paper's
                            Lemma A.3 with an irredundant description), in one batched
                            SVD. Reverse search (Avis, in Kalai and Ziegler, eds., 2000,
                            p. 177) is the output-sensitive way to list the same rays.

On a face of m donors the residual arrangement {r_k(w) = 0} has dimension m - 1 and
the cone arrangement has dimension K - m, so one of them is small. Edges are handled
in the residual arrangement: at most K + 1 intervals, one attainability test each.
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
from scipy.optimize import linprog


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


def lower_face_patterns(X0, sigma, link, tol=1e-10, cap=60_000):
    """S(sigma) for a normal cone of dimension >= 3; None when the subsets exceed `cap`."""
    K = X0.shape[0]
    cols = sorted(sigma)
    a = cols[0]
    E = (X0[:, cols[1:]] - X0[:, [a]]).T                      # E c = 0
    W = (X0[:, sorted(link)] - X0[:, [a]]).T                  # W c <= 0
    d = K - len(cols) + 1
    walls = np.vstack([W, np.eye(K)])
    if comb(len(walls), d - 1) > cap:
        return None
    idx = np.array(list(itertools.combinations(range(len(walls)), d - 1)), dtype=int)
    M = np.concatenate([np.broadcast_to(E, (len(idx),) + E.shape), walls[idx]], axis=1)
    _, sv, Vt = np.linalg.svd(M)
    ok = sv[:, -1] > 1e-9 * np.maximum(sv[:, 0], 1.0)
    U = Vt[ok, -1, :]
    U = np.vstack([U, -U])
    U = U / np.abs(U).max(axis=1, keepdims=True)
    if W.size:
        U = U[(U @ W.T <= tol).all(axis=1)]
    P = np.where(np.abs(U) <= tol, 0, np.sign(U)).astype(np.int8)
    P = np.unique(P[np.any(P != 0, axis=1)], axis=0)
    if len(P) > 1:                                            # drop patterns a weaker one implies
        weaker = np.all((P[:, None, :] == 0) | (P[:, None, :] == P[None, :, :]), axis=2)
        np.fill_diagonal(weaker, False)
        P = P[~weaker.any(axis=0)]
    return [p.astype(int) for p in P]


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


def pattern_lps(X0, sigma):
    """S(sigma) by one LP per sign pattern (the reference's fallback), for oversized cones."""
    K = X0.shape[0]
    out = []
    for s in itertools.product((-1, 1), repeat=K):
        s = np.array(s)
        if cone_compatible(X0, sigma, s.astype(float), np.zeros(K, bool)):
            out.append(s)
    return out


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
