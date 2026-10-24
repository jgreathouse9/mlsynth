"""Sign patterns S(sigma) of a face, attainability tests, and a face's best attainable fit.

A point w on face sigma (its donor support) is in the closure of the attainable set
under positive diagonal predictor weights exactly when some nonzero c in the normal
cone N(sigma) has c_k r_k(w) >= 0 for every k, with r = X1 - X0 w (paper, Lemma A.1 and
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
                            patterns are not listed: ``face_min`` searches the face by
                            branch and bound over sign sub-cubes, so a sub-cube whose
                            relaxation cannot beat the incumbent is never tested. Each
                            test is one least-distance problem whose verdict carries a
                            point or a Farkas certificate (lsq.ldp_verdict; a Phase I by
                            least squares, Dantzig and Thapa, Linear Programming 2, 2003,
                            sec. 6.7), the certificate names the coordinates to branch
                            on, and HiGHS decides the rare test where neither checks.

On a face of m donors the residual arrangement {r_k(w) = 0} has dimension m - 1 and
the cone arrangement has dimension K - m, so one of them is small. Edges are handled
in the residual arrangement: at most K + 1 intervals, one attainability test each.
"""
from __future__ import annotations

import heapq
import itertools
from typing import NamedTuple

import numpy as np
from scipy.optimize import linprog

from ....exceptions import MlsynthDataError, MlsynthEstimationError
from . import lsq


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


class _Cone(NamedTuple):
    """N(sigma) on the slice g'c = 1 as c = c0 + N x, with its equalities and walls."""
    E: np.ndarray            # E c = 0: the face's edges from its first donor
    W: np.ndarray            # W c <= 0: the link vertices, from the same donor
    g: np.ndarray            # g'c > 0 on the pointed cone minus the origin
    c0: np.ndarray
    N: np.ndarray
    WN: np.ndarray
    Wc0: np.ndarray


def _cone(X0, sigma, link):
    """N(sigma) with the link vertices as its walls; an error when the face has no proper cone."""
    if not link:
        raise MlsynthEstimationError("a proper face has a nonempty link; pass the vertices of its star outside the face")
    cols = sorted(sigma)
    a = cols[0]
    E = (X0[:, cols[1:]] - X0[:, [a]]).T
    W = (X0[:, sorted(link)] - X0[:, [a]]).T
    g = -W.sum(axis=0)
    g /= max(np.linalg.norm(g), np.finfo(float).tiny)
    Aeq = np.vstack([E, g[None, :]])
    beq = np.r_[np.zeros(len(E)), 1.0]
    U, sv, Vt = np.linalg.svd(Aeq)
    rank = int((sv > 1e-10 * sv[0]).sum())
    c0 = Vt[:rank].T @ ((U[:, :rank].T @ beq) / sv[:rank])
    if np.abs(Aeq @ c0 - beq).max() > 1e-9:
        raise MlsynthDataError("the normal cone's slice g'c = 1 misses the face's equalities E c = 0: a link "
                         "vertex lies in the face's affine hull, so general position fails")
    N = Vt[rank:].T
    return _Cone(E, W, g, c0, N, W @ N, W @ c0)


def _sign_test(cone, t, stats):
    """Is there c in N(sigma) on the slice with t_k c_k >= 0 wherever t_k != 0?

    Returns (True, None), or (False, the signed coordinates the proof uses): those whose
    rows a Farkas certificate weights, or every signed coordinate when HiGHS decided.
    """
    stats["pattern_tests"] = stats.get("pattern_tests", 0) + 1
    signed = np.flatnonzero(t)
    G = np.vstack([-cone.WN, t[signed, None] * cone.N[signed]])
    h = np.r_[cone.Wc0, -t[signed] * cone.c0[signed]]
    verdict, evidence = lsq.ldp_verdict(G, h)
    if verdict is not None:
        return verdict, (None if verdict else signed[evidence[len(cone.W):] > 0])
    stats["pattern_lp_checks"] = stats.get("pattern_lp_checks", 0) + 1
    bounds = [(0, None) if x > 0 else (None, 0) if x < 0 else (None, None) for x in t]
    res = linprog(np.zeros(len(t)), A_ub=cone.W, b_ub=np.zeros(len(cone.W)),
                  A_eq=np.vstack([cone.E, cone.g[None, :]]), b_eq=np.r_[np.zeros(len(cone.E)), 1.0],
                  bounds=bounds, method="highs")
    return (True, None) if res.status == 0 else (False, signed)


def face_min(X0, X1, Z1, Z0, sigma, link, U, root=None, stats=None):
    """The best attainable fit on face sigma below U, by branch and bound over sign sub-cubes.

    Returns (loss, beta), beta the weights on sorted(sigma), or (inf, None) when no
    attainable point of the face fits below U.

    A node fixes the signs s_k of some coordinates (0 leaves k free) and stands for every
    sign vector that completes them. Its relaxation is the region {w : s_k r_k(w) >= 0
    for fixed k}, which holds every piece below the node, and its bound is the region's
    best fit, one least-squares solve. A node is dropped when the region is empty or its
    fit cannot beat the incumbent. Otherwise the region's minimiser w is tested: if some
    c in N(sigma) has the node's signs and c_k r_k(w) >= 0 on the free coordinates, w is
    attainable and settles the node. If none has, the test's Farkas certificate names
    the free coordinates whose signs at w the cone cannot meet together, so one of them
    is flipped in every attainable piece below the node. The node is split into one
    child per such coordinate k_j, with k_1 to k_{j-1} fixed at w's signs and k_j at the
    opposite one; the children partition the sign vectors the certificate leaves, each
    excludes w, and a certificate on fixed signs alone ends the node.

    ``root`` = (signs, bound, beta) starts the search below `signs` with a lower bound on
    their region's fit, its minimum when beta, the region's minimiser, is given; the
    default is the whole face with bound 0. As in the SYNDES and MAREX searches, a child
    is queued under its parent's bound and solved when it comes off the heap, best
    first. ``stats`` counts solves, tests, and tests HiGHS decided.
    """
    stats = {} if stats is None else stats
    cone = _cone(X0, sigma, link)
    cols = sorted(sigma)
    tol = 1e-9 * max(1.0, float(np.abs(X1).max()))
    signs, bound, beta = (np.zeros(X0.shape[0], int), 0.0, None) if root is None else root
    best, best_beta = np.inf, None
    tie = itertools.count()
    heap = [(bound, next(tie), np.asarray(signs, int), beta)]
    while heap:
        val, _, s, beta = heapq.heappop(heap)
        cap = min(U, best)
        if val >= cap - 1e-12 * max(1.0, abs(cap)):
            break
        if beta is None:
            val, beta = lsq.sign_piece_ls(X0, X1, Z1, Z0, cols, s)
            stats["solves"] = stats.get("solves", 0) + 1
            if val is None or val >= cap:
                continue
        r = X1 - X0[:, cols] @ beta
        t = s.copy()
        on = (s == 0) & (np.abs(r) > tol)
        t[on] = np.sign(r[on])
        attainable, used = _sign_test(cone, t, stats)
        if attainable:
            best, best_beta = val, beta
            continue
        prefix = s.copy()
        for k in used[s[used] == 0]:                          # the certificate's free coordinates
            child = prefix.copy()
            child[k] = -t[k]
            heapq.heappush(heap, (val, next(tie), child, None))
            prefix[k] = t[k]
    return best, best_beta


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
