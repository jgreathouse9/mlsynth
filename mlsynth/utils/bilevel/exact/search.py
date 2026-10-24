"""The exact synthetic control of Abadie and Shang (2026): the searches over donor weights.

A treated unit outside the donor hull is solved here; one inside it is matched exactly,
and ``interior`` selects among the exact matches (paper, Section 5). A unit on the
hull's boundary is covered by neither case and is refused.

Diagonal class (their problem (5)): minimise the validation loss ||Z1 - Z0 w||^2 over the
closure of the donor weights attainable with strictly positive diagonal predictor
weights. By Proposition A.1 that set is the union, over faces sigma of the facets
visible from X1 and patterns s in S(sigma), of the pieces
W_s(sigma) = {w in Delta_sigma : s_k r_k(w) >= 0}.

Faces are visited best first by a valid lower bound, the unconstrained simplex fit of
the parent face, and a face is skipped once that bound cannot beat the incumbent. On
each face the pieces are read off the normal cone (``patterns``) and each piece is one
exact least-squares solve (``lsq``). Before the pieces are listed, a cheaper relaxation
is tried: a sign that every generator of N(sigma) shares in some coordinate is a
constraint common to every piece, hence valid for their union (Balas, Disjunctive
Programming, 2018, Theorem 1.2), and one solve with those signs can rule the face out.

Positive definite class (their problem (6), Proposition 2): the best simplex fit over
the closed visible facets.

Both return (loss, w) with w on the simplex over the J donors.
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from ....exceptions import MlsynthDataError, MlsynthEstimationError
from . import interior
from .hull import Hull, outside_hull
from .lsq import sign_piece_ls, simplex_ls
from .patterns import cone_compatible, edge_min, face_min, facet_patterns, ridge_patterns


def _check(X1, X0, Z1, Z0):
    X1, X0, Z1, Z0 = (np.asarray(a, float) for a in (X1, X0, Z1, Z0))
    if X0.ndim != 2 or Z0.ndim != 2 or X1.shape != (X0.shape[0],) or Z1.shape != (Z0.shape[0],) \
            or X0.shape[1] != Z0.shape[1]:
        raise MlsynthDataError("expected X1 (K,), X0 (K, J), Z1 (L,), Z0 (L, J)")
    if not all(np.isfinite(a).all() for a in (X1, X0, Z1, Z0)):
        raise MlsynthDataError("inputs must be finite")
    return X1, X0, Z1, Z0


def _interior(solve, X1, X0, Z1, Z0, stats, max_cells):
    """The interior search's answer, with its counters and the case it took."""
    st = {}
    loss, w = solve(X1, X0, Z1, Z0, stats=st, max_cells=max_cells)
    if stats is not None:
        stats.update(st, regime="interior")
    return loss, w


def _finish(w, Z1, Z0):
    w = w.copy()
    w[w < 1e-7] = 0.0
    w /= w.sum()
    res = Z1 - Z0 @ w
    return float(res @ res), w


def solve_diag(X1, X0, Z1, Z0, stats=None, max_cells=interior.MAX_CELLS):
    """The diagonal class: (loss, w). ``stats`` receives the search's counters and its ``regime``;
    ``max_cells`` limits an interior walk."""
    X1, X0, Z1, Z0 = _check(X1, X0, Z1, Z0)
    if not outside_hull(X1, X0):
        return _interior(interior.solve_interior_diag, X1, X0, Z1, Z0, stats, max_cells)
    K, J = X0.shape
    hull = Hull(X0, X1)
    vis = hull.visible_facets()
    scale = max(1.0, np.abs(X1).max())
    U, w_best = np.inf, None
    ctr = dict(faces=0, solves=0, relaxation_prunes=0, edges=0, vertices=0, lps=0, pattern_tests=0, pattern_lp_checks=0)
    heap, seen, tie = [], set(), itertools.count()
    for F in vis:
        seen.add(F)
        heapq.heappush(heap, (0.0, next(tie), F, F))   # (bound, tie, face, visible facet containing it)
    while heap:
        bound, _, sigma, F0 = heapq.heappop(heap)
        if bound >= U - 1e-12 * max(1.0, abs(U)):
            break
        cols = sorted(sigma)
        m = len(cols)
        if m == 1:                                       # a vertex: one point, one test
            j = cols[0]
            val = float(np.sum((Z1 - Z0[:, j]) ** 2))
            if val < U:
                ctr["vertices"] += 1
                ctr["lps"] += 1
                r = X1 - X0[:, j]
                if cone_compatible(X0, sigma, np.sign(r), np.abs(r) <= 1e-12 * scale):
                    U = val
                    w_best = np.zeros(J)
                    w_best[j] = 1.0
            continue
        u, beta_u = simplex_ls(Z1, Z0[:, cols])
        ctr["solves"] += 1
        if u is None or u >= U:
            continue
        ctr["faces"] += 1
        if m == 2:                                       # an edge: residual arrangement
            ctr["edges"] += 1
            (val, t), n_lp = edge_min(X0, X1, Z1, Z0, sigma, U)
            ctr["lps"] += n_lp
            if val < U:
                U = val
                w_best = np.zeros(J)
                w_best[cols[0]], w_best[cols[1]] = 1.0 - t, t
        else:                                            # the normal-cone arrangement
            r_u = X1 - X0[:, cols] @ beta_u
            patterns = []
            if m == K:
                patterns = facet_patterns(hull.normal[sigma][0])
            else:
                star = hull.star(sigma, F0)
                gens = np.array([hull.normal[G][0] for G in star])
                gs = gens / np.abs(gens).max(axis=1, keepdims=True)
                common = np.where((gs > 1e-12).all(axis=0), 1, np.where((gs < -1e-12).all(axis=0), -1, 0))
                root = (u, beta_u)                       # the relaxation's fit, and its minimiser
                if not np.all((common == 0) | (common * r_u >= -1e-9 * scale)):
                    root = sign_piece_ls(X0, X1, Z1, Z0, cols, common)
                    ctr["solves"] += 1
                if root[0] is None or root[0] >= U:
                    ctr["relaxation_prunes"] += 1
                elif m == K - 1:
                    if len(gens) != 2:
                        raise MlsynthDataError("a ridge lies in other than two facets; general position fails")
                    patterns = ridge_patterns(gens[0], gens[1])
                else:                                    # below the ridges: branch and bound
                    link = set().union(*star) - sigma
                    val, beta = face_min(X0, X1, Z1, Z0, sigma, link, U, root=(common, *root), stats=ctr)
                    if val < U:
                        U = val
                        w_best = np.zeros(J)
                        w_best[cols] = beta
            if patterns and any(np.all((s == 0) | (s * r_u >= -1e-9 * scale)) for s in patterns):
                if u < U:                                 # the face's own minimiser is attainable
                    U = u
                    w_best = np.zeros(J)
                    w_best[cols] = beta_u
            else:
                for s in patterns:
                    val, beta = sign_piece_ls(X0, X1, Z1, Z0, cols, s)
                    ctr["solves"] += 1
                    if val is not None and val < U:
                        U = val
                        w_best = np.zeros(J)
                        w_best[cols] = beta
        for v in cols:
            child = sigma - {v}
            if child not in seen:
                seen.add(child)
                heapq.heappush(heap, (u, next(tie), child, F0))
    if w_best is None:                                   # pragma: no cover - P has an attainable point
        raise MlsynthEstimationError("no attainable point found")
    if stats is not None:
        stats.update(regime="exterior", n_visible=len(vis), gift_wrap_steps=hull.n_wrap, **ctr)
    return _finish(w_best, Z1, Z0)


def solve_pd(X1, X0, Z1, Z0, stats=None, max_cells=interior.MAX_CELLS):
    """The positive definite class: (loss, w). Outside the hull ``stats`` also receives the number
    of visible facets and the ``facets`` whose simplex fit is the optimum: several when the
    optimum lies on a face they share, each as its sorted donors."""
    X1, X0, Z1, Z0 = _check(X1, X0, Z1, Z0)
    if not outside_hull(X1, X0):
        return _interior(interior.solve_interior_pd, X1, X0, Z1, Z0, stats, max_cells)
    J = X0.shape[1]
    vis = Hull(X0, X1).visible_facets()
    best, fits = (np.inf, None, None), []
    for F in vis:
        cols = sorted(F)
        val, beta = simplex_ls(Z1, Z0[:, cols])
        if val is not None:
            fits.append((val, tuple(cols)))
            if val < best[0]:
                best = (val, cols, beta)
    if stats is not None:
        tied = sorted(F for val, F in fits if val <= best[0] + 1e-9 * max(1.0, best[0]))
        stats.update(regime="exterior", n_visible=len(vis), facets=tuple(tied))
    w = np.zeros(J)
    w[best[1]] = best[2]
    return _finish(w, Z1, Z0)
