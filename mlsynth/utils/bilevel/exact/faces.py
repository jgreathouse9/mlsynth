"""Faces of the donor hull by dimension: all, visible, and attainable (the paper's Table 3).

Under general position conv(X0) is simplicial, so every subset of a facet's K donors is
a face, and the faces are counted from the facets Qhull returns. A face is visible when
it lies on a facet visible from X1, and attainable when some point w of it is in the
closure of the attainable set: some nonzero c in its normal cone has c_k r_k(w) >= 0 for
every k, r = X1 - X0 w (Proposition A.1). Attainability is decided with the search's own
tools, as a best fit with no incumbent: a vertex by one cone test, an edge in its
residual arrangement, a facet or a ridge by its pieces, and a lower face by the branch
and bound over sign sub-cubes. The fit minimised is ||w||^2, which keeps every
least-squares solve well conditioned; only whether a fit exists is used.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
from scipy.spatial import ConvexHull

from ....exceptions import MlsynthDataError
from .hull import TOL, Hull
from .lsq import sign_piece_ls
from .patterns import cone_compatible, edge_min, face_min, facet_patterns, ridge_patterns
from .structures import FaceCounts


def _facets(X0):
    """The facets of conv(X0) as sets of K donors; MlsynthDataError when one holds more."""
    K = X0.shape[0]
    qh = ConvexHull(X0.T)
    planes = {tuple(np.round(eq, 9)): eq for eq in qh.equations}
    out = []
    for eq in planes.values():
        on = np.flatnonzero(np.abs(X0.T @ eq[:-1] + eq[-1]) <= 1e-9 * max(1.0, abs(eq[-1])))
        if len(on) != K:
            raise MlsynthDataError(f"a facet of the donor hull holds {len(on)} donors, not K = {K}; the "
                                   "predictors violate general position")
        out.append((frozenset(int(i) for i in on), eq))
    return out


def _by_dimension(facets, K):
    faces = {frozenset(c) for F in facets for k in range(1, K + 1) for c in itertools.combinations(sorted(F), k)}
    count = Counter(len(f) - 1 for f in faces)
    return faces, tuple(int(count.get(d, 0)) for d in range(K))


def _attainable(hull, sigma, F0, z, Z, scale):
    """Whether face sigma holds an attainable point; F0 is a visible facet that contains it."""
    X0, X1, K = hull.X0, hull.X1, hull.K
    cols = sorted(sigma)
    m = len(cols)
    if m == 1:
        r = X1 - X0[:, cols[0]]
        return cone_compatible(X0, sigma, np.sign(r), np.abs(r) <= 1e-12 * scale)
    if m == 2:
        (val, _), _ = edge_min(X0, X1, z, Z, sigma, np.inf)
        return bool(np.isfinite(val))
    if m == K:
        patterns = facet_patterns(hull.normal[sigma][0])
    else:
        star = hull.star(sigma, F0)
        if m < K - 1:
            val, _ = face_min(X0, X1, z, Z, sigma, set().union(*star) - sigma, np.inf)
            return bool(np.isfinite(val))
        if len(star) != 2:
            raise MlsynthDataError("a ridge lies in other than two facets; general position fails")
        G1, G2 = star
        patterns = ridge_patterns(hull.normal[G1][0], hull.normal[G2][0])
    return any(sign_piece_ls(X0, X1, z, Z, cols, s)[0] is not None for s in patterns)


def face_counts(X1, X0) -> FaceCounts:
    """Faces of conv(X0) by dimension: all, visible from X1, and attainable."""
    X1, X0 = np.asarray(X1, float), np.asarray(X0, float)
    K, J = X0.shape
    hull = Hull(X0, X1)                                   # refuses a hull of fewer than K dimensions
    facets = _facets(X0)
    _, total = _by_dimension([F for F, _ in facets], K)
    visible = [F for F, eq in facets if eq[:-1] @ X1 + eq[-1] > TOL]
    faces, seen = _by_dimension(visible, K)
    for F in visible:
        hull.add(F)
    z, Z = np.zeros(J), np.eye(J)                         # the fit ||w||^2
    scale = max(1.0, float(np.abs(X1).max()))
    count = Counter()
    for sigma in faces:
        F0 = next(F for F in visible if sigma <= F)
        if _attainable(hull, sigma, F0, z, Z, scale):
            count[len(sigma) - 1] += 1
    return FaceCounts(total=total, visible=seen, attainable=tuple(int(count.get(d, 0)) for d in range(K)))
