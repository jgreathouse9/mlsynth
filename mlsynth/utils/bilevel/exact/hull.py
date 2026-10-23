"""Facets of the donor hull, explored lazily by gift wrapping.

The donor hull P = conv(X0) is simplicial under the paper's general-position
assumption, so a facet is a set of K donors. Two walks are needed:

  - the facets visible from X1 (Algorithm 1 of the paper), which are connected in
    the facet-ridge graph (Lemma A.2);
  - the star of a face sigma, the facets that contain it. In a simplicial polytope
    these are the vertices of the dual face of the simple polar polytope, whose graph
    is connected and (K - |sigma|)-regular (Balinski; Kalai; Achatz and Kleinschmidt
    in Kalai and Ziegler, eds., Polytopes: Combinatorics and Computation, 2000,
    p. 155), so walking the ridges that contain sigma reaches all of them.

Crossing a ridge costs one QR factorisation and one J x K product: in the plane
orthogonal to the ridge, the neighbour is the donor that makes the widest angle
with the current facet (Chand and Kapur's gift wrapping). The reference code
instead forms a candidate hyperplane and tests every donor against it for each of
the J candidates.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ....exceptions import MlsynthDataError

TOL = 1e-9


def outside_hull(X1, X0):
    """Whether X1 lies outside conv(X0) (the exterior case)."""
    J = X0.shape[1]
    res = linprog(np.zeros(J), A_eq=np.vstack([X0, np.ones(J)]), b_eq=np.append(X1, 1.0),
                  bounds=(0, None), method="highs")
    return res.status == 2


def affine_rank(X0, tol=1e-7):
    C = X0 - X0.mean(axis=1, keepdims=True)
    sv = np.linalg.svd(C, compute_uv=False)
    return int((sv > tol * sv[0]).sum()) if sv.size and sv[0] > 0 else 0


def _hyperplane(P, interior):
    """Outward unit normal n and offset b of the hyperplane through the columns of P."""
    base = P[:, 0]
    E = P[:, 1:] - base[:, None]
    U, sv, _ = np.linalg.svd(E)
    if sv.size and sv[-1] < 1e-12 * max(sv[0], 1.0):
        return None
    n = U[:, -1]
    b = n @ base
    if n @ interior > b:
        n, b = -n, -b
    return n, b


def _seed_facet(X0, X1, interior):
    """A facet visible from X1: where the ray from an interior point toward X1 leaves P.

    Generically the exit point is in the relative interior of a facet, whose K donors
    carry the exit weights. When the ray leaves through a lower-dimensional face the
    starting point is moved (deterministically) and the ray recast.
    """
    K, J = X0.shape
    rng = np.random.default_rng(0)
    start = interior.copy()
    for _ in range(20):
        d = X1 - start
        c = np.zeros(J + 1)
        c[-1] = -1.0
        A_eq = np.zeros((K + 1, J + 1))
        A_eq[:K, :J] = X0
        A_eq[:K, J] = -d
        A_eq[K, :J] = 1.0
        res = linprog(c, A_eq=A_eq, b_eq=np.concatenate([start, [1.0]]),
                      bounds=[(0, None)] * (J + 1), method="highs")
        if res.status == 0:
            supp = np.flatnonzero(res.x[:J] > 1e-9)
            if len(supp) == K:
                hb = _hyperplane(X0[:, supp], interior)
                if (hb is not None and (X0.T @ hb[0]).max() <= hb[1] + 1e-9
                        and hb[0] @ X1 > hb[1] + TOL):
                    return frozenset(int(j) for j in supp)
        w = rng.dirichlet(np.ones(J))
        start = 0.5 * interior + 0.5 * (X0 @ w)
    raise MlsynthDataError("could not find a visible facet; the donor hull may violate general position")


class Hull:
    """Facets of conv(X0) discovered on demand, with a cached neighbour oracle."""

    def __init__(self, X0, X1):
        K, J = X0.shape
        if J < K + 1 or affine_rank(X0) < K:
            raise MlsynthDataError(
                "the donor predictors span fewer than K dimensions (for example, shares that sum "
                "to a constant); the diagonal class needs a full-dimensional donor hull")
        self.X0, self.X1 = X0, X1
        self.K, self.J = K, J
        self.interior = X0.mean(axis=1)
        self.normal = {}                      # facet -> (outward normal, offset)
        self.across = {}                      # (facet, apex) -> facet across the ridge facet - {apex}
        self.n_wrap = 0

    def add(self, F):
        if F not in self.normal:
            hb = _hyperplane(self.X0[:, sorted(F)], self.interior)
            if hb is None:
                raise MlsynthDataError("degenerate facet; the donor hull violates general position")
            self.normal[F] = hb
        return self.normal[F]

    def visible(self, F):
        n, b = self.normal[F]
        return n @ self.X1 > b + TOL

    def neighbour(self, F, apex):
        key = (F, apex)
        if key in self.across:
            return self.across[key]
        X0, K, J = self.X0, self.K, self.J
        n, _ = self.normal[F]
        ridge = sorted(F - {apex})
        p = X0[:, ridge[0]]
        Er = X0[:, ridge[1:]] - p[:, None]
        Qr = np.linalg.qr(Er)[0] if Er.shape[1] else np.zeros((K, 0))
        v = X0[:, apex] - p
        v = v - Qr @ (Qr.T @ v) - n * (n @ v)
        v /= np.linalg.norm(v)                 # in-facet direction toward the apex
        D = X0 - p[:, None]
        ang = np.arctan2(-(n @ D), v @ D)      # angle swept away from the facet
        mask = np.ones(J, bool)
        mask[ridge] = False
        mask[apex] = False
        d = int(np.flatnonzero(mask)[np.argmax(ang[mask])])
        G = frozenset(ridge + [d])
        self.add(G)
        self.across[key] = G
        self.across[(G, d)] = F
        self.n_wrap += 1
        return G

    def visible_facets(self):
        seed = _seed_facet(self.X0, self.X1, self.interior)
        self.add(seed)
        vis, queue = {seed}, [seed]
        while queue:
            F = queue.pop()
            for apex in F:
                G = self.neighbour(F, apex)
                if G not in vis and self.visible(G):
                    vis.add(G)
                    queue.append(G)
        return vis

    def star(self, sigma, F0):
        """The facets containing face sigma, starting from a facet F0 that contains it."""
        out, queue = {F0}, [F0]
        while queue:
            G = queue.pop()
            for apex in G - sigma:
                H = self.neighbour(G, apex)
                if H not in out:
                    out.add(H)
                    queue.append(H)
        return out
