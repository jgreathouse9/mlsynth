"""The slope-free relaxation on its face (#965).

Every point Z of a size's relaxation (slopefree.Rows) has forced null directions:

  * v = (-m, 1, ..., 1), since sum u = m and its products sum_j U_ij = m u_i say
    exactly Z v = 0;
  * e_j for a control fixed out (U_jj = u_j = 0 makes row j of Z zero);
  * e_0 - e_j for a control fixed in (Z_00 = Z_0j = Z_jj = 1).

So no point is positive definite, the relaxation has no interior (Burer 2012,
Prop. 8.3, in Anjos and Lasserre eds., Handbook on Semidefinite, Conic and
Polynomial Optimization, p. 206), its dual's optimal set is unbounded (p. 212),
and an interior-point solver loses accuracy on it (pp. 697, 724; Sun, Toh, Yuan
and Zhao 2020, SDPNAL+, p. 8). On the beds Clarabel returned optimal_inaccurate on
199 of 222 Shor solves and 254 of 260 full solves of s30_k4.

Every column z of Z then has z_j = 0 (j fixed out), z_j = z_0 (j fixed in) and
sum_F z_j = q z_0 over the free controls F, with q = m - (number fixed in). With a
pivot r in F, z is determined by its entries R = {0} + (F minus r):

    Z = T W T',   W = Z_RR,

with T integer: row 0 is e_0, row j in F - r is e_j, the pivot's row is
q e_0 - sum_{F - r} e_j, a row fixed in is e_0 and a row fixed out is zero. This is
facial reduction onto the face that contains the feasible set (Krislock and
Wolkowicz, in the Handbook, p. 893; Burer's elimination x = X alpha, pp. 206-207),
with a basis chosen to keep the rows sparse: only those touching the pivot densify.

Two uses.

  * The solve (``reduce``): the rows are written in W, the rows the face satisfies
    identically are dropped (sum u = m becomes m W_00 = m, its products 0 = 0, the
    fixings and the diagonal rows of fixed controls vanish), and Clarabel solves on
    W >= 0, a problem the forced directions no longer make degenerate. Which rows
    the solve keeps decides only the multipliers it proposes; the certificate is
    computed on every row.
  * The certificate (``face_bound``): for any multipliers, <S, Z> = <T'ST, W> and
    tr W = 1 + sum_{F - r} u_j <= 1 + q (u_r = U_rr >= 0), so every design's value is
    at most k + (1 + q) max(0, lambda_max(T'ST)), exactly, since T'ST is an integer
    combination of S's dyadic entries, computed in S's exponent from integers. T is
    not orthonormal, so at multipliers far from the face's this can exceed the
    full-space bound k + (1 + m) max(0, lambda_max(S)); slopefree.bound takes the
    smaller of the two.

A structural row (every row but those carrying c) that reduces on the face to a
constant it does not meet means the node holds no design: ``reduce`` raises
FaceError and names it. The branch and bound never builds such a node (its fixings
are consistent, and a node with every size slot fixed in is a leaf), so the error
is a broken invariant, reported where it happens."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from mlsynth.utils.marex_helpers.certify import Dyadic


class FaceError(ValueError):
    """A row of the relaxation is violated on its face: the node holds no design."""


@dataclass(frozen=True)
class Face:
    p: int
    m: int
    q: int                       # m less the controls fixed in
    pivot: int                   # the free control whose row carries the size
    index: Tuple[int, ...]       # W's coordinates, as indices of Z
    ones: Tuple[int, ...]        # the controls fixed in

    @property
    def trace(self) -> int:
        """The bound on tr W: 1 + q."""
        return 1 + self.q

    def columns(self):
        """T's columns as sparse vectors over Z's indices: [(z_index, coefficient)]."""
        cols = [[(0, 1), (self.pivot + 1, self.q)] + [(o + 1, 1) for o in self.ones]]
        for zi in self.index[1:]:
            cols.append([(zi, 1), (self.pivot + 1, -1)])
        return cols

    def integer_basis(self) -> np.ndarray:
        """T as an array of Python ints, for exact use."""
        T = np.zeros((self.p + 1, len(self.index)), dtype=object); T[:] = 0
        for k, col in enumerate(self.columns()):
            for zi, v in col:
                T[zi, k] += v
        return T

    def matrix(self) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for k, col in enumerate(self.columns()):
            for zi, v in col:
                rows.append(zi); cols.append(k); vals.append(float(v))
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.p + 1, len(self.index)))

    def slack(self, D):
        """T'ST for the dyadic S = D.N / 2**D.e, in the same exponent: T is integer, so
        T'NT is integer, built from T's sparse columns (a few integer column and row
        combinations each), not a product of rationals."""
        N, cols = D.N, self.columns()
        k = len(cols)

        def combine(take):
            return [sum((take(zi) * v for zi, v in col[1:]), take(col[0][0]) * col[0][1]) for col in cols]

        Y = np.empty((N.shape[0], k), dtype=object)
        for a, y in enumerate(combine(lambda zi: N[:, zi])):
            Y[:, a] = y
        M = np.empty((k, k), dtype=object)
        for a, row in enumerate(combine(lambda zi: Y[zi, :])):
            M[a, :] = row
        return Dyadic(M, D.e)

    def kept(self, R) -> Tuple[np.ndarray, np.ndarray]:
        """The row indices the face solve keeps, and FaceError for a structural row
        the face violates."""
        out = _reduce(self, R)
        return out[1], out[3]


def face_of(p: int, m: Optional[int], fixes: Sequence[Tuple[int, int]]) -> Optional[Face]:
    """The face of a size-m node with these fixings, or None when there is no size
    (the group rows), no free control, or every size slot fixed in (the node is one
    design, evaluated as a leaf)."""
    if m is None:
        return None
    fx = {}
    for j, v in fixes:
        if j in fx and fx[j] != v:
            raise ValueError(f"control {j} is fixed both in and out")
        fx[j] = v
    ones = tuple(sorted(j for j, v in fx.items() if v))
    q = m - len(ones)
    if q < 0:
        raise ValueError(f"{len(ones)} controls fixed in, more than the size {m}")
    free = [j for j in range(p) if j not in fx]
    if not free or q == 0:
        return None
    pivot = free[0]
    return Face(p=p, m=m, q=q, pivot=pivot, index=(0,) + tuple(j + 1 for j in free[1:]), ones=ones)


class Reduced(NamedTuple):
    """A size's rows on the face, on x = [vec(W); a] with a = Z's pivot row: the rows
    kept (``eq`` = ``beq``, ``ineq`` <= ``bin``) at their indices in R, and the
    definition rows ``define`` x = 0 that tie a to W."""
    eq: sp.csr_matrix
    beq: np.ndarray
    keep_e: np.ndarray
    ineq: sp.csr_matrix
    bin: np.ndarray
    keep_i: np.ndarray
    define: sp.csr_matrix


def _lift(F: Face) -> sp.csr_matrix:
    """vec(Z) from vec(W), column-major and symmetrised: each row's coefficients on a
    symmetric W, so a row the face satisfies identically (the products of sum u = m)
    is exactly zero (halves of integers are exact in doubles)."""
    T = F.matrix()
    k = T.shape[1]
    K = sp.kron(T, T, format="csr")
    perm = np.arange(k * k).reshape(k, k).T.ravel()
    return ((K + K[:, perm]) / 2).tocsr()


def _reduce(F: Face, R):
    """The kept rows on [vec(W); a] and their indices, and the definition rows.

    Substituting Z = T W T' into a row puts every free control into each entry on the
    pivot's row, since that row of T is q e_0 less the others: at 96 controls the full
    level's 73,825 nonzeros become 262,300. So the entries Z_rb = Z_br of the pivot's
    row are carried by a_b, defined once by a_b = (T W T')_rb, and every other entry is
    one entry of W; a row keeps its sparsity. Which rows the face satisfies identically
    is read from the substituted form, so the redundant rows are still dropped."""
    n = F.p + 1
    K = _lift(F)
    Me, Mi = R.matrices(1.0)
    c_rows = set(R.c_rows)

    def keep(M, bs, equality):
        M = (M @ K).tocsr()
        M.eliminate_zeros()
        nnz = np.diff(M.indptr)
        first = np.where(nnz > 0, M.indices[np.minimum(M.indptr[:-1], max(len(M.indices) - 1, 0))], -1)
        constant = (nnz == 0) | ((nnz == 1) & (first == 0))   # the row on the face is const * W_00, W_00 = 1
        if equality:
            constant[0] = False                               # Z_00 = 1 is W_00 = 1
        else:
            constant[list(c_rows)] = False                    # rows carrying c are kept whatever they reduce to
        for i in np.flatnonzero(constant):
            const = float(M.data[M.indptr[i]]) if nnz[i] else 0.0
            b = float(bs[i])
            if (equality and const != b) or (not equality and const > b):
                kind = "equality" if equality else "inequality"
                raise FaceError(f"{kind} row {i} of the {R.level} rows reduces on the face to {const:g} "
                                f"{'=' if equality else '<='} {b:g}, which fails: the fixings {list(R.fixes)} "
                                f"leave no design")
        return np.flatnonzero(~constant)

    keep_e, keep_i = keep(Me, R.beq, True), keep(Mi, R.bin, False)
    r = F.pivot + 1
    on_row = np.concatenate([r + np.arange(n) * n, np.arange(n) + r * n])      # vec indices of Z_rb and Z_br
    P = sp.csr_matrix((np.ones(2 * n), (on_row, np.concatenate([np.arange(n)] * 2))), shape=(n * n, n))
    P.sum_duplicates(); P.data[:] = 1.0                                       # Z_rr is listed twice: one entry
    off = np.ones(n * n); off[on_row] = 0.0
    K_off = (sp.diags(off) @ K).tocsr()

    def rows(M, kept):
        M = M[kept]
        return sp.hstack([M @ K_off, M @ P], format="csr")

    define = sp.hstack([K[r + np.arange(n) * n], -sp.identity(n)], format="csr")
    return rows(Me, keep_e), keep_e, rows(Mi, keep_i), keep_i, define


def reduce(F: Face, R, c_scale: float) -> Reduced:
    """The kept rows on [vec(W); a], the rows carrying c divided by ``c_scale`` as
    slopefree.solve scales them, with their right-hand sides, indices and the
    definition rows."""
    eq, keep_e, ineq, keep_i, define = _reduce(F, R)
    scale = np.ones(R.n_in); scale[list(R.c_rows)] = 1.0 / c_scale
    ineq = (sp.diags(scale[keep_i]) @ ineq).tocsr()
    return Reduced(eq, np.asarray(R.beq, dtype=float)[keep_e], keep_e,
                   ineq, np.asarray(R.bin, dtype=float)[keep_i], keep_i, define)


def trace_term(F: Face, D) -> Q:
    """(1 + q) max(0, lambda_max(T'ST)) for the dyadic S = D, bounded above exactly."""
    from mlsynth.utils.marex_helpers.branch import lambda_min_lower
    M = F.slack(D)
    low = lambda_min_lower(Dyadic(-M.N, M.e))   # lambda_max(M) <= -low
    if low is None:  # pragma: no cover - 40 quadrupled shifts factor any finite symmetric matrix
        return None
    return F.trace * max(Q(0), -low)


def face_bound(A, R, y_eq, y_in) -> Optional[Q]:
    """The certificate on the face alone: k + (1 + q) max(0, lambda_max(T'ST))."""
    from mlsynth.utils.marex_helpers.branch import _from_fractions
    from .slopefree import _slack
    S, k = _slack(A, R, y_eq, y_in)
    t = trace_term(R.face, _from_fractions(S.tolist()))
    return None if t is None else k + t
