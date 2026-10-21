"""The closed-form treated bound priced from a prefix, and a form of it that is safe.

A prototype for issue #769. It is not wired into
:mod:`mlsynth.utils.marex_helpers.exact`; ``search.py`` swaps
:func:`updated_bound` in for ``treated_fit_bound`` from the outside.

The bound. On the centred columns ``B~ = B - A 1'`` with Gram matrix
``G = B~'B~``, the treated fit of a set ``S`` is bounded below by
``min { w'G_S w + a'w : 1'w = 1 } = (2 + t)^2 / (4 s) - u / 4``, with
``s = 1'G_S^-1 1``, ``t = 1'G_S^-1 a`` and ``u = a'G_S^-1 a`` (``1 / s`` when
``a = 0``). The shipped code solves an ``m``-by-``m`` system per candidate.

Bordering. Write ``S = P + {j}``, ``H = G_P^-1``, ``g = G[P, j]``, ``z = H g``
and ``sigma = G_jj - g'z``, the Schur complement of ``G_P`` in ``G_S``. The
Banachiewicz inversion formula (Zhang 2005, (0.7.2)) gives, for any ``e`` and
``f``,

    e'G_S^-1 f = e_P'H f_P + (e_j - z'e_P)(f_j - z'f_P) / sigma,

so ``s``, ``t`` and ``u`` extend from the prefix's in ``O(m^2)`` per candidate.
The candidates arrive in lexicographic order, so those sharing a prefix are
contiguous and the prefix's inverse is formed once for all of them. ``sigma``
is positive for a positive definite ``G`` (Zhang 2005, Theorem 1.12); a
non-positive value in floating point marks a breakdown, and the candidate gets
the bound 0, which discards nothing. The formula also shows that ``s`` can only
grow as a market is added, ``s_S - s_P = (1 - g'H 1)^2 / sigma``, so the bound
can only fall: the monotonicity a subtree bound rests on (``subtree.py``).

The safe form. Neumaier and Shcherbina (2004, sec. 3) turn an approximate dual
point into a rigorous bound by charging its residual against bounds on the
variables. The same move applies here, with the simplex as the bounds: for any
vector ``v``, ``||B~_S (w - v)||^2 >= 0`` gives ``w'G_S w >= 2 v'G_S w - v'G_S v``,
and minimising over the simplex,

    g(S) >= -v'G_S v + min over i of (2 (G_S v)_i + a_i),

with equality at the affine minimiser, where the right side is the closed form.
It is Fenchel's inequality for the treated fit (Rockafellar 1970, in the proof
of Theorem 31.1), the simplex entering through its support function
(Corollary 16.5.1). The bound holds for every ``v``, so how accurately ``v`` was computed, by a
fresh solve or by bordering, does not affect its validity; only the arithmetic
of this last evaluation does. :func:`safe_error_bound` bounds that arithmetic a
priori with Higham's inner-product bound (2002, (3.5) and (3.13)),
``|fl(x'y) - x'y| <= gamma_n |x|'|y|``, counting the length-``T`` products that
form ``G``.

References
----------
Zhang, F. (ed.) (2005). The Schur Complement and Its Applications. Springer.
Rockafellar, R. T. (1970). Convex Analysis. Princeton University Press.
Neumaier, A. and Shcherbina, O. (2004). Safe bounds in linear and mixed-integer
linear programming. Mathematical Programming 99, 283-296.
Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms, 2nd ed.
SIAM.
"""
from __future__ import annotations

import numpy as np

#: Unit roundoff of IEEE double precision.
U = 2.0 ** -53

#: Candidates priced per batch, as the shipped search.
CHUNK = 20_000


def gram(B: np.ndarray, A: np.ndarray):
    """The centred columns and their Gram matrix."""
    Bc = B - A[:, None]
    return Bc, Bc.T @ Bc


def prefix_groups(subsets: np.ndarray):
    """The distinct prefixes of contiguous runs, and each candidate's run."""
    pre = subsets[:, :-1]
    new = np.ones(len(subsets), dtype=bool)
    new[1:] = np.any(pre[1:] != pre[:-1], axis=1)
    return pre[new], np.cumsum(new) - 1


def bordered(G: np.ndarray, subsets: np.ndarray, lin=None):
    """``s``, ``t``, ``u``, ``x = G_S^-1 1``, ``y = G_S^-1 a`` and ``sigma`` per candidate.

    ``t``, ``u`` and ``y`` are ``None`` without a linear term.
    """
    P, gid = prefix_groups(subsets)
    j = subsets[:, -1]
    H = np.linalg.inv(G[P[:, :, None], P[:, None, :]])
    xP = H.sum(axis=2)
    sP = xP.sum(axis=1)
    g = G[subsets[:, :-1], j[:, None]]
    z = np.einsum("lij,lj->li", H[gid], g)
    sigma = G[j, j] - np.einsum("li,li->l", g, z)
    xPl = xP[gid]
    a1 = 1.0 - np.einsum("li,li->l", g, xPl)
    s = sP[gid] + a1 * a1 / sigma
    x = np.concatenate([xPl - z * (a1 / sigma)[:, None],
                        (a1 / sigma)[:, None]], axis=1)
    if lin is None:
        return s, None, None, x, None, sigma
    aP = lin[P]
    yP = np.einsum("kij,kj->ki", H, aP)
    tP = np.einsum("ki,ki->k", xP, aP)
    uP = np.einsum("ki,ki->k", yP, aP)
    aa = lin[j] - np.einsum("li,li->l", z, lin[subsets[:, :-1]])
    t = tP[gid] + a1 * aa / sigma
    u = uP[gid] + aa * aa / sigma
    y = np.concatenate([yP[gid] - z * (aa / sigma)[:, None],
                        (aa / sigma)[:, None]], axis=1)
    return s, t, u, x, y, sigma


def closed_form(s, t, u):
    """The bound from ``s``, ``t`` and ``u``; 0 where they are not usable."""
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 1.0 / s if t is None else (2.0 + t) ** 2 / (4.0 * s) - u / 4.0
    good = np.isfinite(val) & (s > 0)
    return np.where(good, np.maximum(val, 0.0), 0.0)


def updated_bound(B, A, subsets, linear=None):
    """``exact.treated_fit_bound`` by bordering: the same values, to rounding."""
    subsets = np.asarray(subsets, dtype=np.int64)
    G = gram(B, A)[1]
    lin = None if linear is None else np.asarray(linear, dtype=float)
    if subsets.shape[1] == 1:
        j = subsets[:, 0]
        s = 1.0 / G[j, j]
        t = None if lin is None else lin[j] * s
        u = None if lin is None else lin[j] ** 2 * s
        return closed_form(s, t, u)
    out = np.empty(len(subsets))
    for lo in range(0, len(subsets), CHUNK):
        blk = subsets[lo:lo + CHUNK]
        s, t, u, _, _, sigma = bordered(G, blk, lin)
        out[lo:lo + len(blk)] = np.where(sigma > 0, closed_form(s, t, u), 0.0)
    return out


def affine_minimiser(s, t, x, y):
    """The affine relaxation's minimiser ``v`` from bordering's output."""
    if t is None:
        return x / s[:, None]
    mu = (2.0 + t) / s
    return (mu[:, None] * x - y) / 2.0


def safe_bound(G, subsets, v, lin=None):
    """``-v'G_S v + min_i (2 (G_S v)_i + a_i)``, a lower bound on ``g(S)`` for any ``v``."""
    GS = G[subsets[:, :, None], subsets[:, None, :]]
    Gv = np.einsum("lij,lj->li", GS, v)
    a = 0.0 if lin is None else lin[subsets]
    return -np.einsum("li,li->l", v, Gv) + (2.0 * Gv + a).min(axis=1)


def safe_error_bound(Habs, subsets, v, T, lin=None):
    """An a priori bound on the rounding error of :func:`safe_bound`.

    ``Habs = |B~|'|B~|``. Every quantity is an inner product of length at most
    ``T + m + 3`` over terms bounded by ``|v|'Habs|v|`` and ``(Habs |v|)_i``, so
    Higham's (3.5) and (3.13) bound the error by ``gamma_{T + m + 3}`` times
    their sum; the factor 2 covers the two places they enter.
    """
    m = subsets.shape[1]
    k = T + m + 3
    gamma = k * U / (1.0 - k * U)
    HS = Habs[subsets[:, :, None], subsets[:, None, :]]
    av = np.abs(v)
    Hv = np.einsum("lij,lj->li", HS, av)
    quad = np.einsum("li,li->l", av, Hv)
    lin_term = 0.0 if lin is None else np.abs(lin[subsets]).max(axis=1)
    return 2.0 * gamma * (quad + 2.0 * Hv.max(axis=1) + lin_term)
