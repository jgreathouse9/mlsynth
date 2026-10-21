"""The perspective node bound: a diagonal split of the Gram matrix with indicators.

On the simplex, ``w'Gw = w'(G + 1a' + a1')w - 2a'w`` for any vector ``a``.
Choosing ``a`` and a diagonal ``D = diag(d) >= 0`` with

    M = G + 1a' + a1' - D  positive semidefinite,

the objective of a design splits as ``w'Mw + sum_j d_j w_j^2 - 2a'w`` plus the
same in ``v``. The separable part is where cardinality and disjoint support
enter: with ``z_j = 1`` for a treated market, ``w_j = 0`` whenever ``z_j = 0``
and ``v_j = 0`` whenever ``z_j = 1``, so ``d_j w_j^2 = d_j w_j^2 / z_j`` and
``d_j v_j^2 = d_j v_j^2 / (1 - z_j)`` on every design. Relaxing ``z`` to
``[0, 1]`` keeps the perspective terms convex (Frangioni and Gentile 2006;
the survey of Sun et al. 2013 calls this the perspective relaxation). Per
node it is a second-order cone program in ``3n`` variables:

    min  w'Mw + sum_j d_j w_j^2/z_j - 2a'w + v'Mv + sum_j d_j v_j^2/(1-z_j) - 2a'v
    s.t. sum w = sum v = 1, w, v >= 0, z in [0, 1], sum_F z = r,
         z_I = 1, z_O = 0, w_O = 0, v_I = 0.

``(a, d)`` is chosen once per panel, by a semidefinite program maximising
``sum d``. Only ``G`` on the zero-sum directions limits ``d``: when it is
singular there, which happens once the markets outnumber the time points plus
one, ``d`` is zero and the bound is the monotone one.
"""
from __future__ import annotations

import numpy as np


def diagonal_split(G):
    """``(a, d)`` maximising ``sum d`` with ``G + 1a' + a1' - diag(d)`` PSD."""
    import cvxpy as cp
    n = G.shape[0]
    s = float(np.max(np.diag(G)))
    Gs = G / s
    a = cp.Variable(n)
    d = cp.Variable(n, nonneg=True)
    one = np.ones((n, 1))
    Mexpr = Gs + one @ cp.reshape(a, (1, n)) + cp.reshape(a, (n, 1)) @ one.T - cp.diag(d)
    prob = cp.Problem(cp.Maximize(cp.sum(d)), [(Mexpr + Mexpr.T) / 2 >> 0])
    prob.solve(solver=cp.CLARABEL)
    return a.value * s, d.value * s


def perspective_value(G, a, d, I, O, F, r, shrink=1.0):
    """The perspective relaxation's value at one node (floating point)."""
    import cvxpy as cp
    n = G.shape[0]
    d = d * shrink
    M = G + np.outer(np.ones(n), a) + np.outer(a, np.ones(n)) - np.diag(d)
    lam = np.linalg.eigvalsh((M + M.T) / 2)[0]
    if lam < 0:
        M = M - lam * np.eye(n) * 1.000001          # repair rounding for the float solve
        d = np.maximum(d + lam * 1.000001, 0.0)
    M = (M + M.T) / 2
    s = float(np.max(np.diag(G)))
    w = cp.Variable(n, nonneg=True)
    v = cp.Variable(n, nonneg=True)
    z = cp.Variable(n)
    cons = [cp.sum(w) == 1, cp.sum(v) == 1, z >= 0, z <= 1]
    if F:
        cons.append(cp.sum(z[F]) == r)
    if I:
        cons += [z[I] == 1, v[I] == 0]
    if O:
        cons += [z[O] == 0, w[O] == 0]
    obj = cp.quad_form(w, cp.psd_wrap(M / s)) + cp.quad_form(v, cp.psd_wrap(M / s)) \
        - 2 * (a / s) @ (w + v)
    for j in range(n):
        if d[j] <= 0:
            continue
        if j in O:
            obj += d[j] / s * cp.square(v[j])
        elif j in I:
            obj += d[j] / s * cp.square(w[j])
        else:
            obj += d[j] / s * (cp.quad_over_lin(w[j], z[j]) + cp.quad_over_lin(v[j], 1 - z[j]))
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value) * s, prob.status
