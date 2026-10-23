"""The node relaxation with treated indicators, after Wiegele and Zhao.

Wiegele and Zhao ("Tight SDP relaxations for cardinality-constrained
problems", 2024) relax a cardinality-constrained quadratic program over the
simplex by lifting the weights together with binary indicators and writing
the complementarity on the lifted matrix's off-diagonal block. On portfolio
instances with up to 400 assets their bound closed the gap to within 0.48%.

For MAREX the variables are ``x = (w, v, z)``: treated weights, control
weights and treated indicators, ``z_j = 1`` when market ``j`` is treated. A
design satisfies

    sum w = 1,  sum v = 1,  sum z = m,  w, v >= 0,  z binary,
    w_j (1 - z_j) = 0,  v_j z_j = 0,  w_j v_j = 0,

and its objective is ``w'Gw + v'Gv``. Lifting ``x x'`` to ``X`` with
``[[1, x'], [x, X]]`` positive semidefinite, each relation becomes linear in
``X``:

- binary ``z``: ``diag(X_zz) = z``;
- ``w_j (1 - z_j) = 0``: ``diag(X_wz) = w``; ``v_j z_j = 0``:
  ``diag(X_vz) = 0``; ``w_j v_j = 0``: ``diag(X_wv) = 0``;
- products of each linear equality with every variable: ``X_aw 1 = a``,
  ``X_av 1 = a`` and ``X_az 1 = m a`` for each block ``a``.

A node fixes ``z_j = 1`` for markets fixed in and ``z_j = 0`` for markets fixed
out. With ``dnn=True`` the lifted matrix is also kept elementwise nonnegative,
the doubly nonnegative relaxation; Wiegele and Zhao do without it.

The value is approximate, as for ``sdp_bound``.
"""
from __future__ import annotations

import numpy as np


def indicator_bound(B, A, I, O, F, r, m, dnn=False, solver="CLARABEL"):
    import cvxpy as cp
    n = B.shape[1]
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    scale = float(np.max(np.diag(G)))
    G = G / scale
    k = 3 * n
    Y = cp.Variable((k + 1, k + 1), PSD=True)
    x = Y[0, 1:]
    X = Y[1:, 1:]
    W, V, Z = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
    w, v, z = x[W], x[V], x[Z]
    one = np.ones(n)
    cons = [Y[0, 0] == 1, cp.sum(w) == 1, cp.sum(v) == 1, cp.sum(z) == m,
            w >= 0, v >= 0, z >= 0, z <= 1,
            cp.diag(X[Z, Z]) == z, cp.diag(X[W, Z]) == w,
            cp.diag(X[V, Z]) == 0, cp.diag(X[W, V]) == 0]
    for a in (W, V, Z):
        xa = x[a]
        cons += [X[a, W] @ one == xa, X[a, V] @ one == xa, X[a, Z] @ one == m * xa]
    if dnn:
        cons.append(X >= 0)
    if I:
        cons.append(z[I] == 1)
    if O:
        cons.append(z[O] == 0)
    obj = cp.trace(G @ X[W, W]) + cp.trace(G @ X[V, V])
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        if solver == "SCS":
            prob.solve(solver=cp.SCS, eps_abs=1e-9, eps_rel=1e-9, max_iters=200_000)
        else:
            prob.solve(solver=getattr(cp, solver))
    except cp.error.SolverError:
        return float("nan"), "solver_error"
    if prob.value is None:
        return float("nan"), prob.status
    return max(float(prob.value), 0.0) * scale, prob.status
