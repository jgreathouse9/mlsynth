"""A semidefinite node bound for MAREX's design, from the paper's own program.

Abadie and Zhao's supplementary appendix (OA.6, program (OA.7)) writes the
unconstrained design as a quadratic program in ``x = (w, v)`` with the
complementarity ``sum_j w_j v_j = 0``: a market carries weight on one side at
most. Lifting ``x x'`` to a matrix ``X`` with ``[[1, x'], [x, X]]`` positive
semidefinite gives a convex relaxation. Every constraint below holds for every
real design, so for every treated set ``S`` in a node's subtree, with its
weights ``w`` and ``v``, the matrix ``X = x x'`` is feasible and the objective
equals ``f(S)``. The relaxation's value is therefore a lower bound on ``f`` over
the subtree:

- ``sum w = sum v = 1``, ``w, v >= 0`` and ``X >= 0`` elementwise;
- from ``sum v = 1`` and ``sum w = 1``: ``X_ww 1 = w``, ``X_wv 1 = w``,
  ``X_vv 1 = v``, ``X_vw 1 = v``;
- complementarity: ``diag(X_wv) = 0``, since ``w_j v_j = 0``;
- the node's fixings: ``w_j = 0`` for a market fixed out of the treated set,
  ``v_j = 0`` for one fixed in;
- the cardinality: ``w`` has at most ``r`` nonzeros on the free markets and
  ``m`` overall, so by Cauchy-Schwarz ``sum_F w_j^2 >= (sum_F w_j)^2 / r`` and
  ``sum w_j^2 >= 1 / m``. In the lift these are linear:
  ``r tr(X_FF) >= 1' X_FF 1`` and ``m tr(X_ww) >= 1``.

The objective is ``<G, X_ww> + <G, X_vv>``, with ``G`` the Gram matrix of the
markets measured from the target, scaled to unit largest diagonal for the
solver and scaled back.

This module solves the relaxation with an interior-point solver and returns
its value. That value is approximate; a pruning bound needs a certificate
from the dual solution with the rounding bounded, which is the next step.
"""
from __future__ import annotations

import numpy as np


def sdp_bound(B, A, I, O, F, r, m, solver="CLARABEL"):
    import cvxpy as cp
    n = B.shape[1]
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    scale = float(np.max(np.diag(G)))
    G = G / scale
    k = 2 * n
    Y = cp.Variable((k + 1, k + 1), PSD=True)
    x = Y[0, 1:]
    X = Y[1:, 1:]
    w, v = x[:n], x[n:]
    Xww, Xvv = X[:n, :n], X[n:, n:]
    Xwv, Xvw = X[:n, n:], X[n:, :n]
    one = np.ones(n)
    cons = [Y[0, 0] == 1, cp.sum(w) == 1, cp.sum(v) == 1, x >= 0, X >= 0,
            Xww @ one == w, Xwv @ one == w, Xvv @ one == v, Xvw @ one == v,
            cp.diag(Xwv) == 0, m * cp.trace(Xww) >= 1]
    if O:
        cons.append(w[O] == 0)
    if I:
        cons.append(v[I] == 0)
    if F and r < len(F):
        XFF = Xww[F, :][:, F]
        cons.append(r * cp.trace(XFF) >= cp.sum(XFF))
    obj = cp.trace(G @ Xww) + cp.trace(G @ Xvv)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=getattr(cp, solver))
    return max(float(prob.value), 0.0) * scale, prob.status
