"""Node bounds for a branch and bound over the global SYNDES designs (#803, #820).

A node fixes a set ``I`` of treated units and a set ``O`` of controls and leaves
``F`` free; its subtree is every treated set ``S`` of size ``K`` with
``I <= S <= I + F``. With ``sigma`` the design's +/-1 indicator, the two-way
closed form of ``gram.py`` bounds each set,

    f2(S) >= lb(S) = 4 alpha / sigma' R sigma,

and one-way is two-way with the treated weights fixed, so ``f1(S) >= f2(S)``.
A node bound is then ``4 alpha / U`` for any ``U >= max sigma' R sigma`` over the
subtree. Split ``sigma`` into its decided part ``D = I + O`` and free part:

    sigma' R sigma = c + 2 b' x + x' A x,    x = sigma_F in {-1, +1}^n,
    c = sigma_D' R_DD sigma_D,  b = R_FD sigma_D,  A = R_FF,

with ``1' x = s = 2 r - n``, ``r = K - |I|`` positives among the ``n`` free units.

``trs_upper``: relax ``x`` to the sphere ``|x|^2 = n`` cut by the plane
``1'x = s``. Writing ``x = (s/n) 1 + z`` with ``z`` orthogonal to ``1`` and
``|z|^2 = rho^2 = n - s^2/n`` leaves a trust-region subproblem in ``z``. Its
Lagrangian dual,

    d(mu) = const + mu rho^2 + g' (mu I - B)^+ g,     mu > lambda_max(B),

is an upper bound for every such ``mu`` (``B`` the projected ``A``, ``g`` the
projected linear term), and its minimum over ``mu`` is the subproblem's maximum.
So an inexact ``mu`` costs tightness, never validity.

``spectral_upper``: the cheap bound ``c + 2 |b|_1 + lambda_max(A) n``.
Neither dominates the other, so ``node_upper`` takes the smaller.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def split(R: np.ndarray, I: Sequence[int], O: Sequence[int], F: Sequence[int]):
    """``c``, ``b``, ``A`` of the decomposition above."""
    D = list(I) + list(O)
    sD = np.concatenate([np.ones(len(I)), -np.ones(len(O))])
    F = list(F)
    c = float(sD @ R[np.ix_(D, D)] @ sD) if D else 0.0
    b = R[np.ix_(F, D)] @ sD if D else np.zeros(len(F))
    return c, b, R[np.ix_(F, F)]


def spectral_upper(R, I, O, F, K) -> float:
    c, b, A = split(R, I, O, F)
    n = len(F)
    if n == 0:
        return c
    top = float(np.linalg.eigvalsh(A)[-1])
    return c + 2.0 * float(np.abs(b).sum()) + max(top, 0.0) * n


def trs_upper(R, I, O, F, K) -> Tuple[float, np.ndarray]:
    """``U >= max sigma' R sigma`` over the subtree, and the relaxed maximiser
    ``x`` (for branching). Exact when ``n <= 1``."""
    c, b, A = split(R, I, O, F)
    return trs_parts(c, b, A, 2 * (K - len(I)) - len(F))


def trs_parts(c: float, b: np.ndarray, A: np.ndarray, s: int) -> Tuple[float, np.ndarray]:
    """The trust-region bound on ``c + 2 b'x + x'Ax`` over ``|x|^2 = n``,
    ``1'x = s``, and its maximiser."""
    n = len(b)
    if n == 0:
        return c, np.zeros(0)
    x0 = np.full(n, s / n)
    rho2 = n - s * s / n
    const = c + 2.0 * float(b @ x0) + float(x0 @ A @ x0)
    if rho2 <= 1e-12 * n:                         # r = 0 or r = n: x is decided
        return const, x0
    # Work in an orthonormal basis P of the plane's direction space, 1-perp,
    # so the constant direction never enters the spectrum: with it, its zero
    # eigenvalue would force mu > 0 when A is negative on the plane.
    P = np.linalg.svd(np.ones((1, n)))[2][1:].T
    B = P.T @ A @ P
    B = 0.5 * (B + B.T)
    g = P.T @ (b + A @ x0)
    lam, Q = np.linalg.eigh(B)
    gt = Q.T @ g
    top = float(lam[-1])

    def d(mu: float) -> float:
        return const + mu * rho2 + float(np.sum(gt ** 2 / (mu - lam)))

    # d is convex in mu with d'(mu) = rho^2 - |z(mu)|^2, z = (mu I - B)^-1 g.
    # Its minimiser solves h(mu) = 1 / |z(mu)| = 1 / rho, and h is concave and
    # increasing on (top, inf) (More and Sorensen 1983), so Newton's method on
    # h started left of the root stays left of it and converges monotonically.
    # Every iterate is above top, so every d(mu) is a valid bound.
    scale = max(abs(top), float(np.abs(lam).max()), 1e-300)
    mu = top + 1e-12 * scale
    target = 1.0 / math.sqrt(rho2)
    for _ in range(60):
        u = gt / (mu - lam)
        z2 = float(u @ u)
        if z2 <= rho2:                            # at or right of the root: the hard case, or g = 0
            break
        h = 1.0 / math.sqrt(z2)
        dh = float(u @ (u / (mu - lam))) * h ** 3
        step = (target - h) / dh
        mu += step
        if step <= 1e-15 * max(abs(mu), scale):
            break
    z = P @ (Q @ (gt / (mu - lam)))
    return d(mu), x0 + z


def _lift_problem(c: float, b: np.ndarray, A: np.ndarray, s: int):
    """The lifted program and its constraint groups, for the certificate."""
    import cvxpy as cp

    n = len(b)
    X = cp.Variable((n + 1, n + 1), symmetric=True)
    x, XX = X[0, 1:], X[1:, 1:]
    col = cp.reshape(x, (n, 1), order="F") @ np.ones((1, n))
    row = np.ones((n, 1)) @ cp.reshape(x, (1, n), order="F")
    eq = [X[0, 0] == 1, cp.diag(XX) == 1, cp.sum(x) == s, cp.sum(XX, axis=1) == s * x]
    rlt = [1 + XX + col + row >= 0, 1 + XX - col - row >= 0,
           1 - XX + col - row >= 0, 1 - XX - col + row >= 0]
    prob = cp.Problem(cp.Maximize(c + 2 * b @ x + cp.trace(A @ XX)),
                      [X >> 0] + eq + rlt)
    return prob, eq, rlt


def _certify_lift(c, b, A, s, eq, rlt) -> float:
    """An upper bound from any multipliers. With ``y`` on the equalities and
    ``N_k >= 0`` on the products, the Lagrangian ``L(X)`` is at least the
    objective at every design's ``X = [1; x][1; x]'``, and is affine in ``X``:
    ``const + <M, X>``. A design's ``X`` is semidefinite with trace ``n + 1``,
    so ``const + (n + 1) lambda_max(M)`` bounds the subtree. The objective's
    constant ``c`` stays in ``const``, as the solver writes it: on a design
    ``c X_00`` is the same number, but the solver's multipliers are stationary
    only for the first form, and the second left the bound loose by ``c`` times
    ``n + 1``. Clipping ``N`` to
    its sign keeps this valid; the equality multipliers may have either sign,
    so every sign convention is tried and the smallest bound kept."""
    n = len(b)
    y0 = float(eq[0].dual_value)
    yd = np.asarray(eq[1].dual_value, dtype=float).reshape(n)
    ys = float(eq[2].dual_value)
    w = np.asarray(eq[3].dual_value, dtype=float).reshape(n)
    N = [np.maximum(np.asarray(r.dual_value, dtype=float).reshape(n, n), 0.0) for r in rlt]
    # Products, written as const + <M, X> with M symmetric; the x-coefficients
    # v go on the border as v / 2 on each side.
    rlt_const = float(sum(Nk.sum() for Nk in N))
    rlt_block = N[0] + N[1] - N[2] - N[3]
    rlt_v = ((N[0].sum(1) + N[0].sum(0)) - (N[1].sum(1) + N[1].sum(0))
             + (N[2].sum(1) - N[2].sum(0)) + (-N[3].sum(1) + N[3].sum(0)))
    best = np.inf
    for g0 in (1.0, -1.0):
        for gd in (1.0, -1.0):
            for gs in (1.0, -1.0):
                for gw in (1.0, -1.0):
                    a0, ad, as_, aw = g0 * y0, gd * yd, gs * ys, gw * w
                    M = np.zeros((n + 1, n + 1))
                    M[0, 0] = -a0
                    M[1:, 1:] = A - np.diag(ad) - 0.5 * (aw[:, None] + aw[None, :])
                    M[1:, 1:] += 0.5 * (rlt_block + rlt_block.T)
                    v = 2.0 * b - as_ + s * aw + rlt_v
                    M[0, 1:] = M[1:, 0] = 0.5 * v
                    const = c + a0 + float(ad.sum()) + as_ * s + rlt_const
                    top = float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])
                    best = min(best, const + (n + 1) * top)
    return best


def lift_upper(R, I, O, F, K, solver: str = "CLARABEL") -> float:
    """Shor's relaxation of the node with the four product inequalities.

    With ``X = [1, x'; x, x x']`` for a sign completion ``x`` of the free units,
    every constraint below holds, so the relaxation's maximum bounds the
    subtree's: ``X >= 0`` (semidefinite), ``X_00 = 1``, ``diag = 1``
    (``x_i^2 = 1``), ``1'x = s`` and ``X 1 = s x`` (the cardinality, and its
    products with ``x``), and, for every pair, ``(1 +- x_i)(1 +- x_j) >= 0``
    written on the lift. The last four are the reformulation-linearization
    (RLT) products of the box ``-1 <= x <= 1``, the sign-vector form of the
    doubly nonnegative relaxation; ``benchmarks/studies/marex_bnb`` found the
    same entrywise nonnegativity carries MAREX's bound.

    The parts are divided by their largest entry before the solve. ``R``
    scales as the outcome's units to the power -4, so on GeoLift its entries
    are near 1e-11, under the solver's absolute tolerances; unscaled, the
    solver reported ``optimal`` at a value 12% under the true maximum, a bound
    above the optimum. The value returned is not the solver's: it is rebuilt
    from the solver's multipliers by ``_certify_lift`` (in floating point), so
    an inaccurate solve costs tightness, never validity. A solve with no
    multipliers falls back to the trust-region bound."""
    import cvxpy as cp

    c, b, A = split(R, I, O, F)
    n = len(F)
    s = 2 * (K - len(I)) - n
    if n <= 1:
        return trs_parts(c, b, A, s)[0]
    unit = max(abs(c), float(np.abs(b).max(initial=0.0)), float(np.abs(A).max()))
    if unit == 0.0:
        return c
    c, b, A = c / unit, b / unit, A / unit
    prob, eq, rlt = _lift_problem(c, b, A, s)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:  # pragma: no cover - falls back below
        pass
    fallback = trs_parts(c, b, A, s)[0]
    if eq[0].dual_value is None:  # pragma: no cover - a solver that returned nothing
        return unit * fallback
    return unit * min(_certify_lift(c, b, A, s, eq, rlt), fallback)


def node_upper(R, I, O, F, K) -> float:
    """The smaller of the two upper bounds. Neither dominates: the spectral
    bound takes the linear term over the box, ``b'x <= |b|_1``, the
    trust-region bound over the sphere, where ``b'x`` reaches
    ``sqrt(n) |b|_2 >= |b|_1``, and the trust-region bound is the tighter on
    the quadratic term."""
    return min(trs_upper(R, I, O, F, K)[0], spectral_upper(R, I, O, F, K))


def node_lower(alpha: float, upper: float) -> float:
    """``4 alpha / U``, the node's bound on f2, hence on f1."""
    return 4.0 * alpha / upper if upper > 0 else math.inf
