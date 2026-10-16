"""Candidate solvers for ``min ||A - B w||^2`` over the probability simplex.

Each is a callable ``(B, A) -> Candidate``. Two are already in the library and
are here as the baseline and the oracle; the rest are not implemented anywhere in
mlsynth and are the point of the study.

The methods split on one axis, and the session's measurements predict which side
wins: a method with few expensive steps pays little Python overhead, while a
first-order method with many cheap steps pays it once per step. A converged
Basque solve is 54 us of which about 3 us is arithmetic, so an iteration that is
almost free in flops is not almost free here.

    active_set          the shipped primal active set, priced seed (baseline)
    active_set_cold     the same, starting from the uniform point
    clarabel            cvxpy's interior point, the correctness oracle
    osqp                ADMM on the QP directly, warm-startable, no factorization
                        per iteration
    frank_wolfe         simplex-native: the linear oracle over the simplex is an
                        argmin, so an iteration is one matvec and no projection
    away_frank_wolfe    Frank-Wolfe with away steps, which is what gives linear
                        convergence on a polytope
    pairwise_smo        pairwise coordinate descent. The sum-to-one constraint
                        forbids moving one coordinate, so the smallest move is a
                        transfer between two -- SVM's SMO, on this program
    fista_projected     accelerated projected gradient, exact simplex projection
    becker_kloessner    the special-case cascade of Becker & Kloessner (2017):
                        no sunny donors means an exact fit exists and the answer
                        comes from one restricted QP; a single sunny donor is the
                        answer outright; otherwise prune shady donors and hand
                        the reduced program to the active set
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np

from mlsynth.utils.solvers.accelerate import simplex_project
from mlsynth.utils.solvers.active_set import solve_simplex_qp

MAX_ITER = 20000
FTOL = 1e-12


class Candidate(NamedTuple):
    w: np.ndarray
    iterations: int
    status: str


def _obj(B, A, w):
    r = B @ w - A
    return float(r @ r)


# --------------------------------------------------------------------------- #
# in the library already
# --------------------------------------------------------------------------- #
def active_set(B, A) -> Candidate:
    w, info = solve_simplex_qp(B, A, return_info=True)
    return Candidate(np.asarray(w, float), int(info["pivots"]),
                     "optimal" if info["converged"] else "unconverged")


def active_set_cold(B, A) -> Candidate:
    w, info = solve_simplex_qp(B, A, return_info=True, accelerate=False)
    return Candidate(np.asarray(w, float), int(info["pivots"]),
                     "optimal" if info["converged"] else "unconverged")


def clarabel(B, A) -> Candidate:
    import cvxpy as cp

    J = B.shape[1]
    w = cp.Variable(J)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(B @ w - A)),
                      [w >= 0, cp.sum(w) == 1])
    for solver in ("CLARABEL", "SCS", "ECOS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            prob.solve(solver=solver)
        except Exception:
            continue
        if w.value is not None:
            return Candidate(np.clip(np.asarray(w.value, float).ravel(), 0, None),
                             int(prob.solver_stats.num_iters or 0), prob.status)
    return Candidate(np.full(J, 1.0 / J), 0, "unavailable")


# --------------------------------------------------------------------------- #
# not in the library
# --------------------------------------------------------------------------- #
def osqp(B, A) -> Candidate:
    """ADMM on the QP. One factorization at setup, then cheap iterations."""
    import osqp as _osqp
    import scipy.sparse as sp

    m, J = B.shape
    P = sp.csc_matrix(2.0 * (B.T @ B))
    q = -2.0 * (B.T @ A)
    # rows: the sum-to-one equality, then the J non-negativity bounds
    Acon = sp.vstack([sp.csc_matrix(np.ones((1, J))), sp.eye(J, format="csc")],
                     format="csc")
    lo = np.concatenate([[1.0], np.zeros(J)])
    hi = np.concatenate([[1.0], np.full(J, np.inf)])
    prob = _osqp.OSQP()
    prob.setup(P=P, q=q, A=Acon, l=lo, u=hi, verbose=False,
               eps_abs=1e-9, eps_rel=1e-9, max_iter=MAX_ITER, polish=True)
    res = prob.solve()
    w = np.clip(np.asarray(res.x, float).ravel(), 0.0, None)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return Candidate(np.full(J, 1.0 / J), 0, "failed")
    return Candidate(w / s, int(res.info.iter), str(res.info.status))


def frank_wolfe(B, A) -> Candidate:
    """The linear oracle over the simplex is ``argmin`` of the gradient, so an
    iteration is one matvec and a convex step -- no projection, no factorization,
    and the iterate never leaves the simplex."""
    m, J = B.shape
    w = np.zeros(J)
    w[int(np.argmin(np.einsum("ij,i->j", B, -A)))] = 1.0
    Bw = B @ w
    for k in range(MAX_ITER):
        grad = 2.0 * (B.T @ (Bw - A))
        s = int(np.argmin(grad))
        d_gap = float(grad @ w - grad[s])          # Frank-Wolfe gap
        if d_gap <= FTOL:
            return Candidate(w, k, "optimal")
        Bs = B[:, s]
        diff = Bs - Bw
        denom = float(diff @ diff)
        if denom <= 0.0:
            return Candidate(w, k, "optimal")
        gamma = min(1.0, max(0.0, float((A - Bw) @ diff) / denom))
        w = (1.0 - gamma) * w
        w[s] += gamma
        Bw = Bw + gamma * diff
    return Candidate(w, MAX_ITER, "maxiter")


def away_frank_wolfe(B, A) -> Candidate:
    """Frank-Wolfe with away steps: also allowed to *remove* weight from the
    worst active vertex, which is what buys linear convergence on a polytope."""
    m, J = B.shape
    w = np.zeros(J)
    w[int(np.argmin(np.einsum("ij,i->j", B, -A)))] = 1.0
    Bw = B @ w
    for k in range(MAX_ITER):
        grad = 2.0 * (B.T @ (Bw - A))
        s = int(np.argmin(grad))
        active = np.flatnonzero(w > 1e-14)
        v = int(active[np.argmax(grad[active])])
        fw_gap = float(grad @ w - grad[s])
        away_gap = float(grad[v] - grad @ w)
        if max(fw_gap, away_gap) <= FTOL:
            return Candidate(w, k, "optimal")
        if fw_gap >= away_gap:                     # toward vertex s
            diff = B[:, s] - Bw
            gmax = 1.0
            idx, sign = s, +1
        else:                                      # away from vertex v
            diff = Bw - B[:, v]
            gmax = w[v] / max(1.0 - w[v], 1e-300)
            idx, sign = v, -1
        denom = float(diff @ diff)
        if denom <= 0.0:
            return Candidate(w, k, "optimal")
        gamma = float((A - Bw) @ diff) / denom
        gamma = min(max(gamma, 0.0), gmax)
        if gamma <= 0.0:
            return Candidate(w, k, "optimal")
        if sign > 0:
            w = (1.0 - gamma) * w
            w[idx] += gamma
        else:
            w = (1.0 + gamma) * w
            w[idx] -= gamma
            np.clip(w, 0.0, None, out=w)
            w /= w.sum()
        Bw = B @ w
    return Candidate(w, MAX_ITER, "maxiter")


def pairwise_smo(B, A) -> Candidate:
    """Pairwise coordinate descent. ``1'w = 1`` forbids moving one coordinate, so
    the smallest feasible move transfers weight between two -- pick the most
    negative and most positive reduced gradients and solve the 1-D problem
    exactly. This is SVM's SMO applied to this program."""
    m, J = B.shape
    G = B.T @ B
    c = B.T @ A
    w = np.full(J, 1.0 / J)
    Gw = G @ w
    for k in range(MAX_ITER):
        g = Gw - c                                 # half the gradient
        up = np.where(w > 1e-14, g, -np.inf)       # can give weight away
        i = int(np.argmax(up))                     # donor to shrink
        j = int(np.argmin(g))                      # donor to grow
        if g[i] - g[j] <= FTOL:
            return Candidate(w, k, "optimal")
        denom = G[i, i] + G[j, j] - 2.0 * G[i, j]
        if denom <= 0.0:
            return Candidate(w, k, "degenerate")
        t = min(w[i], (g[i] - g[j]) / denom)
        if t <= 0.0:
            return Candidate(w, k, "optimal")
        w[i] -= t
        w[j] += t
        Gw = Gw + t * (G[:, j] - G[:, i])
    return Candidate(w, MAX_ITER, "maxiter")


def fista_projected(B, A) -> Candidate:
    """Accelerated projected gradient with the exact simplex projection."""
    m, J = B.shape
    G = B.T @ B
    c = B.T @ A
    L = 2.0 * float(np.linalg.norm(G, 2)) or 1.0
    w = np.full(J, 1.0 / J)
    y = w.copy()
    t = 1.0
    for k in range(MAX_ITER):
        grad = 2.0 * (G @ y - c)
        w_new = simplex_project(y - grad / L)
        if float(np.abs(w_new - w).max()) <= 1e-13:
            return Candidate(w_new, k, "optimal")
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
    return Candidate(w, MAX_ITER, "maxiter")


# --------------------------------------------------------------------------- #
# Becker & Kloessner (2017), section 3.1
# --------------------------------------------------------------------------- #
def sunny_alphas(B, A):
    """Equation (9) per donor: ``min alpha`` s.t. ``alpha x_j`` is in the hull of
    the difference vectors. ``alpha* < 1`` means the donor is shady."""
    from scipy.optimize import linprog

    m, J = B.shape
    Xt = B - A[:, None]
    alphas = np.ones(J)
    for j in range(J):
        cost = np.zeros(J + 1); cost[0] = 1.0
        Aeq = np.hstack([Xt[:, j : j + 1], -Xt])
        Aeq = np.vstack([Aeq, np.concatenate([[0.0], np.ones(J)])])
        beq = np.concatenate([np.zeros(m), [1.0]])
        res = linprog(cost, A_eq=Aeq, b_eq=beq,
                      bounds=[(0.0, None)] * (J + 1), method="highs")
        if res.success:
            alphas[j] = float(res.x[0])
    return alphas


def becker_kloessner(B, A, tol=1e-7) -> Candidate:
    """The cascade. Proposition 1: no sunny donors iff ``0`` is in the hull iff an
    exact fit exists, and then the answer is the minimum-norm point of that face
    rather than whatever the pivot order lands on. A single sunny donor is the
    answer outright. Otherwise Proposition 2 lets the shady columns be dropped.
    """
    m, J = B.shape
    alphas = sunny_alphas(B, A)
    sunny = alphas >= 1.0 - tol
    n_sunny = int(sunny.sum())
    if n_sunny == 0:
        # 0 in H: an exact fit exists. Among the exact-fit weights, take the
        # least-norm point, which is equation (10) with the outer objective
        # standing in as ||w||^2 -- the outcome-only case has no separate Z.
        from scipy.optimize import lsq_linear
        Aeq = np.vstack([B, np.ones((1, J))])
        beq = np.concatenate([A, [1.0]])
        res = lsq_linear(Aeq, beq, bounds=(0.0, np.inf), tol=1e-12)
        w = np.clip(np.asarray(res.x, float), 0.0, None)
        s = w.sum()
        return Candidate(w / s if s > 0 else np.full(J, 1.0 / J),
                         int(res.nit or 0), "exact-fit-face")
    if n_sunny == 1:
        w = np.zeros(J)
        w[int(np.flatnonzero(sunny)[0])] = 1.0
        return Candidate(w, 0, "single-sunny")
    keep = np.flatnonzero(sunny)
    wr, info = solve_simplex_qp(np.ascontiguousarray(B[:, keep]), A,
                                return_info=True)
    w = np.zeros(J)
    w[keep] = np.asarray(wr, float)
    return Candidate(w, int(info["pivots"]),
                     f"pruned:{J - n_sunny}" if n_sunny < J else "no-pruning")


ALGORITHMS: dict = {
    "active_set": active_set,
    "active_set_cold": active_set_cold,
    "clarabel": clarabel,
    "osqp": osqp,
    "frank_wolfe": frank_wolfe,
    "away_frank_wolfe": away_frank_wolfe,
    "pairwise_smo": pairwise_smo,
    "fista_projected": fista_projected,
    "becker_kloessner": becker_kloessner,
}
