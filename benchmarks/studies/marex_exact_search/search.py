"""An exact MAREX design without branch-and-bound.

MAREX's standard design chooses a treated set ``S`` of ``m`` markets, treated
weights ``w`` on the simplex over ``S`` and control weights ``v`` on the simplex
over the rest, to minimise

    ||A - B w||^2 + ||A - B v||^2,

with ``A`` the cluster's mean over the fit window and ``B`` its markets. Once
``S`` is named the two terms share nothing, so the objective is

    f(S) = g(S) + h(N \\ S),

each a simplex-constrained least-squares fit -- the program
``mlsynth.utils.solvers.active_set.solve_simplex_qp`` solves exactly. The
design is therefore a search over treated sets with a cheap oracle, not a
mixed-integer program.

``exact_design`` makes that search fast with a bound. Both terms are
non-negative, so ``f(S) >= g(S) >= lb(S)`` for any lower bound ``lb`` on the
treated fit. Price every candidate by ``lb`` in one batched pass, sort, and
walk up: solve ``g`` exactly only while the bound is below the best total so
far, and ``h`` only while ``g`` is. The moment the bound reaches the incumbent
every remaining candidate is dominated, and the search stops with the optimum.

The bound. Because the weights sum to one, subtracting the target from every
column changes nothing: ``||A - B w|| = ||(B - A 1') w||``. On those centred
columns the target is zero, and dropping only ``w >= 0`` leaves

    lb(S) = min { w' G w : 1'w = 1 } = 1 / (1' G_SS^-1 1),   G = B~' B~,

a closed form that equals ``g(S)`` whenever its minimiser is non-negative.
This is the construction SYNDES's exact backend uses for its own objective
(``mlsynth/utils/syndes_helpers/gram.py``), applied to MAREX's.

``prototype_bound`` is the first version, kept for comparison. It dropped both
constraints and computed ``||A||^2 - c' G^-1 c`` on the raw columns, with a
``1e-12`` ridge to survive a singular block. Two things are wrong with it. The
ridge shrinks ``c' G^-1 c`` and so raises the bound, which is the direction
that can make it invalid. And on these panels ``||A||^2`` is near 7,200 while
the residuals are near 1 to 7, so the subtraction cancels most of the
precision. ``check_bounds.py`` measures whether either ever mattered.

Floating point. A bound is used to discard a candidate only when it exceeds
the incumbent by ``PRUNE_MARGIN`` relative, and a block whose solve does not
reproduce ``G x = 1`` to ``1e-8`` gets the bound zero, so it is never
discarded; both guards can only make the search look at more candidates.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from mlsynth.utils.solvers.active_set import solve_simplex_qp

PRUNE_MARGIN = 1e-6
CHUNK = 20_000


def fit_value(B: np.ndarray, A: np.ndarray, cols) -> float:
    """``min ||A - B[:, cols] w||^2`` over the simplex, exactly."""
    cols = np.asarray(cols)
    w = solve_simplex_qp(B[:, cols], A, accelerate=False)
    r = A - B[:, cols] @ w
    return float(r @ r)


def all_subsets(J: int, m: int) -> np.ndarray:
    flat = np.fromiter(itertools.chain.from_iterable(
        itertools.combinations(range(J), m)), dtype=np.int64)
    return flat.reshape(-1, m)


# ---------------------------------------------------------------- the bounds
def centred_bound(B: np.ndarray, A: np.ndarray, subs: np.ndarray) -> np.ndarray:
    """``1 / (1' G_SS^-1 1)`` on the centred columns, for every row of ``subs``."""
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    m = subs.shape[1]
    ones = np.ones(m)
    out = np.empty(len(subs))
    for lo in range(0, len(subs), CHUNK):
        idx = subs[lo:lo + CHUNK]
        Gs = G[idx[:, :, None], idx[:, None, :]]
        try:
            x = np.linalg.solve(Gs, np.broadcast_to(ones, (len(idx), m))[..., None])[..., 0]
        except np.linalg.LinAlgError:          # pragma: no cover - singular block
            x = np.full((len(idx), m), np.nan)
            for i, g in enumerate(Gs):
                try:
                    x[i] = np.linalg.solve(g, ones)
                except np.linalg.LinAlgError:
                    pass
        s = x.sum(axis=1)
        resid = np.abs(np.einsum("bij,bj->bi", Gs, x) - 1.0).max(axis=1)
        good = np.isfinite(s) & (s > 0) & (resid <= 1e-8)
        lb = np.zeros(len(idx))
        lb[good] = 1.0 / s[good]
        out[lo:lo + CHUNK] = lb
    return out


def prototype_bound(B: np.ndarray, A: np.ndarray, subs: np.ndarray) -> np.ndarray:
    """The first version: ``||A||^2 - c_S' (G_SS + 1e-12 I)^-1 c_S``, uncentred."""
    G, c, nrmA = B.T @ B, B.T @ A, float(A @ A)
    out = np.empty(len(subs))
    for lo in range(0, len(subs), CHUNK):
        idx = subs[lo:lo + CHUNK]
        Gs = G[idx[:, :, None], idx[:, None, :]]
        cs = c[idx]
        Gs[:, np.arange(idx.shape[1]), np.arange(idx.shape[1])] += 1e-12
        sol = np.linalg.solve(Gs, cs[:, :, None])[:, :, 0]
        out[lo:lo + CHUNK] = nrmA - np.einsum("ij,ij->i", cs, sol)
    return out


BOUNDS = {"centred": centred_bound, "prototype": prototype_bound}


# -------------------------------------------------------------- the searches
def greedy_incumbent(B: np.ndarray, A: np.ndarray, m: int):
    """Forward selection on the treated side; a feasible design to prune on."""
    J = B.shape[1]
    chosen: list[int] = []
    for _ in range(m):
        scores = [(fit_value(B, A, chosen + [j]), j)
                  for j in range(J) if j not in chosen]
        chosen.append(min(scores)[1])
    S = np.asarray(sorted(chosen))
    total = fit_value(B, A, S) + fit_value(B, A, np.setdiff1d(np.arange(J), S))
    return total, S


def exact_design(B: np.ndarray, A: np.ndarray, m: int,
                 bound: str = "centred") -> dict:
    """The optimal treated set, by bounded search. Returns value and counts."""
    J = B.shape[1]
    allj = np.arange(J)
    t = {}
    t0 = time.perf_counter()
    best, best_S = greedy_incumbent(B, A, m)
    t["greedy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    subs = all_subsets(J, m)
    lb = BOUNDS[bound](B, A, subs)
    order = np.argsort(lb, kind="stable")
    t["bound"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_g = n_h = 0
    for i in order:
        if lb[i] >= best + PRUNE_MARGIN * (1.0 + abs(best)):
            break
        S = subs[i]
        g = fit_value(B, A, S)
        n_g += 1
        if g >= best:
            continue
        total = g + fit_value(B, A, np.setdiff1d(allj, S))
        n_h += 1
        if total < best:
            best, best_S = total, S
    t["sweep"] = time.perf_counter() - t0
    return dict(objective=best, treated=tuple(int(j) for j in best_S),
                candidates=len(subs), exact_g=n_g, exact_h=n_h, stage_secs=t)


def enumerate_design(B: np.ndarray, A: np.ndarray, m: int) -> dict:
    """Every treated set, both fits each: the reference the search must match."""
    J = B.shape[1]
    allj = np.arange(J)
    best, best_S = np.inf, None
    for S in all_subsets(J, m):
        total = fit_value(B, A, S) + fit_value(B, A, np.setdiff1d(allj, S))
        if total < best:
            best, best_S = total, S
    return dict(objective=best, treated=tuple(int(j) for j in best_S))
