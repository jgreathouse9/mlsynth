"""One-way SYNDES by enumeration under closed-form bounds (#807).

For a treated set S of size K, with F = S^c and M = G + lam I on contrasts,

    f(S) = min { u'Mu : u_S = 1/K, u_F <= 0, 1'u = 0 }
         = lam/K + min over c on the simplex over F of
           (1/T)||ybar_S - Y_F c||^2 + lam ||c||^2.

Two lower bounds, each a relaxation:

- two-way: drop u_S = 1/K, keep sigma'u = 2 and 1'u = 0 (gram.py's lb(S));
- ridge: keep u_S = 1/K and 1'u = 0, drop u_F <= 0. With C = [E_S, 1] and
  d = [1/K; 0], the value is d'(C'HC)^{-1}d, H = M^{-1}: a (K+1) x (K+1) solve.

ridge >= two-way, since its constraints imply the two-way ones.
"""
from __future__ import annotations

import math
from itertools import combinations
from time import perf_counter
from typing import Optional

import numpy as np

from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem, signs_from_subsets


class OneWay:
    """One instance: the Gram pieces for the bounds, and exact pricing."""

    def __init__(self, Y: np.ndarray, lam: float):
        Y = np.asarray(Y, dtype=float)
        self.T, self.N = Y.shape
        self.lam = float(lam)
        centred = Y - Y.mean(axis=1, keepdims=True)
        scale = float(np.sqrt(np.mean(centred ** 2))) or 1.0
        self.scale = scale
        self.Yc = centred / scale                       # exact pricing works here
        self.problem = TwoWayProblem.from_panel(Y, lam=self.lam)
        self.engine = BoundEngine.from_problem(self.problem)
        self.H = self.engine.H
        self.p = self.engine.p
        self.alpha = self.engine.alpha
        self.n_fits = 0
        self.max_rel_gap = 0.0

    def two_way_bound(self, sets: np.ndarray) -> np.ndarray:
        return self.engine.bound(signs_from_subsets(self.N, sets)).lower

    def ridge_bound(self, sets: np.ndarray) -> np.ndarray:
        sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
        B, K = sets.shape
        A = np.empty((B, K + 1, K + 1))
        A[:, :K, :K] = self.H[sets[:, :, None], sets[:, None, :]]
        A[:, :K, K] = self.p[sets]
        A[:, K, :K] = self.p[sets]
        A[:, K, K] = self.alpha
        d = np.zeros((B, K + 1))
        d[:, :K] = 1.0 / K
        x = np.linalg.solve(A, d[..., None])[..., 0]
        return np.einsum("bi,bi->b", d, x)

    def price(self, S) -> float:
        """f(S), exactly, in the outcome's units."""
        S = list(S)
        K = len(S)
        F = [j for j in range(self.N) if j not in S]
        lam_s = self.lam / self.scale ** 2
        target = self.Yc[:, S].mean(axis=1)
        root_t = math.sqrt(self.T)
        B = self.Yc[:, F] / root_t
        A = target / root_t
        if lam_s > 0:
            B = np.vstack([B, math.sqrt(lam_s) * np.eye(len(F))])
            A = np.concatenate([A, np.zeros(len(F))])
        c, info = solve_simplex_qp(B, A, return_info=True)
        if not info["converged"]:
            raise RuntimeError("fit did not converge")
        r = A - B @ c
        value = float(r @ r)
        grad = -2.0 * (B.T @ r)
        if value > 0:
            self.max_rel_gap = max(self.max_rel_gap, abs(float(grad @ c - grad.min())) / value)
        self.n_fits += 1
        return (value + lam_s / K) * self.scale ** 2


def search(Y: np.ndarray, K: int, lam: float, *, chunk: int = 200_000,
           limit: int = 20_000_000) -> dict:
    """Enumerate, bound every set, price in order of the ridge bound."""
    t0 = perf_counter()
    inst = OneWay(Y, lam)
    N = inst.N
    count = math.comb(N, K)
    if count > limit:
        return {"skipped": True, "count": count}
    sets_all, tw_all, rg_all = [], [], []
    it = combinations(range(N), K)
    while True:
        block = np.array([s for _, s in zip(range(chunk), it)], dtype=np.int64)
        if block.size == 0:
            break
        sets_all.append(block)
        tw_all.append(inst.two_way_bound(block))
        rg_all.append(inst.ridge_bound(block))
    sets = np.concatenate(sets_all)
    tw = np.concatenate(tw_all)
    rg = np.concatenate(rg_all)
    t_bounds = perf_counter() - t0
    order = np.argsort(rg, kind="stable")
    best, best_set = math.inf, None
    priced = 0
    for k in order:
        if rg[k] >= best:
            break
        value = inst.price(sets[k])
        priced += 1
        if value < best:
            best, best_set = value, tuple(int(i) for i in sets[k])
    return {
        "skipped": False, "count": count, "design": best_set, "value": best,
        "gap_two_way": (best - tw.min()) / best, "gap_ridge": (best - rg.min()) / best,
        "pruned_two_way": float(np.mean(tw >= best)), "pruned_ridge": float(np.mean(rg >= best)),
        "ridge_ge_two_way": bool(np.all(rg >= tw * (1 - 1e-9))),
        "priced": priced, "max_rel_gap": inst.max_rel_gap,
        "t_bounds": t_bounds, "t_total": perf_counter() - t0,
    }


def brute_force(Y: np.ndarray, K: int, lam: float):
    inst = OneWay(Y, lam)
    best, best_set = math.inf, None
    for S in combinations(range(inst.N), K):
        v = inst.price(S)
        if v < best:
            best, best_set = v, S
    return best_set, best


def scip(Y: np.ndarray, K: int, lam: float, *, time_limit: Optional[float] = 60.0,
         gap_limit: Optional[float] = 0.05,
         wall_limit: Optional[float] = None) -> dict:
    """:func:`scip_here` in a fresh process, killed at ``wall_limit``.

    The per-unit study (``syndes_per_unit_exact``) found SCIP corrupting the heap,
    running past its time limit and returning ``solver_error`` there; in a child
    process with a wall clock those are outcomes: ``crashed``, ``hung`` and
    ``error``.
    """
    import multiprocessing as mp
    import queue

    if wall_limit is None:
        wall_limit = 600.0 + 2.0 * (time_limit if time_limit is not None else 300.0)
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    child = ctx.Process(target=_scip_child,
                        args=(results, Y, K, lam, time_limit, gap_limit))
    t0 = perf_counter()
    child.start()
    out = None
    while out is None and perf_counter() - t0 < wall_limit:
        try:
            out = results.get(timeout=1.0)
        except queue.Empty:
            if not child.is_alive():
                break
    if out is None:
        status = "hung" if child.is_alive() else f"crashed (exit {child.exitcode})"
        out = {"status": status, "wall": perf_counter() - t0, "scip_time": math.nan,
               "design": None, "value": math.nan, "scip_gap": math.nan}
    child.join(timeout=5)
    if child.is_alive():
        child.kill()
        child.join()
    return out


def _scip_child(results, Y, K, lam, time_limit, gap_limit) -> None:
    results.put(scip_here(Y, K, lam, time_limit=time_limit, gap_limit=gap_limit))


def scip_here(Y: np.ndarray, K: int, lam: float, *, time_limit: Optional[float] = 60.0,
              gap_limit: Optional[float] = 0.05) -> dict:
    """The library's one-way MIP on SCIP, timed from building to the answer."""
    import cvxpy as cp

    from mlsynth.utils.miqp_accel import solve_warm_cut
    from mlsynth.utils.syndes_helpers.formulation import build_syndes_problem_components

    N = Y.shape[1]
    t0 = perf_counter()
    D = cp.Variable(N, boolean=True)
    comp = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam,
                                           mode="global_equal_weights")
    prob = cp.Problem(cp.Minimize(comp.objective), list(comp.constraints))
    try:
        _, info = solve_warm_cut(prob, D, gap_limit=gap_limit, time_limit=time_limit)
    except Exception as exc:
        return {"status": f"error: {str(exc)[:60]}", "wall": perf_counter() - t0,
                "scip_time": math.nan, "design": None, "value": math.nan,
                "scip_gap": math.nan}
    wall = perf_counter() - t0
    if D.value is None:
        return {"status": info.status, "wall": wall, "scip_time": info.solve_time,
                "design": None, "value": math.nan, "scip_gap": info.gap}
    S = tuple(int(i) for i in np.flatnonzero(np.asarray(D.value) > 0.5))
    return {"status": info.status, "wall": wall, "scip_time": info.solve_time,
            "design": S, "value": OneWay(Y, lam).price(S), "scip_gap": info.gap}


def fallback(Y: np.ndarray, K: int, lam: float, *, n_restarts: int = 32) -> dict:
    """Above the enumeration limit: the two-way swap search's candidates, priced
    as one-way designs, against the Rayleigh bound.

    The two-way backend's ``maxcut_candidates`` maximises ``sigma' R sigma``,
    which minimises the two-way bound; each distinct local optimum is priced
    exactly. ``one-way >= two-way >= Rayleigh``, so the Rayleigh bound over every
    size-``K`` set is a valid, if loose, certificate for the best candidate.
    """
    from mlsynth.utils.syndes_helpers.gram import maxcut_candidates

    t0 = perf_counter()
    inst = OneWay(Y, lam)
    cands = maxcut_candidates(inst.engine, K, n_restarts=n_restarts)
    values = [inst.price(S) for S in cands]
    j = int(np.argmin(values))
    rayleigh = inst.engine.global_lower_bound(K)
    return {"design": tuple(int(i) for i in cands[j]), "value": float(values[j]),
            "n_candidates": len(cands), "rayleigh": rayleigh,
            "rayleigh_gap": (values[j] - rayleigh) / values[j],
            "t_total": perf_counter() - t0}
