"""
Fast_scm_bb_helpers.py - GUROBI-style QP-reduced BnB helpers
Now includes:
- active-set simplex QP solver
- incremental warm-start cache
- strong branching heuristic
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from math import comb


# ============================================================
# GLOBAL STATE
# ============================================================

_qp_call_count = 0
_warm_start_cache: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}


def get_qp_call_count() -> int:
    return _qp_call_count


def reset_qp_call_count():
    global _qp_call_count
    _qp_call_count = 0


def clear_solver_cache():
    global _qp_call_count, _warm_start_cache
    _qp_call_count = 0
    _warm_start_cache.clear()


# ============================================================
# SOLUTION CONTAINER
# ============================================================

@dataclass(order=True)
class Solution:
    loss: float
    indices: List[int] = field(compare=False)
    weights: np.ndarray = field(compare=False)
    labels: Optional[List[Any]] = field(default=None, compare=False)
    full_weights: Optional[np.ndarray] = field(default=None, compare=False)
    weight_dict: Optional[Dict[Any, float]] = field(default=None, compare=False)
    cost: float = 0.0
    label: Optional[str] = field(default=None, compare=False)


# ============================================================
# SEARCH SPACE SIZE
# ============================================================

def compute_search_space_size(M: int, m: int):
    total_subsets = comb(M, m)
    total_nodes = sum(comb(M, k) for k in range(1, m + 1))
    return total_subsets, total_nodes


# ============================================================
# FAST ACTIVE-SET SIMPLEX QP SOLVER (CORE)
# ============================================================

def solve_qp_simplex_value(
    Q: np.ndarray,
    w_init: Optional[np.ndarray] = None,
    indices: Optional[List[int]] = None,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> Tuple[float, np.ndarray]:

    global _qp_call_count, _warm_start_cache
    _qp_call_count += 1

    k = Q.shape[0]

    # -----------------------------
    # WARM START INIT
    # -----------------------------
    if w_init is not None:
        w = np.maximum(w_init, 0.0)
    else:
        w = np.ones(k) / k

    # reuse parent solution if available
    if indices is not None:
        key = tuple(indices)
        if key in _warm_start_cache:
            prev = _warm_start_cache[key]
            if len(prev) == k:
                w = prev.copy()
            elif len(prev) == k - 1:
                w = np.append(prev, 0.0)

    w /= (np.sum(w) + 1e-12)
    active = w > 1e-12

    # -----------------------------
    # ACTIVE SET ITERATIONS
    # -----------------------------
    for _ in range(max_iter):

        if not np.any(active):
            active = np.ones(k, dtype=bool)
            w = np.ones(k) / k

        A = np.where(active)[0]

        Q_A = Q[np.ix_(A, A)]
        ones = np.ones(len(A))

        KKT = np.block([
            [2 * Q_A, ones[:, None]],
            [ones[None, :], np.zeros((1, 1))]
        ])

        rhs = np.zeros(len(A) + 1)
        rhs[-1] = 1.0

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

        w_A = sol[:-1]

        w_new = np.zeros(k)
        w_new[A] = w_A

        w_new = np.maximum(w_new, 0.0)
        w_new /= (np.sum(w_new) + 1e-12)

        if np.linalg.norm(w - w_new) < tol:
            w = w_new
            break

        w = w_new
        active = w > 1e-12

    if indices is not None:
        _warm_start_cache[tuple(indices)] = w.copy()

    loss = float(w @ Q @ w)
    return loss, w


def solve_qp_simplex(Q: np.ndarray, w_init: Optional[np.ndarray] = None) -> np.ndarray:
    return solve_qp_simplex_value(Q, w_init)[1]


# ============================================================
# LOWER BOUND
# ============================================================

def simplex_lower_bound(Q: np.ndarray) -> float:

    k = Q.shape[0]

    if k == 1:
        return float(Q[0, 0])

    diag_min = float(np.min(np.diag(Q)))

    w_uniform = np.ones(k) / k
    uniform_loss = float(w_uniform @ Q @ w_uniform)

    row_sums = np.sum(np.abs(Q), axis=1) - np.abs(np.diag(Q))
    gersh_bound = float(np.min(np.diag(Q) - row_sums))

    cheap_lb = max(
        diag_min * 0.75,
        gersh_bound,
        uniform_loss * 0.65
    )

    if cheap_lb < 0.9:
        try:
            val, _ = solve_qp_simplex_value(Q)
            return max(val, cheap_lb)
        except Exception:
            pass

    return cheap_lb


# ============================================================
# UPPER BOUND (DETERMINISTIC)
# ============================================================

def feasible_upper_bound(Q: np.ndarray) -> float:

    k = Q.shape[0]

    if k == 1:
        return float(Q[0, 0])

    w_uniform = np.ones(k) / k

    idx_min = np.argmin(np.diag(Q))
    w_diag = np.zeros(k)
    w_diag[idx_min] = 1.0

    # 2-support heuristic (no randomness)
    i, j = np.unravel_index(np.argmin(Q), Q.shape)
    w_pair = np.zeros(k)
    w_pair[i] = 0.5
    w_pair[j] = 0.5

    return min([
        float(w_uniform @ Q @ w_uniform),
        float(w_diag @ Q @ w_diag),
        float(w_pair @ Q @ w_pair),
    ])


# ============================================================
# GREEDY INITIAL SOLUTION
# ============================================================

def greedy_initial_solution(G, candidate_idx, m):
    selected = list(candidate_idx[:m])
    Q = G[np.ix_(selected, selected)]
    loss, w = solve_qp_simplex_value(Q)
    return loss, selected, w


# ============================================================
# EXPAND WEIGHTS
# ============================================================

def expand_weights_to_full(indices, weights, total_units):
    w_full = np.zeros(total_units)
    w_full[indices] = weights
    return w_full


# ============================================================
# STRONG BRANCHING SCORE (LIGHTWEIGHT LOOKAHEAD)
# ============================================================

def strong_branch_score(G, Q_partial, candidate_idx, j, indices):

    k = Q_partial.shape[0]

    if k == 0:
        return -G[j, j]

    g = G[j, indices]

    return -G[j, j] - 2.0 * np.sum(g) / (k + 1e-12)


# ============================================================
# BnB CORE EXPANSION
# ============================================================

def expand_tuple(
    G,
    candidate_idx,
    m,
    top_K,
    top_tuples,
    indices,
    stats,
    start_pos,
    Q_partial,
    unit_costs=None,
    budget=None,
    current_cost=0.0,
):

    stats["nodes_visited"] += 1

    if budget is not None and unit_costs is not None:
        if current_cost > budget:
            stats["branches_pruned"] += 1
            return

        remaining = m - len(indices)
        if remaining > 0:
            costs = unit_costs[candidate_idx[start_pos:]]
            if len(costs) >= remaining:
                if current_cost + np.sort(costs)[:remaining].sum() > budget:
                    stats["branches_pruned"] += 1
                    return

    if len(indices) == m:
        stats["subsets_evaluated"] += 1
        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)

        if len(top_tuples) > top_K:
            top_tuples.pop()

        return

    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # -----------------------------
    # STRONG BRANCHING ORDERING
    # -----------------------------
    candidates = candidate_idx[start_pos:]

    scored = [
        (j, strong_branch_score(G, Q_partial, candidate_idx, j, indices))
        for j in candidates
    ]

    scored.sort(key=lambda x: x[1])

    # -----------------------------
    # EXPAND
    # -----------------------------
    for j, _ in scored:

        stats["branches_considered"] += 1

        k = Q_partial.shape[0]

        Q_new = np.empty((k + 1, k + 1))
        Q_new[:k, :k] = Q_partial

        g = G[j, indices]

        Q_new[k, :k] = g
        Q_new[:k, k] = g
        Q_new[k, k] = G[j, j]

        if k == 0 and G[j, j] >= current_ub:
            stats["branches_pruned"] += 1
            continue

        ub_local = feasible_upper_bound(Q_new)
        effective_ub = min(current_ub, ub_local)

        lb = simplex_lower_bound(Q_new)

        tol = 1e-8 * max(1.0, abs(effective_ub))

        if lb >= effective_ub - tol:
            stats["branches_pruned"] += 1
            continue

        expand_tuple(
            G,
            candidate_idx,
            m,
            top_K,
            top_tuples,
            indices + [j],
            stats,
            start_pos + 1,
            Q_new,
            unit_costs,
            budget,
            current_cost + (unit_costs[j] if unit_costs is not None else 0.0),
        )
