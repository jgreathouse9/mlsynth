"""
Fast_scm_bb_helpers.py
----------------------
Helper primitives for the branch-and-bound synthetic control solver.

Key fix:
    ✔ introduce partial lower bound applied at EVERY node (not just leaves)
    ✔ enable early subtree pruning
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. GREEDY INITIAL SOLUTION
# ============================================================

def greedy_initial_solution(G: np.ndarray, candidate_idx: np.ndarray, m: int):
    selected = []
    Q_partial = None

    for _ in range(m):
        best_j = None
        best_loss = np.inf
        best_Q = None

        for j in candidate_idx:
            if j in selected:
                continue

            if not selected:
                Q_new = np.array([[G[j, j]]])
            else:
                k = len(selected)
                Q_new = np.empty((k + 1, k + 1))
                Q_new[:k, :k] = Q_partial
                g = G[j, selected]
                Q_new[k, :k] = g
                Q_new[:k, k] = g
                Q_new[k, k] = G[j, j]

            loss, _ = solve_qp_simplex_value(Q_new)

            if loss < best_loss:
                best_loss = loss
                best_j = j
                best_Q = Q_new

        selected.append(best_j)
        Q_partial = best_Q

    loss, w = solve_qp_simplex_value(Q_partial)
    return selected, loss, w


# ============================================================
# 2. GLOBAL STATE
# ============================================================

_qp_call_count: int = 0
_warm_start_cache: Dict[Tuple[int, ...], np.ndarray] = {}


def get_qp_call_count() -> int:
    return _qp_call_count


def reset_qp_call_count() -> None:
    global _qp_call_count
    _qp_call_count = 0


def clear_solver_cache() -> None:
    global _qp_call_count, _warm_start_cache
    _qp_call_count = 0
    _warm_start_cache.clear()


# ============================================================
# 3. SOLUTION CONTAINER
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
# 4. SEARCH SPACE SIZE
# ============================================================

def compute_search_space_size(M: int, m: int) -> Tuple[int, int]:
    leaves = comb(M, m)
    nodes = sum(comb(M, k) for k in range(1, m + 1))
    return leaves, nodes


# ============================================================
# 5. PARTIAL LOWER BOUND (KEY FIX)
# ============================================================

def partial_lower_bound(
    G: np.ndarray,
    indices: List[int],
    remaining: np.ndarray,
    m: int,
) -> float:
    """
    Cheap but effective optimistic lower bound.

    Intuition:
        - current loss ≈ average diagonal energy of selected set
        - future loss ≈ best possible remaining diagonal entries
        - combine proportionally
    """

    k = len(indices)
    if k == 0:
        return 0.0

    current = np.mean([G[i, i] for i in indices])

    slots_left = m - k
    if slots_left <= 0:
        return current

    if len(remaining) > 0:
        best_remaining = np.sort(np.diag(G)[remaining])[:slots_left]
        future = np.mean(best_remaining)
    else:
        future = 0.0

    return (k * current + slots_left * future) / m


# ============================================================
# 6. QP SOLVER
# ============================================================

def solve_qp_simplex_value(
    Q: np.ndarray,
    w_init: Optional[np.ndarray] = None,
    indices: Optional[List[int]] = None,
) -> Tuple[float, np.ndarray]:

    global _qp_call_count
    _qp_call_count += 1

    k = Q.shape[0]

    if w_init is None and indices is not None:
        cached = _warm_start_cache.get(tuple(indices))
        if cached is not None and len(cached) == k:
            w_init = cached

    w = cp.Variable(k, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Q)), [cp.sum(w) == 1])

    if w_init is not None:
        w_val = np.maximum(w_init, 0.0)
        s = w_val.sum()
        w.value = w_val / (s + 1e-12) if s > 0 else np.ones(k) / k

    prob.solve(solver=cp.OSQP, verbose=False, warm_start=(w_init is not None))

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        best = int(np.argmin(np.diag(Q)))
        w_out = np.zeros(k)
        w_out[best] = 1.0
        return float(Q[best, best]), w_out

    w_out = np.maximum(w.value, 0.0)
    w_out /= w_out.sum() + 1e-12

    if indices is not None:
        _warm_start_cache[tuple(indices)] = w_out.copy()

    return float(prob.value), w_out


# ============================================================
# 7. UTILITY
# ============================================================

def expand_weights_to_full(indices: List[int], weights: np.ndarray, total_units: int):
    w = np.zeros(total_units)
    w[indices] = weights
    return w


# ============================================================
# 8. SCORING
# ============================================================

def strong_branch_score(G, Q_partial, candidate_idx, j, indices):
    if len(indices) == 0:
        return -G[j, j]

    cross = np.mean(G[j, indices])
    return -(G[j, j] + 0.5 * cross)


# ============================================================
# 9. BnB CORE (FIXED PRUNING)
# ============================================================

def expand_tuple(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int,
    top_K: int,
    top_tuples: List[Solution],
    indices: List[int],
    stats: Dict[str, int],
    Q_partial: np.ndarray,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    current_cost: float = 0.0,
):

    stats["nodes_visited"] += 1
    stats["nodes_generated"] = stats.get("nodes_generated", 0) + 1

    k = len(indices)
    slots_left = m - k
    current_ub = top_tuples[-1].loss if len(top_tuples) >= top_K else np.inf

    # =========================================================
    # LEAF NODE
    # =========================================================
    if k == m:
        stats["subsets_evaluated"] += 1
        stats["leaf_nodes"] = stats.get("leaf_nodes", 0) + 1

        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)

        if len(top_tuples) > top_K:
            top_tuples.pop()

        return

    # =========================================================
    # CHILD SELECTION
    # =========================================================
    if indices:
        start_pos = int(np.searchsorted(candidate_idx, indices[-1])) + 1
    else:
        start_pos = 0

    remaining = candidate_idx[start_pos:]

    if len(indices) == 0:
        ordered = sorted(remaining, key=lambda j: -G[j, j])
    else:
        ordered = sorted(
            remaining,
            key=lambda j: strong_branch_score(G, Q_partial, candidate_idx, j, indices),
        )

    # =========================================================
    # EXPAND
    # =========================================================
    for j in ordered:

        stats["branches_considered"] += 1

        new_cost = current_cost + (
            float(unit_costs[j]) if unit_costs is not None else 0.0
        )

        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            stats["nodes_pruned"] += 1
            continue

        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial

        if k > 0:
            g = G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g

        Q_new[k, k] = G[j, j]

        # =====================================================
        # 🔥 KEY FIX: PARTIAL BOUND AT EVERY NODE
        # =====================================================
        lb = partial_lower_bound(
            G,
            indices + [j],
            remaining,
            m,
        )

        if lb >= current_ub:
            stats["branches_pruned"] += 1
            stats["nodes_pruned"] += 1
            continue

        expand_tuple(
            G=G,
            candidate_idx=candidate_idx,
            m=m,
            top_K=top_K,
            top_tuples=top_tuples,
            indices=indices + [j],
            stats=stats,
            Q_partial=Q_new,
            unit_costs=unit_costs,
            budget=budget,
            current_cost=new_cost,
        )
