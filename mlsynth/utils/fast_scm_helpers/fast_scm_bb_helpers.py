"""
Fast_scm_bb_helpers.py
----------------------
Helper primitives for branch-and-bound synthetic control.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. GLOBAL STATE
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
# 2. SOLUTION OBJECT
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
# 3. SEARCH SPACE
# ============================================================

def compute_search_space_size(M: int, m: int) -> Tuple[int, int]:
    leaves = comb(M, m)
    nodes = sum(comb(M, k) for k in range(1, m + 1))
    return leaves, nodes


# ============================================================
# 4. GREEDY INITIALIZER (STRONG UB)
# ============================================================

def greedy_initial_solution(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int,
) -> Tuple[List[int], float, np.ndarray]:

    selected: List[int] = []
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
# 5. LOWER BOUND (PARTIAL RELAXATION)
# ============================================================

def partial_lower_bound(Q: np.ndarray) -> float:
    diag = np.diag(Q)
    interaction = np.sum(np.abs(Q), axis=1) - diag
    adjusted = diag - interaction
    return float(max(0.0, np.min(adjusted)))


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
# 7. UTILITY: EXPAND WEIGHTS (RESTORED)
# ============================================================

def expand_weights_to_full(
    indices: List[int],
    weights: np.ndarray,
    total_units: int,
) -> np.ndarray:
    """Embed sparse solution into full-dimensional weight vector."""
    w = np.zeros(total_units)
    w[indices] = weights
    return w


# ============================================================
# 8. SCORING
# ============================================================

def strong_branch_score(
    G: np.ndarray,
    Q_partial: np.ndarray,
    candidate_idx: np.ndarray,
    j: int,
    indices: List[int],
) -> float:
    if len(indices) == 0:
        return -G[j, j]
    return -G[j, j] - 2.0 * float(np.mean(G[j, indices]))


# ============================================================
# 9. BnB CORE
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
) -> None:

    stats["nodes_visited"] += 1
    stats["nodes_generated"] = stats.get("nodes_generated", 0) + 1

    assert np.all(candidate_idx[:-1] <= candidate_idx[1:])

    k = len(indices)
    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # -----------------------------
    # CHILD SELECTION
    # -----------------------------
    start_pos = (
        int(np.searchsorted(candidate_idx, indices[-1])) + 1
        if indices else 0
    )

    remaining = candidate_idx[start_pos:]

    if len(indices) == 0:
        ordered = sorted(remaining, key=lambda j: -G[j, j])
    else:
        ordered = sorted(
            remaining,
            key=lambda j: strong_branch_score(G, Q_partial, candidate_idx, j, indices)
        )

    # -----------------------------
    # EXPAND
    # -----------------------------
    for j in ordered:

        stats["branches_considered"] += 1

        new_cost = current_cost + (
            float(unit_costs[j]) if unit_costs is not None else 0.0
        )

        if budget is not None and new_cost > budget:
            stats["branches_pruned"] = stats.get("branches_pruned", 0) + 1
            stats["nodes_pruned"] = stats.get("nodes_pruned", 0) + 1
            continue

        # build Q
        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial

        if k > 0:
            g = G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g

        Q_new[k, k] = G[j, j]

        # -----------------------------
        # PARTIAL LOWER BOUND PRUNING
        # -----------------------------
        lb = partial_lower_bound(Q_new)

        if lb >= current_ub:
            stats["branches_pruned"] = stats.get("branches_pruned", 0) + 1
            stats["nodes_pruned"] = stats.get("nodes_pruned", 0) + 1
            continue

        # recurse
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
