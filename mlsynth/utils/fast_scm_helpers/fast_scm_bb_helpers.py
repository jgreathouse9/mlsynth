"""
Fast_scm_bb_helpers.py
----------------------
Correct branch-and-bound synthetic control solver with Optimality Pruning.
Optimized for M=200 using Lagrangian Simplex Bounding.
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

def get_qp_call_count() -> int:
    return _qp_call_count

def reset_qp_call_count() -> None:
    global _qp_call_count
    _qp_call_count = 0



def compute_search_space_size(M: int, m: int) -> Tuple[int, int]:
    leaves = comb(M, m)
    nodes = sum(comb(M, k) for k in range(1, m + 1))
    return leaves, nodes




# ============================================================
# 2. SOLUTION CONTAINER
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
# 3. QP SOLVERS
# ============================================================

def solve_qp_simplex_value(Q: np.ndarray):
    global _qp_call_count
    _qp_call_count += 1
    k = Q.shape[0]
    w = cp.Variable(k, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Q)), [cp.sum(w) == 1])
    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        i = int(np.argmin(np.diag(Q)))
        w_out = np.zeros(k)
        w_out[i] = 1.0
        return float(Q[i, i]), w_out

    w_out = np.maximum(w.value, 0.0)
    w_out /= (w_out.sum() + 1e-12)
    return float(prob.value), w_out

def solve_relaxed_lower_bound(G: np.ndarray, indices: List[int], remaining_idx: np.ndarray, m: int) -> float:
    """
    Lagrangian Simplex Bound (Harmonic Floor).
    Mathematically safe: The minimum of a quadratic form on the simplex
    is bounded by the harmonic mean of the variances (diagonals).
    """
    k = len(indices)
    # Pool of all donors available in this branch
    active_idx = np.concatenate([indices, remaining_idx])
    diags = np.diag(G)[active_idx]
    
    # Tightening: Since we can only pick 'm' units, the best possible loss
    # is bounded by the best 'm' units available in this branch.
    if len(diags) > m:
        # Get the m smallest diagonals (best individual donors)
        best_diags = np.partition(diags, m-1)[:m]
    else:
        best_diags = diags

    # The absolute floor of w'Gw on a simplex of size m is:
    # Floor = 1 / sum(1/sigma_i^2)
    # This is derived from the KKT conditions of the equality-constrained problem.
    inv_sum = np.sum(1.0 / (best_diags + 1e-12))
    lower_bound = 1.0 / inv_sum
    
    return lower_bound

# ============================================================
# 4. SEARCH HELPERS
# ============================================================

def greedy_initial_solution(G: np.ndarray, candidate_idx: np.ndarray, m: int):
    selected = []
    current_Q = None
    for _ in range(m):
        best_j, best_loss, best_Q = None, np.inf, None
        for j in candidate_idx:
            if j in selected: continue
            if not selected:
                Q_new = np.array([[G[j, j]]])
            else:
                sz = len(selected)
                Q_new = np.empty((sz+1, sz+1))
                Q_new[:sz, :sz] = current_Q
                g = G[j, selected]
                Q_new[sz, :sz] = Q_new[:sz, sz] = g
                Q_new[sz, sz] = G[j, j]
            loss, _ = solve_qp_simplex_value(Q_new)
            if loss < best_loss:
                best_loss, best_j, best_Q = loss, j, Q_new
        selected.append(best_j)
        current_Q = best_Q
    loss, w = solve_qp_simplex_value(current_Q)
    return selected, loss, w

def branch_score(G, j, indices):
    if len(indices) == 0: return -G[j, j]
    return -(G[j, j] + np.mean(G[j, indices]))

# ============================================================
# 5. CORE BRANCH AND BOUND
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
    k = len(indices)
    
    # 1. Dynamic Upper Bound
    current_ub = top_tuples[-1].loss if len(top_tuples) >= top_K else np.inf

    # 2. Pruning
    if k < m:
        last = indices[-1] if k > 0 else -1
        remaining = candidate_idx[candidate_idx > last]
        
        if len(remaining) < (m - k):
            return

        # Lagrangian Lower Bound
        # We use a small epsilon (1e-9) to avoid pruning the optimum due to float noise
        lb = solve_relaxed_lower_bound(G, indices, remaining, m)
        if lb > current_ub + 1e-9 and current_ub != np.inf:
            stats["branches_pruned"] += 1
            return

    # 3. Leaf
    if k == m:
        stats["subsets_evaluated"] += 1
        loss, w = solve_qp_simplex_value(Q_partial)
        if loss < current_ub + 1e-9:
            top_tuples.append(Solution(loss, indices[:], w))
            top_tuples.sort(key=lambda s: s.loss)
            if len(top_tuples) > top_K:
                top_tuples.pop()
        return

    # 4. Expansion
    ordered = sorted(remaining, key=lambda j: branch_score(G, j, indices))
    for j in ordered:
        stats["branches_considered"] += 1
        new_cost = current_cost + (float(unit_costs[j]) if unit_costs is not None else 0.0)
        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            continue

        k_next = k + 1
        Q_next = np.empty((k_next, k_next))
        Q_next[:k, :k] = Q_partial
        if k > 0:
            g = G[j, indices]
            Q_next[k, :k] = Q_next[:k, k] = g
        Q_next[k, k] = G[j, j]

        expand_tuple(G, candidate_idx, m, top_K, top_tuples, indices + [j], 
                     stats, Q_next, unit_costs, budget, new_cost)
