"""
Fast_scm_bb_helpers.py
----------------------
Correct branch-and-bound synthetic control solver.

Fixes:
    ✔ full combinatorial correctness restored
    ✔ no accidental subset exclusion
    ✔ safe pruning only via explicit budget
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
# 3. GREEDY INITIAL SOLUTION
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
# 4. QP SOLVER
# ============================================================

def solve_qp_simplex_value(Q: np.ndarray,
                          w_init=None,
                          indices=None):

    global _qp_call_count
    _qp_call_count += 1

    k = Q.shape[0]

    w = cp.Variable(k, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Q)),
                      [cp.sum(w) == 1])

    if w_init is not None:
        w0 = np.maximum(w_init, 0.0)
        s = w0.sum()
        w.value = w0 / (s + 1e-12) if s > 0 else np.ones(k) / k

    prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)

    if w.value is None:
        i = int(np.argmin(np.diag(Q)))
        w_out = np.zeros(k)
        w_out[i] = 1.0
        return float(Q[i, i]), w_out

    w_out = np.maximum(w.value, 0.0)
    w_out /= w_out.sum() + 1e-12

    return float(prob.value), w_out


# ============================================================
# 5. UTILITIES
# ============================================================

def expand_weights_to_full(indices, weights, total_units):
    w = np.zeros(total_units)
    w[indices] = weights
    return w


def branch_score(G, j, indices):
    if len(indices) == 0:
        return -G[j, j]
    return -(G[j, j] + np.mean(G[j, indices]))


# ============================================================
# 6. CORE BnB (FIXED)
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

    current_ub = top_tuples[-1].loss if len(top_tuples) >= top_K else np.inf

    # =========================================================
    # LEAF
    # =========================================================
    if k == m:
        stats["subsets_evaluated"] += 1

        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        if loss < current_ub:
            top_tuples.append(Solution(loss, indices[:], w))
            top_tuples.sort(key=lambda s: s.loss)

            if len(top_tuples) > top_K:
                top_tuples.pop()

        return

    # =========================================================
    # CRITICAL FIX: correct remaining set (NO searchsorted slicing)
    # =========================================================
    if len(indices) == 0:
        remaining = candidate_idx
    else:
        last = indices[-1]
        remaining = candidate_idx[candidate_idx > last]

    ordered = sorted(remaining, key=lambda j: branch_score(G, j, indices))

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
            continue

        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial

        if k > 0:
            Q_new[k, :k] = G[j, indices]
            Q_new[:k, k] = G[j, indices]

        Q_new[k, k] = G[j, j]

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
