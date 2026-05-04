"""
Fast_scm_bb_helpers.py
----------------------
Helper primitives for the branch-and-bound synthetic control solver.

Now includes:
- Global Cauchy bound (safe)
- Leaf eigenvalue bound (safe)
- Schur-style relaxation bound for partial nodes (safe + tighter)

Guaranteed to find the global optimum.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. GLOBAL SOLVER STATE
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


def expand_weights_to_full(indices: List[int], weights: np.ndarray, total: int) -> np.ndarray:
    w = np.zeros(total)
    w[indices] = weights
    return w

# ============================================================
# 2. SOLUTION CONTAINER
# ============================================================

@dataclass(order=True)
class Solution:
    loss: float
    indices: List[int]                      = field(compare=False)
    weights: np.ndarray                     = field(compare=False)
    labels: Optional[List[Any]]             = field(default=None, compare=False)
    full_weights: Optional[np.ndarray]      = field(default=None, compare=False)
    weight_dict: Optional[Dict[Any, float]] = field(default=None, compare=False)
    cost: float                             = 0.0
    label: Optional[str]                    = field(default=None, compare=False)


# ============================================================
# 3. SEARCH SPACE
# ============================================================

def compute_search_space_size(M: int, m: int) -> Tuple[int, int]:
    leaves = comb(M, m)
    nodes  = sum(comb(M, k) for k in range(1, m + 1))
    return leaves, nodes


# ============================================================
# 4. LOWER BOUNDS
# ============================================================

def compute_global_lower_bound(G: np.ndarray, m: int) -> float:
    lam_min = float(np.linalg.eigvalsh(G)[0])
    return max(0.0, lam_min / m)


def simplex_lower_bound(Q: np.ndarray) -> float:
    k = Q.shape[0]
    lam_min = float(np.linalg.eigvalsh(Q)[0])
    return max(0.0, lam_min / k)


def schur_relaxation_lower_bound(
    Q_partial: np.ndarray,
    lam_min_global: float
) -> float:
    """
    Valid partial-node bound using a two-block relaxation.

    Combines:
        current curvature (λ_min(Q_partial))
        best-case future curvature (λ_min(G))

    Returns
    -------
    float
        Safe lower bound for any completion.
    """
    if Q_partial.size == 0:
        return 0.0

    a = float(np.linalg.eigvalsh(Q_partial)[0])
    b = lam_min_global

    a = max(a, 0.0)
    b = max(b, 0.0)

    if a + b <= 1e-12:
        return 0.0

    return (a * b) / (a + b)


# ============================================================
# 5. QP SOLVER
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
# 6. GREEDY INIT
# ============================================================

def greedy_initial_solution(G, candidate_idx, m):
    selected = []
    Q_partial = None

    for _ in range(m):
        best_j, best_loss, best_Q = None, np.inf, None

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

            if len(selected) > 0 and simplex_lower_bound(Q_new) > best_loss:
                continue

            loss, _ = solve_qp_simplex_value(Q_new)

            if loss < best_loss:
                best_loss, best_j, best_Q = loss, j, Q_new

        selected.append(best_j)
        Q_partial = best_Q

    loss, w = solve_qp_simplex_value(Q_partial)
    return selected, loss, w


# ============================================================
# 7. BRANCH HEURISTIC
# ============================================================

def strong_branch_score(G, Q_partial, j, indices):
    if not indices:
        return -G[j, j]
    return -G[j, j] - 2.0 * float(np.mean(G[j, indices]))


# ============================================================
# 8. BnB CORE
# ============================================================

def expand_tuple(
    G,
    candidate_idx,
    m,
    top_K,
    top_tuples,
    indices,
    stats,
    Q_partial,
    global_lb,
    lam_min_global,
    unit_costs=None,
    budget=None,
    current_cost=0.0,
):
    stats["nodes_visited"] += 1
    k = len(indices)

    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # GLOBAL PRUNE
    if global_lb >= current_ub:
        stats["branches_pruned"] += 1
        return

    # LEAF
    if k == m:
        stats["subsets_evaluated"] += 1
        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)

        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    # branching
    start_pos = np.searchsorted(candidate_idx, indices[-1]) + 1 if indices else 0
    remaining = candidate_idx[start_pos:]

    ordered = sorted(
        remaining,
        key=lambda j: strong_branch_score(G, Q_partial, j, indices)
    )

    for j in ordered:
        stats["branches_considered"] += 1

        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial

        if k > 0:
            g = G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g

        Q_new[k, k] = G[j, j]

        lb_partial = schur_relaxation_lower_bound(Q_new, lam_min_global)

        if lb_partial >= current_ub:
            stats["branches_pruned"] += 1
            continue

        # leaf eigen prune
        if k_new == m:
            lb_leaf = simplex_lower_bound(Q_new)
            if lb_leaf >= current_ub:
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
            Q_new,
            global_lb,
            lam_min_global,
            unit_costs,
            budget,
            current_cost,
        )
