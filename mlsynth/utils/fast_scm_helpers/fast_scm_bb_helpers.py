"""
Fast_scm_bb_helpers.py
----------------------
Helper primitives for the branch-and-bound synthetic control solver.

Sections
    1. Global solver state
    2. Solution container
    3. Search-space sizing
    4. Lower-bound functions
    5. QP solver
    6. Utility
    7. Scoring / pruning helpers
    8. BnB recursive core
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1.  GLOBAL SOLVER STATE
# ============================================================

_qp_call_count: int = 0
_warm_start_cache: Dict[Tuple[int, ...], np.ndarray] = {}


def get_qp_call_count() -> int:
    """Return the number of QP solves since the last reset."""
    return _qp_call_count


def reset_qp_call_count() -> None:
    """Zero the QP call counter."""
    global _qp_call_count
    _qp_call_count = 0


def clear_solver_cache() -> None:
    """Zero the QP counter and flush the warm-start cache."""
    global _qp_call_count, _warm_start_cache
    _qp_call_count = 0
    _warm_start_cache.clear()


# ============================================================
# 2.  SOLUTION CONTAINER
# ============================================================

@dataclass(order=True)
class Solution:
    """A single candidate tuple with its QP loss and weights."""

    loss: float
    indices: List[int]                      = field(compare=False)
    weights: np.ndarray                     = field(compare=False)
    labels: Optional[List[Any]]             = field(default=None, compare=False)
    full_weights: Optional[np.ndarray]      = field(default=None, compare=False)
    weight_dict: Optional[Dict[Any, float]] = field(default=None, compare=False)
    cost: float                             = 0.0
    label: Optional[str]                    = field(default=None, compare=False)


# ============================================================
# 3.  SEARCH-SPACE SIZING
# ============================================================

def compute_search_space_size(M: int, m: int) -> Tuple[int, int]:
    """Return (number of m-subsets, total BnB nodes up to depth m)."""
    leaves = comb(M, m)
    nodes  = sum(comb(M, k) for k in range(1, m + 1))
    return leaves, nodes


# ============================================================
# 4.  LOWER BOUNDS
# ============================================================

def compute_global_lower_bound(G: np.ndarray, m: int) -> float:
    """
    Universal lower bound λ_min(G) / m, valid at any partial depth.

    Proof: Cauchy interlacing guarantees λ_min(Q_S) ≥ λ_min(G) for any
    principal submatrix Q_S of G.  For any w ∈ Δ_m:
        w'Q_S w ≥ λ_min(Q_S) / m ≥ λ_min(G) / m.

    Compute once before the search; pass the scalar into expand_tuple.
    Empirically validated against all partial depths in test_partial_bounds.py.
    """
    lam_min = float(np.linalg.eigvalsh(G)[0])
    return max(0.0, lam_min / m)


def simplex_lower_bound(Q: np.ndarray) -> float:
    """
    Tighter leaf-only bound: λ_min(Q_full) / m for the complete m-tuple.

    Only valid when Q is the full m×m submatrix (k_new == m).  At partial
    depth, off-diagonal cancellation invalidates diagonal-based or partial
    eigenvalue bounds — use global_lb there instead.
    """
    k = Q.shape[0]
    lam_min = float(np.linalg.eigvalsh(Q)[0])
    return max(0.0, lam_min / k)


# ============================================================
# 5.  QP SOLVER
# ============================================================

def solve_qp_simplex_value(
    Q: np.ndarray,
    w_init: Optional[np.ndarray] = None,
    indices: Optional[List[int]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Solve  min_{w ≥ 0, 1'w = 1}  w'Qw  via OSQP.

    Warm-starts from cache when w_init is absent; stores result when
    indices is provided.  Falls back to best single-unit on solver failure.
    """
    global _qp_call_count
    _qp_call_count += 1

    k = Q.shape[0]

    if w_init is None and indices is not None:
        cached = _warm_start_cache.get(tuple(indices))
        if cached is not None and len(cached) == k:
            w_init = cached

    w    = cp.Variable(k, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Q)), [cp.sum(w) == 1])

    if w_init is not None:
        w_val   = np.maximum(w_init, 0.0)
        s       = w_val.sum()
        w.value = w_val / (s + 1e-12) if s > 0 else np.ones(k) / k

    prob.solve(solver=cp.OSQP, verbose=False, warm_start=(w_init is not None))

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        best  = int(np.argmin(np.diag(Q)))
        w_out = np.zeros(k)
        w_out[best] = 1.0
        return float(Q[best, best]), w_out

    w_out  = np.maximum(w.value, 0.0)
    w_out /= w_out.sum() + 1e-12

    if indices is not None:
        _warm_start_cache[tuple(indices)] = w_out.copy()

    return float(prob.value), w_out


# ============================================================
# 6.  UTILITY
# ============================================================

def expand_weights_to_full(
    indices: List[int],
    weights: np.ndarray,
    total_units: int,
) -> np.ndarray:
    """Embed a sparse weight vector into a zero-padded full-length array."""
    w = np.zeros(total_units)
    w[indices] = weights
    return w


def greedy_initial_solution(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int,
) -> Tuple[List[int], float, np.ndarray]:
    """
    Build a feasible m-tuple greedily by minimizing QP loss at each step.
    Returns (indices, loss, weights).
    """
    selected: List[int]             = []
    Q_partial: Optional[np.ndarray] = None

    for _ in range(m):
        best_j    = None
        best_loss = np.inf
        best_Q    = None

        for j in candidate_idx:
            if j in selected:
                continue
            if not selected:
                Q_new = np.array([[G[j, j]]])
            else:
                k     = len(selected)
                Q_new = np.empty((k + 1, k + 1))
                Q_new[:k, :k] = Q_partial
                g = G[j, selected]
                Q_new[k, :k] = g
                Q_new[:k, k] = g
                Q_new[k, k]  = G[j, j]

            loss, _ = solve_qp_simplex_value(Q_new)
            if loss < best_loss:
                best_loss, best_j, best_Q = loss, j, Q_new

        selected.append(best_j)
        Q_partial = best_Q

    loss, w = solve_qp_simplex_value(Q_partial)
    return selected, loss, w


# ============================================================
# 7.  SCORING / PRUNING HELPERS
# ============================================================

def strong_branch_score(
    G: np.ndarray,
    Q_partial: np.ndarray,
    candidate_idx: np.ndarray,
    j: int,
    indices: List[int],
) -> float:
    """
    Cheap heuristic branching score (lower = more promising).

    Uses diagonal self-interaction and mean cross-term with current selection.
    """
    if len(indices) == 0:
        return -G[j, j]
    return -G[j, j] - 2.0 * float(np.mean(G[j, indices]))


# ============================================================
# 8.  BnB RECURSIVE CORE
# ============================================================

def expand_tuple(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int,
    top_K: int,
    top_tuples: List[Solution],
    indices: List[int],
    stats: Dict[str, Any],
    Q_partial: np.ndarray,
    global_lb: float,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    current_cost: float = 0.0,
) -> None:
    """
    Recursive BnB node expansion.

    Two-layer pruning:
        1. Any depth  — global_lb = λ_min(G)/m.  Validated in
           test_partial_bounds.py; prunes entire subtrees for free.
        2. Leaf only  — simplex_lower_bound(Q_new) = λ_min(Q_full)/m.
           Tighter than the global bound; avoids the QP call when it fires.
    """
    stats["nodes_visited"]  += 1
    stats["nodes_generated"] = stats.get("nodes_generated", 0) + 1

    assert np.all(candidate_idx[:-1] <= candidate_idx[1:]),\
        "candidate_idx must be sorted ascending"

    k          = len(indices)
    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # ------------------------------------------------------------------
    # LAYER 1: partial-depth global bound
    # Prune this node immediately if no completion can beat the incumbent.
    # ------------------------------------------------------------------
    if global_lb >= current_ub:
        stats["branches_pruned"] += 1
        stats["nodes_pruned"]     = stats.get("nodes_pruned", 0) + 1
        key = f"depth_{k}_global"
        stats["pruned_by_depth"][key] = stats["pruned_by_depth"].get(key, 0) + 1
        return

    # ------------------------------------------------------------------
    # BASE CASE
    # ------------------------------------------------------------------
    if k == m:
        stats["subsets_evaluated"] += 1
        stats["leaf_nodes"]         = stats.get("leaf_nodes", 0) + 1

        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)
        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    # ------------------------------------------------------------------
    # CHILD ORDERING
    # Position-based start_pos ensures each subset is visited exactly once.
    # Heuristic ordering puts promising children first to tighten current_ub
    # early and increase pruning effectiveness downstream.
    # ------------------------------------------------------------------
    if indices:
        last_pos  = int(np.searchsorted(candidate_idx, indices[-1]))
        start_pos = last_pos + 1
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

    # ------------------------------------------------------------------
    # EXPAND CHILDREN
    # ------------------------------------------------------------------
    for j in ordered:
        stats["branches_considered"] += 1

        # budget gate
        new_cost = current_cost + (float(unit_costs[j]) if unit_costs is not None else 0.0)
        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            stats["nodes_pruned"]     = stats.get("nodes_pruned", 0) + 1
            key = f"depth_{k}_budget"
            stats["pruned_by_depth"][key] = stats["pruned_by_depth"].get(key, 0) + 1
            continue

        # incremental Gram sub-matrix
        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial
        if k > 0:
            g = G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g
        Q_new[k, k] = G[j, j]

        # LAYER 2: tighter leaf-only bound (saves the QP call when it fires)
        if k_new == m:
            lb_leaf = simplex_lower_bound(Q_new)
            if lb_leaf >= current_ub:
                stats["branches_pruned"] += 1
                stats["nodes_pruned"]     = stats.get("nodes_pruned", 0) + 1
                key = f"depth_{k_new}_leaf_eig"
                stats["pruned_by_depth"][key] = stats["pruned_by_depth"].get(key, 0) + 1
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
            global_lb=global_lb,
            unit_costs=unit_costs,
            budget=budget,
            current_cost=new_cost,
        )
