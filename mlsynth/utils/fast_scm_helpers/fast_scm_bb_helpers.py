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


def greedy_initial_solution(G: np.ndarray, candidate_idx: np.ndarray, m: int):
    """
    Build a feasible m-tuple greedily by minimizing QP loss at each step.
    Returns (indices, loss, weights).
    """
    selected = []
    Q_partial = None

    for _ in range(m):
        best_j = None
        best_loss = np.inf
        best_Q = None

        for j in candidate_idx:
            if j in selected:
                continue

            # build candidate Q
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
    indices: List[int]           = field(compare=False)
    weights: np.ndarray          = field(compare=False)
    labels: Optional[List[Any]]  = field(default=None, compare=False)
    full_weights: Optional[np.ndarray] = field(default=None, compare=False)
    weight_dict: Optional[Dict[Any, float]] = field(default=None, compare=False)
    cost: float                  = 0.0
    label: Optional[str]         = field(default=None, compare=False)


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

def simplex_lower_bound(Q: np.ndarray) -> float:
    """
    Lower bound on min_{w in Δ} w'Qw via the minimum eigenvalue of Q.

    For w in the probability simplex, w'Qw ≥ λ_min(Q)/k where k = dim(Q).
    Only meaningful when Q is the *complete* m-tuple sub-matrix — off-diagonal
    cancellation means partial-tuple diagonals are not valid lower bounds for
    the completed loss.  Callers must only apply this when k_new == m.
    """
    k = Q.shape[0]
    lam_min = float(np.linalg.eigvalsh(Q)[0])
    return max(0.0, lam_min / k)


def lookahead_lower_bound(
    G: np.ndarray,
    lb_so_far: float,
    remaining: np.ndarray,
    slots_left: int,
    m: int,
) -> float:
    """
    Always returns 0.0.

    A diagonal-based lookahead is not a valid lower bound for w'Qw over the
    m-tuple simplex because off-diagonal cancellation can drive the completed
    loss arbitrarily below any individual G[j,j].  This stub is kept so call
    sites compile; it will be replaced once a provably valid partial bound is
    derived.
    """
    return 0.0


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

    Looks up a warm-start from the cache when *w_init* is not given.
    Stores the solution back into the cache when *indices* is provided.
    Falls back to the best single-unit solution on solver failure.
    """
    global _qp_call_count
    _qp_call_count += 1

    k = Q.shape[0]

    # --- warm-start lookup ---
    if w_init is None and indices is not None:
        cached = _warm_start_cache.get(tuple(indices))
        if cached is not None and len(cached) == k:
            w_init = cached

    w = cp.Variable(k, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Q)), [cp.sum(w) == 1])

    if w_init is not None:
        w_val = np.maximum(w_init, 0.0)
        s     = w_val.sum()
        w.value = w_val / (s + 1e-12) if s > 0 else np.ones(k) / k

    prob.solve(solver=cp.OSQP, verbose=False, warm_start=(w_init is not None))

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        # fallback: put all weight on the unit with smallest diagonal
        best = int(np.argmin(np.diag(Q)))
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
    Cheap heuristic score for branching priority (lower = more promising).

    Approximates the reduction in QP loss from adding unit *j* by considering
    its self-interaction and its mean cross-term with current selections.
    """
    if len(indices) == 0:
        return -G[j, j]
    return -G[j, j] - 2.0 * float(np.mean(G[j, indices]))


def prune_by_correlation(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    threshold: float = 0.999,
) -> np.ndarray:
    """Remove near-duplicate candidates (pairwise correlation > threshold)."""
    keep:    List[int] = []
    removed: set       = set()

    for i in candidate_idx:
        if i in removed:
            continue
        keep.append(i)
        for j in candidate_idx:
            if i == j:
                continue
            corr = G[i, j] / (np.sqrt(G[i, i] * G[j, j]) + 1e-12)
            if corr > threshold:
                removed.add(j)

    return np.array(keep, dtype=int)


def prune_by_cost(
    candidate_idx: np.ndarray,
    unit_costs: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Keep only candidates whose unit cost fits within the budget."""
    return np.array([i for i in candidate_idx if unit_costs[i] <= budget], dtype=int)


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
    stats: Dict[str, int],
    Q_partial: np.ndarray,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    current_cost: float = 0.0,
) -> None:

    stats["nodes_visited"] += 1
    stats["nodes_generated"] = stats.get("nodes_generated", 0) + 1

    assert np.all(candidate_idx[:-1] <= candidate_idx[1:]), \
        "candidate_idx must be sorted ascending before entering expand_tuple"

    k = len(indices)
    slots_left = m - k
    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # ---------------------------------------------------------------
    # BASE CASE (leaf node)
    # ---------------------------------------------------------------
    if k == m:
        stats["subsets_evaluated"] += 1
        stats["leaf_nodes"] = stats.get("leaf_nodes", 0) + 1

        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)

        if len(top_tuples) > top_K:
            top_tuples.pop()

        return

    # ---------------------------------------------------------------
    # CHILD SELECTION (ordering constraint)
    # ---------------------------------------------------------------
    if indices:
        last_pos = int(np.searchsorted(candidate_idx, indices[-1]))
        start_pos = last_pos + 1
    else:
        start_pos = 0

    remaining = candidate_idx[start_pos:]

    # ---------------------------------------------------------------
    # HEURISTIC ORDERING
    # ---------------------------------------------------------------
    if len(indices) == 0:
        ordered = sorted(remaining, key=lambda j: -G[j, j])
    else:
        ordered = sorted(
            remaining,
            key=lambda j: strong_branch_score(G, Q_partial, candidate_idx, j, indices)
        )

    # ---------------------------------------------------------------
    # EXPAND CHILDREN
    # ---------------------------------------------------------------
    for j in ordered:

        stats["branches_considered"] += 1

        new_cost = current_cost + (
            float(unit_costs[j]) if unit_costs is not None else 0.0
        )

        # ---- budget pruning ----
        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            stats["nodes_pruned"] = stats.get("nodes_pruned", 0) + 1
            continue

        # ---- build Q_new ----
        k_new = k + 1
        Q_new = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial

        if k > 0:
            g = G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g

        Q_new[k, k] = G[j, j]

        # ---- bound pruning (only at full depth) ----
        if k_new == m:
            lb = simplex_lower_bound(Q_new)
            if lb >= current_ub:
                stats["branches_pruned"] += 1
                stats["nodes_pruned"] = stats.get("nodes_pruned", 0) + 1
                continue

        # ---- recurse ----
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
