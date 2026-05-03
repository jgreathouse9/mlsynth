"""
Fast_scm_bb_helpers.py
----------------------
Helper primitives for the branch-and-bound synthetic control solver.

Sections
    1. Global solver state
    2. Solution container
    3. Search-space sizing
    4. Pre-solve candidate filtering
    5. Lower-bound functions
    6. QP solver
    7. Initialisation utilities
    8. Scoring / pruning helpers
    9. BnB recursive core
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
# 4.  PRE-SOLVE CANDIDATE FILTERING
# ============================================================

def presolve_candidates(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    budget: Optional[float] = None,
    unit_costs: Optional[np.ndarray] = None,
    m: int = 1,
) -> np.ndarray:
    """
    Remove obviously poor candidates before BnB.

    Steps
        1. Drop units that exceed the budget.
        2. Drop units whose diagonal G[j,j] is an outlier (> 50× median).
        3. Deduplicate near-identical units (correlation > 0.999), keeping
           the cheaper one when costs are supplied.
    """
    idx = np.asarray(candidate_idx, dtype=int)

    # --- budget gate ---
    if budget is not None and unit_costs is not None:
        idx = idx[unit_costs[idx] <= budget]

    if len(idx) == 0:
        return idx

    # --- diagonal outlier removal ---
    diags     = np.diag(G)[idx]
    threshold = np.median(diags) * 50.0
    idx       = idx[diags < threshold]

    if len(idx) == 0:
        return idx

    # --- near-duplicate removal ---
    if unit_costs is not None:
        to_drop: set = set()
        for i in range(len(idx)):
            u = idx[i]
            if u in to_drop:
                continue
            for j in range(i + 1, len(idx)):
                v = idx[j]
                if v in to_drop:
                    continue
                corr = G[u, v] / (np.sqrt(G[u, u] * G[v, v]) + 1e-12)
                if corr > 0.999:
                    drop = v if unit_costs[v] >= unit_costs[u] else u
                    to_drop.add(drop)
        idx = np.array([c for c in idx if c not in to_drop], dtype=int)

    return idx


# ============================================================
# 5.  LOWER BOUNDS
# ============================================================

def simplex_lower_bound(Q: np.ndarray) -> float:
    """
    Lower bound on min_{w in Δ} w'Qw via the minimum eigenvalue of Q.

    For any w in the probability simplex, w'Qw ≥ λ_min(Q) · ||w||² ≥ λ_min(Q)/k
    where k = dim(w).  We clamp at zero because losses are non-negative.
    """
    if Q.shape[0] == 1:
        return float(Q[0, 0])
    lam_min = float(np.linalg.eigvalsh(Q)[0])
    return max(0.0, lam_min / Q.shape[0])


def lookahead_lower_bound(
    G: np.ndarray,
    lb_so_far: float,
    remaining: np.ndarray,
    slots_left: int,
    m: int,
) -> float:
    """
    Cheap predictive lower bound that blends the current partial bound with
    the average diagonal of the best `slots_left` remaining candidates.

    Ignores cross-terms, so it is optimistic but fast.
    """
    if slots_left <= 0 or len(remaining) == 0:
        return lb_so_far

    best_diags = np.sort(np.diag(G)[remaining])[: max(1, slots_left)]
    avg_best   = float(np.mean(best_diags))

    k_filled = m - slots_left
    return max(lb_so_far, (k_filled / m) * lb_so_far + (slots_left / m) * avg_best)


# ============================================================
# 6.  QP SOLVER
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
# 7.  INITIALISATION UTILITIES
# ============================================================

def greedy_initial_solution(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int,
) -> Tuple[float, List[int], np.ndarray]:
    """
    Build a greedy incumbent by sequentially adding the candidate that
    most reduces the QP loss, starting from the best single unit.

    This gives a tighter initial upper bound than arbitrary first-m selection,
    which improves BnB pruning at the root.
    """
    # best single unit
    diags   = np.diag(G)[candidate_idx]
    best_i  = int(candidate_idx[int(np.argmin(diags))])
    selected = [best_i]

    for _ in range(m - 1):
        best_loss = np.inf
        best_j    = -1
        best_w    = None

        for j in candidate_idx:
            if j in selected:
                continue
            trial = selected + [j]
            Q_trial = G[np.ix_(trial, trial)]
            loss, w = solve_qp_simplex_value(Q_trial, indices=trial)
            if loss < best_loss:
                best_loss, best_j, best_w = loss, j, w

        if best_j == -1:
            break
        selected.append(best_j)

    Q_final = G[np.ix_(selected, selected)]
    loss, w = solve_qp_simplex_value(Q_final, indices=selected)
    return loss, selected, w


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
# 8.  SCORING / PRUNING HELPERS
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
# 9.  BnB RECURSIVE CORE
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
    """
    Recursive BnB node expansion.

    Invariants
        • *indices* is strictly increasing (enforced by `j > last` gate).
        • *Q_partial* is the Gram sub-matrix for the units in *indices*.
        • Children are only spawned for j > last, guaranteeing each subset
          is visited exactly once.

    Pruning
        • Budget cut: skip j if adding its cost exceeds the budget.
        • Eigenvalue lower bound: skip j if lb(Q_new) ≥ current upper bound.
        • Lookahead bound: tighter prediction using best remaining diagonals.
    """
    stats["nodes_visited"] += 1

    k          = len(indices)
    slots_left = m - k
    current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

    # ------------------------------------------------------------------
    # BASE CASE: tuple is complete
    # ------------------------------------------------------------------
    if k == m:
        stats["subsets_evaluated"] += 1
        loss, w = solve_qp_simplex_value(Q_partial, indices=indices)

        top_tuples.append(Solution(loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)
        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    # ------------------------------------------------------------------
    # ENUMERATE CHILDREN (strictly larger indices only)
    # ------------------------------------------------------------------
    last = indices[-1] if indices else -1

    # Use searchsorted to skip candidates ≤ last in O(log M) instead of O(M)
    start_pos = int(np.searchsorted(candidate_idx, last + 1))

    for pos in range(start_pos, len(candidate_idx)):
        j = candidate_idx[pos]

        stats["branches_considered"] += 1

        # --- budget gate ---
        new_cost = current_cost + (float(unit_costs[j]) if unit_costs is not None else 0.0)
        if budget is not None and new_cost > budget:
            stats["branches_pruned"] += 1
            continue

        # --- build Q_new by appending row/column for j ---
        k_new   = k + 1
        Q_new   = np.empty((k_new, k_new))
        Q_new[:k, :k] = Q_partial
        if k > 0:
            g = G[j, indices]
            Q_new[k, :k] = g
            Q_new[:k, k] = g
        Q_new[k, k] = G[j, j]

        # --- lower bounds ---
        lb = simplex_lower_bound(Q_new)

        if slots_left > 1:
            remaining = candidate_idx[pos + 1:]          # already sorted, O(1) slice
            lb = max(lb, lookahead_lower_bound(G, lb, remaining, slots_left - 1, m))

        if lb >= current_ub:
            stats["branches_pruned"] += 1
            continue

        # --- recurse ---
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
