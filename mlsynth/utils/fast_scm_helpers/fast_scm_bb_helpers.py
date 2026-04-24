import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from math import comb

# ============================================================
# SOLUTION CONTAINER (PRESERVED)
# ============================================================

@dataclass(order=True)
class Solution:
    """Container for a candidate solution in the branch-and-bound search."""
    loss: float
    indices: List[int] = field(compare=False)
    weights: np.ndarray = field(compare=False)
    labels: Optional[List[Any]] = field(default=None, compare=False)
    full_weights: Optional[np.ndarray] = field(default=None, compare=False)
    weight_dict: Optional[Dict[Any, float]] = field(default=None, compare=False)
    cost: float = 0.0
    label: Optional[str] = field(default=None, compare=False)

# ============================================================
# UTILITIES & PROJECTION
# ============================================================

def expand_weights_to_full(indices: List[int], weights: np.ndarray, total_units: int) -> np.ndarray:
    """Expand a subset weight vector into a full-length vector."""
    w_full = np.zeros(total_units)
    w_full[indices] = weights
    return w_full

def compute_search_space_size(M: int, m: int) -> Tuple[int, int]:
    """Compute number of size-m subsets and total potential nodes."""
    total_subsets = comb(M, m)
    total_nodes = sum(comb(M, k) for k in range(1, m + 1))
    return total_subsets, total_nodes

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex (O(n log n))."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)

def solve_qp_simplex(Q: np.ndarray, max_iter: int = 200, lr: float = 0.01) -> np.ndarray:
    """Solve min w^T Q w s.t. w in simplex using PGD."""
    m = Q.shape[0]
    w = np.ones(m) / m
    for _ in range(max_iter):
        grad = 2 * Q @ w
        w -= lr * grad
        w = project_to_simplex(w)
    return w

# ============================================================
# ADMISSIBLE BOUNDS (GERSHGORIN)
# ============================================================

def get_gershgorin_bound(Q: np.ndarray) -> float:
    """
    Computes a guaranteed lower bound on the smallest eigenvalue 
    multiplied by ||w||^2 min (1/k) to bound the quadratic form.
    """
    k = Q.shape[0]
    if k == 0: return 0.0
    diags = np.diag(Q)
    radii = np.sum(np.abs(Q), axis=1) - np.abs(diags)
    eig_lb = np.min(diags - radii)
    # Valid lower bound for quadratic form on simplex is lambda_min / k
    return float(max(0.0, eig_lb / k))

# ============================================================
# INITIALIZATION HELPERS
# ============================================================

def compute_seed_tuples(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    top_P: int
) -> List[Tuple[float, List[int], np.ndarray]]:
    """Generate initial 1-unit seeds sorted by diagonal loss."""
    seeds = [(float(G[i, i]), [i], np.array([1.0])) for i in candidate_idx]
    seeds.sort(key=lambda x: x[0])
    return seeds[:top_P]

def greedy_initial_solution(G: np.ndarray, unit_costs: np.ndarray, candidate_idx: np.ndarray, m: int, lam: float):
    """Construct an initial feasible Solution object to prime the search."""
    selected = list(candidate_idx[:m])
    Q = G[np.ix_(selected, selected)]
    w = solve_qp_simplex(Q)
    loss = float(w @ Q @ w)
    total_cost = float(np.sum(unit_costs[selected]))
    score = loss + (lam * total_cost)
    return Solution(loss=score, indices=selected, weights=w, cost=total_cost)

# ============================================================
# CORE RECURSION
# ============================================================

def expand_tuple(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    unit_costs: np.ndarray,
    m: int,
    lam: float,
    top_K: int,
    top_tuples: List[Solution],
    indices: List[int],
    stats: Dict,
    start_pos: int,
    Q_partial: np.ndarray,
    current_cost: float = 0.0
):
    """
    Admissible Branch and Bound using Lagrangian relaxation and 
    Gershgorin eigenvalue bounds.
    """
    stats["nodes_visited"] += 1
    k = len(indices)

    # --- Leaf Node ---
    if k == m:
        stats["subsets_evaluated"] += 1
        w = solve_qp_simplex(Q_partial)
        loss = float(w @ Q_partial @ w)
        score = loss + (lam * current_cost)
        
        top_tuples.append(Solution(loss=score, indices=indices[:], weights=w, cost=current_cost))
        top_tuples.sort()
        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    # --- Pruning Calculations ---
    remaining_needed = m - k
    available_indices = candidate_idx[start_pos:]
    if len(available_indices) < remaining_needed:
        return

    # Lookahead: Minimum possible future cost
    # np.partition is O(N) compared to O(N log N) for full sort
    future_costs = np.partition(unit_costs[available_indices], remaining_needed)[:remaining_needed]
    min_future_cost = np.sum(future_costs)

    # Analytical Lower Bound (Quadratic + Linear Cost)
    lb_quad = get_gershgorin_bound(Q_partial)
    lb_total = lb_quad + lam * (current_cost + min_future_cost)

    if len(top_tuples) >= top_K and lb_total >= top_tuples[-1].loss:
        stats["branches_pruned"] += 1
        return

    # --- Branching ---
    for j_idx in range(start_pos, len(candidate_idx)):
        j = candidate_idx[j_idx]
        stats["branches_considered"] += 1

        # Incremental Q update
        g = G[j, indices]
        Q_next = np.empty((k + 1, k + 1))
        Q_next[:k, :k] = Q_partial
        Q_next[k, :k] = g
        Q_next[:k, k] = g
        Q_next[k, k] = G[j, j]

        expand_tuple(
            G, candidate_idx, unit_costs, m, lam, top_K, top_tuples,
            indices + [j], stats, j_idx + 1, Q_next, 
            current_cost + unit_costs[j]
        )
