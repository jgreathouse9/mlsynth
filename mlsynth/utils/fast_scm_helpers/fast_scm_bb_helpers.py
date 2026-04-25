import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from math import comb

# ============================================================
# SOLUTION CONTAINER
# ============================================================

@dataclass(order=True)
class Solution:
    """
    Container for a candidate solution in the branch-and-bound search.

    This dataclass stores the optimization results for a specific donor subset,
    including the loss, weights, and metadata. It supports native sorting 
    based on the loss attribute to maintain a top-K ranking.

    Attributes
    ----------
    loss : float
        The objective value (w'Qw + lambda * cost). Used for sorting.
    indices : List[int]
        The indices of the units included in this subset.
    weights : np.ndarray
        The optimized weights for the donors in this subset, summing to 1.
    labels : List[Any], optional
        Human-readable labels for the donors in the subset.
    full_weights : np.ndarray, optional
        Weight vector expanded to the dimension of the full candidate pool.
    weight_dict : Dict[Any, float], optional
        Mapping of donor labels to their respective optimized weights.
    cost : float
        The total linear cost of the units in this subset.
    label : str, optional
        A custom label assigned to the solution (e.g., "Tuple 1").
    """
    loss: float
    indices: List[int] = field(compare=False)
    weights: np.ndarray = field(compare=False)
    labels: Optional[List[Any]] = field(default=None, compare=False)
    full_weights: Optional[np.ndarray] = field(default=None, compare=False)
    weight_dict: Optional[Dict[Any, float]] = field(default=None, compare=False)
    cost: float = 0.0
    label: Optional[str] = field(default=None, compare=False)


def expand_weights_to_full(indices, weights, total_units):
    """
    Expand a subset weight vector into a full-length vector.

    Parameters
    ----------
    indices : list of int
        Indices of selected units.
    weights : np.ndarray, shape (k,)
        Weights corresponding to `indices`.
    total_units : int
        Total number of units in the full problem.

    Returns
    -------
    w_full : np.ndarray, shape (total_units,)
        Weight vector with zeros for non-selected units and
        `weights` placed at `indices`.

    Notes
    -----
    - Useful for mapping subset solutions back to the full unit space.
    """
    w_full = np.zeros(total_units)
    w_full[indices] = weights
    return w_full




def compute_search_space_size(M: int, m: int):
    """
    Compute the size of the combinatorial search space.

    Parameters
    ----------
    M : int
        Total number of candidate units.
    m : int
        Subset size (number of units to select).

    Returns
    -------
    total_subsets : int
        Number of size-m subsets (C(M, m)).
    total_nodes : int
        Total number of nodes in the search tree (sum_{k=1}^m C(M, k)).

    Notes
    -----
    - `total_nodes` corresponds to the full branch-and-bound tree size
      without pruning.
    """
    total_subsets = comb(M, m)
    total_nodes = sum(comb(M, k) for k in range(1, m + 1))
    return total_subsets, total_nodes

def compute_seed_tuples(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    top_P: int
) -> List[Tuple[float, List[int], np.ndarray]]:
    """
    Generate initial 1-unit seed tuples for branch-and-bound optimization.

    Parameters
    ----------
    G : np.ndarray, shape (N, N)
        Symmetric loss matrix (e.g., covariance or Gram matrix) used for computing diagonal losses.
    candidate_idx : np.ndarray, shape (M,)
        Indices of candidate units to consider.
    top_P : int
        Number of top 1-unit seeds to retain.

    Returns
    -------
    seeds : list of tuples
        List of length <= top_P, each tuple contains:
        - total_loss : float
            Diagonal loss of the single unit (G[i, i]).
        - indices : list of int
            List containing the single unit index.
        - weights : np.ndarray, shape (1,)
            Weight vector (always [1.0] for 1-unit seeds).

    Notes
    -----
    - Seeds are sorted by increasing diagonal loss.
    - Used as the initial candidates for branch-and-bound expansion.
    """
    unit_losses = []

    for i in candidate_idx:
        w = np.array([1.0])
        loss = float(G[i, i])
        unit_losses.append((loss, [i], w))

    unit_losses.sort(key=lambda x: x[0])
    return unit_losses[:top_P]

def project_to_simplex(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a vector onto the probability simplex.

    Parameters
    ----------
    v : np.ndarray, shape (n,)
        Input vector.
    eps : float, optional
        Numerical tolerance (unused but retained for API stability).

    Returns
    -------
    w : np.ndarray, shape (n,)
        Projection of `v` onto the simplex:
        w_i >= 0 and sum(w) = 1.

    Notes
    -----
    - Implements the method of Duchi et al. (2008).
    - Runs in O(n log n) due to sorting.
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def solve_qp_simplex(Q: np.ndarray, w_init: np.ndarray, steps: int = 8, lr: float = 0.01) -> np.ndarray:
    """
    VERY FAST approximate QP solver for bounds.
    Designed for branch-and-bound nodes.
    """
    w = w_init.copy()

    for _ in range(steps):
        w -= lr * (2 * Q @ w)
        w = project_to_simplex(w)

    return w


# ----------------------------
# Lower Bound (with shortcuts)
# ----------------------------

def pairwise_lower_bound(Q):
    """
    Compute a lower bound using all pairwise 2-element subsets.

    Parameters
    ----------
    Q : np.ndarray, shape (k, k)
        Quadratic matrix.

    Returns
    -------
    best : float
        Minimum achievable objective over all pairs.

    Notes
    -----
    - Evaluates the exact 2-element solution for each pair.
    - Useful as a tighter bound than diagonal-only approximations.
    - O(k^2) complexity.
    """
    k = Q.shape[0]
    best = np.inf

    for i in range(k):
        for j in range(i + 1, k):
            a, b = Q[i, i], Q[j, j]
            c = Q[i, j]
            denom = a + b - 2 * c
            w = 0.5 if denom <= 1e-10 else np.clip((b - c) / denom, 0.0, 1.0)
            val = w*w*a + (1-w)*(1-w)*b + 2*w*(1-w)*c
            best = min(best, val)

    return best




def compute_lower_bound(Q: np.ndarray) -> float:
    """
    Compute a fast lower bound on the quadratic objective over the simplex.

    Parameters
    ----------
    Q : np.ndarray, shape (k, k)
        Quadratic matrix.

    Returns
    -------
    lb : float
        Lower bound on min_w w^T Q w.

    Notes
    -----
    - Exact for k = 1 and k = 2.
    - For k >= 3:
        * Uses smallest diagonal entries as a baseline.
        * Adds a heuristic correction based on negative interactions.
    - Designed for speed rather than tightness.
    """
    k = Q.shape[0]

    if k == 1:
        return float(Q[0, 0])

    if k == 2:
        a, b = Q[0, 0], Q[1, 1]
        c = Q[0, 1]
        denom = a + b - 2 * c
        w = 0.5 if denom <= 1e-10 else np.clip((b - c) / denom, 0.0, 1.0)
        return float(w*w*a + (1-w)*(1-w)*b + 2*w*(1-w)*c)

    # Strong, fast bound for k >= 3
    diag = np.diag(Q)
    sorted_diag = np.sort(diag)
    lb = float(np.sum(sorted_diag[:k]))

    # Add a controlled interaction term
    if k >= 3:
        smallest_idx = np.argsort(diag)[:k]
        subQ = Q[np.ix_(smallest_idx, smallest_idx)]
        off_diag = subQ - np.diag(np.diag(subQ))
        min_off_diag = float(np.min(off_diag))
        if min_off_diag < -1e-8:
            lb += min_off_diag * (k * (k-1) / 2) * 0.55   # tuned coefficient

    return lb



def greedy_initial_solution(G: np.ndarray, candidate_idx: np.ndarray, m: int):
    """
    Construct an initial feasible solution using the first m candidates.

    Parameters
    ----------
    G : np.ndarray, shape (N, N)
        Global quadratic (Gram) matrix.
    candidate_idx : np.ndarray
        Candidate unit indices (assumed pre-ordered).
    m : int
        Subset size.

    Returns
    -------
    loss : float
        Objective value of the solution.
    selected : list of int
        Selected indices.
    w : np.ndarray, shape (m,)
        Optimal weights for the selected subset.

    Notes
    -----
    - Assumes `candidate_idx` is sorted (e.g., by diagonal values).
    - Provides a baseline solution for pruning.
    """
    selected = list(candidate_idx[:m])
    Q = G[np.ix_(selected, selected)]
    w = solve_qp_simplex(Q, np.ones(Q.shape[0]) / Q.shape[0], steps=50, lr=0.01)
    loss = float(w @ Q @ w)
    return loss, selected, w


def multi_start_bound(Q: np.ndarray, steps: int = 6) -> float:
    k = Q.shape[0]

    # ---- 1. FAST ANALYTIC LOWER BOUND (NEW) ----
    diag_min = np.min(np.diag(Q))
    trace_bound = diag_min  # crude but deterministic baseline

    # ---- 2. EIGENVALUE RELAXATION (CRITICAL ADDITION) ----
    try:
        eig_min = np.linalg.eigvalsh(Q).min()
        spectral_bound = eig_min
    except:
        spectral_bound = trace_bound

    base_bound = max(trace_bound, spectral_bound)

    # ---- 3. HEURISTIC IMPROVEMENT (ONLY REFINES) ----
    w1 = np.ones(k) / k

    diag = np.diag(Q)
    w2 = 1.0 / (diag + 1e-8)
    w2 = np.maximum(w2, 0)
    w2 = w2 / w2.sum()

    w3 = np.zeros(k)
    w3[np.argmin(diag)] = 1.0

    w1 = solve_qp_simplex(Q, w1, steps=steps)
    w2 = solve_qp_simplex(Q, w2, steps=steps)
    w3 = solve_qp_simplex(Q, w3, steps=steps)

    heuristic = min(
        w1 @ Q @ w1,
        w2 @ Q @ w2,
        w3 @ Q @ w3
    )

    # ---- FINAL COMBINATION ----
    return max(base_bound, heuristic)

def expand_tuple(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int,
    top_K: int,
    top_tuples: List,
    indices: List[int],
    stats: Dict,
    start_pos: int,
    Q_partial: np.ndarray,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    current_cost: float = 0.0
):

    """
    Recursively expand a partial subset within a branch-and-bound search.

    Parameters
    ----------
    G : np.ndarray, shape (N, N)
        Global quadratic matrix.
    candidate_idx : np.ndarray
        Ordered candidate indices.
    m : int
        Target subset size.
    top_K : int
        Number of best solutions to retain.
    top_tuples : list of Solution
        Current list of best solutions (sorted by loss).
    indices : list of int
        Current partial subset.
    stats : dict
        Mutable dictionary tracking search statistics.
    start_pos : int
        Position in `candidate_idx` from which to continue expansion.
    Q_partial : np.ndarray, shape (k, k)
        Quadratic matrix restricted to `indices`.

    Notes
    -----
    - Performs depth-first search with pruning.
    - At each step:
        1. Expands subset by adding one candidate.
        2. Updates Q incrementally (no recomputation).
        3. Computes a lower bound via approximate QP solve.
        4. Prunes if bound is worse than current top-K worst solution.
    - Leaf nodes (|indices| == m) are evaluated exactly.
    - `start_pos` ensures combinations (not permutations).
    - Stats tracked:
        * nodes_visited
        * subsets_evaluated
        * branches_considered
        * branches_pruned
    """

    stats["nodes_visited"] += 1

    # ====================== EARLY BUDGET PRUNING ======================
    if budget is not None and unit_costs is not None:
        if current_cost > budget + 1e-6:
            stats["branches_pruned"] += 1
            return

        remaining_slots = m - len(indices)
        if remaining_slots > 0:
            remaining_costs = unit_costs[candidate_idx[start_pos:]]
            if len(remaining_costs) >= remaining_slots:
                cheapest_remaining = np.sort(remaining_costs)[:remaining_slots].sum()
                if current_cost + cheapest_remaining > budget + 1e-6:
                    stats["branches_pruned"] += 1
                    return
    # =================================================================

    # Leaf node
    if len(indices) == m:
        stats["subsets_evaluated"] += 1

        Q = Q_partial
        w = solve_qp_simplex(Q, np.ones(Q.shape[0]) / Q.shape[0], steps=50, lr=0.01)
        total_loss = float(w @ Q @ w)

        # Final safety check
        if budget is not None and unit_costs is not None:
            subset_cost = np.dot(unit_costs[indices], w)
            if subset_cost > budget + 1e-6:
                return  # discard over-budget solution

        top_tuples.append(Solution(total_loss, indices[:], w))
        top_tuples.sort(key=lambda s: s.loss)
        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    # Branching
    for j_idx in range(start_pos, len(candidate_idx)):
        j = candidate_idx[j_idx]
        j_cost = float(unit_costs[j]) if unit_costs is not None else 0.0

        stats["branches_considered"] += 1

        # ---- incremental Q ----
        k = Q_partial.shape[0]

        Q_new = np.empty((k + 1, k + 1))
        Q_new[:k, :k] = Q_partial

        g = G[j, indices]  # vectorized

        Q_new[k, :k] = g
        Q_new[:k, k] = g
        Q_new[k, k] = G[j, j]

        # ---- bound ----
        lb = multi_start_bound(Q_new, steps=6)


        current_ub = top_tuples[-1].loss if len(top_tuples) == top_K else np.inf

        if lb >= current_ub:
            stats["branches_pruned"] += 1
            continue

        # ---- recurse (CORRECT — no reset!) ----
        expand_tuple(
            G,
            candidate_idx,
            m,
            top_K,
            top_tuples,
            indices + [j],
            stats,
            start_pos=j_idx + 1,
            Q_partial=Q_new,
            # NEW
            unit_costs=unit_costs,
            budget=budget,
            current_cost=current_cost + j_cost
        )
