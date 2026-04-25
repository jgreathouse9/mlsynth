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


# ============================================================
# UTILITIES & PROJECTION
# ============================================================

def expand_weights_to_full(indices: List[int], weights: np.ndarray, total_units: int) -> np.ndarray:
    """
    Expand a subset weight vector into a full-length vector.

    Parameters
    ----------
    indices : List[int]
        The indices corresponding to the chosen donors.
    weights : np.ndarray
        The optimized weights corresponding to `indices`.
    total_units : int
        The total number of candidate units in the pool.

    Returns
    -------
    np.ndarray
        A vector of length `total_units` with subset weights at the specified
        indices and zeros elsewhere.
    """
    w_full = np.zeros(total_units)
    w_full[indices] = weights
    return w_full


def compute_search_space_size(M: int, m: int) -> Tuple[int, int]:
    """
    Compute the total number of possible subsets and nodes in the search tree.

    Parameters
    ----------
    M : int
        The number of total candidate donors.
    m : int
        The number of donors to select for the subset.

    Returns
    -------
    total_subsets : int
        The number of combinations (M choose m).
    total_nodes : int
        The sum of combinations (M choose k) for k=1 to m.
    """
    total_subsets = comb(M, m)
    total_nodes = sum(comb(M, k) for k in range(1, m + 1))
    return total_subsets, total_nodes


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the probability simplex using an O(n log n) algorithm.

    The projection solves: min ||w - v||^2 subject to sum(w) = 1 and w >= 0.

    Parameters
    ----------
    v : np.ndarray
        The input vector to be projected.

    Returns
    -------
    np.ndarray
        The projected vector on the probability simplex.
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def solve_qp_simplex(Q: np.ndarray, max_iter: int = 100, lr: float = 0.05) -> np.ndarray:
    """
    Solve a quadratic program over the probability simplex using Projected Gradient Descent.

    Solves: min w^T Q w subject to sum(w) = 1 and w >= 0.

    Parameters
    ----------
    Q : np.ndarray
        The quadratic cost matrix (Gram matrix of donors).
    max_iter : int, default=100
        Maximum number of gradient descent iterations.
    lr : float, default=0.05
        The learning rate for the gradient steps.

    Returns
    -------
    np.ndarray
        The optimized weight vector `w`.
    """
    m = Q.shape[0]
    w = np.ones(m) / m
    for _ in range(max_iter):
        grad = 2 * Q @ w
        w -= lr * grad
        w = project_to_simplex(w)
    return w


# ============================================================
# ADMISSIBLE BOUNDS (SIMPLEX-CONSTRAINED)
# ============================================================

def get_tighter_bound(Q: np.ndarray, full_G_diag: np.ndarray, available_indices: np.ndarray,
                      remaining_needed: int) -> float:
    """
    Computes a strictly admissible lower bound for the quadratic form.

    This version is designed to be 'safer' than the linear diagonal bound.
    It uses a conservative estimate to ensure we don't prune the global optimum
    in the presence of high covariance.
    """
    k = Q.shape[0]
    if k == 0: return 0.0

    # Minimum possible variance of any single unit currently in the submatrix
    diags = np.diag(Q)

    # We divide by k^2 to account for the 'Worst Case' scenario of 
    # perfectly negatively correlated units (which is mathematically 
    # the most a quadratic form can be reduced on a simplex).
    current_lb = np.min(diags) / (k ** 2)

    if remaining_needed > 0 and len(available_indices) > 0:
        # What is the absolute best donor we haven't picked yet?
        future_min = np.min(full_G_diag[available_indices])

        # We take the lower of the two possibilities to remain admissible.
        total_k = k + remaining_needed
        return float(min(current_lb, future_min / (total_k ** 2)))

    return float(current_lb)


# ============================================================
# INITIALIZATION (SFS)
# ============================================================

def greedy_initial_solution(G: np.ndarray, unit_costs: np.ndarray, candidate_idx: np.ndarray, m: int, lam: float):
    """
    Find a high-quality initial solution using Sequential Forward Selection (SFS).

    This greedy approach iteratively adds the unit that results in the
    minimum objective value (Loss + lambda * Cost) until m units are selected.

    Parameters
    ----------
    G : np.ndarray
        The full Gram matrix.
    unit_costs : np.ndarray
        Vector of linear costs associated with each candidate unit.
    candidate_idx : np.ndarray
        The indices of units available for selection.
    m : int
        The target subset size.
    lam : float
        Penalty parameter for the linear cost term.

    Returns
    -------
    Solution
        A Solution object containing the greedy results.
    """
    selected = []
    remaining = list(candidate_idx)
    curr_w, curr_loss, curr_cost = None, 0.0, 0.0

    for _ in range(m):
        best_cand, best_score = -1, np.inf
        for cand in remaining:
            trial = selected + [cand]
            Q = G[np.ix_(trial, trial)]
            w = solve_qp_simplex(Q)
            loss = float(w @ Q @ w)
            cost = float(np.sum(unit_costs[trial]))
            score = loss + (lam * cost)
            if score < best_score:
                best_score, best_cand, curr_w, curr_loss, curr_cost = score, cand, w, loss, cost
        selected.append(best_cand)
        remaining.remove(best_cand)

    return Solution(loss=curr_loss + (lam * curr_cost), indices=selected, weights=curr_w, cost=curr_cost)


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
    Recursive core of the Branch and Bound algorithm for subset selection.

    Explores the search tree using depth-first search. Prunes branches
    using lower bounds on both the quadratic loss and linear costs.

    Parameters
    ----------
    G : np.ndarray
        The full Gram matrix.
    candidate_idx : np.ndarray
        Ordered indices of candidates to ensure consistent branching.
    unit_costs : np.ndarray
        Linear cost vector for the candidates.
    m : int
        The target subset size.
    lam : float
        Cost penalty parameter.
    top_K : int
        The number of best solutions to track.
    top_tuples : List[Solution]
        The sorted list of top-K Solution objects found so far.
    indices : List[int]
        The indices of the units in the current path.
    stats : Dict
        Dictionary for tracking search metrics (nodes visited, prunes, etc.).
    start_pos : int
        Index in `candidate_idx` from which to begin branching.
    Q_partial : np.ndarray
        The pre-computed sub-matrix G[indices, indices].
    current_cost : float, default=0.0
        The accumulated cost of the units in `indices`.

    Returns
    -------
    None
        Updates `top_tuples` and `stats` in-place.
    """
    stats["nodes_visited"] += 1
    k = len(indices)

    # Base case: full subset reached
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

    remaining_needed = m - k
    available_indices = candidate_idx[start_pos:]
    num_available = len(available_indices)

    # Feasibility check
    if num_available < remaining_needed:
        return

    # Future Cost Bounding
    if num_available == remaining_needed:
        min_future_cost = np.sum(unit_costs[available_indices])
    elif remaining_needed > 0:
        future_costs = np.partition(unit_costs[available_indices], remaining_needed - 1)[:remaining_needed]
        min_future_cost = np.sum(future_costs)
    else:
        min_future_cost = 0.0

    # Quadratic Bounding
    lb_quad = get_tighter_bound(Q_partial, np.diag(G), available_indices, remaining_needed)
    lb_total = lb_quad + lam * (current_cost + min_future_cost)

    # Pruning condition
    if len(top_tuples) >= top_K and lb_total >= top_tuples[-1].loss:
        stats["branches_pruned"] += 1
        return

    # Branching
    for j_idx in range(start_pos, len(candidate_idx)):
        j = candidate_idx[j_idx]
        stats["branches_considered"] += 1

        # Incremental sub-matrix construction
        g = G[j, indices]
        Q_next = np.empty((k + 1, k + 1))
        Q_next[:k, :k] = Q_partial
        Q_next[k, :k], Q_next[:k, k] = g, g
        Q_next[k, k] = G[j, j]

        expand_tuple(
            G, candidate_idx, unit_costs, m, lam, top_K, top_tuples,
            indices + [j], stats, j_idx + 1, Q_next,
            current_cost + unit_costs[j]
        )
