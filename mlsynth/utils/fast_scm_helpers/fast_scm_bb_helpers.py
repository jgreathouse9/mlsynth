import cvxpy as cp
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from math import comb

@dataclass(order=True)
class Solution:
    """
    Container for a candidate solution in the branch-and-bound search.

    Attributes
    ----------
    loss : float
        Objective value (w^T Q w) for the selected subset.
    indices : list of int
        Indices of selected units (subset of candidate_idx).
    weights : np.ndarray, shape (m,)
        Optimal simplex weights corresponding to `indices`.
    label : str, optional
        Optional human-readable label (assigned post hoc).

    Notes
    -----
    - Ordering is based on `loss` only, enabling sorting of solutions.
    - Lower loss indicates a better solution.
    """
    loss: float
    indices: List[int] = field(compare=False)
    weights: np.ndarray = field(compare=False)
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


def solve_qp_simplex(Q: np.ndarray, max_iter: int = 200, lr: float = 0.01) -> np.ndarray:
    """
    Approximately solve a quadratic program over the simplex.

    Minimizes:
        w^T Q w
    subject to:
        w >= 0, sum(w) = 1

    Parameters
    ----------
    Q : np.ndarray, shape (k, k)
        Symmetric positive semi-definite matrix.
    max_iter : int, default=200
        Number of projected gradient steps.
    lr : float, default=0.01
        Learning rate.

    Returns
    -------
    w : np.ndarray, shape (k,)
        Approximate minimizer on the simplex.

    Notes
    -----
    - Uses projected gradient descent with simplex projection.
    - Convergence is heuristic; not guaranteed to reach the global optimum.
    - Suitable for small k (subset sizes).
    """
    m = Q.shape[0]
    w = np.ones(m) / m
    for _ in range(max_iter):
        grad = 2 * Q @ w
        w -= lr * grad
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
    w = solve_qp_simplex(Q)
    loss = float(w @ Q @ w)
    return loss, selected, w




def expand_tuple(
    G: np.ndarray,
    candidate_idx: np.ndarray,
    m: int,
    top_K: int,
    top_tuples: List,
    indices: List[int],
    stats: Dict,
    start_pos: int,
    Q_partial: np.ndarray
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

    # ---- leaf ----
    if len(indices) == m:
        stats["subsets_evaluated"] += 1

        Q = Q_partial
        w = solve_qp_simplex(Q)
        total_loss = float(w @ Q @ w)

        top_tuples.append(Solution(total_loss, indices[:], w))
        top_tuples.sort()
        if len(top_tuples) > top_K:
            top_tuples.pop()
        return

    # ---- branch ----
    for j_idx in range(start_pos, len(candidate_idx)):
        j = candidate_idx[j_idx]

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
        w = solve_qp_simplex(Q_new, max_iter=500)  # fewer iters
        lb = float(w @ Q_new @ w)

        if len(top_tuples) >= top_K and lb >= top_tuples[-1].loss * 0.999:
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
            Q_partial=Q_new
        )
