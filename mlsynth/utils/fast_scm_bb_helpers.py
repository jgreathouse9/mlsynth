import cvxpy as cp
import numpy as np
from typing import List, Tuple, Dict

from math import comb

def compute_search_space_size(M: int, m: int):
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
    Project a vector onto the probability simplex (non-negative, sum=1).

    Parameters
    ----------
    v : np.ndarray, shape (n,)
        Input vector to be projected.
    eps : float, optional
        Small tolerance to avoid numerical issues (default 1e-10).

    Returns
    -------
    w : np.ndarray, shape (n,)
        Projected vector satisfying:
        - w_i >= 0 for all i
        - sum(w) = 1

    Notes
    -----
    - Implements the algorithm from:
      Duchi et al., "Efficient Projections onto the L1-Ball for Learning in High Dimensions", https://doi.org/10.1145/1390156.139019
    - Sorting and cumulative sum are used to compute the threshold.
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def solve_qp_simplex(Q: np.ndarray, max_iter: int = 200, lr: float = 0.01) -> np.ndarray:
    """
    Solve a quadratic program over the simplex using projected gradient descent.

    Parameters
    ----------
    Q : np.ndarray, shape (m, m)
        Quadratic matrix in the objective: minimize w^T Q w.
    max_iter : int, optional
        Maximum number of gradient descent iterations (default 200).
    lr : float, optional
        Learning rate for gradient descent (default 0.01).

    Returns
    -------
    w : np.ndarray, shape (m,)
        Optimal weight vector on the simplex (sum(w)=1, w_i >= 0).

    Notes
    -----
    - Objective: minimize w^T Q w subject to w on the simplex.
    - Gradient: grad = 2 Q w
    - Uses `project_to_simplex` at each step to enforce constraints.
    - Suitable for small- to medium-sized Q (hundreds of units).
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
    stats["nodes_visited"] += 1

    # ---- leaf ----
    if len(indices) == m:
        stats["subsets_evaluated"] += 1

        Q = Q_partial
        w = solve_qp_simplex(Q)
        total_loss = float(w @ Q @ w)

        top_tuples.append((total_loss, indices[:], w))
        top_tuples.sort(key=lambda x: x[0])
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

        if len(top_tuples) >= top_K and lb >= top_tuples[-1][0] * 0.999:
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
