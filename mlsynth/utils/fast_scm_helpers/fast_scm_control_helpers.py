import cvxpy as cp
import numpy as np
from typing import List, Optional, Sequence

from ..bilevel.minnorm import solve_simplex_minnorm_batch

def _build_objective(
    X_E: np.ndarray,
    v: cp.Variable,
    treated_vec: np.ndarray,
    lambda_penalty: float
):
    """
    Construct the synthetic control quadratic objective.

    Parameters
    ----------
    X_E : np.ndarray, shape (T_E, N)
        Standardized feature matrix over the estimation period.
    v : cp.Variable, shape (N,)
        Optimization variable representing control weights.
    treated_vec : np.ndarray, shape (T_E,)
        Synthetic treated series over the estimation period.
    lambda_penalty : float
        Regularization strength for unit-specific mismatch penalties.

    Returns
    -------
    objective : cp.Expression
        CVXPY expression representing:
            ||X_E v - treated_vec||^2
            + lambda_penalty * sum_j v_j ||X_E[:, j] - treated_vec||^2

    Notes
    -----
    - First term enforces global fit to the treated series.
    - Second term penalizes assigning weight to units that individually
      deviate from the treated trajectory.
    - The penalty is linear in `v`, preserving convexity.
    """
    match_term = cp.sum_squares(X_E @ v - treated_vec)

    penalty_term = lambda_penalty * cp.sum([
        v[j] * cp.sum_squares(X_E[:, j] - treated_vec)
        for j in range(X_E.shape[1])
    ])

    return match_term + penalty_term


def _build_constraints(
    v: cp.Variable,
    treated_idx: List[int]
):
    """
    Construct feasibility constraints for synthetic control weights.

    Parameters
    ----------
    v : cp.Variable, shape (N,)
        Control weight vector.
    treated_idx : list of int
        Indices of treated units to exclude from the donor pool.

    Returns
    -------
    constraints : list of cp.Constraint
        Constraints enforcing:
        - Nonnegativity: v >= 0
        - Convex combination: sum(v) == 1
        - Exclusion: v[j] = 0 for j in treated_idx

    Notes
    -----
    - Ensures the synthetic control is a convex combination of donor units.
    - Explicit exclusion prevents contamination from treated units.
    """
    constraints = [
        v >= 0,
        cp.sum(v) == 1
    ]

    for j in treated_idx:
        constraints.append(v[j] == 0)

    return constraints


def _solve_qp_problem(objective, constraints) -> Optional[np.ndarray]:
    """
    Solve a convex quadratic program using CVXPY.

    Parameters
    ----------
    objective : cp.Expression
        Objective function to minimize.
    constraints : list of cp.Constraint
        Feasibility constraints.

    Returns
    -------
    solution : np.ndarray, shape (N,) or None
        Optimal variable values, or None if the solver fails.

    Notes
    -----
    - Uses the OSQP solver with tight tolerances.
    - Returns None if no solution is found (caller must handle).
    - Assumes the problem is convex and well-posed.
    """
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6)

    if prob.variables()[0].value is None:
        return None

    return np.asarray(prob.variables()[0].value).flatten()

def solve_control_qp(
    X_E: np.ndarray,
    treated_vec: np.ndarray,
    treated_idx: List[int],
    lambda_penalty: float = 0.1,
    exclude_idx: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """
    Solve for synthetic control weights given a treated trajectory.

    Parameters
    ----------
    X_E : np.ndarray, shape (T_E, N)
        Standardized feature matrix over the estimation period.
    treated_vec : np.ndarray, shape (T_E,)
        Synthetic treated series constructed from selected units.
    treated_idx : list of int
        Indices of treated units to exclude from the control.
    lambda_penalty : float, default=0.1
        Regularization strength for mismatch penalties.
    exclude_idx : list of int, optional
        Additional indices to drop from the donor pool -- the spillover
        neighbours ``N(S)`` of the treated units (Vives-i-Bastida "exclusion
        restriction": a treated unit's same-cluster/adjacent units may not serve
        as its controls, so the treatment cannot contaminate the counterfactual).

    Returns
    -------
    v : np.ndarray, shape (N,) or None
        Synthetic control weights over all units. Entries at treated AND excluded
        indices are exactly zero. Returns None if optimization fails (including
        when the exclusions empty the donor pool).

    Notes
    -----
    - Solves a convex QP via CVXPY.
    - The solution represents a convex combination of donor units.
    - Used downstream to construct synthetic control trajectories.
    """
    _, N = X_E.shape

    # Exclusion by *construction*: drop the treated columns from the problem
    # entirely rather than pinning ``v[treated] == 0`` as a solver equality.
    # A first-order QP solver only satisfies equalities to its tolerance
    # (~1e-6), which can leave a tiny weight on a treated unit; eliminating the
    # variables makes those entries exactly zero. Optimising over the donor
    # submatrix is mathematically identical to constraining the treated weights
    # to zero, since the penalty term is separable in ``v`` and a zero weight
    # contributes nothing.
    excluded = {int(j) for j in treated_idx}
    if exclude_idx is not None:
        excluded |= {int(j) for j in exclude_idx}
    control_idx = np.array([i for i in range(N) if i not in excluded], dtype=int)
    if control_idx.size == 0:
        return None

    v_control = cp.Variable(control_idx.size)
    objective = _build_objective(X_E[:, control_idx], v_control,
                                 treated_vec, lambda_penalty)
    constraints = [v_control >= 0, cp.sum(v_control) == 1]

    solution_control = _solve_qp_problem(objective, constraints)
    if solution_control is None:
        return None

    # Scatter the donor weights back into a full-length vector with exact zeros
    # on the treated indices.
    v = np.zeros(N)
    v[control_idx] = solution_control
    return v



def _homogeneous_gram(Q: np.ndarray, b: np.ndarray, yy: np.ndarray,
                      lambda_penalty: float) -> np.ndarray:
    """The penalised control objective as a homogeneous form on the simplex.

    With ``Q = A'A``, ``b = A'y`` and ``c_j = ||A[:, j] - y||^2`` the objective is

        ||A v - y||^2 + lam * sum_j v_j c_j  =  v'Q v + d'v + y'y,
        d = lam c - 2 b,   c_j = Q_jj - 2 b_j + y'y

    and the equality constraint absorbs the linear part: ``1'v = 1`` gives
    ``d'v = d'v (1'v) = v'(d 1')v``, so symmetrising,

        S = Q + (d 1' + 1 d') / 2

    leaves ``v'S v`` equal to the objective up to the constant ``y'y``. The
    minimiser is therefore unchanged and the program is the one Wolfe's active
    set solves.

    ``S`` is not itself positive semi-definite -- the rank-one correction sees to
    that, and no shift repairs it, since ``1 1'`` has rank one while ``Q`` is
    singular in every direction the donors do not span. What the active set
    needs is convexity on the *feasible* set, and that holds exactly: for any
    ``z`` with ``1'z = 0``, ``z'S z = z'Q z >= 0``. Every step the method takes
    -- corral solves and line searches alike -- stays on ``{1'v = 1}``, so it
    only ever sees the restriction.

    Parameters
    ----------
    Q : np.ndarray, shape (..., J, J)
        Donor Gram matrices, batched over any leading axes.
    b : np.ndarray, shape (..., J)
        ``A'y`` for each problem.
    yy : np.ndarray, shape (...,)
        ``y'y`` for each problem.
    lambda_penalty : float
        Weight on the Abadie-L'Hour discrepancy penalty.

    Returns
    -------
    np.ndarray, shape (..., J, J)
        The homogeneous forms ``S``.
    """
    diag = np.einsum('...jj->...j', Q)
    c = diag - 2.0 * b + yy[..., None]
    d = lambda_penalty * c - 2.0 * b
    return Q + 0.5 * (d[..., :, None] + d[..., None, :])


def solve_control_qps(
    X_E: np.ndarray,
    treated_vecs: Sequence[np.ndarray],
    treated_idx_list: Sequence[Sequence[int]],
    lambda_penalty: float = 0.1,
    exclude_idx_list: Optional[Sequence[Optional[Sequence[int]]]] = None,
) -> List[Optional[np.ndarray]]:
    """Solve the control QP for a whole shortlist of designs at once.

    Same program as :func:`solve_control_qp`, once per design, but reduced to
    the homogeneous form (see :func:`_homogeneous_gram`) and handed to the
    batched Wolfe active set, which runs the stack in lockstep: iterations are
    set by the hardest design in the shortlist and not by their number.

    Designs are grouped by donor-pool width, since a stack has to be
    rectangular and the spillover exclusions ``N(S)`` differ in size between
    designs. With no conflict graph every pool is ``N - m`` wide and there is
    one group.

    Parameters
    ----------
    X_E : np.ndarray, shape (T_E, N)
        Standardized feature matrix over the estimation period.
    treated_vecs : sequence of np.ndarray, each shape (T_E,)
        Synthetic treated series, one per design.
    treated_idx_list : sequence of sequences of int
        Treated units to exclude from the donor pool, one per design.
    lambda_penalty : float, default=0.1
        Regularization strength for mismatch penalties.
    exclude_idx_list : sequence of (sequence of int or None), optional
        Additional indices to drop per design -- the spillover neighbours
        ``N(S)``. ``None`` or an empty sequence means no extra exclusions.

    Returns
    -------
    list of (np.ndarray, shape (N,) or None)
        Control weights per design, aligned with the input order. Entries at
        treated and excluded indices are exactly zero. A design whose exclusions
        empty the donor pool comes back as ``None`` at its own position, so the
        caller can skip it without the list shifting under the others.
    """
    n = len(treated_vecs)
    if n != len(treated_idx_list):
        raise ValueError(
            f"{n} treated series for {len(treated_idx_list)} designs; "
            "the two must align.")
    if n == 0:
        return []

    _, N = X_E.shape
    out: List[Optional[np.ndarray]] = [None] * n

    # Donor pools, one per design. Exclusion is by construction -- the columns
    # leave the problem -- so the excluded weights are exactly zero and not zero
    # to a solver tolerance.
    pools: List[np.ndarray] = []
    for k in range(n):
        excluded = {int(j) for j in treated_idx_list[k]}
        if exclude_idx_list is not None and exclude_idx_list[k] is not None:
            excluded |= {int(j) for j in exclude_idx_list[k]}
        pools.append(np.array([i for i in range(N) if i not in excluded],
                              dtype=int))

    widths: dict = {}
    for k, ctrl in enumerate(pools):
        if ctrl.size:
            widths.setdefault(ctrl.size, []).append(k)

    G_full = X_E.T @ X_E
    XT = X_E.T

    for width, members in widths.items():
        C = np.stack([pools[k] for k in members])                # (S, width)
        Ys = np.stack([np.asarray(treated_vecs[k], dtype=float)
                       for k in members])                        # (S, T_E)
        Q = G_full[C[:, :, None], C[:, None, :]]                 # (S, w, w)
        b = np.einsum('sjt,st->sj', XT[C], Ys)
        yy = np.einsum('st,st->s', Ys, Ys)
        W = solve_simplex_minnorm_batch(
            _homogeneous_gram(Q, b, yy, lambda_penalty))
        for row, k in enumerate(members):
            v = np.zeros(N)
            v[C[row]] = W[row]
            out[k] = v

    return out


def compute_nmse(X: np.ndarray, w: np.ndarray, target: np.ndarray, idx: np.ndarray, treated_idx: list) -> float:
    """
    Compute normalized mean squared error (NMSE) over a time subset.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full feature matrix.
    w : np.ndarray, shape (k,)
        Weights for treated units.
    target : np.ndarray, shape (T,)
        Target series (e.g., weighted average of outcome units).
    idx : np.ndarray
        Time indices over which to evaluate the error.
    treated_idx : list of int
        Indices of treated units corresponding to `w`.

    Returns
    -------
    nmse : float
        Mean squared error normalized by per-time cross-sectional variance.

    Notes
    -----
    - Synthetic series: X[:, treated_idx] @ w
    - Normalization uses variance across outcome columns at each time step.
    - A small constant is added to avoid division by zero.
    - This normalization emphasizes relative fit rather than scale.
    """
    synth = X[idx][:, treated_idx] @ w
    tgt = target[idx]
    denom = np.var(X[idx, :len(target)], axis=1) + 1e-8  # variance over Y units
    return float(np.mean(((synth - tgt) ** 2) / denom))


def compute_effect_series(X: np.ndarray, treated_idx: list, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the estimated treatment effect time series.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full feature matrix.
    treated_idx : list of int
        Indices of treated units.
    w : np.ndarray, shape (k,)
        Weights defining the synthetic treated unit.
    v : np.ndarray, shape (N,)
        Synthetic control weights.

    Returns
    -------
    effects : np.ndarray, shape (T,)
        Estimated treatment effect at each time:
            (synthetic treated) - (synthetic control)

    Notes
    -----
    - Synthetic treated: X[:, treated_idx] @ w
    - Synthetic control: X @ v
    - Positive values indicate treated > control.
    """
    return X[:, treated_idx] @ w - X @ v
