"""Set-partitioning MIP for PANGEO supergeo-pair design.

Given the admissible supergeo pairs over an arm's units (each with a
pre-period parallelism score), select a subset of pairs that **partitions
every unit exactly once** while minimising total non-parallelism -- the
Supergeo covering formulation (Chen et al. 2023), unchanged except the
per-pair score is the difference-in-differences parallelism of
:mod:`.parallelism` rather than a scalar sum-difference.

.. math::

   \\min_{x \\in \\{0,1\\}^{|\\mathcal F|}}
     \\sum_{G} \\text{score}(G)\\, x_G
   \\quad\\text{s.t.}\\quad
   M^\\top x = \\mathbf 1 \\ (\\text{exact cover}),\\;
   \\mathbf 1^\\top x \\ge \\kappa\\ (\\text{min pairs}).

Solved with cvxpy using the HiGHS mixed-integer backend.
"""

from __future__ import annotations

from typing import List

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError


def solve_partition(
    candidate_pairs: List[dict],
    unit_indices: np.ndarray,
    min_pairs: int = 1,
) -> List[dict]:
    """Select the exact-cover set of supergeo pairs of minimum total score.

    Parameters
    ----------
    candidate_pairs : list of dict
        Output of :func:`.parallelism.enumerate_candidate_pairs`; each has
        ``members`` (unit indices), ``score``, ``side_a``, ``side_b``.
    unit_indices : np.ndarray
        The arm's unit indices that must all be covered exactly once.
    min_pairs : int
        Minimum number of supergeo pairs in the design (>= 1).

    Returns
    -------
    list of dict
        The chosen candidate pairs (a subset of ``candidate_pairs``).
    """
    n_units = len(unit_indices)
    n_cand = len(candidate_pairs)
    if n_cand == 0:
        raise MlsynthEstimationError(
            "PANGEO: no admissible supergeo pairs (need >= 2 units per arm)."
        )

    pos = {int(u): j for j, u in enumerate(unit_indices)}
    # Incidence matrix M (units x candidates): M[i, c] = 1 iff unit i in pair c.
    M = np.zeros((n_units, n_cand), dtype=float)
    for c, pair in enumerate(candidate_pairs):
        for u in pair["members"]:
            M[pos[int(u)], c] = 1.0
    cost = np.array([p["score"] for p in candidate_pairs], dtype=float)

    x = cp.Variable(n_cand, boolean=True)
    constraints = [M @ x == 1]                      # exact cover
    if min_pairs > 1:
        constraints.append(cp.sum(x) >= min_pairs)
    problem = cp.Problem(cp.Minimize(cost @ x), constraints)
    try:
        problem.solve(solver=cp.HIGHS)
    except cp.error.SolverError as exc:
        raise MlsynthEstimationError(
            f"PANGEO partition MIP failed: {exc}"
        ) from exc
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise MlsynthEstimationError(
            f"PANGEO partition MIP did not solve (status={problem.status}); "
            "an exact cover may be infeasible for this Q / arm size."
        )

    chosen = [candidate_pairs[c] for c in range(n_cand)
              if x.value is not None and x.value[c] > 0.5]
    return chosen
