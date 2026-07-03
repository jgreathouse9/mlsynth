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

Solved with cvxpy using the first installed mixed-integer backend (HiGHS by
preference, else SCIP / GLPK_MI / CBC).
"""

from __future__ import annotations

import time
from typing import List

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError


# MILP-capable cvxpy solvers, in preference order. The partition is a boolean
# exact-cover MIP; HiGHS is preferred but is not wired into every cvxpy build
# (e.g. some Python 3.10 wheels), so fall back to the next installed solver --
# SCIP ships with the ``design`` extra's ``pyscipopt``.
_MILP_SOLVERS = ("HIGHS", "SCIP", "GLPK_MI", "CBC")


def _pick_milp_solver():
    """First installed cvxpy MILP solver, else a clear translated error."""
    installed = set(cp.installed_solvers())
    for name in _MILP_SOLVERS:
        if name in installed:
            return getattr(cp, name)
    raise MlsynthEstimationError(
        "PANGEO needs a mixed-integer solver, but none of "
        f"{_MILP_SOLVERS} is installed in cvxpy. Install pyscipopt "
        '(``pip install "mlsynth[design]"``) or a HiGHS-enabled cvxpy.'
    )


def solve_partition(
    candidate_pairs: List[dict],
    unit_indices: np.ndarray,
    min_pairs: int = 1,
    return_diagnostics: bool = False,
):
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
    return_diagnostics : bool
        If ``True``, return ``(chosen, diagnostics)`` where ``diagnostics`` is a
        dict recording the MIP solve: ``path`` (``"exact_mip"``),
        ``n_candidate_pairs`` (``|F|`` -- the size of the candidate set fed to
        the cover), ``objective_value``, ``status``, ``solver_name``,
        ``solve_seconds``, and ``n_selected_pairs``. Default ``False`` keeps the
        original contract (returns just the chosen list).

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
    t0 = time.perf_counter()
    try:
        problem.solve(solver=_pick_milp_solver())
    except cp.error.SolverError as exc:
        raise MlsynthEstimationError(
            f"PANGEO partition MIP failed: {exc}"
        ) from exc
    solve_seconds = time.perf_counter() - t0
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise MlsynthEstimationError(
            f"PANGEO partition MIP did not solve (status={problem.status}); "
            "an exact cover may be infeasible for this Q / arm size."
        )

    chosen = [candidate_pairs[c] for c in range(n_cand)
              if x.value is not None and x.value[c] > 0.5]
    if return_diagnostics:
        stats = getattr(problem, "solver_stats", None)
        diagnostics = {
            "path": "exact_mip",
            "n_candidate_pairs": int(n_cand),
            "objective_value": (float(problem.value)
                                if problem.value is not None else float("nan")),
            "status": problem.status,
            "solver_name": getattr(stats, "solver_name", None),
            "solve_seconds": float(solve_seconds),
            "n_selected_pairs": len(chosen),
        }
        return chosen, diagnostics
    return chosen
