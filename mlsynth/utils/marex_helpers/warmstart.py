"""Warm-start the exact MAREX MIQP with an initial integer design.

cvxpy does not forward a MIP start to SCIP, so we subclass its SCIP conic solver
and inject the binary selection ``z`` as a *partial* solution (via pyscipopt's
``createPartialSol`` / ``setSolVal`` / ``addSol``) before the solve. SCIP
completes the continuous treated/control weights and uses the design as an
initial incumbent.

A warm start is only a hint: the proven optimum is identical with or without it,
so this never changes the answer of a solve that runs to completion. What it does
change is the *path* -- branch-and-bound starts from a good design instead of
hunting for one -- which matters under a wall-clock budget (``time_limit``) at
scale, where finding a strong feasible incumbent is the expensive part.

The binary ``z`` variables are located in cvxpy's canonicalised problem via
``data[BOOL_IDX]`` (``z`` is the only boolean variable in the MIQP); their values
are supplied in cvxpy's column-major variable order.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import cvxpy as cp
import cvxpy.settings as s
import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP as _CvxSCIP


class _WarmStartSCIP(_CvxSCIP):
    """cvxpy SCIP solver that seeds a partial MIP start on the binary vars.

    ``warm_bool`` is a 1-D 0/1 array aligned to ``data[BOOL_IDX]`` -- the binary
    ``z`` positions in the conic objective vector ``c``. We set only the integer
    part; SCIP completes the continuous weights via its repair heuristics. If the
    length does not match (a defensive guard), the warm start is skipped and the
    solve proceeds cold -- correctness is unaffected.
    """

    def __init__(self, warm_bool: np.ndarray) -> None:
        super().__init__()
        self._warm_bool = warm_bool

    def solve_via_data(self, data, warm_start, verbose, solver_opts,
                       solver_cache=None):
        from pyscipopt.scip import Model

        model = Model()
        model.redirectOutput()
        A, b, c, dims = self._define_data(data)
        variables = self._create_variables(model, data, c)
        constraints = self._add_constraints(model, variables, A, b, dims)
        self._set_params(model, verbose, solver_opts, data, dims)

        bidx = list(data[s.BOOL_IDX])
        if len(bidx) == len(self._warm_bool):
            sol = model.createPartialSol()
            for pos, vpos in enumerate(bidx):
                model.setSolVal(sol, variables[vpos], float(self._warm_bool[pos]))
            model.addSol(sol)
        return self._solve(model, variables, constraints, data, dims)


def solve_warmstarted(
    prob: cp.Problem,
    z_var: cp.Variable,
    warm_z: np.ndarray,
    *,
    verbose: bool = False,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> cp.Problem:
    """Solve MIQP ``prob`` (binary ``z_var``) seeding the design ``warm_z``.

    ``warm_z`` is a 0/1 array of ``z_var``'s shape -- ones at the (unit, cluster)
    cells to start treated. The problem is mutated in place so ``prob.value`` and
    ``z_var.value`` are populated, exactly as ``prob.solve`` would. ``solver_opts``
    is forwarded to SCIP (e.g. ``{"scip_params": {"limits/time": 90}}``).

    Returns ``prob`` for convenience.
    """
    warm_z = np.asarray(warm_z, dtype=float)
    if warm_z.shape != tuple(z_var.shape):
        raise ValueError(
            f"warm_z shape {warm_z.shape} does not match z {tuple(z_var.shape)}")
    # cvxpy stacks matrix variables column-major; align to that order.
    warm_bool = warm_z.flatten(order="F")
    data, chain, inv = prob.get_problem_data(cp.SCIP)
    solver = _WarmStartSCIP(warm_bool)
    raw = solver.solve_via_data(data, False, verbose, dict(solver_opts or {}))
    prob.unpack(chain.invert(raw, inv))
    return prob


def lexscm_warm_start(lex_result: Any) -> list:
    """Treated unit labels of a fitted LEXSCM result's top candidate.

    Convenience for ``MAREX(..., warm_start=lexscm_warm_start(lex.fit()))``: pulls
    the winning candidate's treated set (full-precision dict keys) so MAREX seeds
    its MIQP with LEXSCM's recommended design. Duck-typed -- any object exposing
    ``search.winner.treated_weight_dict_full`` works.
    """
    return list(lex_result.search.winner.treated_weight_dict_full)
