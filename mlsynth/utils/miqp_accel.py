"""Warm-start + valid objective-cut accelerator for cvxpy MIQP solves on SCIP.

Two levers a mixed-integer solver uses to finish fast are a good incumbent (so it
stops *hunting* for one) and a tight dual bound (so it can *prove* the incumbent
is good and stop). cvxpy forwards neither to SCIP. This module supplies both:

* a *warm start* -- a binary MIP start injected via pyscipopt's
  ``createPartialSol`` / ``setSolVal`` / ``addSol``; and
* a valid *objective lower-bound cut* ``c^T x >= L`` -- a single linear
  constraint on the canonicalised objective. Because SCIP *minimises* ``c^T x``,
  this forces every node relaxation (hence the global dual bound) up to ``L``. It
  is a valid cut whenever ``L`` is a true lower bound on the optimum: no feasible
  point has objective below the optimum, so none is removed.

The dual-bound lever is the one SCIP cannot supply itself for the hard MIQPs in
this library (SYNDES/MAREX two-way): SCIP's own dual bound is the McCormick
relaxation, which is very loose, so it keeps branching long after its incumbent is
essentially optimal. Handing it an external bound (e.g. an SDP/moment bound) as a
cut lets the ordinary ``gap_limit`` certify against a tight bound and terminate.

Safety. A cut with an ``L`` above the true optimum would remove the optimum (a
correctness bug, not a slowdown). Callers must margin ``L`` below the bound. As a
backstop, if the cut renders the model infeasible (``L`` above *every* feasible
objective), the solve falls back to the un-cut problem and reports ``fell_back``,
so a bad bound degrades to a correct-but-unaccelerated solve rather than a wrong
or infeasible answer.

Offset. The cut is written on the canonical objective vector ``c`` assuming the
canonicalised objective has no constant term (``objective = c^T x``), which holds
for the homogeneous quadratic objectives in this library. A nonzero offset would
misalign the cut, so this helper is intended for those objectives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cvxpy as cp
import cvxpy.settings as s
import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP as _CvxSCIP


@dataclass(frozen=True)
class AccelInfo:
    """Diagnostics from an accelerated solve.

    Attributes
    ----------
    status : str
        SCIP's terminal status (``"optimal"``, ``"gaplimit"``, ``"timelimit"``,
        ``"infeasible"``, ...).
    dual_bound : float
        SCIP's global dual (lower) bound at termination -- lifted to the injected
        cut ``L`` when one was applied.
    gap : float
        SCIP's reported relative optimality gap.
    solve_time : float
        SCIP solving time in seconds.
    cut_applied : bool
        Whether the objective lower-bound cut was added.
    warm_applied : bool
        Whether the binary warm start was accepted (length matched the model's
        boolean variables).
    fell_back : bool
        Whether the cut was dropped and the solve retried un-cut because the cut
        made the model infeasible (a too-high ``L``).
    """

    status: str
    dual_bound: float
    gap: float
    solve_time: float
    cut_applied: bool
    warm_applied: bool
    fell_back: bool


class _WarmCutSCIP(_CvxSCIP):
    """cvxpy SCIP solver that optionally injects a warm start and an objective cut.

    ``warm_bits`` is a 1-D 0/1 array aligned to ``data[BOOL_IDX]`` (cvxpy's
    column-major boolean-variable order); ``objective_lower_bound`` is ``L`` in
    the cut ``c^T x >= L``. Either may be ``None``. The ``*_applied`` attributes
    record what actually took effect.
    """

    def __init__(self, warm_bits: Optional[np.ndarray] = None,
                 objective_lower_bound: Optional[float] = None) -> None:
        super().__init__()
        self._warm = None if warm_bits is None else np.asarray(warm_bits, dtype=float)
        self._L = objective_lower_bound
        self.warm_applied = False
        self.cut_applied = False
        # The built SCIP model, kept so the caller can read the dual bound / gap /
        # status directly rather than through cvxpy's solution dict, whose keys
        # ("model", "scip_status") are internal and vary across cvxpy versions.
        self.model = None

    def solve_via_data(self, data, warm_start, verbose, solver_opts,
                       solver_cache=None):
        from pyscipopt import quicksum
        from pyscipopt.scip import Model

        model = Model()
        model.redirectOutput()
        A, b, c, dims = self._define_data(data)
        variables = self._create_variables(model, data, c)
        constraints = self._add_constraints(model, variables, A, b, dims)
        self._set_params(model, verbose, solver_opts, data, dims)

        # Objective lower-bound cut: c^T x >= L. SCIP minimises c^T x, so this
        # lifts the dual bound to L. Only the nonzero objective coefficients
        # contribute (the objective is a handful of epigraph/aux variables).
        if self._L is not None:
            nz = np.nonzero(c)[0]
            model.addCons(quicksum(float(c[i]) * variables[i] for i in nz)
                          >= float(self._L))
            self.cut_applied = True

        # Binary warm start (partial MIP start on the boolean variables).
        if self._warm is not None:
            bidx = list(data[s.BOOL_IDX])
            if len(bidx) == len(self._warm):
                sol = model.createPartialSol()
                for pos, vpos in enumerate(bidx):
                    model.setSolVal(sol, variables[vpos], float(self._warm[pos]))
                model.addSol(sol)
                self.warm_applied = True

        self.model = model
        return self._solve(model, variables, constraints, data, dims)


def solve_warm_cut(
    prob: cp.Problem,
    bool_var: cp.Variable,
    *,
    warm_bits: Optional[np.ndarray] = None,
    objective_lower_bound: Optional[float] = None,
    gap_limit: Optional[float] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[cp.Problem, AccelInfo]:
    """Solve MIQP ``prob`` on SCIP, injecting a warm start and/or a valid cut.

    Parameters
    ----------
    prob : cvxpy.Problem
        A mixed-integer problem whose binary variable is ``bool_var``.
    bool_var : cvxpy.Variable
        The problem's boolean variable (used to align/validate ``warm_bits``).
    warm_bits : np.ndarray, optional
        A 0/1 array of ``bool_var``'s shape -- the initial incumbent. ``None``
        skips the warm start. Raises ``ValueError`` on a shape mismatch.
    objective_lower_bound : float, optional
        ``L`` in the valid cut ``objective >= L``. Must be a true lower bound on
        the optimum (margin it below any first-order relaxation bound). ``None``
        skips the cut.
    gap_limit, time_limit : float, optional
        Forwarded to SCIP as ``limits/gap`` / ``limits/time``.
    verbose : bool, optional
        SCIP verbosity.

    Returns
    -------
    (prob, AccelInfo)
        ``prob`` is solved in place (``prob.value`` / ``bool_var.value``
        populated). ``AccelInfo`` carries the solve diagnostics.
    """
    if warm_bits is not None:
        warm_bits = np.asarray(warm_bits, dtype=float)
        if warm_bits.shape != tuple(bool_var.shape):
            raise ValueError(
                f"warm_bits shape {warm_bits.shape} does not match bool var "
                f"{tuple(bool_var.shape)}")
        # cvxpy stacks matrix variables column-major; align to that order.
        warm_flat = warm_bits.flatten(order="F")
    else:
        warm_flat = None

    scip_params: dict = {}
    if gap_limit is not None:
        scip_params["limits/gap"] = float(gap_limit)
    if time_limit is not None:
        scip_params["limits/time"] = float(time_limit)
    opts = {"scip_params": scip_params} if scip_params else {}

    data, chain, inv = prob.get_problem_data(cp.SCIP)
    solver = _WarmCutSCIP(warm_flat, objective_lower_bound)
    raw = solver.solve_via_data(data, False, verbose, dict(opts))
    prob.unpack(chain.invert(raw, inv))

    # Validity backstop for a too-high L. The objective is an epigraph variable
    # ``t`` (minimise ``t`` s.t. ``t >= quadratic``), so the cut is really
    # ``t >= L``: an L above the true optimum does not make the model infeasible,
    # it inflates ``t`` and returns a garbage design. The signature is that the
    # *recovered* design's true objective drops below the cut floor -- impossible
    # for a valid L, where every feasible objective is >= optimum >= L. When it
    # happens, drop the cut and re-solve so correctness always holds.
    fell_back = False
    if objective_lower_bound is not None:
        L = float(objective_lower_bound)
        val = None if prob.value is None else float(prob.value)
        if val is None or val < L - 1e-6 * max(1.0, abs(L)):
            solver = _WarmCutSCIP(warm_flat, None)
            raw = solver.solve_via_data(data, False, verbose, dict(opts))
            prob.unpack(chain.invert(raw, inv))
            fell_back = True

    # Read diagnostics from the SCIP model we built (version-robust), not from
    # cvxpy's solution dict whose keys differ across cvxpy releases.
    model = solver.model
    if model is not None:
        dual_bound = float(model.getDualbound())
        gap = float(model.getGap())
        status = str(model.getStatus())
        solve_time = float(model.getSolvingTime())
    else:                                   # pragma: no cover - model always built
        dual_bound = gap = solve_time = float("nan")
        status = "unknown"
    info = AccelInfo(
        status=status,
        dual_bound=dual_bound,
        gap=gap,
        solve_time=solve_time,
        cut_applied=solver.cut_applied,
        warm_applied=solver.warm_applied,
        fell_back=fell_back,
    )
    return prob, info
