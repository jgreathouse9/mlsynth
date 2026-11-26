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

Solved by HiGHS through ``highspy``, or through cvxpy with its first installed
mixed-integer backend (HiGHS by preference, else SCIP / GLPK_MI / CBC).

The MIP is solved to relative and absolute gaps of zero unless stated
otherwise. HiGHS's default relative gap of 1e-4 stops at any design within
0.01% of its dual bound and calls it optimal: on a 20-market panel with one
outlying market both backends returned a design costing 281,977,490 against the
optimum 281,971,264. A solve that does not end optimal is repeated once with
presolve off, since HiGHS's presolve has returned "Solve error" on a feasible
model the simplex solved without it (the supergeo study's fixture,
``presolve_failure_master.json``; Achterberg, Bixby, Gu, Rothberg and Weninger,
"Presolve Reductions in Mixed Integer Programming", INFORMS Journal on
Computing 32 (2020) 473-506, describe the reductions it runs). The status, the
retry, the gaps asked for and the proven gap are returned in the diagnostics.

The MIP does not leave infeasibility to the solver. A unit that no candidate
group holds has an exact-cover row with no variables, ``0 = 1``; cvxpy's SCIP
interface drops every row without variables whatever its right-hand side
(``scip_conif.add_model_lin_constr`` in cvxpy 1.7), and SCIP then reports the
rest of the program optimal, so the returned design leaves that unit out
(jgreathouse9/mlsynth#1169). Both backends therefore check the candidates
before any solve, raising when a unit is in no group (:func:`_require_coverable`),
and check the design after it, raising unless every unit of the arm is in
exactly one chosen group (:func:`_check_exact_cover`).
"""

from __future__ import annotations

import math
import time
from typing import List

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError


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


def _mip_solve_stats(problem) -> dict:
    """Solver optimality accounting from a solved cvxpy problem.

    Pulls the Gurobi-style fields HiGHS reports via ``solver_stats.extra_stats``
    -- optimality gap, best dual bound, branch-and-bound node count, and simplex
    iterations -- guarding each with ``None`` so the schema is stable across
    solvers that omit them.
    """
    stats = getattr(problem, "solver_stats", None)
    out = {
        "solver_name": getattr(stats, "solver_name", None),
        "mip_gap": None,
        "dual_bound": None,
        "node_count": None,
        "simplex_iterations": None,
    }
    extra = getattr(stats, "extra_stats", None)
    if extra is not None:
        mapping = {
            "mip_gap": "mip_gap",
            "dual_bound": "mip_dual_bound",
            "node_count": "mip_node_count",
            "simplex_iterations": "simplex_iteration_count",
        }
        for key, attr in mapping.items():
            val = getattr(extra, attr, None)
            if val is None:
                continue
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                out[key] = val
    return out


def _infeasibility_reason(candidate_pairs: List[dict],
                          unit_indices: np.ndarray) -> str:
    """IIS-style structural explanation of why an exact cover is infeasible.

    Reports the concrete obstruction rather than a generic "did not solve":
    units that appear in no admissible pair (structurally uncoverable), and the
    odd-arm/even-pair parity clash (e.g. ``Q=1`` on an odd arm, where every
    candidate pair has even size so no exact cover can exist).
    """
    covered = set()
    sizes = set()
    for pair in candidate_pairs:
        members = [int(u) for u in pair["members"]]
        covered.update(members)
        sizes.add(len(members))
    units = [int(u) for u in unit_indices]
    n = len(units)
    reasons = []
    uncoverable = [u for u in units if u not in covered]
    if uncoverable:
        shown = uncoverable[:10]
        more = "" if len(uncoverable) <= 10 else f" (+{len(uncoverable) - 10} more)"
        reasons.append(
            f"{len(uncoverable)} unit(s) appear in no admissible supergeo pair "
            f"and are structurally uncoverable: indices {shown}{more}")
    if sizes and n % 2 == 1 and all(s % 2 == 0 for s in sizes):
        reasons.append(
            f"the arm size {n} is odd but every admissible pair has even size "
            f"(min size {min(sizes)}), so no exact cover exists -- raise "
            f"max_supergeo_size to admit odd-size supergeos")
    if not reasons:
        reasons.append(
            "no exact cover exists for the current size bounds (try a larger "
            "max_supergeo_size, or fewer required min_pairs)")
    return "; ".join(reasons)


def _require_coverable(candidate_pairs: List[dict], unit_indices: np.ndarray) -> None:
    """Raise MlsynthEstimationError when a unit of the arm is in no candidate group: its exact-cover row has no
    variables, so no design exists, and a solver may drop such a row instead of reporting the program infeasible
    (module docstring). The message is :func:`_infeasibility_reason`'s."""
    held = {int(u) for pair in candidate_pairs for u in pair["members"]}
    if any(int(u) not in held for u in unit_indices):
        raise MlsynthEstimationError(
            f"PANGEO partition MIP has no exact cover: {_infeasibility_reason(candidate_pairs, unit_indices)}.")


def _check_exact_cover(chosen: List[dict], unit_indices: np.ndarray) -> None:
    """Raise MlsynthEstimationError unless the chosen groups hold every unit of the arm exactly once and no unit
    outside it: the design a solve returns is checked, not trusted."""
    count = {int(u): 0 for u in unit_indices}
    outside = set()
    for pair in chosen:
        for u in pair["members"]:
            if int(u) in count:
                count[int(u)] += 1
            else:
                outside.add(int(u))
    missing = sorted(u for u, c in count.items() if c == 0)
    twice = sorted(u for u, c in count.items() if c > 1)
    if missing or twice or outside:
        what = [f"units {lst[:10]} {why}" for lst, why in ((missing, "in no chosen group"),
                                                           (twice, "in more than one chosen group"),
                                                           (sorted(outside), "outside the arm")) if lst]
        raise MlsynthEstimationError(
            "PANGEO partition MIP returned a design that is not an exact cover of the arm: " + "; ".join(what)
            + ". The solver reported it as solved; the design is refused.")


def _check_gaps(mip_rel_gap, mip_abs_gap) -> None:
    """The gaps a MIP is solved to: a relative gap in [0, 1) and a finite
    absolute gap >= 0."""
    for name, v, top in (("mip_rel_gap", mip_rel_gap, 1.0), ("mip_abs_gap", mip_abs_gap, np.inf)):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
                or not np.isfinite(v) or v < 0 or v >= top:
            bound = "in [0, 1)" if top == 1.0 else "finite and >= 0"
            raise MlsynthConfigError(f"{name} must be {bound}, not {v!r}.")


def _cost_scale(cost) -> float:
    """The power of two that puts the largest cost in [2^29, 2^30), 1 when every
    cost is 0. HiGHS's feasibility and optimality tolerances are absolute, so
    the costs' units decide what it can tell apart: the default-gap instance's
    costs multiplied by 1e-12 gave a design 368 dearer than the optimum in the
    original units, with both gaps at 0, and the same costs scaled into this
    band gave the optimum at every scale from 1e-14 to 1e6. A power of two
    multiplies exactly, so the minimizing design is unchanged. The exponent is
    capped at 1023, the largest power of two: when the largest cost is below
    about 2^-993 it lands at top 2^1023, below the band (2^-51 at the smallest
    subnormal), instead of raising OverflowError (jgreathouse9/mlsynth#1170)."""
    top = float(np.max(np.abs(np.asarray(cost, dtype=float)))) if len(cost) else 0.0
    if top == 0.0 or not np.isfinite(top):
        return 1.0
    _, e = math.frexp(top)                    # top = m 2^e, 0.5 <= m < 1
    return math.ldexp(1.0, min(30 - e, 1023))  # 2^1023 is the largest power of two: below about 2^-993 the
                                              # largest cost lands at top 2^1023 < 2^30 (#1170)


def _run_highs(h):
    """Run a HiGHS model; when its status is not Optimal, run it once more with
    presolve off. Returns (status string, whether it was repeated)."""
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    if status == "Optimal":
        return status, False
    h.setOptionValue("presolve", "off")
    h.clearSolver()
    h.run()
    return h.modelStatusToString(h.getModelStatus()), True


def _cvxpy_options(solver_name: str, mip_rel_gap: float, mip_abs_gap: float,
                   presolve: bool = True):
    """The cvxpy solver options that set the MIP gaps, and presolve off for the
    retry, for the named solver. Returns (options, whether the gaps are set):
    HiGHS and SCIP take both gaps; GLPK_MI and CBC are not given them through
    cvxpy, and the diagnostics say so."""
    if solver_name == "HIGHS":
        opts = {"mip_rel_gap": float(mip_rel_gap), "mip_abs_gap": float(mip_abs_gap)}
        if not presolve:
            opts["presolve"] = "off"
        return opts, True
    if solver_name == "SCIP":
        params = {"limits/gap": float(mip_rel_gap), "limits/absgap": float(mip_abs_gap)}
        if not presolve:
            params["presolving/maxrounds"] = 0
        return {"scip_params": params}, True
    return {}, False


def _highspy_available() -> bool:
    """True if the standalone ``highspy`` HiGHS interface is importable."""
    try:
        import highspy  # noqa: F401
    except ImportError:
        return False
    return True


def solve_partition(
    candidate_pairs: List[dict],
    unit_indices: np.ndarray,
    min_pairs: int = 1,
    return_diagnostics: bool = False,
    backend: str = "auto",
    mip_rel_gap: float = 0.0,
    mip_abs_gap: float = 0.0,
):
    """Select the exact-cover set of supergeo pairs of minimum total score.

    ``backend="auto"`` dispatches to the standalone ``highspy`` HiGHS backend
    when available (dropping the cvxpy middleman for the MIP), otherwise to the
    cvxpy MIP; ``"highspy"`` and ``"cvxpy"`` ask for one. Both solve the
    identical exact-cover program and return the same optimum within the
    solver's tolerances; see :func:`_solve_partition_highspy` /
    :func:`_solve_partition_cvxpy`. The diagnostics record the backend used.

    ``mip_rel_gap`` and ``mip_abs_gap`` are the gaps the solver must close, 0
    by default (the optimum). The diagnostics add ``status``,
    ``presolve_retry``, ``mip_rel_gap_requested``, ``mip_abs_gap_requested``
    and ``gap_enforced`` to the solver's own ``mip_gap`` and ``dual_bound``.
    """
    _check_gaps(mip_rel_gap, mip_abs_gap)
    if backend not in ("auto", "highspy", "cvxpy"):
        raise MlsynthConfigError(
            f"mip_backend must be 'auto', 'highspy' or 'cvxpy', not {backend!r}.")
    if backend == "highspy" and not _highspy_available():
        raise MlsynthConfigError(
            "mip_backend='highspy' needs the highspy package, which is not "
            "installed; use 'cvxpy' or 'auto'.")
    use = "highspy" if backend == "highspy" or (
        backend == "auto" and _highspy_available()) else "cvxpy"
    solve = _solve_partition_highspy if use == "highspy" else _solve_partition_cvxpy
    out = solve(candidate_pairs, unit_indices, min_pairs, return_diagnostics,
                mip_rel_gap=float(mip_rel_gap), mip_abs_gap=float(mip_abs_gap))
    if return_diagnostics:
        chosen, diagnostics = out
        return chosen, {**diagnostics, "backend": use}
    return out


def _solve_partition_highspy(
    candidate_pairs: List[dict],
    unit_indices: np.ndarray,
    min_pairs: int = 1,
    return_diagnostics: bool = False,
    mip_rel_gap: float = 0.0,
    mip_abs_gap: float = 0.0,
):
    """Exact-cover partition MIP solved with ``highspy`` (HiGHS) directly.

    Builds the same program as :func:`_solve_partition_cvxpy` -- one boolean
    variable per candidate pair, an exact-cover equality per unit, an optional
    ``sum(x) >= min_pairs``, minimise total score -- but hands it to HiGHS via
    ``highspy`` with no cvxpy canonicalisation. Returns the chosen pairs (or
    ``(chosen, diagnostics)``), matching the cvxpy backend's contract.
    """
    import highspy

    n_units = len(unit_indices)
    n_cand = len(candidate_pairs)
    if n_cand == 0:
        raise MlsynthEstimationError(
            "PANGEO: no admissible supergeo pairs (need >= 2 units per arm)."
        )

    _require_coverable(candidate_pairs, unit_indices)
    pos = {int(u): j for j, u in enumerate(unit_indices)}
    cost = [float(p["score"]) for p in candidate_pairs]
    scale = _cost_scale(cost)
    # Candidates covering each unit position.
    cover: List[List[int]] = [[] for _ in range(n_units)]
    for c, pair in enumerate(candidate_pairs):
        for u in pair["members"]:
            cover[pos[int(u)]].append(c)

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", float(mip_rel_gap))
    h.setOptionValue("mip_abs_gap", float(mip_abs_gap) * scale)   # in the scaled units
    x = h.addBinaries(n_cand)
    for i in range(n_units):                            # every unit is in a group: _require_coverable
        h.addConstr(sum(x[c] for c in cover[i]) == 1)    # exact cover
    if min_pairs > 1:
        h.addConstr(sum(x[c] for c in range(n_cand)) >= min_pairs)

    h.changeColsCost(n_cand, np.arange(n_cand, dtype=np.int32), np.asarray(cost, dtype=np.float64) * scale)
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    t0 = time.perf_counter()
    status, retried = _run_highs(h)
    solve_seconds = time.perf_counter() - t0

    if status != "Optimal":
        raise MlsynthEstimationError(
            f"PANGEO partition MIP did not solve (status={status}"
            f"{', after a retry with presolve off' if retried else ''}); "
            f"{_infeasibility_reason(candidate_pairs, unit_indices)}."
        )

    values = np.asarray(h.getSolution().col_value, dtype=float)
    chosen = [candidate_pairs[c] for c in range(n_cand) if values[c] > 0.5]
    _check_exact_cover(chosen, unit_indices)
    if not return_diagnostics:
        return chosen

    info = h.getInfo()

    def _f(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    diagnostics = {
        "path": "exact_mip",
        "n_candidate_pairs": int(n_cand),
        "objective_value": float(sum(p["score"] for p in chosen)),
        "status": "optimal",                          # kOptimal -> cvxpy's label
        "solve_seconds": float(solve_seconds),
        "n_selected_pairs": len(chosen),
        "presolve_retry": bool(retried),
        "mip_rel_gap_requested": float(mip_rel_gap),
        "mip_abs_gap_requested": float(mip_abs_gap),
        "gap_enforced": True,
        "solver_name": "HiGHS",
        "mip_gap": _f(getattr(info, "mip_gap", None)),
        "dual_bound": (None if getattr(info, "mip_dual_bound", None) is None
                       else float(info.mip_dual_bound) / scale),
        "node_count": _f(getattr(info, "mip_node_count", None)),
        "simplex_iterations": _f(getattr(info, "simplex_iteration_count", None)),
    }
    return chosen, diagnostics


def _solve_partition_cvxpy(
    candidate_pairs: List[dict],
    unit_indices: np.ndarray,
    min_pairs: int = 1,
    return_diagnostics: bool = False,
    mip_rel_gap: float = 0.0,
    mip_abs_gap: float = 0.0,
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

    _require_coverable(candidate_pairs, unit_indices)
    pos = {int(u): j for j, u in enumerate(unit_indices)}
    # Incidence matrix M (units x candidates): M[i, c] = 1 iff unit i in pair c.
    M = np.zeros((n_units, n_cand), dtype=float)
    for c, pair in enumerate(candidate_pairs):
        for u in pair["members"]:
            M[pos[int(u)], c] = 1.0
    cost = np.array([p["score"] for p in candidate_pairs], dtype=float)
    scale = _cost_scale(cost)

    x = cp.Variable(n_cand, boolean=True)
    constraints = [M @ x == 1]                      # exact cover
    if min_pairs > 1:
        constraints.append(cp.sum(x) >= min_pairs)
    problem = cp.Problem(cp.Minimize((cost * scale) @ x), constraints)
    solver = _pick_milp_solver()
    opts, enforced = _cvxpy_options(str(solver), mip_rel_gap, mip_abs_gap * scale, presolve=True)
    retried = False
    t0 = time.perf_counter()
    try:
        problem.solve(solver=solver, **opts)
        if problem.status != "optimal":
            retried = True
            opts, _ = _cvxpy_options(str(solver), mip_rel_gap, mip_abs_gap * scale, presolve=False)
            problem.solve(solver=solver, **opts)
    except cp.error.SolverError as exc:
        raise MlsynthEstimationError(
            f"PANGEO partition MIP failed: {exc}"
        ) from exc
    solve_seconds = time.perf_counter() - t0
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise MlsynthEstimationError(
            f"PANGEO partition MIP did not solve (status={problem.status}"
            f"{', after a retry with presolve off' if retried else ''}); "
            f"{_infeasibility_reason(candidate_pairs, unit_indices)}."
        )

    chosen = [candidate_pairs[c] for c in range(n_cand)
              if x.value is not None and x.value[c] > 0.5]
    _check_exact_cover(chosen, unit_indices)
    if return_diagnostics:
        stats = _mip_solve_stats(problem)
        diagnostics = {
            "path": "exact_mip",
            "n_candidate_pairs": int(n_cand),
            "objective_value": float(sum(p["score"] for p in chosen)),
            "status": problem.status,
            "solve_seconds": float(solve_seconds),
            "n_selected_pairs": len(chosen),
            "presolve_retry": retried,
            "mip_rel_gap_requested": float(mip_rel_gap),
            "mip_abs_gap_requested": float(mip_abs_gap),
            "gap_enforced": enforced,
            **stats,
        }
        if diagnostics.get("dual_bound") is not None:
            diagnostics["dual_bound"] = float(diagnostics["dual_bound"]) / scale
        return chosen, diagnostics
    return chosen
