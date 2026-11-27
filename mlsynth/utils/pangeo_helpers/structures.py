"""Frozen dataclasses for the PANGEO design estimator.

PANGEO is a *prospective experimental design* method: from historical
(pre-treatment) sales it partitions each treatment arm's geos into
supergeo pairs whose treatment/control halves are maximally parallel over
the pre-period, so a later difference-in-differences / synthetic-control
analysis has clean parallel trends. The output is a **design**
(supergeo pairs + treatment/control assignment + achieved parallelism),
not a treatment effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field

from ...config_models import DesignResult


class PangeoExcludedUnitWarning(UserWarning):
    """A market left out of the design (an odd arm's pairs on the exact path)."""


class PangeoCertifiedGapWarning(UserWarning):
    """An arm's design returned without a proof of optimality, with the gap its node budget certified (the exact
    path's ``certify_node_limit``; the arm's ``mip_reports`` entry has the design's bound and the gap)."""


@dataclass(frozen=True)
class SupergeoPair:
    """One supergeo pair within an arm.

    Attributes
    ----------
    treatment : list
        Unit names assigned to treatment in this pair.
    control : list
        Unit names assigned to control in this pair.
    gap_variance : float
        Pre-period level-removed gap variance between the two halves
        (lower = more parallel; the DiD pre-period residual SS).
    parallelism_r2 : float
        R^2 of the within-pair parallel-trends fit (1 = perfectly parallel).
    treatment_mean : np.ndarray
        Pre-period mean trajectory of the treatment half.
    control_mean : np.ndarray
        Pre-period mean trajectory of the control half.
    covariate_smd : dict
        ``{covariate_name: standardized mean difference}`` between the
        treatment and control halves (empty if no covariates were used).
    gap_level : float
        DiD counterfactual gap level :math:`\\delta` -- the mean gap over the
        *estimation* window E (the periods the split was optimised on).
    holdout_resid : np.ndarray
        Gap residuals on the held-out blank window B (``gap[B] - gap_level``).
        B is excluded from the optimisation, so these residuals are an honest
        out-of-sample estimate of the parallel-trends noise -- the reservoir
        for conformal inference and the variance behind the MDE.
    first_half_treated : bool
        The pair's coin: True when the half holding the pair's first unit (the
        lowest index in the panel) is the treated half. Drawn with probability
        one half, independently across pairs, from the design's
        ``assignment_seed`` (Chen et al. 2023, arXiv:2301.12044, section 3).
    """

    treatment: List[Any]
    control: List[Any]
    gap_variance: float
    parallelism_r2: float
    treatment_mean: np.ndarray
    control_mean: np.ndarray
    covariate_smd: Dict[str, float] = field(default_factory=dict)
    gap_level: float = 0.0
    holdout_resid: np.ndarray = field(default_factory=lambda: np.empty(0))
    first_half_treated: bool = True


@dataclass(frozen=True)
class ArmDesign:
    """The supergeo-pair design for a single treatment arm.

    Attributes
    ----------
    arm : Any
        Arm label.
    pairs : list of SupergeoPair
        The chosen supergeo pairs partitioning the arm's units.
    n_units : int
        Number of geos eligible for the arm.
    total_gap_variance : float
        Sum of the pairs' gap variances -- the design objective value.
    mean_parallelism_r2 : float
        Mean within-pair parallel-trends R^2 across pairs.
    treatment_units : list
        All units assigned to treatment in the arm.
    control_units : list
        All units assigned to control in the arm.
    excluded_units : list
        Units left out of the design: with pairs on the exact path, the one
        market an odd arm leaves out so that the rest pair at the least cost
        (``pairs.solve_pairs``). Empty otherwise.
    """

    arm: Any
    pairs: List[SupergeoPair]
    n_units: int
    total_gap_variance: float
    mean_parallelism_r2: float
    treatment_units: List[Any]
    control_units: List[Any]
    excluded_units: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class MipSolveReport:
    """How one arm's exact-cover MIP was solved.

    HiGHS stops a MIP at its tolerance and reports "optimal" there, so the
    design alone does not say how close to the optimum it is. The report
    carries what the solver proved: the gaps asked for, whether the solver
    could be given them, the status, whether the solve was repeated with
    presolve off, and the proven gap and dual bound.

    Attributes
    ----------
    backend : str
        ``"highspy"`` or ``"cvxpy"``.
    solver_name : str or None
        The MIP solver that ran (HiGHS, SCIP, ...).
    status : str
        The final status: ``"optimal"`` or ``"optimal_inaccurate"`` from a
        MIP or a finished search, or ``"certified_gap"`` when the exact
        path's node budget ran out first and the design comes with the gap
        it certified (``search`` is then ``"certified"``). Other statuses
        raise.
    presolve_retry : bool
        True when the first solve was not optimal and was repeated once with
        presolve off.
    requested_rel_gap, requested_abs_gap : float
        The relative and absolute gaps the solver was asked to close.
    gap_enforced : bool
        False when the solver cannot be given the gaps through cvxpy
        (GLPK_MI, CBC); the design is then optimal only to that solver's own
        tolerance.
    proven_gap : float or None
        The relative gap between the design's cost and the solver's dual
        bound at the end, when the solver reports it.
    dual_bound : float or None
        The solver's lower bound on the optimum, when reported.
    objective_value : float
        The design's total cost.
    node_count : float or None
        Branch-and-bound nodes, when reported.
    solve_seconds : float
        Wall time of the solve, the retry included; with groups left out by
        the root bound, the total over the MIPs solved.
    columns_scored : int or None
        Candidate groups scored, every group of 2 to 2Q units, when the root
        bound left groups out of the MIP (:mod:`.columns`); None otherwise.
    columns_solved : int or None
        Candidate groups in the last MIP: those whose reduced cost the
        Lagrangian bound could not rule out, and a fixed design's groups.
    lagrangian_bound : float or None
        The root LP's Lagrangian bound, a lower bound on the optimum.
    search : str or None
        The exact-cover search before the MIP (:mod:`.cover`): ``"proved"``
        when it proved the design optimal and no MIP ran (``backend`` is then
        ``"search"``, ``proven_gap`` its certified gap and ``dual_bound`` a
        rigorous lower bound on the optimum), ``"node_limit"`` when its node
        cap stopped it and the MIP took over, ``"certified"`` when the proof's
        cap and then the certify budget stopped it and the design is returned
        with its certified gap (``proven_gap``, ``dual_bound``; no MIP ran),
        ``"off"`` or ``"unavailable"`` (no numba); None when the design did not
        come from :mod:`.columns`.
    search_nodes : int or None
        Groups the search placed.
    """

    backend: str
    solver_name: Optional[str]
    status: str
    presolve_retry: bool
    requested_rel_gap: float
    requested_abs_gap: float
    gap_enforced: bool
    proven_gap: Optional[float]
    dual_bound: Optional[float]
    objective_value: float
    node_count: Optional[float]
    solve_seconds: float
    columns_scored: Optional[int] = None
    columns_solved: Optional[int] = None
    lagrangian_bound: Optional[float] = None
    search: Optional[str] = None
    search_nodes: Optional[int] = None

    @classmethod
    def from_diagnostics(cls, d: Dict[str, Any]) -> "MipSolveReport":
        """From :func:`.mip.solve_partition`'s diagnostics."""
        return cls(
            backend=d["backend"], solver_name=d.get("solver_name"),
            status=d["status"], presolve_retry=bool(d["presolve_retry"]),
            requested_rel_gap=float(d["mip_rel_gap_requested"]),
            requested_abs_gap=float(d["mip_abs_gap_requested"]),
            gap_enforced=bool(d["gap_enforced"]), proven_gap=d.get("mip_gap"),
            dual_bound=d.get("dual_bound"),
            objective_value=float(d["objective_value"]),
            node_count=d.get("node_count"),
            solve_seconds=float(d["solve_seconds"]),
            columns_scored=d.get("columns_scored"),
            columns_solved=d.get("columns_solved"),
            lagrangian_bound=d.get("lagrangian_bound"),
            search=d.get("search"),
            search_nodes=d.get("search_nodes"),
        )


class PangeoResults(DesignResult):
    """Top-level container returned by :meth:`mlsynth.PANGEO.fit`.

    A :class:`~mlsynth.config_models.DesignResult` (the experimental-design
    family): it chooses the treatment/control assignment *before* any
    intervention, and -- once post-period outcomes exist -- resolves to an
    :class:`~mlsynth.config_models.EffectResult` via :attr:`report`.

    Inherited design fields
    -----------------------
    report : EffectResult or None
        The realized effect report (program-level Augmented-DiD ATT mapped to
        the standard effect surface); ``None`` for a design-only fit.
    assignment : dict
        Flat ``{unit_name: "treatment"|"control"}`` map across all arms.
    selected_units : list
        Units assigned to treatment (the design's chosen treated set).
    power : PangeoPower or None
        Program- and arm-level MDE / power analysis.
    metadata : dict
        Free-form design diagnostics (solver, q_sweep, solver_diagnostics, ...).

    PANGEO-specific fields
    ----------------------
    arm_designs : dict
        ``{arm_label: ArmDesign}`` -- the supergeo-pair design per arm.
    max_supergeo_size : int
        The Q used (max size of either supergeo within a pair).
    time_labels : np.ndarray
        Pre-period time labels the design was built on.
    effects : PangeoEffects or None
        The rich realized-ATT object (per-pair, per-arm, program, plus the
        design-based randomization inference) when post-period data is given.
    assignment_seed : int
        The seed of the coins that labelled each pair's halves treatment and
        control. A run with the same seed on the same pre-period panel labels
        the pairs the same way, which is how an analysis run (``post_col``)
        reads the assignment the experiment ran.
    excluded_units : list
        Every arm's ``ArmDesign.excluded_units``, in arm order: the markets in
        no pair, absent from ``assignment``.
    mip_reports : dict
        ``{arm_label: MipSolveReport}`` for every arm designed by the
        exact-cover MIP; empty when no arm was (the fast path, the pairs
        path).
    """

    arm_designs: Dict[Any, ArmDesign] = Field(default_factory=dict)
    max_supergeo_size: Optional[int] = None
    time_labels: Optional[np.ndarray] = None
    effects: Optional[Any] = None  # PangeoEffects when post-period data given
    assignment_seed: Optional[int] = None
    excluded_units: List[Any] = Field(default_factory=list)
    mip_reports: Dict[Any, MipSolveReport] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        frozen = True  # preserve the immutable-result contract (use model_copy)
