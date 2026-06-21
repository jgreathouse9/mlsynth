"""Frozen dataclasses for the SPILLSYNTH estimator.

SPILLSYNTH bundles synthetic-control estimators that explicitly model
spillover (interference) on potentially-affected control units, behind
a single ``method`` dispatcher.

Current methods
---------------
* **cd** (Cao & Dowd 2023) -- Ferman-Pinto demeaned SCM weights for
  every unit, stacked into a leave-one-out weight matrix ``B``; the
  treatment-and-spillover effect vector is recovered as
  ``alpha = A * (A' M A)^{-1} A' (I - B)' [(I - B) Y_{T+1} - a]`` where
  ``A`` encodes the user's spillover structure (Example 3 of the paper).

The four layers below (inputs, per-method fit, spillover panel,
top-level results) keep the pipeline pluggable as additional methods
are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .cd.inference import KappaATestResult, PTestResult
    from .cd.sensitivity import PureDonorSensitivity
    from .sar.structures import SARFit


@dataclass(frozen=True)
class SpillSynthInputs:
    """Preprocessed panel for SPILLSYNTH.

    Parameters
    ----------
    Y : np.ndarray
        Outcome panel of shape ``(N, T)`` with row 0 the treated unit,
        rows ``1 .. p`` the affected (spillover) units, and rows
        ``p+1 .. N-1`` the clean controls. Ordering is enforced by
        :func:`prepare_spillsynth_inputs`.
    Y_pre : np.ndarray
        Pre-treatment slice ``Y[:, :T0]``.
    Y_post : np.ndarray
        Post-treatment slice ``Y[:, T0:]``.
    A : np.ndarray
        Spillover-structure matrix of shape ``(N, 1 + p)``. Column 0 is
        a unit basis vector for the treated unit, columns ``1 .. p``
        are basis vectors for each affected control unit.
    treated_label : Any
        Label of the treated unit.
    affected_labels : Tuple[Any, ...]
        Labels of the ``p`` potentially-affected control units, in row
        order of ``Y``.
    clean_labels : Tuple[Any, ...]
        Labels of the unaffected control units, in row order of ``Y``.
    time_labels : np.ndarray
        Length-``T`` time labels in pre/post order.
    pre_time : np.ndarray
        Length-``T0`` pre-period time labels.
    post_time : np.ndarray
        Length-``T1`` post-period time labels.
    N : int
        Total number of units.
    T : int
        Total number of periods.
    T0 : int
        Number of pre-treatment periods.
    T1 : int
        Number of post-treatment periods.
    p : int
        Number of potentially-affected control units. ``A`` has
        ``1 + p`` columns under ``"per_unit"`` (Cao-Dowd v3 Example 1),
        or ``2`` columns under ``"homogeneous"`` (v3 Example 2) and
        ``"distance_decay"`` (v3 Example 3).
    spillover_structure : str
        One of ``"per_unit"`` (Cao-Dowd v3 Example 1, leading case),
        ``"homogeneous"`` (v3 Example 2), ``"distance_decay"``
        (v3 Example 3). Selects which A-matrix construction was used.
    """

    Y: np.ndarray
    Y_pre: np.ndarray
    Y_post: np.ndarray
    A: np.ndarray
    treated_label: Any
    affected_labels: Tuple[Any, ...]
    clean_labels: Tuple[Any, ...]
    time_labels: np.ndarray
    pre_time: np.ndarray
    post_time: np.ndarray
    N: int
    T: int
    T0: int
    T1: int
    p: int
    spillover_structure: str = "per_unit"
    treated_labels: Tuple[Any, ...] = ()
    n_treated: int = 1
    # Optional per-unit predictor block (pre-period covariate aggregates),
    # shape ``(N, K)`` in the same row order as ``Y``. Used by the ``iscm``
    # method for covariate matching; ``None`` for outcome-only matching.
    predictors: Optional[np.ndarray] = None
    predictor_names: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class CDFit:
    """Cao-Dowd per-method fit artifacts.

    Parameters
    ----------
    a : np.ndarray
        Length-``N`` vector of estimated intercepts ``a_i`` from each
        leave-one-out demeaned SCM fit (eq. 1 of Cao-Dowd 2023).
    B : np.ndarray
        Shape-``(N, N)`` matrix of leave-one-out SCM weights. ``B[i, :]``
        is the convex donor weight vector for the leave-one-out fit of
        unit ``i`` (and ``B[i, i] == 0``).
    M : np.ndarray
        Shape-``(N, N)`` Gram matrix ``(I - B)' (I - B)`` plus a tiny
        ridge for numerical stability.
    gamma : np.ndarray
        Shape-``(1 + p, T1)`` matrix of per-period parameter estimates
        ``gamma_hat`` (eq. 5 of Cao-Dowd 2023). The first row is the
        treatment effect on the treated unit; the remaining ``p`` rows
        are spillover effects for the affected control units.
    alpha : np.ndarray
        Shape-``(N, T1)`` matrix of per-period effect estimates
        ``alpha_hat = A @ gamma_hat``. Row 0 is the treated unit's
        spillover-adjusted ATT; rows ``1 .. p`` are the affected units'
        spillover paths; rows ``p+1 ..`` are identically zero.
    counterfactual_sp : np.ndarray
        Length-``T1`` post-period counterfactual for the treated unit
        under the spillover-adjusted model: ``Y_treated_post - alpha[0]``.
    counterfactual_scm : np.ndarray
        Length-``T1`` post-period counterfactual under vanilla SCM
        (treated unit's leave-one-out fit, ignoring spillover).
    gap_sp : np.ndarray
        Length-``T1`` per-period treatment effects from the SP path
        (``Y_treated_post - counterfactual_sp``, equal to ``alpha[0]``).
    gap_scm : np.ndarray
        Length-``T1`` per-period treatment effects from the vanilla
        SCM path.
    att_sp : float
        Mean of ``gap_sp`` over the post-period (spillover-adjusted
        ATT).
    att_scm : float
        Mean of ``gap_scm`` over the post-period (vanilla SCM ATT).
    spillover_panel : Dict[Any, np.ndarray]
        Mapping from each affected unit's label to its length-``T1``
        spillover trajectory ``alpha_k(t)``. Convenience accessor for
        plotting and downstream analysis.
    cond_AMA : float
        Condition number of ``A' M A`` (the matrix the per-period
        formula inverts). Diagnostic for the Assumption 1(d)
        invertibility requirement.
    treatment_test : Optional[PTestResult]
        Cao-Dowd Section 4.2 P-test for ``H_0: alpha_1(t) = 0`` at each
        post-period, using selector :math:`C = e_1^\\prime` and weight
        :math:`W_T = I`.
    spillover_tests : Dict[Any, PTestResult]
        Per-affected-unit Cao-Dowd P-test for ``H_0: alpha_k(t) = 0``
        at each post-period. Keyed by affected-unit label.
    treatment_ci_95 : Optional[np.ndarray]
        Shape ``(T1, 2)``. 95% confidence interval on the treated
        unit's treatment effect in each post-period, obtained by
        inverting the level-5% P-test (Cao-Dowd Section 6.2 / R
        reference).
    spillover_ci_95 : Dict[Any, np.ndarray]
        Per-affected-unit 95% confidence interval on the spillover
        effect in each post-period.
    joint_spillover_test : Optional[PTestResult]
        Cao-Dowd MATLAB-reference *joint* spillover test (single
        rejection per post-period) with selector
        :math:`C = [0_{p \\times 1} \\mid I_p \\mid 0]`. ``None`` when
        no affected units are declared (``p == 0``).
    kappa_A_test : Optional[KappaATestResult]
        Cao-Dowd v3 Section 5.1.2 specification test for the chosen
        A-matrix.
    pure_donor_sensitivity : Optional[PureDonorSensitivity]
        Cao-Dowd v3 Section 5.2 worst-case misspecification-bias
        weights for SP vs the pure-donor SCM, as a function of the
        number ``p`` of missed spillovers. ``None`` when ``A`` declares
        every unit affected (no clean controls to bound bias against).
    efficient_fit : Optional[Dict[str, np.ndarray]]
        When the user requests ``weighting='efficient'``, this holds
        the GMM-weighted estimator artefacts (Cao-Dowd v3 Section
        S.1.1, Proposition S.1): keys are ``gamma_W``, ``alpha_W``,
        ``W`` (the weighting matrix used, typically the inverse of the
        sample residual covariance), and ``cond_AMA_W``. Otherwise
        ``None``.
    """

    a: np.ndarray
    B: np.ndarray
    M: np.ndarray
    gamma: np.ndarray
    alpha: np.ndarray
    counterfactual_sp: np.ndarray
    counterfactual_scm: np.ndarray
    gap_sp: np.ndarray
    gap_scm: np.ndarray
    att_sp: float
    att_scm: float
    spillover_panel: Dict[Any, np.ndarray]
    cond_AMA: float
    treatment_test: Optional["PTestResult"] = None
    spillover_tests: Dict[Any, "PTestResult"] = field(default_factory=dict)
    treatment_ci_95: Optional[np.ndarray] = None
    spillover_ci_95: Dict[Any, np.ndarray] = field(default_factory=dict)
    joint_spillover_test: Optional["PTestResult"] = None
    kappa_A_test: Optional["KappaATestResult"] = None
    pure_donor_sensitivity: Optional["PureDonorSensitivity"] = None
    efficient_fit: Optional[Dict[str, Any]] = None
    # Multi-treated extensions (Cao-Dowd v3 Section S.1.2). Keyed by
    # treated-unit label. For the single-treated case (the default),
    # each dict has exactly one entry corresponding to the sole treated
    # unit.
    gaps_sp_by_unit: Dict[Any, np.ndarray] = field(default_factory=dict)
    gaps_scm_by_unit: Dict[Any, np.ndarray] = field(default_factory=dict)
    atts_sp_by_unit: Dict[Any, float] = field(default_factory=dict)
    atts_scm_by_unit: Dict[Any, float] = field(default_factory=dict)
    treatment_tests: Dict[Any, "PTestResult"] = field(default_factory=dict)
    treatment_cis_95: Dict[Any, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class ISCMFit:
    """Inclusive SCM (Di Stefano & Mellace 2024) per-method fit artifacts.

    A synthetic control is built for the treated unit *and* each affected
    unit, every fit keeping the other affected units in its donor pool. The
    weight each affected unit receives in another's synthetic control forms
    the cross-weight system ``Omega`` (eq. 6); inverting it de-contaminates
    the observed gaps into the true treated-unit effect and the spillover
    effects on the affected units.

    Parameters
    ----------
    att : float
        Inclusive (de-contaminated) ATT on the treated unit, post-period mean.
    att_scm : float
        Naive SCM ATT for the treated unit (affected units left in the donor
        pool without correction) -- the contaminated comparison.
    gap : np.ndarray
        Per-period inclusive treatment effect on the treated unit, shape
        ``(T1,)`` (row 0 of ``theta``).
    gap_scm : np.ndarray
        Per-period naive SCM gap on the treated unit, shape ``(T1,)``.
    counterfactual : np.ndarray
        Inclusive post-period counterfactual for the treated unit, shape
        ``(T1,)`` (observed minus inclusive effect).
    counterfactual_scm : np.ndarray
        Naive SCM post-period counterfactual for the treated unit.
    theta : np.ndarray
        De-contaminated effects for the affected set, shape ``(m, T1)`` with
        ``m = 1 + p``; row 0 is the treated unit, rows ``1 .. p`` the
        affected units (aligned with ``affected_labels``).
    omega : np.ndarray
        ``(m, m)`` cross-weight system matrix (unit diagonal, ``-w`` off).
    omega_det : float
        Determinant of ``omega``.
    cross_weights : Dict[str, float]
        Human-readable cross-weights between treated and affected units.
    weight_matrix : np.ndarray
        ``(m, N)`` donor-weight rows for the affected set (each unit's
        synthetic control over all other units).
    donor_weights : Dict[Any, float]
        Treated unit's synthetic-control donor weights (label -> weight).
    pre_rmspe : float
        Treated unit's pre-treatment RMSPE (inclusive donor pool).
    pre_rmspe_restricted : float
        Treated unit's pre-treatment RMSPE with the affected units *excluded*
        from the donor pool (the cost of the conventional exclude strategy).
    spillover_panel : Dict[Any, np.ndarray]
        Per-affected-unit de-contaminated spillover trajectory ``theta_k(t)``.
    spillover_att : Dict[Any, float]
        Per-affected-unit post-period mean spillover effect.
    bilevel_solver : str
        Bilevel backend used for covariate matching (``"malo"`` / ``"mscmt"``)
        or ``"outcome-only"`` when no covariates were supplied.
    predictor_weights : Optional[Dict[Any, float]]
        Treated unit's optimized predictor weights ``V`` (covariate mode).
    """

    att: float
    att_scm: float
    gap: np.ndarray
    gap_scm: np.ndarray
    counterfactual: np.ndarray
    counterfactual_scm: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    omega_det: float
    cross_weights: Dict[str, float]
    weight_matrix: np.ndarray
    donor_weights: Dict[Any, float]
    pre_rmspe: float
    pre_rmspe_restricted: float
    spillover_panel: Dict[Any, np.ndarray] = field(default_factory=dict)
    spillover_att: Dict[Any, float] = field(default_factory=dict)
    bilevel_solver: str = "outcome-only"
    predictor_weights: Optional[Dict[Any, float]] = None
    # Treated unit's pre-period synthetic path, shape ``(T0,)``. The shared
    # plotter needs the pre-period synthetic to draw the counterfactual; the
    # inclusive method has no Cao-Dowd ``a``/``B`` artifacts, so it is stored
    # directly.
    treated_synthetic_pre: Optional[np.ndarray] = None

    # SP-dialect aliases so the shared plotter (written for the Cao-Dowd
    # "spillover-adjusted" naming) consumes an ISCMFit unchanged. The
    # inclusive effect IS the spillover-adjusted effect.
    @property
    def att_sp(self) -> float:
        return self.att

    @property
    def gap_sp(self) -> np.ndarray:
        return self.gap

    @property
    def counterfactual_sp(self) -> np.ndarray:
        return self.counterfactual


@dataclass(frozen=True)
class GrossiFit:
    """Grossi et al. (2025) direct + spillover effects under partial interference.

    Under partial interference (interference only within the treated unit's
    cluster/neighbourhood), the treated unit *and* its potentially-affected
    cluster-mates are each given a penalized synthetic control built from the
    **far/clean controls only** (units in other clusters). The treated unit's
    gap is the direct effect (eq. 3.4); each cluster-mate's gap is a spillover
    effect, averaged into the average spillover (eq. 3.5).

    Attributes
    ----------
    direct_att : float
        Direct ATT on the treated unit (post-period mean).
    att_scm : float
        Naive SCM ATT using **all** controls (incl. the affected units) -- the
        contaminated comparison the partial-interference design avoids.
    gap : np.ndarray
        Per-period direct effect on the treated unit, shape ``(T1,)``.
    gap_scm : np.ndarray
        Per-period naive SCM gap (all controls).
    counterfactual, counterfactual_scm : np.ndarray
        Post-period counterfactuals (partial-interference and naive).
    avg_spillover_att : float
        Average spillover ATT across the cluster-mates (post-period mean).
    avg_spillover_gap : np.ndarray
        Per-period average spillover effect, shape ``(T1,)``.
    spillover_panel : dict
        Per-affected-unit spillover trajectory, shape ``(T1,)`` each.
    spillover_att : dict
        Per-affected-unit post-period mean spillover.
    direct_pre_rmspe : float
        Treated unit's pre-treatment RMSPE (far-control synthetic).
    donor_weights : dict
        Treated unit's synthetic-control weights over the clean controls.
    treated_synthetic_pre : np.ndarray
        Treated unit's pre-period synthetic path, shape ``(T0,)``.
    n_clean : int
        Number of clean (far) control units used as donors.
    lam : float
        Penalty selected for the treated unit's penalized SC.
    direct_ci, avg_spillover_ci : np.ndarray, optional
        ``(T1, 2)`` pivotal bias-corrected CIs from residual resampling
        (eq. 3.6-3.7); ``None`` when ``n_boot=0``.
    bilevel_solver : str
        Backend used (``"penalized"`` by default).
    """

    direct_att: float
    att_scm: float
    gap: np.ndarray
    gap_scm: np.ndarray
    counterfactual: np.ndarray
    counterfactual_scm: np.ndarray
    avg_spillover_att: float
    avg_spillover_gap: np.ndarray
    spillover_panel: Dict[Any, np.ndarray]
    spillover_att: Dict[Any, float]
    direct_pre_rmspe: float
    donor_weights: Dict[Any, float]
    treated_synthetic_pre: np.ndarray
    n_clean: int
    lam: float
    direct_ci: Optional[np.ndarray] = None
    avg_spillover_ci: Optional[np.ndarray] = None
    bilevel_solver: str = "penalized"

    # SP-dialect aliases for the shared plotter.
    @property
    def att_sp(self) -> float:
        return self.direct_att

    @property
    def gap_sp(self) -> np.ndarray:
        return self.gap

    @property
    def counterfactual_sp(self) -> np.ndarray:
        return self.counterfactual


@dataclass(frozen=True)
class IterativeFit:
    """Melnychuk (2024) Iterative ("waterfall") SCM.

    Each spillover-affected control is **cleaned** in turn: a synthetic control
    is built for it from the clean controls (and already-cleaned affected units,
    excluding the treated unit), and its **post-treatment** outcomes are replaced
    by that spillover-free synthetic (its pre-treatment outcomes are kept). The
    treated unit's synthetic control is then refit on the cleaned donor pool, so
    the affected donors no longer carry the treatment's spillover. Because the
    pre-period outcomes are untouched, the refit weights equal the naive ones --
    the correction enters only through the cleaned post-period counterfactual.

    Attributes
    ----------
    att : float
        Iterative ATT on the treated unit (post-period mean), cleaned pool.
    att_scm : float
        Naive SCM ATT (same weights, original contaminated donor outcomes).
    gap, gap_scm : np.ndarray
        Per-period iterative / naive treatment effect, shape ``(T1,)``.
    counterfactual, counterfactual_scm : np.ndarray
        Post-period counterfactuals (cleaned / naive).
    spillover_panel : dict
        Per-affected-unit cleaned synthetic post-period trajectory ``(T1,)``.
    spillover_att : dict
        Per-affected-unit post-period mean of (observed - cleaned synthetic),
        i.e. the spillover removed from that donor.
    donor_weights : dict
        Treated unit's synthetic-control weights over the cleaned pool.
    cleaned_units : list
        Affected units cleaned, in waterfall order.
    n_clean : int
        Number of clean (never-affected) controls.
    pre_rmspe : float
        Treated unit's pre-treatment RMSPE on the cleaned pool.
    treated_synthetic_pre : np.ndarray
        Treated unit's pre-treatment synthetic fit, shape ``(T0,)`` -- the same
        series ``ISCMFit`` and ``GrossiFit`` expose, so observed-vs-fitted plots
        and pre-treatment RMSE are uniform across every spillover method. By
        construction ``sqrt(mean((Y_treated_pre - treated_synthetic_pre)**2))``
        equals ``pre_rmspe``.
    bilevel_solver : str
        Backend used for the per-unit SCM (``"mscmt"`` / ``"malo"`` /
        ``"outcome-only"``).
    """

    att: float
    att_scm: float
    gap: np.ndarray
    gap_scm: np.ndarray
    counterfactual: np.ndarray
    counterfactual_scm: np.ndarray
    spillover_panel: Dict[Any, np.ndarray]
    spillover_att: Dict[Any, float]
    donor_weights: Dict[Any, float]
    cleaned_units: list
    n_clean: int
    pre_rmspe: float
    treated_synthetic_pre: np.ndarray
    bilevel_solver: str = "mscmt"

    # SP-dialect aliases for the shared accessors / plotter.
    @property
    def att_sp(self) -> float:
        return self.att

    @property
    def gap_sp(self) -> np.ndarray:
        return self.gap

    @property
    def counterfactual_sp(self) -> np.ndarray:
        return self.counterfactual


@dataclass(frozen=True)
class SpillSynthResults:
    """Top-level SPILLSYNTH result container.

    Parameters
    ----------
    inputs : SpillSynthInputs
        The preprocessed panel and spillover structure.
    cd : Optional[CDFit]
        Fit artifacts for the Cao-Dowd method, when ``method='cd'``.
    iscm : Optional[ISCMFit]
        Fit artifacts for the inclusive SCM method, when ``method='iscm'``.
    grossi : Optional[GrossiFit]
        Fit artifacts for the Grossi et al. (2025) partial-interference
        method, when ``method='grossi'``.
    method : str
        Method string used (``'cd'``, ``'iscm'`` or ``'grossi'``).
    """

    inputs: SpillSynthInputs
    method: str
    cd: Optional[CDFit] = None
    iscm: Optional["ISCMFit"] = None
    grossi: Optional["GrossiFit"] = None
    sar: Optional["SARFit"] = None
    iterative: Optional["IterativeFit"] = None

    # ------------------------------------------------------------------
    # Convenience accessors (route to the active method's fit).
    # ------------------------------------------------------------------
    @property
    def _active(self):
        if self.method == "cd":
            if self.cd is None:
                raise AttributeError(
                    "SPILLSYNTH method='cd' but no Cao-Dowd fit present."
                )
            return self.cd
        if self.method == "iscm":
            if self.iscm is None:
                raise AttributeError(
                    "SPILLSYNTH method='iscm' but no inclusive-SCM fit present."
                )
            return self.iscm
        if self.method == "grossi":
            if self.grossi is None:
                raise AttributeError(
                    "SPILLSYNTH method='grossi' but no partial-interference fit present."
                )
            return self.grossi
        if self.method == "sar":
            if self.sar is None:
                raise AttributeError(
                    "SPILLSYNTH method='sar' but no SAR spillover fit present."
                )
            return self.sar
        if self.method == "iterative":
            if self.iterative is None:
                raise AttributeError(
                    "SPILLSYNTH method='iterative' but no Iterative-SCM fit present."
                )
            return self.iterative
        raise AttributeError(f"Unknown SPILLSYNTH method {self.method!r}.")

    @property
    def att(self) -> float:
        """Spillover-adjusted / direct ATT on the treated unit (post-period mean)."""
        f = self._active
        if self.method == "iscm":
            return f.att
        if self.method == "grossi":
            return f.direct_att
        return f.att_sp

    @property
    def att_scm(self) -> float:
        """Vanilla SCM ATT (post-period mean), no spillover correction."""
        return self._active.att_scm

    @property
    def gap(self) -> np.ndarray:
        """Per-period spillover-adjusted treatment effect on the treated unit."""
        f = self._active
        return f.gap if self.method == "iscm" else f.gap_sp

    @property
    def gap_scm(self) -> np.ndarray:
        """Per-period vanilla SCM treatment effect (no spillover correction)."""
        return self._active.gap_scm

    @property
    def counterfactual(self) -> np.ndarray:
        """Spillover-adjusted post-period counterfactual for the treated unit."""
        f = self._active
        return f.counterfactual if self.method == "iscm" else f.counterfactual_sp

    @property
    def counterfactual_scm(self) -> np.ndarray:
        """Vanilla SCM post-period counterfactual."""
        return self._active.counterfactual_scm

    @property
    def spillover_effects(self) -> Dict[Any, np.ndarray]:
        """Per-affected-unit, per-period spillover trajectories."""
        return self._active.spillover_panel
