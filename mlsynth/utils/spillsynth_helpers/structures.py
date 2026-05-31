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
    from .cd.inference import PTestResult


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
        Number of potentially-affected control units (so ``A`` has
        ``1 + p`` columns).
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


@dataclass(frozen=True)
class SpillSynthResults:
    """Top-level SPILLSYNTH result container.

    Parameters
    ----------
    inputs : SpillSynthInputs
        The preprocessed panel and spillover structure.
    cd : Optional[CDFit]
        Fit artifacts for the Cao-Dowd method, when ``method='cd'``.
    method : str
        Method string used (``'cd'`` for now).
    """

    inputs: SpillSynthInputs
    method: str
    cd: Optional[CDFit] = None

    # ------------------------------------------------------------------
    # Convenience accessors (route to the active method's fit).
    # ------------------------------------------------------------------
    @property
    def _active(self) -> CDFit:
        if self.method == "cd":
            if self.cd is None:
                raise AttributeError(
                    "SPILLSYNTH method='cd' but no Cao-Dowd fit present."
                )
            return self.cd
        raise AttributeError(f"Unknown SPILLSYNTH method {self.method!r}.")

    @property
    def att(self) -> float:
        """Spillover-adjusted ATT on the treated unit (post-period mean)."""
        return self._active.att_sp

    @property
    def att_scm(self) -> float:
        """Vanilla SCM ATT (post-period mean), no spillover correction."""
        return self._active.att_scm

    @property
    def gap(self) -> np.ndarray:
        """Per-period spillover-adjusted treatment effect on the treated unit."""
        return self._active.gap_sp

    @property
    def gap_scm(self) -> np.ndarray:
        """Per-period vanilla SCM treatment effect (no spillover correction)."""
        return self._active.gap_scm

    @property
    def counterfactual(self) -> np.ndarray:
        """Spillover-adjusted post-period counterfactual for the treated unit."""
        return self._active.counterfactual_sp

    @property
    def counterfactual_scm(self) -> np.ndarray:
        """Vanilla SCM post-period counterfactual."""
        return self._active.counterfactual_scm

    @property
    def spillover_effects(self) -> Dict[Any, np.ndarray]:
        """Per-affected-unit, per-period spillover trajectories."""
        return self._active.spillover_panel
