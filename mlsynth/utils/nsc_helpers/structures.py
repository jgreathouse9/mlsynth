"""Frozen dataclasses for the Nonlinear Synthetic Control (NSC) estimator.

NSC implements Tian (2023), *"The Synthetic Control Method with
Nonlinear Outcomes"* (arXiv:2306.01967). The estimator generalises
the Abadie-Diamond-Hainmueller (2010) synthetic-control procedure
to settings where the outcome is a nonlinear function of the
underlying predictors. The key modifications versus the canonical
Abadie SC are:

* the non-negativity constraint on donor weights is **relaxed** (only
  the adding-up constraint :math:`\\sum_j w_j = 1` remains);
* the objective adds an **L1 penalty weighted by pairwise pretreatment
  matching discrepancies** (favours donors close to the treated unit)
  plus an **L2 penalty** (spreads the weights);
* the tuning parameters :math:`a`, :math:`b` are **scaled by the
  eigenvalues of** :math:`Z_0 Z_0'`, so the dimensionless tuning
  parameters :math:`a^*, b^* \\in [0, 1]` can be cross-validated
  on a coarse grid.

The five layers below (inputs, design, fold-level CV summary, the
full grid trace, inference) keep the pipeline pluggable and
inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class NSCInputs:
    """Preprocessed panel data for NSC estimation.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Outcome series for the treated unit over the full panel,
        shape ``(T,)``.
    donor_outcomes : np.ndarray
        Outcome matrix for the J donor units over the full panel,
        shape ``(T, J)``.
    matching_matrix : np.ndarray
        Pre-period matching matrix ``Z_0`` of donor units used by the
        eigenvalue scaling, shape ``(J, p)`` where ``p = T0`` (paper
        default: pretreatment outcomes; additional covariates may
        be stacked downstream).
    treated_matching_vector : np.ndarray
        Pre-period matching vector ``Z_1`` for the treated unit,
        shape ``(p,)``.
    donor_names : np.ndarray
        Names / labels of the donor units, length ``J``.
    treated_unit_name : Any
        Label of the treated unit.
    T : int
        Total number of panel periods.
    T0 : int
        Number of pre-treatment periods.
    time_labels : np.ndarray
        Labels of the time periods, length ``T``.
    """

    treated_outcome: np.ndarray
    donor_outcomes: np.ndarray
    matching_matrix: np.ndarray
    treated_matching_vector: np.ndarray
    donor_names: np.ndarray
    treated_unit_name: Any
    T: int
    T0: int
    time_labels: np.ndarray

    @property
    def J(self) -> int:
        """Number of donor units."""
        return int(self.donor_outcomes.shape[1])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class NSCDesign:
    """Optimised NSC weights and the tuning state that produced them.

    Parameters
    ----------
    w : np.ndarray
        Donor weight vector ``(J,)``. Sums to 1; may contain negative
        entries because NSC drops the non-negativity restriction.
    donor_weights : dict
        ``{donor_name: weight}`` -- convenience view of ``w``.
    a_star : float
        Dimensionless L1-discrepancy tuning parameter on ``[0, 1]``.
    b_star : float
        Dimensionless L2 tuning parameter on ``[0, 1]``.
    a_scaled : float
        Raw L1 penalty multiplier the optimiser actually used
        (after eigenvalue scaling).
    b_scaled : float
        Raw L2 penalty multiplier (after eigenvalue scaling).
    eigvals : np.ndarray
        Sorted (ascending) non-zero eigenvalues of ``Z_0 Z_0'`` used
        for the (a, b) scaling. Length :math:`n = \\min(J, p)`.
    """

    w: np.ndarray
    donor_weights: Dict[Any, float]
    a_star: float
    b_star: float
    a_scaled: float
    b_scaled: float
    eigvals: np.ndarray


@dataclass(frozen=True)
class NSCCVTrace:
    """Coordinate-descent trace of the (a_star, b_star) sweep.

    Parameters
    ----------
    a_grid : np.ndarray
        Candidate values of ``a_star`` on the grid.
    b_grid : np.ndarray
        Candidate values of ``b_star`` on the grid.
    a_mspe_curve : np.ndarray
        MSPE at each ``a_star`` (with ``b_star`` held at its current
        value) on the final coordinate-descent iteration.
    b_mspe_curve : np.ndarray
        MSPE at each ``b_star`` (with ``a_star`` held at the
        coordinate-descent estimate) on the final iteration.
    iterations : int
        Number of coordinate-descent iterations actually run.
    converged : bool
        ``True`` iff ``(a_star, b_star)`` did not move on the last
        iteration.
    target : str
        Which CV target was used: ``"controls"`` (predict each
        donor's pre-period from the others -- paper default) or
        ``"treated"`` (predict the treated unit's pre-period
        outcomes from the donors).
    """

    a_grid: np.ndarray
    b_grid: np.ndarray
    a_mspe_curve: np.ndarray
    b_mspe_curve: np.ndarray
    iterations: int
    converged: bool
    target: str


@dataclass(frozen=True)
class NSCInference:
    """Doudchenko-Imbens (2017) inference for NSC.

    The variance of the per-period estimator is approximated by the
    MSE of predicting each control unit's outcome at that period
    using the other controls (the "leave-one-control-out"
    estimator). A normal CI is then formed period-by-period.

    Parameters
    ----------
    method : str
        ``"doudchenko_imbens"`` or ``"none"``.
    alpha : float
        Two-sided significance level used to build the CIs.
    period_variance : np.ndarray
        Per-period variance estimate, shape ``(T,)``.
    period_se : np.ndarray
        Per-period standard error :math:`\\sqrt{\\text{period\\_variance}}`.
    gap : np.ndarray
        Treated minus counterfactual at every period, shape ``(T,)``.
    gap_lower : np.ndarray
        Lower CI bound for the per-period gap, shape ``(T,)``.
    gap_upper : np.ndarray
        Upper CI bound for the per-period gap, shape ``(T,)``.
    att : float
        Average post-treatment gap.
    att_se : float
        Standard error of ``att`` (post-period variance averaged
        and divided by ``n_post``).
    att_lower, att_upper : float
        ``(1 - alpha)`` CI for the ATT.
    p_value : float
        Two-sided z-test of ``H_0: ATT = 0``.
    """

    method: str
    alpha: float
    period_variance: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    period_se: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    gap: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    gap_lower: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    gap_upper: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    att: float = float("nan")
    att_se: float = float("nan")
    att_lower: float = float("nan")
    att_upper: float = float("nan")
    p_value: float = float("nan")


@dataclass(frozen=True)
class NSCResults:
    """Top-level container returned by :meth:`mlsynth.NSC.fit`.

    Parameters
    ----------
    inputs : NSCInputs
        Preprocessed panel.
    design : NSCDesign
        Optimised weights plus the tuning state.
    cv_trace : NSCCVTrace or None
        Coordinate-descent diagnostics; ``None`` when the user
        supplied ``a`` and ``b`` explicitly.
    inference : NSCInference
        Per-period and ATT inference.
    counterfactual : np.ndarray
        Synthetic-control imputation of the treated outcome at
        every period, shape ``(T,)``.
    gap : np.ndarray
        Treated minus counterfactual, shape ``(T,)``.
    att : float
        Mean post-treatment gap.
    pre_rmse : float
        Root mean squared pre-treatment fit error.
    metadata : dict
        Free-form pipeline diagnostics (anchor of (a*, b*), iteration
        count, condition number of ``Z_0 Z_0'``, ...).
    """

    inputs: NSCInputs
    design: NSCDesign
    cv_trace: Optional[NSCCVTrace]
    inference: NSCInference
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    pre_rmse: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def donor_weights(self) -> Dict[Any, float]:
        """Alias for :py:attr:`design.donor_weights`."""
        return self.design.donor_weights
