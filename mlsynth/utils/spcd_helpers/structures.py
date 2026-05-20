"""Structured containers for the SPCD synthetic design pipeline.

Implements the data containers used throughout the SPCD estimator,
which itself implements:

    Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
    Synthetic Principal Component Design: Fast Covariate Balancing
    with Synthetic Controls. arXiv:2211.15241v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..fast_scm_helpers.structure import IndexSet

if TYPE_CHECKING:
    from ...config_models import BaseEstimatorResults
    from .inference import SPCDConformalResult
    from .power import SPCDPowerAnalysis


@dataclass(frozen=True)
class SPCDInputs:
    """Preprocessed panel data for SPCD estimation.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(T_pre, N)``. Note that
        rows are time periods and columns are units, matching the
        convention used by ``prepare_scdi_inputs``. The paper's
        equations use ``Y in R^{N x T}``, so when implementing
        Eq. (2) of the paper, the iteration matrix is built as
        ``Y_pre.T @ Y_pre + alpha I + lambda 1 1.T``.
    Y_post : np.ndarray or None
        Post-treatment outcome matrix of shape ``(T_post, N)``.
    unit_index : IndexSet
        Mapping from unit labels to integer indices.
    time_index : IndexSet
        Mapping from time labels to integer indices.
    pre_time_index : IndexSet
        Index set for pre-treatment periods.
    post_time_index : IndexSet or None
        Index set for post-treatment periods.
    outcome : str
        Name of outcome variable.
    """

    Y_pre: np.ndarray
    Y_post: Optional[np.ndarray]
    unit_index: IndexSet
    time_index: IndexSet
    pre_time_index: IndexSet
    post_time_index: Optional[IndexSet]
    outcome: str


@dataclass(frozen=True)
class SPCDDesign:
    """Optimized SPCD design solution.

    The "assignment" is represented in two equivalent forms:

    * ``assignment_pm1`` — the paper's internal ``{-1, +1}`` sign vector
      ``y* in {-1, +1}^N`` (see Algorithm 1, page 7).
    * ``selected_mask`` — a ``{0, 1}`` indicator marking the minority
      group that is treated, following the rule "Treat Unit i if
      ``gamma(i) = -sgn(sum_j gamma(j))``" at the bottom of Algorithm 1.

    Parameters
    ----------
    variant : str
        Iteration variant used: ``"spcd"`` (Eq. (4)/(7)) or
        ``"norm_spcd"`` (Eq. (5)/(8)).
    weights_mode : str
        Final weight step used: ``"empirical"`` (Eq. (9), Algorithm 2) or
        ``"exact"`` (Eq. (6), Algorithm 1).
    assignment_pm1 : np.ndarray
        Final sign vector ``y* in {-1, +1}^N`` produced by the
        iteration. See Algorithm 1, page 7.
    selected_mask : np.ndarray
        Binary ``{0, 1}`` indicator of treated units, following the
        minority-group convention from the bottom of Algorithm 1.
    raw_weights : np.ndarray
        Signed weights ``w in R^N`` computed by the final weight step
        (Eq. (9) for ``weights="empirical"`` or Eq. (6) for
        ``weights="exact"``).
    treated_weights : np.ndarray
        Weights restricted to the treated group, normalized to sum to 1.
    control_weights : np.ndarray
        Weights restricted to the control group, normalized to sum to 1.
    contrast_weights : np.ndarray
        Signed contrast weights forming ``treated_weights - control_weights``
        (with appropriate signs), used to construct the synthetic gap.
    synthetic_treated : np.ndarray
        Synthetic treated trajectory ``Y @ treated_weights`` of length
        ``T_pre + T_post``.
    synthetic_control : np.ndarray
        Synthetic control trajectory ``Y @ control_weights`` of length
        ``T_pre + T_post``.
    synthetic_gap : np.ndarray
        Pointwise difference ``synthetic_treated - synthetic_control``.
    selected_unit_indices : np.ndarray
        Integer indices of treated units.
    selected_unit_labels : np.ndarray
        Original labels of treated units.
    n_treated : int
        Number of treated units.
    n_iterations : int
        Number of iterations executed by Algorithm 1's while loop.
    converged : bool
        True if the sign vector stabilized before ``max_iter``.
    alpha_ridge : float
        Value of ``alpha`` used in Eq. (2) (ridge on ``Y Y^T``).
    lam_balance : float
        Value of ``lambda`` used in Eq. (2) (penalty on ``(1^T W)^2``).
    beta : float
        Value of ``beta`` used in the iteration update.
    """

    variant: str
    weights_mode: str
    assignment_pm1: np.ndarray
    selected_mask: np.ndarray
    raw_weights: np.ndarray
    treated_weights: np.ndarray
    control_weights: np.ndarray
    contrast_weights: np.ndarray
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray
    synthetic_gap: np.ndarray
    selected_unit_indices: np.ndarray
    selected_unit_labels: np.ndarray
    n_treated: int
    n_iterations: int
    converged: bool
    alpha_ridge: float
    lam_balance: float
    beta: float

@dataclass(frozen=True)
class SPCDResults:
    """User-facing output of the SPCD estimator.

    Parameters
    ----------
    design : SPCDDesign
        Optimization solution.
    inputs : SPCDInputs, optional
        Preprocessed data used in estimation. Attached by the
        :class:`mlsynth.estimators.SPCD` orchestrator so the result is
        self-contained for plotting.
    summary : BaseEstimatorResults, optional
        Standardized result bundle containing ATT, pre/post fit RMSEs,
        synthetic-paths time series, per-unit signed weights, and method
        diagnostics. Attached by the SPCD orchestrator so users get a
        single object whose shape matches the rest of the mlsynth
        estimator suite.

    Notes
    -----
    ``mode`` always reports ``"spcd"`` so plotting and dispatch code can
    branch on it uniformly with :class:`SCDIResults`.

    Convenience properties (``att``, ``rmse_pre``, ``rmse_post``,
    ``donor_weights``) forward to the corresponding fields of
    ``summary`` when it is attached.
    """

    design: SPCDDesign
    inputs: Optional[SPCDInputs] = None
    summary: Optional["BaseEstimatorResults"] = None
    conformal: Optional["SPCDConformalResult"] = None
    power: Optional["SPCDPowerAnalysis"] = None

    @property
    def mode(self) -> str:
        """Solver mode reported to downstream consumers."""
        return "spcd"

    @property
    def assignment(self) -> np.ndarray:
        """Alias for ``design.selected_mask`` (0/1 indicator of treated units)."""
        return self.design.selected_mask

    @property
    def selected_unit_indices(self) -> np.ndarray:
        """Integer indices of units selected into treatment."""
        return self.design.selected_unit_indices

    @property
    def selected_unit_labels(self) -> np.ndarray:
        """Labels of units selected into treatment."""
        if self.inputs is None:
            return self.design.selected_unit_indices
        return self.inputs.unit_index.get_labels(self.design.selected_unit_indices)

    @property
    def att(self) -> Optional[float]:
        """Average treatment effect on the treated, or ``None`` if no summary."""
        if self.summary is None or self.summary.effects is None:
            return None
        return self.summary.effects.att

    @property
    def rmse_pre(self) -> Optional[float]:
        """Pre-treatment RMSE of the synthetic gap."""
        if self.summary is None or self.summary.fit_diagnostics is None:
            return None
        return self.summary.fit_diagnostics.rmse_pre

    @property
    def rmse_post(self) -> Optional[float]:
        """Post-treatment RMSE of the synthetic gap, if a post period exists."""
        if self.summary is None or self.summary.fit_diagnostics is None:
            return None
        return self.summary.fit_diagnostics.rmse_post

    @property
    def donor_weights(self) -> Optional[dict]:
        """Per-unit signed contrast weights as a label-to-float dict."""
        if self.summary is None or self.summary.weights is None:
            return None
        return self.summary.weights.donor_weights



    @property
    def p_value(self) -> Optional[float]:
        """Conformal p-value vs. H0: tau = 0, if computed."""
        return self.conformal.p_value if self.conformal is not None else None

    @property
    def ci_lower(self) -> Optional[float]:
        """Conformal lower bound of the ATT CI, if computed."""
        return self.conformal.ci_lower if self.conformal is not None else None

    @property
    def ci_upper(self) -> Optional[float]:
        """Conformal upper bound of the ATT CI, if computed."""
        return self.conformal.ci_upper if self.conformal is not None else None

    @property
    def mde(self) -> Optional[float]:
        """Minimum detectable effect on the absolute scale, if computed."""
        return self.power.mde_tau if self.power is not None else None

    @property
    def mde_pct(self) -> Optional[float]:
        """Minimum detectable effect as a percentage of the holdout baseline."""
        return self.power.mde_pct if self.power is not None else None
