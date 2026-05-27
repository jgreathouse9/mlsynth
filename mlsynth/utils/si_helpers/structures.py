"""Frozen dataclass containers for the Synthetic Interventions (SI) pipeline.

Implements the containers used throughout SI, which itself implements:

    Agarwal, A., Shah, D., & Shen, D. (2026). "Synthetic Interventions:
    Extending Synthetic Controls to Multiple Treatments." Operations Research
    74(2):840-859.

All containers are frozen (immutable) per the repository convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SIDonorPool:
    """Donor pool for one alternative intervention.

    Parameters
    ----------
    name : str
        Intervention label (a column in ``inters``).
    matrix : np.ndarray
        Donor outcomes over the full timeline, shape ``(T, Nd)`` -- rows are
        periods, columns the units that received this intervention.
    names : list of str
        Donor unit labels (length ``Nd``).
    """

    name: str
    matrix: np.ndarray
    names: List[str]


@dataclass(frozen=True)
class SIInputs:
    """Preprocessed panel data for SI.

    The focal (target) unit is the one flagged by the ``treat`` column; SI
    estimates its counterfactual outcome under each alternative intervention.

    Parameters
    ----------
    treated_unit_name : Any
        Label of the focal target unit.
    y_target : np.ndarray
        Focal unit's observed outcome over the full timeline, shape ``(T,)``.
    T0 : int
        Number of (common) pre-treatment periods.
    time_labels : np.ndarray
        Time-period labels (length ``T``).
    pools : dict of {str: SIDonorPool}
        Donor pool per alternative intervention.
    """

    treated_unit_name: Any
    y_target: np.ndarray
    T0: int
    time_labels: np.ndarray
    pools: Dict[str, SIDonorPool]

    @property
    def T(self) -> int:
        """Total number of periods."""
        return int(self.y_target.shape[0])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods (``T1``)."""
        return self.T - self.T0

    @property
    def Y_pre(self) -> np.ndarray:
        """Focal unit's pre-treatment outcomes, shape ``(T0,)``."""
        return self.y_target[: self.T0]

    @property
    def Y_post(self) -> np.ndarray:
        """Focal unit's post-treatment outcomes, shape ``(T1,)``."""
        return self.y_target[self.T0:]


@dataclass(frozen=True)
class SIArm:
    """SI estimate of the focal unit's outcome under one intervention.

    Parameters
    ----------
    name : str
        Intervention label.
    donor_names : list of str
        Full donor pool for this intervention.
    weights : dict of {str: float}
        Donor weights with non-trivial magnitude. For the bias-corrected
        estimator these are supported on the active subset ``omega_names``.
    selected_rank : int
        Spectral rank ``k`` used by SI-PCR.
    omega_names : list of str
        Active (rank-complete) donor subset for the bias-corrected estimator
        (Agarwal-Shah-Shen Section 4.3); equals ``donor_names`` for the plain
        SI-PCR estimator.
    counterfactual : np.ndarray
        Focal unit's counterfactual outcome under this intervention over the
        full timeline, shape ``(T,)``.
    gap : np.ndarray
        Observed minus counterfactual, shape ``(T,)``.
    att : float
        Average post-treatment effect (mean of the post-period gap).
    cf_mean : float
        Average post-treatment counterfactual outcome ``theta_i(d)``.
    pre_rmse : float
        Pre-treatment root-mean-square fit error.
    bias_corrected : bool
        Whether the bias-corrected estimator (and hence inference) was used.
    sigma_hat : float, optional
        Estimated noise standard deviation (eq. 14); ``None`` without bias
        correction.
    weight_norm : float, optional
        ``||w(i, d, Omega)||_2``, the bias-corrected weight norm entering the
        CI; ``None`` without bias correction.
    cf_mean_ci : tuple of (float, float), optional
        Asymptotic-normality confidence interval for ``cf_mean`` (eq. 13).
    att_ci : tuple of (float, float), optional
        Confidence interval for ``att`` (the CI half-width is shared with
        ``cf_mean_ci`` since the observed outcome is fixed).
    """

    name: str
    donor_names: List[str]
    weights: Dict[str, float]
    selected_rank: int
    omega_names: List[str]
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    cf_mean: float
    pre_rmse: float
    bias_corrected: bool
    sigma_hat: Optional[float] = None
    weight_norm: Optional[float] = None
    cf_mean_ci: Optional[Tuple[float, float]] = None
    att_ci: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class SIResults:
    """User-facing output of the SI estimator.

    Parameters
    ----------
    inputs : SIInputs
        Preprocessed panel data for the focal unit and donor pools.
    arms : dict of {str: SIArm}
        One :class:`SIArm` per alternative intervention.
    alpha : float
        Two-sided significance level used for the confidence intervals.
    bias_corrected : bool
        Whether the bias-corrected SI-PCR estimator (with CIs) was used.
    """

    inputs: SIInputs
    arms: Dict[str, SIArm]
    alpha: float
    bias_corrected: bool

    @property
    def mode(self) -> str:
        """Solver mode reported to downstream consumers."""
        return "si"

    @property
    def treated_unit_name(self) -> Any:
        """Label of the focal target unit."""
        return self.inputs.treated_unit_name

    @property
    def observed(self) -> np.ndarray:
        """Focal unit's observed outcome over the full timeline, shape ``(T,)``."""
        return self.inputs.y_target

    @property
    def att_by_intervention(self) -> Dict[str, float]:
        """``{intervention: att}`` across the fitted arms."""
        return {name: arm.att for name, arm in self.arms.items()}
