"""Frozen dataclasses for the Factor Model Approach (FMA) estimator.

FMA implements Li and Sonnier (2023), *"Statistical Inference for the
Factor Model Approach to Estimate Causal Effects in
Quasi-Experimental Settings"*, JMR 60(3):449-472. The estimator
constructs a counterfactual for a single treated unit by (i)
extracting principal-component factors from the control panel,
(ii) projecting the treated unit's pre-period outcomes onto those
factors, and (iii) using the resulting loadings to predict the
treated unit's untreated potential outcomes in the post-period.

Inference is the paper's main contribution. Three procedures live
side-by-side in :class:`FMAInference`:

* **asymptotic** -- Theorem 3.1 (stationary) / Theorem 3.3
  (non-stationary) normal CIs for the ATT.
* **bootstrap** -- Web Appendix F residual bootstrap for per-period
  ATT_t CIs.
* **placebo** -- Web Appendix G control-as-pseudo-treated bands.

The four layers below (inputs, design, inference, results) keep that
pipeline pluggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class FMAInputs:
    """Preprocessed panel data for FMA estimation.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Outcome series for the treated unit, shape ``(T,)``.
    control_outcomes : np.ndarray
        Outcome matrix for the ``N_co`` control units, shape
        ``(T, N_co)``.
    donor_names : np.ndarray
        Labels of the control units, length ``N_co``.
    treated_unit_name : Any
        Label of the treated unit.
    T : int
        Total number of panel periods.
    T0 : int
        Number of pre-treatment periods.
    time_labels : np.ndarray
        Labels of the time periods, length ``T``.
    preprocessing : str
        ``"demean"`` (default) or ``"standardize"`` -- preprocessing
        applied to the control panel before PCA.
    stationarity : str
        ``"stationary"`` or ``"nonstationary"`` -- selects the
        appropriate factor-selection criterion (MBN vs Bai-IPC1).
    """

    treated_outcome: np.ndarray
    control_outcomes: np.ndarray
    donor_names: np.ndarray
    treated_unit_name: Any
    T: int
    T0: int
    time_labels: np.ndarray
    preprocessing: str
    stationarity: str

    @property
    def N_co(self) -> int:
        """Number of control units."""
        return int(self.control_outcomes.shape[1])

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class FMADesign:
    """Factor-model design produced by the FMA fit.

    Parameters
    ----------
    n_factors : int
        Number of common factors selected (or user-supplied).
    n_factors_source : str
        ``"MBN"`` (modified Bai-Ng), ``"IPC1"`` (Bai 2004),
        ``"user"`` (user override).
    factors : np.ndarray
        Estimated factors ``F_hat``, shape ``(T, r)``.
    lambda_hat : np.ndarray
        Estimated factor loading for the treated unit, shape
        ``(r + 1,)`` (intercept + loadings).
    counterfactual : np.ndarray
        Predicted treated potential outcome under no treatment,
        shape ``(T,)``.
    gap : np.ndarray
        Observed treated minus counterfactual, shape ``(T,)``.
    common_component : np.ndarray
        Reconstructed control panel via factor projection,
        shape ``(T, N_co)``.
    residual_variance : float
        Variance of the pre-treatment residuals
        ``y_1t - F_hat_t' lambda_hat``.
    """

    n_factors: int
    n_factors_source: str
    factors: np.ndarray
    lambda_hat: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray
    common_component: np.ndarray
    residual_variance: float


@dataclass(frozen=True)
class FMAInference:
    """Combined asymptotic + bootstrap + placebo inference output.

    Parameters
    ----------
    method : str
        Primary inference method tag (``"asymptotic"``,
        ``"bootstrap"``, ``"placebo"``, or ``"none"``).
    alpha : float
        Two-sided significance level.
    att : float
        Mean post-treatment gap.

    asymptotic_att_se : float
        Standard error of the ATT from the paper's Theorem 3.1.
    asymptotic_att_lower : float
    asymptotic_att_upper : float
        ``(1 - alpha)`` asymptotic CI bounds.
    asymptotic_att_p_value : float
        Two-sided z-test of ``H_0: ATT = 0``.

    bootstrap_att_t_lower : np.ndarray
    bootstrap_att_t_upper : np.ndarray
        Per-period bootstrap CI bounds for ATT_t, shape
        ``(n_post,)``. Empty when ``method != "bootstrap"``.
    bootstrap_replicates : np.ndarray
        ``(B, n_post)`` matrix of bootstrap replicate ATT_t.
    bootstrap_n_replicates : int
        Number of bootstrap draws actually completed.

    placebo_att_curves : np.ndarray
        ``(N_co, T)`` matrix of pseudo-ATT curves (one per control
        used as a placebo treated unit). Empty when
        ``method != "placebo"``.
    placebo_quantile_lower : np.ndarray
    placebo_quantile_upper : np.ndarray
        Pointwise (alpha/2, 1 - alpha/2) quantile bands across the
        placebo curves at each period. Shape ``(T,)``.
    """

    method: str
    alpha: float
    att: float

    asymptotic_att_se: float = float("nan")
    asymptotic_att_lower: float = float("nan")
    asymptotic_att_upper: float = float("nan")
    asymptotic_att_p_value: float = float("nan")

    bootstrap_att_t_lower: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    bootstrap_att_t_upper: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    bootstrap_replicates: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    bootstrap_n_replicates: int = 0

    placebo_att_curves: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    placebo_quantile_lower: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    placebo_quantile_upper: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )


@dataclass(frozen=True)
class FMAResults:
    """Top-level container returned by :meth:`mlsynth.FMA.fit`.

    Parameters
    ----------
    inputs : FMAInputs
        Preprocessed panel.
    design : FMADesign
        Factor-model design.
    inference : FMAInference
        Inference output.
    counterfactual : np.ndarray
        Convenience alias of ``design.counterfactual``.
    gap : np.ndarray
        Convenience alias of ``design.gap``.
    att : float
        Mean post-treatment gap.
    pre_rmse : float
        Root mean squared pre-treatment fit error.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    inputs: FMAInputs
    design: FMADesign
    inference: FMAInference
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    pre_rmse: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        """Selected number of factors."""
        return self.design.n_factors
