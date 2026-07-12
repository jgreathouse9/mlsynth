"""Typed result containers for SparseSC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class SparseSCInputs:
    """Pre-processed panel + predictor matrices for SparseSC.

    Parameters
    ----------
    Y0 : np.ndarray
        Donor outcome matrix, shape ``(T, N)`` (rows = time, columns =
        donors), aligned with ``donor_names``.
    Y1 : np.ndarray
        Treated outcome series, shape ``(T,)``.
    X0 : np.ndarray
        Donor predictor matrix, shape ``(P, N)`` (rows = predictors,
        columns = donors), already standardized.
    X1 : np.ndarray
        Treated predictor vector, shape ``(P,)``, already standardized.
    T : int
        Total number of time periods.
    T0_total : int
        End of the full pre-treatment window (exclusive).
    T0_train : int
        End of the training block within the pre-period (exclusive).
        Validation block is ``[T0_train, T0_total)``.
    treated_unit_name : Any
        Label of the treated unit.
    donor_names : Sequence
        Donor labels in column order of ``Y0`` / ``X0``.
    predictor_names : Sequence
        Predictor labels in row order of ``X0`` / ``X1``.
    time_labels : np.ndarray
        Time labels in row order of ``Y0``.
    Ywide : Any
        Wide outcome frame preserved for plotting.
    outcome : str
        Outcome variable name.
    """

    Y0: np.ndarray
    Y1: np.ndarray
    X0: np.ndarray
    X1: np.ndarray
    T: int
    T0_total: int
    T0_train: int
    treated_unit_name: Any
    donor_names: Sequence
    predictor_names: Sequence
    time_labels: np.ndarray
    Ywide: Any
    outcome: str

    @property
    def N(self) -> int:
        """Number of donor units."""
        return self.Y0.shape[1]

    @property
    def P(self) -> int:
        """Number of predictors."""
        return self.X0.shape[0]


@dataclass(frozen=True)
class SparseSCDesign:
    """Optimization outputs of the lambda sweep.

    Parameters
    ----------
    v : np.ndarray
        Final V-weights, shape ``(P,)``. First entry is 1 (the anchor).
    w : np.ndarray
        Final donor weights, shape ``(N,)``, on the simplex.
    opt_lambda : float
        Selected L1 penalty.
    lambda_grid : np.ndarray
        Full grid of lambdas swept.
    train_loss_curve : np.ndarray
        Training loss at each grid point, length equal to
        ``lambda_grid``.
    val_mse_curve : np.ndarray
        Validation MSE at each grid point.
    v_path : np.ndarray
        Per-grid-point V-weights, shape ``(len(grid), P)``.
    """

    v: np.ndarray
    w: np.ndarray
    opt_lambda: float
    lambda_grid: np.ndarray
    train_loss_curve: np.ndarray
    val_mse_curve: np.ndarray
    v_path: np.ndarray


@dataclass(frozen=True)
class SparseSCInference:
    """Inference results for SparseSC.

    Either the Abadie-style placebo permutation or the validation-block
    conformal inference of Chernozhukov, Wuethrich and Zhu (2021)
    adapted to the SparseSC pre/post layout. The ``method`` tag
    identifies which fields are populated.

    Parameters
    ----------
    method : str
        ``"abadie_placebo_permutation"``, ``"conformal_validation"``,
        ``"conformal_pre"``, or ``"none"``.
    p_value : float
        Two-sided p-value for ``H_0: ATT = 0``. NaN when no inference
        was run.
    att_observed : float
        Point estimate of ATT, copied here for convenience.
    ci_lower, ci_upper : float
        Lower/upper bounds of the (1 - alpha) confidence interval for
        the ATT. NaN for ``method="none"``.
    alpha : float
        Two-sided significance level used to build ``ci_*``.
    placebo_atts : np.ndarray
        Placebo ATTs, populated only when ``method`` is the placebo
        permutation. Empty array otherwise.
    n_placebo : int
        Number of placebo runs (placebo method only; 0 otherwise).
    calibration_residuals : np.ndarray
        Residuals used to build the conformity scores (conformal
        method only). Empty for the placebo method.
    pointwise_lower, pointwise_upper : np.ndarray
        Per-period pointwise band around each post-period gap from
        the (1 - alpha)-quantile of the conformity scores. Empty for
        non-conformal methods.
    """

    method: str
    p_value: float
    att_observed: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    alpha: float = float("nan")
    placebo_atts: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    n_placebo: int = 0
    calibration_residuals: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    pointwise_lower: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    pointwise_upper: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )


class SparseSCResults(BaseEstimatorResults):
    """Public ``SparseSC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the SparseSC-specific fields below it exposes the
    standardized sub-models (``effects``, ``time_series``, ``weights``,
    ``inference``, ``fit_diagnostics``, ``method_details``) and the flat
    accessors ``att`` / ``counterfactual`` / ``gap`` / ``att_ci`` /
    ``pre_rmse`` / ``donor_weights``.

    Parameters
    ----------
    inputs : SparseSCInputs
        Pre-processed panel + predictors.
    design : SparseSCDesign
        Lambda-selection results, V and W weights.
    inference_detail : SparseSCInference
        The raw placebo / conformal inference object (``method`` / ``p_value``
        / ``placebo_atts`` / ``pointwise_*`` / ...) or ``method="none"``.
        (Renamed from ``inference``; the standardized
        :class:`~mlsynth.config_models.InferenceResults` is mirrored into the
        ``inference`` slot so ``res.att_ci`` resolves.)
    predictor_weights : Dict[Any, float]
        ``{predictor_name: v_p}``.

    Notes
    -----
    The donor weights (``{donor_name: w_j}``) live in the standardized
    ``weights`` slot and are served by ``res.donor_weights``; the predictor
    weights are also mirrored into ``weights.summary_stats``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SparseSCInputs
    design: SparseSCDesign
    inference_detail: SparseSCInference
    predictor_weights: Dict[Any, float]
    scpi: Optional[Any] = None        # ScpiPIInference (simplex PI), when computed
