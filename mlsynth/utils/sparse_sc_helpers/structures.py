"""Typed result containers for SparseSC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


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
    """Abadie-style placebo permutation inference."""

    method: str                  # "abadie_placebo_permutation" or "none"
    placebo_atts: np.ndarray
    p_value: float
    n_placebo: int


@dataclass(frozen=True)
class SparseSCResults:
    """Public ``SparseSC.fit()`` return container.

    Parameters
    ----------
    inputs : SparseSCInputs
        Pre-processed panel + predictors.
    design : SparseSCDesign
        Lambda-selection results, V and W weights.
    inference : SparseSCInference
        Placebo p-value or ``method = "none"``.
    counterfactual : np.ndarray
        ``Y0 @ w`` over all ``T`` periods.
    gap : np.ndarray
        ``Y1 - counterfactual``, shape ``(T,)``.
    att : float
        Mean post-treatment gap.
    pre_rmse : float
        Root-mean-squared pre-treatment fit error.
    donor_weights : Dict[Any, float]
        ``{donor_name: w_j}``.
    predictor_weights : Dict[Any, float]
        ``{predictor_name: v_p}``.
    """

    inputs: SparseSCInputs
    design: SparseSCDesign
    inference: SparseSCInference
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    pre_rmse: float
    donor_weights: Dict[Any, float]
    predictor_weights: Dict[Any, float]
