"""Structured containers for the Harmonic Synthetic Control (HSC) pipeline.

Implements the dataclasses used throughout HSC, which itself implements:

    Liu, Z., & Xu, Y. (2026). "The Harmonic Synthetic Control Method."

All containers are frozen (immutable) per the repository convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class HSCInputs:
    """Preprocessed panel data for HSC.

    Parameters
    ----------
    y_target : np.ndarray
        Treated unit's outcome over the full timeline, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcomes, shape ``(T, N)`` (rows = periods, cols = donors).
    T : int
        Total number of periods.
    T0 : int
        Number of pre-treatment periods.
    treated_unit_name : Any
        Label of the treated unit.
    donor_names : Sequence
        Labels of the donor units (length ``N``).
    time_labels : np.ndarray
        Time-period labels (length ``T``).
    q : int
        Smoothness order of the difference operator (1 or 2).
    """

    y_target: np.ndarray
    donor_matrix: np.ndarray
    T: int
    T0: int
    treated_unit_name: Any
    donor_names: Sequence
    time_labels: np.ndarray
    q: int

    @property
    def Y_pre(self) -> np.ndarray:
        """Treated pre-treatment outcomes, shape ``(T0,)``."""
        return self.y_target[: self.T0]

    @property
    def X_pre(self) -> np.ndarray:
        """Donor pre-treatment matrix, shape ``(T0, N)``."""
        return self.donor_matrix[: self.T0]

    @property
    def Y_post(self) -> np.ndarray:
        """Treated post-treatment outcomes, shape ``(T - T0,)``."""
        return self.y_target[self.T0:]

    @property
    def X_post(self) -> np.ndarray:
        """Donor post-treatment matrix, shape ``(T - T0, N)``."""
        return self.donor_matrix[self.T0:]

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0


@dataclass(frozen=True)
class HSCDesign:
    """Fitted HSC design.

    Parameters
    ----------
    selected_rho : float
        Cross-validated allocation parameter in ``[0, 1]``. Near 1 the design
        matches on levels (with an intercept/trend); near 0 it matches on
        ``q``-th differences and the smooth component absorbs the trend.
    q : int
        Smoothness order used.
    omega : np.ndarray
        Donor weights on the simplex, shape ``(N,)``.
    smooth_pre : np.ndarray
        Fitted treated-unit smooth residual ``E`` over the pre-period,
        shape ``(T0,)`` (``S_{rho,q} (Y_pre - X_pre omega)``).
    donor_match_pre : np.ndarray
        Donor-matched component over the pre-period, ``X_pre @ omega``.
    counterfactual_post : np.ndarray
        Post-treatment counterfactual, shape ``(n_post,)``
        (``X_post @ omega + forecast(E)``).
    smooth_forecast : np.ndarray
        Forecast of the smooth residual over the post-period.
    donor_match_post : np.ndarray
        Donor-matched component over the post-period, ``X_post @ omega``.
    cv_curve : dict
        ``{rho: rolling-origin CV error}`` used to select ``selected_rho``.
    forecaster : str
        Name of the residual forecaster used (e.g. ``"arima110"``).
    """

    selected_rho: float
    q: int
    omega: np.ndarray
    smooth_pre: np.ndarray
    donor_match_pre: np.ndarray
    counterfactual_post: np.ndarray
    smooth_forecast: np.ndarray
    donor_match_post: np.ndarray
    cv_curve: Dict[float, float]
    forecaster: str


@dataclass(frozen=True)
class HSCResults:
    """User-facing output of the HSC estimator.

    Parameters
    ----------
    inputs : HSCInputs
        Preprocessed panel data.
    design : HSCDesign
        Fitted design (weights, smooth component, counterfactual).
    att : float
        Average post-treatment effect (mean of ``treatment_effect``).
    counterfactual_full : np.ndarray
        Counterfactual over the full timeline, shape ``(T,)`` (pre-period
        in-sample fit followed by the post-period counterfactual).
    treatment_effect : np.ndarray
        Post-period treated minus counterfactual, shape ``(n_post,)``.
    weights_by_donor : dict
        ``{donor_label: weight}`` for donors with non-trivial weight.
    """

    inputs: HSCInputs
    design: HSCDesign
    att: float
    counterfactual_full: np.ndarray
    treatment_effect: np.ndarray
    weights_by_donor: Dict[Any, float]

    @property
    def mode(self) -> str:
        """Solver mode reported to downstream consumers."""
        return "hsc"

    @property
    def selected_rho(self) -> float:
        """Cross-validated allocation parameter."""
        return self.design.selected_rho
