"""Typed result containers for the MASC estimator.

Kellogg, Mogstad, Pouliot, and Torgovitsky (2021), *Combining Matching
and Synthetic Control to Trade off Biases from Extrapolation and
Interpolation*. The estimator forms a convex combination of a
nearest-neighbour matching weight vector and a synthetic-control
simplex weight vector, with both tuning parameters (``m``, ``phi``)
selected by rolling-origin cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MASCInputs:
    """Pre-pivoted inputs for a single-treated-unit MASC fit.

    Covariate panels (when supplied) are stored as full ``(T, J + 1, P)``
    tensors where the last axis indexes predictors and the second axis
    indexes units with the treated unit in slot 0; this lets each CV
    fold aggregate covariates over its own pre-window (matching the
    R reference, which re-averages within every fold).
    """

    Y_treated: np.ndarray         # (T,)
    Y_donors: np.ndarray          # (T, J)
    treated_label: Any
    donor_labels: Tuple[Any, ...]
    time_index: np.ndarray
    intervention_time: Any
    treatment_period: int         # 1-indexed position in `time_index`
    T: int                        # total periods
    T0: int                       # pre-period length
    T1: int                       # post-period length
    J: int                        # number of donors
    cov_treated_panel: Optional[np.ndarray] = None   # (T, P)
    cov_donors_panel: Optional[np.ndarray] = None    # (T, J, P)
    covariate_names: Tuple[Any, ...] = ()
    covariate_windows: Optional[dict] = None  # name -> (start, end) inclusive

    @property
    def has_covariates(self) -> bool:
        return self.cov_treated_panel is not None and self.cov_treated_panel.size > 0


@dataclass(frozen=True)
class MASCFit:
    """Single MASC point-estimate fit."""

    att: float
    weights: np.ndarray           # (J,) -- phi * match + (1-phi) * sc
    weights_match: np.ndarray     # (J,) -- nearest-neighbour weights
    weights_sc: np.ndarray        # (J,) -- simplex SC weights
    phi_hat: float
    m_hat: int
    counterfactual: np.ndarray    # (T,)
    gap: np.ndarray               # (T,) -- treated - counterfactual
    pre_rmse: float
    cv_error: float               # min CV error at (m_hat, phi_hat)
    cv_error_by_fold: np.ndarray  # (len(folds),) at the selected (m_hat, phi_hat)
    cv_grid: np.ndarray           # (len(m_grid), 3) -- columns: m, phi, cv_error
    donor_weights: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MASCResults:
    """Top-level container returned by ``MASC.fit``."""

    inputs: MASCInputs
    fit: MASCFit

    @property
    def att(self) -> float:
        return self.fit.att

    @property
    def weights(self) -> np.ndarray:
        return self.fit.weights

    @property
    def phi_hat(self) -> float:
        return self.fit.phi_hat

    @property
    def m_hat(self) -> int:
        return self.fit.m_hat

    @property
    def counterfactual(self) -> np.ndarray:
        return self.fit.counterfactual

    @property
    def gap(self) -> np.ndarray:
        return self.fit.gap

    @property
    def donor_weights(self) -> dict:
        return self.fit.donor_weights
