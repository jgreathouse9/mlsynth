"""Frozen dataclasses for the DSC estimator pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DSCARInputs:
    """Preprocessed panel for DSC.

    Parameters
    ----------
    Y : np.ndarray
        Shape ``(N, T)`` outcome panel ordered with the ``n_treated``
        treated units first (rows ``0 .. n_treated - 1``), then donor
        units.
    Y_lag1 : np.ndarray
        Shape ``(N, T)`` one-period-lag outcome. Column ``t = 0``
        carries the user-provided pre-period lag; columns
        ``t >= 1`` equal ``Y[:, t - 1]``.
    X : np.ndarray
        Shape ``(N, T, p)`` exogenous-covariate cube. ``p`` may be 0.
    var_names : tuple of str
        Length-``p`` names of the exogenous covariates (informational).
    y_name : str
        Outcome column name (informational).
    treated_labels : tuple
        Labels of the directly-treated units, in panel row order.
    donor_labels : tuple
        Labels of the donor units, in panel row order.
    time_labels : np.ndarray
        Length-``T`` time labels.
    N : int
        Total number of units.
    T : int
        Total number of time periods.
    T0 : int
        Number of pre-treatment periods.
    T1 : int
        Number of post-treatment periods.
    n_treated : int
        Number of directly-treated units.
    """

    Y: np.ndarray
    Y_lag1: np.ndarray
    X: np.ndarray
    var_names: Tuple[str, ...]
    y_name: str
    treated_labels: Tuple[Any, ...]
    donor_labels: Tuple[Any, ...]
    time_labels: np.ndarray
    N: int
    T: int
    T0: int
    T1: int
    n_treated: int


@dataclass(frozen=True)
class DSCARFit:
    """Per-period DSC weights + counterfactual + treatment-effect path.

    Parameters
    ----------
    weights : np.ndarray
        Shape ``(T, n_donors)`` per-period simplex weight matrix.
    Y0_hat : np.ndarray
        Length-``T`` estimated counterfactual outcome for the treated
        group (per-hour mean across treated units, following Zheng &
        Chen 2024 Section 5).
    Y_treated_mean : np.ndarray
        Length-``T`` observed per-hour mean across treated units.
    gap : np.ndarray
        Length-``T`` per-period effect ``Y_treated_mean - Y0_hat``.
    att : float
        Mean of ``gap`` over the post-period.
    att_relative : float
        ``1 - mu1 / mu0`` where ``mu1, mu0`` are post-period means of
        ``Y_treated_mean`` and ``Y0_hat`` respectively.
    se : Optional[float]
        Standard error of ``att`` from the normalised placebo run
        (Section 3.2). ``None`` when ``placebo_reps == 0``.
    placebo_atts : Optional[np.ndarray]
        Length-``placebo_reps`` post-period mean effects from the
        normalised placebo runs.
    pre_period_pvalues : Optional[np.ndarray]
        Length-``T0`` per-pre-period two-sided p-values for
        ``H_0: gap_t = 0`` (Section 3.1).
    pre_period_min_pvalue_adj : Optional[float]
        Benjamini-Yekutieli-adjusted minimum pre-period p-value.
    n_exact_matched_periods : int
        Number of periods at which the EL refinement step succeeded
        (``T_matched`` in the paper's notation).
    v_diagonal : Optional[np.ndarray]
        Shape ``(T, p + 1)`` per-period variable-importance vector
        used in the QP (the diagonal of ``V_t``).
    """

    weights: np.ndarray
    Y0_hat: np.ndarray
    Y_treated_mean: np.ndarray
    gap: np.ndarray
    att: float
    att_relative: float
    se: Optional[float] = None
    placebo_atts: Optional[np.ndarray] = None
    pre_period_pvalues: Optional[np.ndarray] = None
    pre_period_min_pvalue_adj: Optional[float] = None
    n_exact_matched_periods: int = 0
    v_diagonal: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DSCARResults:
    """Top-level DSC result container."""

    inputs: DSCARInputs
    fit: DSCARFit
    method: str = "dsc"

    # ---- Convenience accessors -----------------------------------------
    @property
    def att(self) -> float:
        return self.fit.att

    @property
    def att_relative(self) -> float:
        return self.fit.att_relative

    @property
    def gap(self) -> np.ndarray:
        return self.fit.gap

    @property
    def counterfactual(self) -> np.ndarray:
        return self.fit.Y0_hat

    @property
    def weights(self) -> np.ndarray:
        return self.fit.weights

    @property
    def se(self) -> Optional[float]:
        return self.fit.se
