"""Frozen dataclasses for the DSC estimator pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, model_validator

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)


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


class DSCARResults(BaseEstimatorResults):
    """Top-level DSC result container.

    An :class:`~mlsynth.config_models.EffectResult`: it derives the standardized
    sub-models (``effects``/``time_series``/``weights``/``inference``) from the
    ``fit`` so the flat accessors ``att`` / ``counterfactual`` / ``gap`` /
    ``pre_rmse`` resolve through the base contract. DSCAR's weights are
    time-varying (a ``(T, J)`` matrix); they live in ``weights.unit_weights`` and
    are also exposed via the ``weight_matrix`` accessor (the field was named
    ``weights`` before the result-contract migration).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: DSCARInputs
    fit: DSCARFit
    method: str = "dsc"

    # ---- Convenience accessors (non-colliding; att/gap/counterfactual now
    # resolve through the base contract via the populated sub-models) ----------
    @property
    def att_relative(self) -> float:
        return self.fit.att_relative

    @property
    def se(self) -> Optional[float]:
        return self.fit.se

    @property
    def weight_matrix(self) -> np.ndarray:
        """The time-varying ``(T, J)`` donor-weight matrix (legacy ``weights``)."""
        return self.fit.weights

    @model_validator(mode="after")
    def _populate_contract(self) -> "DSCARResults":
        if self.effects is not None:           # already populated (e.g. round-trip)
            return self
        times = np.asarray(self.inputs.time_labels)
        fit = self.fit
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen)
        set_("effects", EffectsResults(
            att=float(fit.att),
            att_percent=float(fit.att_relative) * 100.0,
            att_std_err=(None if fit.se is None else float(fit.se)),
        ))
        set_("time_series", TimeSeriesResults(
            observed_outcome=np.asarray(fit.Y_treated_mean, dtype=float),
            counterfactual_outcome=np.asarray(fit.Y0_hat, dtype=float),
            estimated_gap=np.asarray(fit.gap, dtype=float),
            time_periods=times,
        ))
        set_("weights", WeightsResults(unit_weights=np.asarray(fit.weights, dtype=float)))
        if fit.se is not None:
            set_("inference", InferenceResults(
                standard_error=float(fit.se),
                method="DSCAR asymptotic standard error"))
        set_("fit_diagnostics", FitDiagnosticsResults())
        set_("method_details", MethodDetailsResults(method_name=f"DSCAR[{self.method}]"))
        return self
