"""Frozen, NumPy-first containers for CPDA.

CPDA is Hsiao and Zhou (2019) Section 3. It sits between the parametric route,
which models the factor structure explicitly, and the panel data approach,
which regresses the treated unit on donor outcomes and models nothing. CPDA
removes the covariate part of the outcome first and runs the donor regression
on what is left, so the donors have to explain only the factor part.

Everything below is pure NumPy; units and periods are addressed through
:class:`IndexSet`, and the only DataFrame touchpoint is ``setup``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..fast_scm_helpers.structure import IndexSet


@dataclass(frozen=True)
class CPDAInputs:
    """Preprocessed, NumPy-only panel for the CPDA engine.

    Parameters
    ----------
    unit_index : IndexSet
        The ``N`` donor units, in the column order of ``Yco`` and ``Xco``.
    time_index : IndexSet
        The ``T`` periods, in the row order of everything else.
    y : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    x : np.ndarray
        Treated covariates, shape ``(T, k)``.
    Yco : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    Xco : np.ndarray
        Donor covariates, shape ``(T, N, k)``.
    T0 : int
        Number of pre-treatment periods; the post window is ``T - T0``.
    treated_label : Any
        Identifier of the treated unit.
    covariate_names : tuple of str
        Column order of the last axis of ``x`` and ``Xco``.
    metadata : dict
        Free-form provenance.
    """

    unit_index: IndexSet
    time_index: IndexSet
    y: np.ndarray
    x: np.ndarray
    Yco: np.ndarray
    Xco: np.ndarray
    T0: int
    treated_label: Any
    covariate_names: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def T2(self) -> int:
        return self.T - self.T0

    @property
    def N(self) -> int:
        return int(self.Yco.shape[1])

    @property
    def k(self) -> int:
        return int(self.x.shape[1])


@dataclass(frozen=True)
class CPDAFit:
    """One CPDA fit, with the two choices the method leaves open recorded on it.

    Parameters
    ----------
    beta : np.ndarray
        The covariate slope of Equation 2, shape ``(k,)``.
    beta_method : str
        Which estimator produced it: ``cce`` (Pesaran 2006, Equation 16) or
        ``bai`` (Bai 2009 interactive fixed effects).
    selector : str
        Which rule chose the donor subset for Equation 13.
    selected_donors : list
        Labels of the donors that rule kept.
    standardize_selection : bool
        Whether the selection design was standardized before the penalty.
    intercept : float
        ``mu`` in Equation 11.
    weights : dict
        ``w`` in Equation 11, by donor label. Regression coefficients, not a
        simplex: CPDA places no sign or sum constraint on them.
    counterfactual : np.ndarray
        ``y^0_1t`` of Equation 15 over all periods, shape ``(T,)``.
    gap : np.ndarray
        Observed minus counterfactual, shape ``(T,)``.
    att : float
        Mean gap over the post window.
    att_se : float
        HAC standard error of the ATT.
    ci : tuple
        Two-sided interval at the configured level.
    p_value : float
        Two-sided normal p-value against a zero ATT.
    sensitivity : dict, optional
        ``{selector: {"att": float, "n_selected": int}}`` across every
        available selector, present only when the caller asked for it. The
        selector is not pinned by the paper and the estimate moves with it, so
        this is the spread the point estimate was drawn from.
    """

    beta: np.ndarray
    beta_method: str
    selector: str
    selected_donors: List[Any]
    standardize_selection: bool
    intercept: float
    weights: Dict[Any, float]
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    att_se: float
    ci: Tuple[float, float]
    p_value: float
    sensitivity: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CPDAResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.CPDA.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: the fit's quantities are
    lifted into the standardized sub-models so the flat accessors resolve
    through the base contract, and the CPDA-specific choices stay on ``fit``.

    ``weights.donor_weights`` are regression coefficients. CPDA constrains
    neither their sign nor their sum, which is what separates it from the
    synthetic control family and is recorded in ``summary_stats``.

    Parameters
    ----------
    inputs : CPDAInputs
        The preprocessed panel.
    fit : CPDAFit
        The fit, including the slope, the selector and any sensitivity sweep.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: CPDAInputs
    fit: CPDAFit
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "CPDAResults":
        """Born standardized. ``object.__setattr__`` because the model is frozen."""
        if self.effects is not None:
            return self
        f = self.fit
        labels = np.asarray(self.inputs.time_index.labels)
        T0, T = self.inputs.T0, self.inputs.T
        gap = np.asarray(f.gap, dtype=float)
        pre = gap[:T0]
        pre_rmse = float(np.sqrt(np.mean(pre ** 2))) if T0 > 0 else float("nan")
        post_rmse = (float(np.sqrt(np.mean(gap[T0:] ** 2)))
                     if T > T0 else float("nan"))
        lo, hi = f.ci

        object.__setattr__(self, "effects", EffectsResults(
            att=float(f.att), att_std_err=float(f.att_se)))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(f.counterfactual, dtype=float),
            estimated_gap=gap,
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None)))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in f.weights.items()},
            summary_stats={
                "constraint": "unconstrained regression coefficients",
                "intercept": float(f.intercept),
                "n_selected": len(f.selected_donors)}))
        object.__setattr__(self, "fit_diagnostics", FitDiagnosticsResults(
            rmse_pre=pre_rmse, rmse_post=post_rmse))
        object.__setattr__(self, "inference", InferenceResults(
            standard_error=float(f.att_se),
            ci_lower=None if (lo is None or not np.isfinite(lo)) else float(lo),
            ci_upper=None if (hi is None or not np.isfinite(hi)) else float(hi),
            p_value=(None if (f.p_value is None or not np.isfinite(f.p_value))
                     else float(f.p_value)),
            method="cpda_hac"))
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="CPDA",
            parameters_used={
                "beta_method": f.beta_method,
                "selector": f.selector,
                "standardize_selection": f.standardize_selection,
                "n_selected": len(f.selected_donors),
                "covariates": list(self.inputs.covariate_names)}))
        return self
