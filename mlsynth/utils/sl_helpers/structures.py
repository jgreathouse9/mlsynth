"""Frozen, NumPy-first containers for SL.

SL is Viviano and Bradic (2023): a library of counterfactual forecasters combined
by exponential weights on a held-out slice of the pre-period, with a block
bootstrap test of the no-effect null.

The fit carries more than the estimate because the estimate alone does not say
what happened. Three diagnostics decide whether an SL number means anything, and
none of them is in the paper: how concentrated the weights are (did the
weighting select, or average), how correlated the experts' errors are (can
averaging cancel anything), and whether any member is degenerate. All three are
typed fields here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

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
class SLInputs:
    """Preprocessed, NumPy-only panel for the SL engine.

    Parameters
    ----------
    unit_index : IndexSet
        The ``N`` donor units, in the column order of ``Yco``.
    time_index : IndexSet
        The ``T`` periods, in row order.
    y : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    Yco : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    T0 : int
        Pre-treatment period count.
    treated_label : Any
        Identifier of the treated unit.
    covariates : np.ndarray, optional
        Time-varying covariate block, shape ``(T, m)``, read by the forest
        expert only.
    covariate_names : tuple of str
        The column names the block was built from.
    metadata : dict
        Free-form provenance.
    """

    unit_index: IndexSet
    time_index: IndexSet
    y: np.ndarray
    Yco: np.ndarray
    T0: int
    treated_label: Any
    covariates: Optional[np.ndarray] = None
    covariate_names: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def N(self) -> int:
        return int(self.Yco.shape[1])


@dataclass(frozen=True)
class SLFit:
    """One SL fit: the ensemble, the test, and what the library was doing.

    Parameters
    ----------
    experts : tuple of str
        The members that fit, in the column order of ``predictions``.
    predictions : np.ndarray
        Each expert's counterfactual path, shape ``(T, K)``.
    weights : dict
        Equation 12's weights by expert name. A simplex point, not regression
        coefficients.
    eta : float
        The learning rate actually used, resolved from the config.
    effective_k : float
        Perplexity of the weights, in ``[1, K]``. At 1 the weighting picked one
        expert; at ``K`` it is the simple average. The paper's own ``eta`` puts
        this at 3.69 of 4, so its ensemble averages.
    counterfactual : np.ndarray
        ``predictions @ weights``, shape ``(T,)``.
    gap : np.ndarray
        Observed minus counterfactual, shape ``(T,)``.
    att : float
        Equation 10's bias-adjusted average effect on the measured post window.
    bias : float
        The in-sample gap the adjustment removes.
    test_statistic : float
        Equations 7-8 on the measured post window.
    critical_values : dict
        Level to the null distribution's upper quantile, from Algorithm 2.
        Critical values for a non-negative quadratic, not interval endpoints.
    p_value : float
        Upper-tail bootstrap p-value against the no-effect null.
    expert_ssr : dict
        In-window squared loss per expert, the input to Equation 12.
    error_correlation : np.ndarray
        Correlation between the experts' weighting-window errors, ``(K, K)``.
    error_participation_ratio : float
        Effective number of independent error directions, in ``[1, K]``. Near 1
        the members err alike and averaging cannot cancel anything.
    degenerate_experts : dict
        Member to the reason it is not doing the job the library assumes.
    dropped_experts : dict
        Member to the reason it could not be built at all.
    train_periods, weight_periods, post_periods : int
        Algorithm 1's split, and the measured post window's length.
    details : dict
        Per-expert provenance, including the lasso's chosen penalty.
    """

    experts: Tuple[str, ...]
    predictions: np.ndarray
    weights: Dict[str, float]
    eta: float
    effective_k: float
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    bias: float
    test_statistic: float
    critical_values: Dict[float, float]
    p_value: float
    expert_ssr: Dict[str, float]
    error_correlation: np.ndarray
    error_participation_ratio: float
    degenerate_experts: Dict[str, str]
    dropped_experts: Dict[str, str]
    train_periods: int
    weight_periods: int
    post_periods: int
    details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.SL.fit`.

    An :class:`~mlsynth.config_models.EffectResult`. Two things about the
    standardized sub-models are deliberate.

    ``effects.att_std_err`` and ``inference.standard_error`` are ``None``. The
    method has no standard error: Theorem 3.1 controls the size of a test, and
    the statistic it controls is a non-negative quadratic whose null quantiles
    are critical values. Populating a standard error field from them would be an
    invention. ``inference.p_value`` and the critical values on ``fit`` are the
    inference. With ``interval="conformal"`` the interval fields carry a
    conformal ATT interval and ``method`` names it.

    ``weights.donor_weights`` holds the expert weights. The units being combined
    here are forecasters, not donors, and ``summary_stats`` records that along
    with ``effective_k``.

    Parameters
    ----------
    inputs : SLInputs
        The preprocessed panel.
    fit : SLFit
        The ensemble, the test, and the library diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SLInputs
    fit: SLFit
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "SLResults":
        """Born standardized. ``object.__setattr__`` because the model is frozen."""
        if self.effects is not None:
            return self
        f = self.fit
        labels = np.asarray(self.inputs.time_index.labels)
        T0, T = self.inputs.T0, self.inputs.T
        gap = np.asarray(f.gap, dtype=float)
        w0 = T0 - f.weight_periods
        pre = gap[w0:T0]
        pre_rmse = float(np.sqrt(np.mean(pre ** 2))) if pre.size else float("nan")
        post_rmse = (float(np.sqrt(np.mean(gap[T0:] ** 2)))
                     if T > T0 else float("nan"))

        object.__setattr__(self, "effects", EffectsResults(
            att=float(f.att), att_std_err=None))
        object.__setattr__(self, "time_series", TimeSeriesResults(
            observed_outcome=np.asarray(self.inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(f.counterfactual, dtype=float),
            estimated_gap=gap,
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None)))
        object.__setattr__(self, "weights", WeightsResults(
            donor_weights={str(k): float(v) for k, v in f.weights.items()},
            summary_stats={
                "combined": "experts, not donors",
                "constraint": "simplex (exponential weights, Equation 12)",
                "eta": float(f.eta),
                "effective_k": float(f.effective_k),
                "n_experts": len(f.experts)}))
        object.__setattr__(self, "fit_diagnostics", FitDiagnosticsResults(
            rmse_pre=pre_rmse, rmse_post=post_rmse,
            additional_metrics={
                "error_participation_ratio": float(f.error_participation_ratio),
                "expert_ssr": {k: float(v) for k, v in f.expert_ssr.items()}}))
        object.__setattr__(self, "inference", InferenceResults(
            standard_error=None,
            ci_lower=None,
            ci_upper=None,
            p_value=(None if (f.p_value is None or not np.isfinite(f.p_value))
                     else float(f.p_value)),
            confidence_level=None,
            method="sl_block_bootstrap",
            details={
                "test_statistic": float(f.test_statistic),
                "critical_values": {str(k): float(v)
                                    for k, v in f.critical_values.items()},
                "n_boot": int(f.metadata.get("n_boot", 0)),
                "block": int(f.metadata.get("block", 0)),
                "note": ("the critical values gate a non-negative quadratic "
                         "statistic; they are not interval endpoints")}))
        object.__setattr__(self, "method_details", MethodDetailsResults(
            method_name="SL",
            parameters_used={
                "experts": list(f.experts),
                "dropped_experts": f.dropped_experts,
                "degenerate_experts": f.degenerate_experts,
                "eta": float(f.eta),
                "effective_k": float(f.effective_k),
                "error_participation_ratio": float(f.error_participation_ratio),
                "train_periods": int(f.train_periods),
                "weight_periods": int(f.weight_periods),
                "post_periods": int(f.post_periods),
                "covariates": list(self.inputs.covariate_names)}))
        return self
