"""Frozen, NumPy-first containers for GPITS.

Implements the containers for

    Cho, S. (2026). "Let Time Tell: Identification and Gaussian Process
    Estimation for Interrupted Time Series." arXiv:2608.20610.

GPITS estimates a single treated unit's untreated counterfactual with no
cross-sectional controls at all, which is the situation a universal treatment
creates: an event that reaches every unit at once leaves nobody untreated to
compare against, so the counterfactual has to be extrapolated from the unit's
own pre-treatment history. The estimator puts a Gaussian-process prior on that
history's trend, conditions on the pre-period, and reads the posterior at the
post-treatment inputs. Its band widens as the forecast leaves the data.

Everything below is pure NumPy; period labels are addressed through the
repository's :class:`IndexSet`. The only DataFrame touchpoint is ``setup``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ConfigDict as _ConfigDict
from pydantic import Field as _PydField
from pydantic import model_validator as _model_validator

from ...config_models import BaseEstimatorResults as _BaseEstimatorResults
from ...config_models import EffectsResults as _EffectsResults
from ...config_models import FitDiagnosticsResults as _FitDiagnosticsResults
from ...config_models import InferenceResults as _InferenceResults
from ...config_models import MethodDetailsResults as _MethodDetailsResults
from ...config_models import TimeSeriesResults as _TimeSeriesResults
from ..helperutils import IndexSet

__all__ = ["GPITSInputs", "GPITSDesign", "GPITSPlacebo", "GPITSResults"]


@dataclass(frozen=True)
class GPITSInputs:
    """Preprocessed, NumPy-only inputs for the GP engine.

    Parameters
    ----------
    time_index : IndexSet
        All ``T`` period labels, in the row order of ``y``.
    y : np.ndarray
        Treated-unit outcome over all periods, shape ``(T,)``.
    design : np.ndarray
        GP design matrix, shape ``(T, d)``. Column order is the one-hot
        encoded categorical covariates first, then the continuous columns
        with the time index leading them, which is the reference's order and
        the one the period rescaling depends on.
    T0 : int
        Number of pre-treatment periods; the post-period is ``T - T0``.
    n_categorical : int
        Count of leading one-hot columns in ``design``.
    column_names : list of str
        Names of the ``design`` columns, for diagnostics.
    treated_label : Any
        Identifier of the treated unit.
    metadata : dict
        Free-form provenance.
    """

    time_index: IndexSet
    y: np.ndarray
    design: np.ndarray
    T0: int
    n_categorical: int
    column_names: List[str]
    treated_label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_post(self) -> int:
        return self.T - self.T0


@dataclass(frozen=True)
class GPITSDesign:
    """The fitted Gaussian process.

    Parameters
    ----------
    kernel : str
        Covariance function used.
    length_scale : float
        The kernel length-scale ``b`` in force, selected or supplied.
    noise_variance : float
        The observation noise ``sigma^2``, which also plays the ridge role
        in Eq. (16).
    period_scaled : float or None
        The seasonal period expressed on the standardised time axis.
    length_scale_selected, noise_variance_selected : bool
        Whether each hyperparameter was fitted, not supplied. A user
        who pins one should be able to see that from the result.
    counterfactual : np.ndarray
        Posterior mean over all ``T`` periods; pre-period entries are the
        in-sample fit, post-period entries the counterfactual.
    counterfactual_se : np.ndarray
        Posterior standard deviation over all ``T`` periods.
    post_covariance : np.ndarray
        Full posterior covariance over the post-period, shape
        ``(n_post, n_post)``. The cumulative-effect interval needs the whole
        block, not its diagonal, because the post-period errors covary.
    """

    kernel: str
    length_scale: float
    noise_variance: float
    period_scaled: Optional[float]
    length_scale_selected: bool
    noise_variance_selected: bool
    counterfactual: np.ndarray
    counterfactual_se: np.ndarray
    post_covariance: np.ndarray


@dataclass(frozen=True)
class GPITSPlacebo:
    """Temporal placebo checks (Section 3.3).

    Each entry refits the GP on everything before a pre-treatment period and
    predicts that period one step ahead. No treatment applies there, so an
    adequate kernel should put the estimate near zero with intervals that
    cover it. The check can falsify Mean Sufficiency but never confirm it: a
    confounder inside the training window is absorbed into the fit and leaves
    the placebo clean even when the same disturbance breaks the assumption
    after the intervention.

    Parameters
    ----------
    time_labels : np.ndarray
        Period label of each placebo target.
    tau, se : np.ndarray
        Placebo estimate and its standard error.
    ci_lower, ci_upper : np.ndarray
        Pointwise interval at the configured level.
    cover : np.ndarray of bool
        Whether each interval covers zero.
    """

    time_labels: np.ndarray
    tau: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    cover: np.ndarray

    @property
    def all_cover(self) -> bool:
        return bool(np.all(self.cover))


class GPITSResults(_BaseEstimatorResults):
    """Public container returned by :meth:`mlsynth.GPITS.fit`.

    Parameters
    ----------
    inputs : GPITSInputs
        Preprocessed series and design.
    design : GPITSDesign
        Fitted hyperparameters, posterior mean, and post-period covariance.
    att_value : float
        Mean post-treatment gap.
    att_interval : tuple of float
        Interval for the ATT, from the post-period covariance block. The
        base class exposes it as ``att_ci``.
    observed, cf_series, gap_series : np.ndarray
        Length-``T`` series. The base class exposes the latter two as
        ``counterfactual`` and ``gap``.
    counterfactual_lower, counterfactual_upper : np.ndarray
        Pointwise band on the counterfactual.
    cumulative_effect : np.ndarray
        Running sum of the post-period gaps.
    cumulative_ci : list of (float, float)
        Interval for each cumulative total, using the full covariance block
        up to that period.
    time_labels : np.ndarray
        Period labels, length ``T``.
    fit_diagnostics_detail : dict
        Pre/post RMSE and pre-period R-squared.
    placebo : GPITSPlacebo or None
        Placebo checks, when requested.
    metadata : dict
        Free-form diagnostics.
    """

    model_config = _ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: GPITSInputs
    design: GPITSDesign
    att_value: float
    att_interval: Tuple[float, float]
    observed: np.ndarray
    cf_series: np.ndarray
    gap_series: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    cumulative_effect: np.ndarray
    cumulative_ci: List[Tuple[float, float]]
    time_labels: np.ndarray
    fit_diagnostics_detail: Dict[str, Any]
    placebo: Optional[GPITSPlacebo] = None
    metadata: Dict[str, Any] = _PydField(default_factory=dict)

    @_model_validator(mode="after")
    def _populate_contract(self) -> "GPITSResults":
        if self.effects is not None:
            return self
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen)
        fd = self.fit_diagnostics_detail or {}
        times = np.asarray(self.time_labels)
        T0 = int(self.inputs.T0)

        cf_mean = float(np.mean(self.cf_series[T0:]))
        set_("effects", _EffectsResults(
            att=float(self.att_value),
            att_percent=(100.0 * self.att_value / cf_mean
                         if cf_mean not in (0.0, np.nan) else None)))
        set_("time_series", _TimeSeriesResults(
            observed_outcome=np.asarray(self.observed, dtype=float),
            counterfactual_outcome=np.asarray(self.cf_series, dtype=float),
            estimated_gap=np.asarray(self.gap_series, dtype=float),
            counterfactual_lower=np.asarray(self.counterfactual_lower, dtype=float),
            counterfactual_upper=np.asarray(self.counterfactual_upper, dtype=float),
            time_periods=times,
            intervention_time=(times[T0] if 0 < T0 < len(times) else None)))
        set_("fit_diagnostics", _FitDiagnosticsResults(
            rmse_pre=fd.get("rmse_pre"), rmse_post=fd.get("rmse_post"),
            r_squared_pre=fd.get("r_squared_pre")))
        set_("inference", _InferenceResults(
            method="gaussian_process_posterior",
            ci_lower=float(self.att_interval[0]),
            ci_upper=float(self.att_interval[1]),
            confidence_level=self.metadata.get("confidence_level"),
            details={"placebo": self.placebo,
                     "cumulative_ci": self.cumulative_ci}))
        set_("method_details", _MethodDetailsResults(
            method_name="GPITS",
            parameters_used={
                "kernel": self.design.kernel,
                "length_scale": self.design.length_scale,
                "noise_variance": self.design.noise_variance,
                "length_scale_selected": self.design.length_scale_selected,
                "noise_variance_selected": self.design.noise_variance_selected,
            }))
        return self
