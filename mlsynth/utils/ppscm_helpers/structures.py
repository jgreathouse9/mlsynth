"""Typed, NumPy-first result containers for Partially Pooled SCM (staggered).

PPSCM ports augsynth::multisynth (Ben-Michael, Feller & Rothstein 2022): a
partially-pooled synthetic control for staggered adoption that interpolates,
via ``nu``, between a separate SCM per treated unit (``nu`` small) and a fully
pooled SCM (``nu`` large), on top of two-way fixed effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy as _np
from pydantic import (
    ConfigDict as _ConfigDict,
    Field as _PydField,
    model_validator as _model_validator,
)

from ...config_models import (
    BaseEstimatorResults as _BaseEstimatorResults,
    EffectsResults as _EffectsResults,
    FitDiagnosticsResults as _FitDiagnosticsResults,
    InferenceResults as _InferenceResults,
    MethodDetailsResults as _MethodDetailsResults,
    TimeSeriesResults as _TimeSeriesResults,
    WeightsResults as _WeightsResults,
)


@dataclass(frozen=True)
class PPSCMInputs:
    """Preprocessed staggered panel (the only pandas touchpoint is ``setup``).

    Parameters
    ----------
    Xy : np.ndarray
        Full outcome matrix, shape ``(n, T)`` (units x all periods).
    trt : np.ndarray
        Adoption index per unit (position in ``time_labels``); ``inf`` for
        never-treated controls.
    n_pre : int
        Number of pre-treatment periods (columns before the last adoption).
    time_labels : np.ndarray
        Sorted time labels, length ``T``.
    units : np.ndarray
        Unit labels, length ``n``.
    outcome : str
        Outcome column name.
    intervention_time : Any
        The last adoption time (pre/post split point).
    """

    Xy: np.ndarray
    trt: np.ndarray
    n_pre: int
    time_labels: np.ndarray
    units: np.ndarray
    outcome: str
    intervention_time: Any

    @property
    def n(self) -> int:
        return int(self.Xy.shape[0])

    @property
    def treated_units(self) -> np.ndarray:
        return self.units[np.isfinite(self.trt)]

    @property
    def control_units(self) -> np.ndarray:
        return self.units[~np.isfinite(self.trt)]


@dataclass(frozen=True)
class PPSCMDesign:
    """The fitted design: pooling level and balance diagnostics."""

    nu_used: float
    lam: float
    fixedeff: bool
    time_cohort: bool
    n_leads: int
    n_lags: int
    global_l2: float
    ind_l2: float
    scaled_global_l2: float
    scaled_ind_l2: float

    @property
    def pct_improve_global(self) -> float:
        return 100.0 * (1.0 - self.scaled_global_l2)

    @property
    def pct_improve_ind(self) -> float:
        return 100.0 * (1.0 - self.scaled_ind_l2)


@dataclass(frozen=True)
class PPSCMEventStudy:
    """Relative-time (time-since-treatment) average ATT path."""

    horizons: np.ndarray          # 0, 1, ..., n_leads-1
    tau: np.ndarray               # n1-weighted average effect per horizon
    se: np.ndarray
    ci: np.ndarray                # (H, 2)


@dataclass(frozen=True)
class PPSCMInference:
    """Overall (post-period average) ATT and its inference."""

    att: float
    se: float
    ci: Tuple[float, float]
    method: str


class PPSCMResults(_BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.PPSCM.fit`.

    An :class:`~mlsynth.config_models.EffectResult`. PPSCM is a staggered /
    partially-pooled estimator, so the standardized ``time_series`` carries the
    pooled event-time effect path (``gap`` = horizon effect, ``counterfactual`` =
    no-effect baseline), and ``effects.att`` is the aggregate ATT -- mirroring the
    SequentialSDID convention. The native objects are preserved: ``inference_detail``
    (the :class:`PPSCMInference`, formerly ``inference``) and ``donor_weights_by_cohort``
    (the nested per-cohort weights, formerly ``donor_weights``); the contract names
    ``inference`` / ``donor_weights`` are taken by the base contract.
    """

    model_config = _ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: PPSCMInputs
    design: PPSCMDesign
    event_study: PPSCMEventStudy
    inference_detail: PPSCMInference
    donor_weights_by_cohort: Dict[Any, Dict[Any, float]]
    metadata: Dict[str, Any] = _PydField(default_factory=dict)

    @property
    def nu(self) -> float:
        return self.design.nu_used

    @_model_validator(mode="after")
    def _populate_contract(self) -> "PPSCMResults":
        if self.effects is not None:
            return self
        inf = self.inference_detail
        es = self.event_study
        tau = _np.asarray(es.tau, dtype=float)
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen)
        se = float(inf.se) if _np.isfinite(inf.se) else None
        ci_lo, ci_hi = float(inf.ci[0]), float(inf.ci[1])
        finite_ci = _np.isfinite(ci_lo) and _np.isfinite(ci_hi)
        set_("effects", _EffectsResults(att=float(inf.att), att_std_err=se))
        set_("time_series", _TimeSeriesResults(
            observed_outcome=tau, counterfactual_outcome=_np.zeros_like(tau),
            estimated_gap=tau, time_periods=_np.asarray(es.horizons),
            intervention_time=0))
        set_("weights", _WeightsResults(summary_stats={
            "constraint": "partially-pooled SC donor weights (per cohort)"}))
        set_("inference", _InferenceResults(
            method=inf.method, standard_error=se,
            ci_lower=(ci_lo if finite_ci else None),
            ci_upper=(ci_hi if finite_ci else None), details=inf))
        set_("fit_diagnostics", _FitDiagnosticsResults())
        set_("method_details", _MethodDetailsResults(method_name="PPSCM"))
        return self
