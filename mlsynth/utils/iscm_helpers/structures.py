"""Frozen dataclasses for the Imperfect Synthetic Controls (ISCM) estimator.

Powell, D. (2026). *"Imperfect Synthetic Controls."* Journal of Applied
Econometrics 41(3):253-264.

The standard SCM assumes a *perfect* synthetic control exists: the
treated unit lies inside the convex hull of the donors and its
pre-period outcomes are matched exactly. With transitory shocks this is
implausible -- an exact fit is impossible even in expectation. ISCM
relaxes this by

1. constructing synthetic controls for **every** unit (not just the
   treated one), so the treatment effect is identified even when the
   treated unit is *outside* the convex hull -- it can still appear as a
   donor for control units, and those units' post-treatment residuals
   carry information about the effect (paper eq. 6);
2. weighting units by a data-driven fit metric :math:`a_i` that
   asymptotically excludes units lacking a valid synthetic control
   (paper eq. 14), removing the researcher's eyeball "is the pre-fit
   good enough" judgment;
3. estimating the effect by weighted least squares across all units
   (paper eq. 8 / 15);
4. conducting inference via the Ibragimov-Muller t-statistic over the
   per-unit estimates (paper eq. 16), valid even with a very small
   donor pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
)


@dataclass(frozen=True)
class ISCMInputs:
    """Preprocessed panel for ISCM (synthetic controls for all units).

    Attributes
    ----------
    Y : np.ndarray
        Outcomes for every unit, shape ``(N, T)``.
    D : np.ndarray
        Treatment indicators, shape ``(N, T)``; ``D[i, t] = 1`` iff unit
        ``i`` is treated at period ``t``.
    T0 : int
        Number of pre-treatment periods (treatment starts at ``T0``).
    unit_names : list
        Length-``N`` unit identifiers.
    time_labels : np.ndarray
        Length-``T`` period labels.
    treated_idx : np.ndarray
        Indices of the ever-treated units.
    """

    Y: np.ndarray
    D: np.ndarray
    T0: int
    unit_names: List[Any]
    time_labels: np.ndarray
    treated_idx: np.ndarray

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]

    @property
    def n_post(self) -> int:
        return self.T - self.T0


class ISCMResults(_BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.ISCM.fit`.

    Attributes
    ----------
    inputs : ISCMInputs
        Preprocessed panel.
    att : float
        Average treatment effect on the treated, aggregated over the
        post-treatment period (paper eq. 15).
    unit_weight_matrix : np.ndarray
        All-units synthetic-control weight matrix, shape ``(N, N)``. Row
        ``i`` is the synthetic control for unit ``i`` (``[i, i] =
        0``, each row non-negative and summing to one).
    fit_metric : np.ndarray
        Per-unit fit weights :math:`a_i \\in (0, 1]`, shape ``(N,)``;
        ``1`` for the best-fitting unit, smaller for poorer synthetic
        controls (paper eq. 14).
    unit_att : np.ndarray
        Per-unit treatment-effect estimates, shape ``(N,)``; ``NaN`` for
        units that carry no identifying variation. Only units in the
        contributing set ``C`` (non-zero treatment exposure) are finite.
    contribution : np.ndarray
        Per-unit share :math:`v_i` of the aggregate ATT, shape ``(N,)``;
        sums to one over the contributing set (paper, before eq. 16).
    residuals : np.ndarray
        Synthetic-control residuals :math:`Y_{it} - \\sum_j w_{ij}
        Y_{jt}`, shape ``(N, T)``.
    exposure : np.ndarray
        Treatment exposure :math:`D_{it} - \\sum_j w_{ij} D_{jt}`, shape
        ``(N, T)`` -- the regressor in the WLS effect estimate.
    weights : WeightsResults, optional
        Standardized donor weights for the treated unit(s) -- the treated
        unit's row of ``unit_weight_matrix`` (a convex synthetic control
        over the other units). For multiple treated units, the cross-unit
        average; per-unit rows live in ``summary_stats``.
    inference : object, optional
        :class:`ISCMInference` when ``inference=True``; ``None`` otherwise.
    metadata : dict
        Free-form diagnostics.
    """

    model_config = _ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: ISCMInputs
    unit_weight_matrix: np.ndarray
    fit_metric: np.ndarray
    unit_att: np.ndarray
    contribution: np.ndarray
    residuals: np.ndarray
    exposure: np.ndarray
    inference_detail: Optional["ISCMInference"] = None
    metadata: Dict[str, Any] = _PydField(default_factory=dict)

    @_model_validator(mode="after")
    def _populate_contract(self) -> "ISCMResults":
        # Representative treated-average path so the contract's time_series has a
        # 1-D observed/counterfactual/gap (ISCM identifies the effect across all
        # contributing units; res.att is the aggregate, populated via effects).
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen)
        if self.time_series is None:
            Y = _np.asarray(self.inputs.Y, dtype=float)
            tr = _np.asarray(self.inputs.treated_idx).ravel()
            W = _np.asarray(self.unit_weight_matrix, dtype=float)
            obs = Y[tr].mean(axis=0)
            synth = (W[tr] @ Y).mean(axis=0)
            times = _np.asarray(self.inputs.time_labels)
            T0 = int(self.inputs.T0)
            set_("time_series", _TimeSeriesResults(
                observed_outcome=obs, counterfactual_outcome=synth,
                estimated_gap=obs - synth, time_periods=times,
                intervention_time=(times[T0] if T0 < len(times) else None)))
        if self.inference is None and self.inference_detail is not None:
            inf = self.inference_detail
            ci = getattr(inf, "ci", None)
            lo = float(ci[0]) if ci is not None and _np.isfinite(ci[0]) else None
            hi = float(ci[1]) if ci is not None and _np.isfinite(ci[1]) else None
            set_("inference", _InferenceResults(
                method=getattr(inf, "method", "ISCM"),
                ci_lower=lo, ci_upper=hi, details=inf))
        if self.fit_diagnostics is None:
            set_("fit_diagnostics", _FitDiagnosticsResults())
        if self.method_details is None:
            set_("method_details", _MethodDetailsResults(method_name="ISCM"))
        return self


@dataclass(frozen=True)
class ISCMInference:
    """Ibragimov-Muller inference for ISCM (paper Section 5, eq. 16).

    ISCM produces one treatment-effect estimate per contributing unit.
    Their (weighted) spread calibrates uncertainty via a sign-flip
    (Rademacher) randomization test -- conservative but valid even with a
    handful of donors, where a permutation test cannot reach standard
    significance thresholds.

    Attributes
    ----------
    method : str
        ``"ibragimov_muller"``.
    null_value : float
        The tested null effect :math:`\\alpha_0`.
    t_stat : float
        The Ibragimov-Muller test statistic (paper eq. 16).
    p_value : float
        Two-sided sign-flip randomization p-value.
    se : float
        Standard error implied by the per-unit estimate spread.
    ci : tuple of float
        Approximate two-sided confidence interval for the ATT.
    alpha_level : float
        Two-sided level used for ``ci``.
    n_contributing : int
        Size of the contributing set ``C``.
    n_draws : int
        Number of Rademacher draws.
    """

    method: str
    null_value: float
    t_stat: float
    p_value: float
    se: float
    ci: tuple
    alpha_level: float
    n_contributing: int
    n_draws: int
