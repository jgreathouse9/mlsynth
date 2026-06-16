"""Frozen, NumPy-first containers for the Synthetic Historical Control (SHC).

Implements the containers for

    Chen, Y.-T., Yang, J.-C., & Yang, T.-T. (2024). "Synthetic Historical
    Control for Policy Evaluation." SSRN 4995085.

SHC reconstructs a single treated unit's untreated counterfactual *without
cross-sectional controls*. It estimates a smooth latent trend
:math:`\\ell_t` by kernel regression on the pre-period, partitions the
series into a "treated block" and a set of overlapping "historical
blocks", and matches the treated block's pre-segment with a simplex
combination of the historical blocks (Section 2.2). The same combination,
applied to the historical blocks' forward segments, yields the
post-intervention counterfactual.

Everything below is pure NumPy; time periods are addressed through the
repository's :class:`IndexSet`. The only DataFrame touchpoint is
``setup``.
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

from ..helperutils import IndexSet
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
class SHCInputs:
    """Preprocessed, NumPy-only inputs for the SHC engine.

    Parameters
    ----------
    time_index : IndexSet
        All ``T`` period labels (row order of ``y``).
    y : np.ndarray
        Treated-unit outcome over all periods, shape ``(T,)``.
    T0 : int
        Number of pre-treatment periods; post is ``n = T - T0``.
    m : int
        Pre-intervention window length of the treated/historical blocks.
    treated_label : Any
        Identifier of the treated unit.
    metadata : dict
        Free-form provenance (e.g. the wide frame from ``dataprep``).
    """

    time_index: IndexSet
    y: np.ndarray
    T0: int
    m: int
    treated_label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def n(self) -> int:
        """Post-intervention horizon."""
        return self.T - self.T0

    @property
    def N(self) -> int:
        """Number of historical blocks (Eq. 7): ``T0 - n - (m - 1)``."""
        return self.T0 - self.n - (self.m - 1)


@dataclass(frozen=True)
class SHCInference:
    """Conformal permutation inference (Chen-Yang-Yang 2024, footnote 21).

    Parameters
    ----------
    method : str
        Always ``"conformal_permutation"``.
    test_statistic : float
        :math:`S = n^{-1/2} \\sum_t |\\hat\\varepsilon_t^0|` over the post
        period.
    p_value : float
        :math:`\\Pr(S^* \\ge S)` under the resampled null.
    critical_values : dict
        Mapping significance level -> upper-tail critical value of ``S^*``.
    reject : dict
        Mapping significance level -> reject decision (``S > cv``).
    num_resamples : int
        Number of null resamples (1000 in the paper).
    null_distribution : np.ndarray
        The resampled ``S^*`` values.
    conformal_lower, conformal_upper : np.ndarray
        Post-period Andrews-Genton conformal bands, retained for plotting.
    confidence_level : float
        Coverage of the conformal bands (e.g. 0.90).
    """

    method: str
    test_statistic: float
    p_value: float
    critical_values: Dict[float, float]
    reject: Dict[float, bool]
    num_resamples: int
    null_distribution: np.ndarray
    conformal_lower: np.ndarray
    conformal_upper: np.ndarray
    confidence_level: float


@dataclass(frozen=True)
class SHCDesign:
    """SHC fitted design.

    Parameters
    ----------
    bandwidth : float
        LOOCV-selected kernel bandwidth for the latent-trend smoother.
    latent_pre : np.ndarray
        Kernel-smoothed latent trend over the pre-period, shape ``(T0,)``
        (the first-stage :math:`\\hat\\ell_t`).
    weights : np.ndarray
        Full length-``N`` historical-block weights (mostly zero).
    selected_blocks : list of int
        Indices of the historical blocks with non-zero weight.
    block_weights : dict
        Mapping ``block_label -> weight`` for the selected blocks.
    counterfactual_window : np.ndarray
        SHC counterfactual over the ``m + n`` treated-block window
        (pre-segment reconstruction followed by the post-intervention
        prediction).
    use_augmented : bool
        Whether the augmented (ASHC) ridge refinement was applied.
    best_lambda : float or None
        ASHC ridge penalty chosen by tuning; ``None`` for plain SHC.
    """

    bandwidth: float
    latent_pre: np.ndarray
    weights: np.ndarray
    selected_blocks: List[int]
    block_weights: Dict[Any, float]
    counterfactual_window: np.ndarray
    use_augmented: bool
    best_lambda: Optional[float] = None


class SHCResults(_BaseEstimatorResults):
    """Public container returned by :meth:`mlsynth.SHC.fit`.

    Parameters
    ----------
    inputs : SHCInputs
        Preprocessed series.
    design : SHCDesign
        Bandwidth, latent trend, block weights, and counterfactual.
    att : float
        Mean post-intervention gap ``mean(observed - counterfactual)``.
    att_percent : float
        ATT as a percentage of the mean counterfactual.
    observed : np.ndarray
        Observed treated series over the ``m + n`` block window.
    counterfactual : np.ndarray
        SHC counterfactual over the same window
        (= ``design.counterfactual_window``).
    gap : np.ndarray
        ``observed - counterfactual`` over the window.
    time_labels : np.ndarray
        Period labels for the ``m + n`` window.
    fit_diagnostics : dict
        Pre/post RMSE and pre-period R-squared.
    inference : SHCInference or None
        Conformal permutation test output.
    metadata : dict
        Free-form diagnostics (m, n, N, bandwidth, augmentation).
    """

    model_config = _ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SHCInputs
    design: SHCDesign
    att_percent: float
    observed: np.ndarray
    cf_window: np.ndarray
    gap_window: np.ndarray
    time_labels: np.ndarray
    fit_diagnostics_detail: Dict[str, Any]
    att_value: float
    inference_detail: Optional[SHCInference] = None
    metadata: Dict[str, Any] = _PydField(default_factory=dict)

    @property
    def weights_by_block(self) -> Dict[Any, float]:
        return self.design.block_weights

    @_model_validator(mode="after")
    def _populate_contract(self) -> "SHCResults":
        if self.effects is not None:
            return self
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen)
        fd = self.fit_diagnostics_detail or {}
        times = _np.asarray(self.time_labels)
        m = int(getattr(self.inputs, "m", 0))
        set_("effects", _EffectsResults(
            att=float(self.att_value), att_percent=float(self.att_percent)))
        set_("time_series", _TimeSeriesResults(
            observed_outcome=_np.asarray(self.observed, dtype=float),
            counterfactual_outcome=_np.asarray(self.cf_window, dtype=float),
            estimated_gap=_np.asarray(self.gap_window, dtype=float),
            time_periods=times,
            intervention_time=(times[m] if 0 < m < len(times) else None)))
        set_("weights", _WeightsResults(donor_weights={
            str(k): float(v) for k, v in self.design.block_weights.items()}))
        set_("fit_diagnostics", _FitDiagnosticsResults(
            rmse_pre=fd.get("rmse_pre"), rmse_post=fd.get("rmse_post"),
            r_squared_pre=fd.get("r_squared_pre")))
        if self.inference_detail is not None:
            inf = self.inference_detail
            set_("inference", _InferenceResults(
                method=getattr(inf, "method", None),
                p_value=getattr(inf, "p_value", None), details=inf))
        set_("method_details", _MethodDetailsResults(method_name="SHC"))
        return self
