"""Standardized result types for mlsynth.

mlsynth has exactly two output families, because panel causal inference has
exactly two modes -- measure an effect, or design to measure one -- and the
design mode resolves to the measurement mode:

* :class:`EffectResult` (an alias of :class:`BaseEstimatorResults`) -- the
  *observational report* returned by effect estimators (ATT, counterfactual,
  weights, inference).
* :class:`DesignResult` -- the *research design* returned by experimental
  design estimators; its :attr:`~DesignResult.report` is an
  :class:`EffectResult`.

Both share the common base :class:`MlsynthResult`. Estimators populate the
standardized sub-models (:class:`EffectsResults`, :class:`TimeSeriesResults`,
:class:`WeightsResults`, :class:`InferenceResults`,
:class:`FitDiagnosticsResults`, :class:`MethodDetailsResults`); every
:class:`EffectResult` additionally exposes the flat convenience accessors
``att``, ``att_ci``, ``counterfactual``, ``gap``, ``donor_weights`` and
``pre_rmse``.
"""

from .config_models import (
    BaseEstimatorResults,
    DesignResult,
    EffectResult,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    MlsynthResult,
    TimeSeriesResults,
    WeightsResults,
)

__all__ = [
    "MlsynthResult",
    "EffectResult",
    "DesignResult",
    "BaseEstimatorResults",
    "EffectsResults",
    "TimeSeriesResults",
    "WeightsResults",
    "InferenceResults",
    "FitDiagnosticsResults",
    "MethodDetailsResults",
]
