"""Frozen, NumPy-first containers for Ensemble Synthetic Control (ESC).

Everything below is pure NumPy; the only DataFrame touchpoint is
:func:`mlsynth.utils.esc_helpers.setup.prepare_esc_inputs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from ...config_models import BaseEstimatorResults, InferenceResults, WeightsResults
from ...utils.results_helpers import build_effect_submodels


@dataclass(frozen=True)
class ESCInputs:
    """Preprocessed, NumPy-only panel for the ESC engine.

    Parameters
    ----------
    treated_name : Any
        Label of the treated unit.
    donor_names : Sequence
        Length-``J`` donor-unit labels (columns of ``donor_matrix``).
    y : np.ndarray
        Treated-unit outcome series, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcome pool, shape ``(T, J)``.
    T0 : int
        Number of pre-treatment periods.
    time_labels : Sequence
        Length-``T`` period labels.
    metadata : dict
        Free-form provenance.
    """

    treated_name: Any
    donor_names: Sequence
    y: np.ndarray
    donor_matrix: np.ndarray
    T0: int
    time_labels: Sequence
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.y.shape[0])

    @property
    def J(self) -> int:
        return int(self.donor_matrix.shape[1])


@dataclass(frozen=True)
class ESCFit:
    """The aggregated ESC fit plus its ensemble prediction interval."""

    counterfactual: np.ndarray            # (T,) ensemble point path (eq. 5)
    gap: np.ndarray                       # (T,) observed - counterfactual
    att: float                            # average post-treatment gap
    lower: np.ndarray                     # (T,) pointwise counterfactual PI lower (eq. 7)
    upper: np.ndarray                     # (T,) pointwise counterfactual PI upper
    ensemble: np.ndarray                  # (B, T) base-learner counterfactuals
    donor_weights: Dict[Any, float]       # mean nonzero weight per donor across learners
    pre_rmse: float                       # ensemble-mean pre-treatment RMSE
    att_ci: Tuple[Optional[float], Optional[float]] = (None, None)
    att_se: Optional[float] = None
    level: float = 0.90
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ESCResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.ESC.fit` (an
    :class:`~mlsynth.config_models.EffectResult`)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: ESCInputs
    fit: ESCFit
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "ESCResults":
        if self.effects is not None:  # pragma: no cover - idempotency guard
            return self
        fit = self.fit
        labels = np.asarray(self.inputs.time_labels)
        T0, T = self.inputs.T0, self.inputs.T
        inference = None
        if fit.att_ci[0] is not None:
            inference = InferenceResults(
                ci_lower=float(fit.att_ci[0]), ci_upper=float(fit.att_ci[1]),
                standard_error=(None if fit.att_se is None else float(fit.att_se)),
                confidence_level=float(fit.level),
                method="ensemble prediction interval (Lian-Pu 2026)",
                details={"n_learners": int(fit.ensemble.shape[0])},
            )
        subs = build_effect_submodels(
            observed_outcome=np.asarray(self.inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(fit.counterfactual, dtype=float),
            n_pre_periods=T0, n_post_periods=T - T0,
            time_periods=labels,
            weights=WeightsResults(
                donor_weights={str(k): float(v) for k, v in fit.donor_weights.items()}),
            inference=inference,
            method_name="ESC",
            intervention_time=(labels[T0] if T0 < T else None),
            prediction_interval={
                "lower": np.asarray(fit.lower, dtype=float),
                "upper": np.asarray(fit.upper, dtype=float),
                "level": float(fit.level),
                "kind": "ensemble:esc",
            },
            additional_metrics={"n_learners": int(fit.ensemble.shape[0]),
                                **{k: fit.params.get(k) for k in fit.params}},
        )
        for name, model in subs.items():
            object.__setattr__(self, name, model)
        # method_details carries the ESC hyper-parameters used
        md = self.method_details
        if md is not None:
            object.__setattr__(md, "parameters_used", dict(fit.params))
        return self


ESCResults.model_rebuild()
