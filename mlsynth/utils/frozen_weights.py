"""Freeze a fitted SCM's weights and reapply them to new data without refitting.

This is the "commission once, refresh forever" mechanic. A synthetic-control fit
is reduced to a small, serializable artifact -- the donor-weight matrix plus the
pre-period residuals -- and that artifact is reapplied to an extended panel to get
an updated counterfactual and refreshed inference, with no optimization.

For the pre-period-weighting family (MSQRT and kin) the donor weights are
estimated only on the pre-period, so extending the post window cannot change
them: the reapplication is exact, not an approximation. :func:`apply_frozen_weights`
returns a standard :class:`~mlsynth.config_models.BaseEstimatorResults`, so a
refresh flows through the same tooling (comparison, plotting, result contract) as
a fit.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from ..config_models import BaseEstimatorResults, InferenceResults, WeightsResults
from ..exceptions import MlsynthConfigError, MlsynthDataError
from .results_helpers import build_effect_submodels
from .scpi_helpers import out_of_sample_intervals

_TOL = 1e-3


class FrozenWeights(BaseModel):
    """A fitted SCM's frozen donor weights, ready to reapply to new data.

    Attributes
    ----------
    estimator : str
        Estimator that produced the weights (e.g. ``"MSQRT"``).
    weights : np.ndarray
        Donor-weight matrix, shape ``(n_donors, m_treated)``.
    donor_names, treated_names : list of str
        Row/column labels for ``weights`` -- the ordering the refresh aligns to.
    pre_residuals : np.ndarray
        Pre-period gap ``Y_pre - X_pre @ weights``, shape ``(T0, m_treated)``;
        the fixed input to the refreshed out-of-sample inference.
    intervention_time : Any
        Pre/post boundary; periods at or after it are post-treatment.
    alpha : float
        Miscoverage level for the prediction intervals.
    time_dependence : {"iid", "general"}
        Time-averaging assumption passed to the interval routine.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    estimator: str
    weights: np.ndarray
    donor_names: List[str]
    treated_names: List[str]
    pre_residuals: np.ndarray
    intervention_time: Any
    alpha: float = 0.1
    time_dependence: str = "iid"

    @field_validator("weights", "pre_residuals", mode="before")
    @classmethod
    def _to_array(cls, v: Any) -> np.ndarray:
        return np.asarray(v, dtype=float)

    @field_validator("intervention_time", mode="before")
    @classmethod
    def _native(cls, v: Any) -> Any:
        return v.item() if hasattr(v, "item") else v       # numpy scalar -> python

    @field_serializer("weights", "pre_residuals")
    def _arr_to_list(self, v: np.ndarray) -> list:
        return np.asarray(v).tolist()


def freeze(
    result: Any,
    *,
    alpha: Optional[float] = None,
    time_dependence: str = "iid",
) -> FrozenWeights:
    """Extract a :class:`FrozenWeights` artifact from a fitted result.

    Supports MSQRT results (donor-weight matrix ``theta`` fit on the pre-period).

    Raises
    ------
    MlsynthConfigError
        If the result does not expose the MSQRT freeze surface.
    """
    theta = getattr(result, "theta", None)
    inputs = getattr(result, "inputs", None)
    if theta is None or inputs is None or not hasattr(inputs, "control_names"):
        raise MlsynthConfigError(
            "freeze() supports MSQRT results (need .theta and .inputs); "
            f"got {type(result).__name__}."
        )
    theta = np.asarray(theta, dtype=float)
    pre_residuals = np.asarray(inputs.Y_pre) - np.asarray(inputs.X_pre) @ theta

    if alpha is None:
        inf = getattr(result, "inference", None)
        cl = getattr(inf, "confidence_level", None) if inf is not None else None
        alpha = float(1.0 - cl) if cl is not None else 0.1

    return FrozenWeights(
        estimator=str(getattr(getattr(result, "method_details", None),
                              "method_name", None) or "MSQRT"),
        weights=theta,
        donor_names=[str(c) for c in inputs.control_names],
        treated_names=[str(t) for t in inputs.treated_names],
        pre_residuals=pre_residuals,
        intervention_time=inputs.time_labels[inputs.T0],
        alpha=float(alpha),
        time_dependence=time_dependence,
    )


def _overall_weights(frozen: FrozenWeights) -> WeightsResults:
    """Donor weights for the standard surface: per-treated mean, MSQRT-style."""
    theta, donors = frozen.weights, frozen.donor_names
    per_unit = {
        t: {donors[k]: float(theta[k, j]) for k in range(len(donors))
            if abs(theta[k, j]) > _TOL}
        for j, t in enumerate(frozen.treated_names)
    }
    pooled = {d: float(np.mean(theta[k])) for k, d in enumerate(donors)}
    pooled = {d: w for d, w in pooled.items() if abs(w) > _TOL}
    return WeightsResults(
        donor_weights=pooled,
        summary_stats={"per_unit_donor_weights": per_unit,
                       "n_treated": len(frozen.treated_names),
                       "n_donors": len(donors)},
    )


def apply_frozen_weights(
    frozen: FrozenWeights,
    panel: pd.DataFrame,
    *,
    outcome: str,
    unitid: str,
    time: str,
) -> BaseEstimatorResults:
    """Reapply frozen weights to an extended panel; refresh the readout. No refit.

    Builds the donor matrix from ``panel`` aligned to ``frozen.donor_names``,
    forms ``counterfactual = donors @ weights``, splits pre/post at
    ``frozen.intervention_time``, and re-runs the out-of-sample intervals using
    the frozen ``pre_residuals`` plus the new post-period gap. Returns a standard
    :class:`~mlsynth.config_models.BaseEstimatorResults`; the full
    :class:`SCPIResults` (per-unit and per-period bands) is on
    ``inference.details`` and the per-unit matrices on ``additional_outputs``.

    Raises
    ------
    MlsynthDataError
        If a donor or treated unit is missing from ``panel``, or the panel has no
        period at or after ``frozen.intervention_time``.
    """
    for col in (outcome, unitid, time):
        if col not in panel.columns:
            raise MlsynthDataError(f"panel is missing column {col!r}.")

    wide = panel.pivot(index=unitid, columns=time, values=outcome).sort_index(axis=1)
    missing = [u for u in frozen.donor_names + frozen.treated_names
               if u not in wide.index]
    if missing:
        raise MlsynthDataError(
            f"panel is missing {len(missing)} unit(s) the artifact needs: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
        )

    weeks = list(wide.columns)
    is_post = np.array([w >= frozen.intervention_time for w in weeks])
    n_post = int(is_post.sum())
    if n_post < 1:
        raise MlsynthDataError(
            f"panel has no period at or after intervention_time "
            f"{frozen.intervention_time!r}; nothing to refresh."
        )
    T0 = int((~is_post).sum())

    X = wide.loc[frozen.donor_names].to_numpy().T          # (T, n_donors)
    Y = wide.loc[frozen.treated_names].to_numpy().T        # (T, m_treated)
    synth = X @ frozen.weights                             # frozen weights
    post_gap = (Y - synth)[is_post]
    post_weeks = [w for w, p in zip(weeks, is_post) if p]

    scpi = out_of_sample_intervals(
        effects=post_gap, pre_residuals=frozen.pre_residuals,
        unit_names=frozen.treated_names, period_labels=post_weeks,
        alpha=frozen.alpha, time_dependence=frozen.time_dependence,
    )

    observed_mean = Y.mean(axis=1)
    synth_mean = synth.mean(axis=1)
    synth_post_mean = float(synth[is_post].mean())
    att = float(post_gap.mean())
    att_pct = float(100.0 * att / synth_post_mean) if synth_post_mean else float("nan")
    pre_rmse = float(np.sqrt(np.mean(frozen.pre_residuals ** 2)))

    std_inference = InferenceResults(
        method=scpi.method, ci_lower=float(scpi.taua.lower),
        ci_upper=float(scpi.taua.upper),
        confidence_level=float(1.0 - frozen.alpha), details=scpi,
    )
    submodels = build_effect_submodels(
        observed_outcome=observed_mean, counterfactual_outcome=synth_mean,
        n_pre_periods=T0, n_post_periods=n_post,
        time_periods=np.asarray(weeks), weights=_overall_weights(frozen),
        inference=std_inference, method_name=f"{frozen.estimator}-refresh",
        effects_overrides={"att": att, "att_percent": att_pct},
        fit_overrides={"rmse_pre": pre_rmse},
        intervention_time=frozen.intervention_time,
    )
    unit_att = {t: float((Y[is_post, j] - synth[is_post, j]).mean())
                for j, t in enumerate(frozen.treated_names)}
    return BaseEstimatorResults(
        **submodels,
        additional_outputs={
            "refresh": True,
            "counterfactual_matrix": synth,
            "observed_matrix": Y,
            "treated_names": frozen.treated_names,
            "unit_att": unit_att,
            "asof": weeks[-1],
        },
    )
