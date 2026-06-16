"""Typed result containers for MicroSynth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import numpy as _np
from pydantic import (
    ConfigDict as _ConfigDict,
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
class MicroSynthInputs:
    """Pre-processed user-level matrices for MicroSynth.

    Parameters
    ----------
    X_T : np.ndarray
        Treated-user covariate matrix, shape ``(n_T, d)``. Already
        standardized if ``standardize_covariates=True``.
    X_C : np.ndarray
        Control-user covariate matrix, shape ``(n_C, d)``. Same
        standardization as ``X_T``.
    Y_T : np.ndarray
        Treated-user post-treatment outcomes, shape
        ``(n_T, T_post)`` where ``T_post`` is the number of
        post-treatment periods. If ``T_post = 1`` this is collapsed
        to ``(n_T,)``.
    Y_C : np.ndarray
        Control-user post-treatment outcomes, shape ``(n_C, T_post)``
        or ``(n_C,)`` matching ``Y_T``.
    treated_unit_names : Sequence
        Identifiers of the treated users, in row order of ``X_T``.
    control_unit_names : Sequence
        Identifiers of the control users, in row order of ``X_C``.
    covariate_names : Sequence[str]
        Labels of the balancing constraints in column order of ``X_T``
        / ``X_C``. Includes both the user-supplied ``covariates`` and
        any ``outcome_lag_periods`` columns.
    n_T, n_C, d, T_post : int
        Cached shapes.
    cohort_time : Any
        The treatment-onset time inferred from ``df``.
    covariate_sd : np.ndarray
        Pooled SD used for standardization, shape ``(d,)``. ``None``
        if standardization was disabled.
    outcome : str
        Outcome column name.
    cov_T_raw, cov_C_raw : np.ndarray
        Raw (un-standardized) covariate matrices, shapes ``(n_T, d_cov)``
        and ``(n_C, d_cov)`` -- the ``covariates`` columns only, excluding
        any lagged-outcome predictors. Used by the panel-method QP, which
        balances treated **totals** and therefore needs raw values.
    lag_T_raw, lag_C_raw : np.ndarray
        Raw lagged-outcome matrices, shapes ``(n_T, m)`` and ``(n_C, m)``
        where ``m = len(outcome_lag_periods)``. Empty (zero-column) arrays
        when no lags were requested.
    """

    X_T: np.ndarray
    X_C: np.ndarray
    Y_T: np.ndarray
    Y_C: np.ndarray
    treated_unit_names: Sequence
    control_unit_names: Sequence
    covariate_names: Sequence
    cohort_time: Any
    covariate_sd: Optional[np.ndarray]
    outcome: str
    cov_T_raw: Optional[np.ndarray] = None
    cov_C_raw: Optional[np.ndarray] = None
    lag_T_raw: Optional[np.ndarray] = None
    lag_C_raw: Optional[np.ndarray] = None

    @property
    def n_T(self) -> int:
        return int(self.X_T.shape[0])

    @property
    def n_C(self) -> int:
        return int(self.X_C.shape[0])

    @property
    def d(self) -> int:
        return int(self.X_T.shape[1])

    @property
    def T_post(self) -> int:
        return 1 if self.Y_T.ndim == 1 else int(self.Y_T.shape[1])


@dataclass(frozen=True)
class MicroSynthDesign:
    """Outputs of the dual ascent + balance diagnostics.

    Parameters
    ----------
    w : np.ndarray
        Control-side weights on the simplex, shape ``(n_C,)``.
        ``sum(w) == 1``, ``w >= 0``.
    dual_lambda : np.ndarray
        Lagrange multipliers for the covariate balance constraints,
        shape ``(d,)``.
    dual_nu : float
        Lagrange multiplier for the sum-to-one constraint.
    smd_before : np.ndarray
        Per-covariate standardized mean difference between treated
        and unweighted controls, shape ``(d,)``.
    smd_after : np.ndarray
        Per-covariate SMD after applying ``w``, shape ``(d,)``.
        Should be near zero on every constraint.
    ess : float
        Effective sample size of the weighted control group,
        ``1 / sum(w^2)``.
    max_weight : float
        Largest single control-user weight.
    feasible : bool
        ``True`` if every ``|smd_after_k| < balance_tol``. ``False``
        signals that the QP did not achieve balance and the treated
        group may lie outside the convex hull of controls.
    feasibility_message : str
        Human-readable diagnostic.
    n_iterations : int
        L-BFGS-B iterations to convergence.
    converged : bool
        Whether the optimizer reported success.
    """

    w: np.ndarray
    dual_lambda: np.ndarray
    dual_nu: float
    smd_before: np.ndarray
    smd_after: np.ndarray
    ess: float
    max_weight: float
    feasible: bool
    feasibility_message: str
    n_iterations: int
    converged: bool


@dataclass(frozen=True)
class MicroSynthInference:
    """Inference summary: bootstrap (simplex) or permutation (panel)."""

    method: str                     # "paired_bootstrap", "permutation", or "none"
    att: float                       # point estimate
    se: float                        # bootstrap / permutation SE
    ci: np.ndarray                   # [low, high]
    n_bootstrap: int                 # successful reps (bootstrap or permutation)
    bootstrap_atts: np.ndarray       # full distribution (bootstrap or placebo)
    # Panel-method permutation extras (NaN / empty for other methods).
    p_value: float = float("nan")            # ATT-level placebo p-value
    p_values_by_period: Optional[np.ndarray] = None  # per-post-period p-values
    test: Optional[str] = None               # "lower" / "upper" / "twosided"


class MicroSynthResults(_BaseEstimatorResults):
    """Public return container for ``MicroSynth.fit()``.

    Parameters
    ----------
    inputs : MicroSynthInputs
        Pre-processed inputs.
    design : MicroSynthDesign
        Weights, dual variables, balance diagnostics.
    inference : MicroSynthInference
        Bootstrap CI on the ATT (or ``method = "none"`` if disabled).
    counterfactual : np.ndarray
        Weighted-control outcomes per post-treatment period,
        shape matches ``Y_T``.
    gap : np.ndarray
        Treated mean minus counterfactual, per post-treatment period.
        Shape matches ``Y_T``.
    gap_trajectory : np.ndarray
        Per-post-period gap, always 1-D (length ``T_post``).
    att : float
        Mean of ``gap_trajectory``.
    donor_weights : Dict[Any, float]
        ``{control_user_name: w_i}`` for all controls with
        ``w_i > 0``.
    """

    model_config = _ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MicroSynthInputs
    design: MicroSynthDesign
    inference_detail: MicroSynthInference
    counterfactual_post: np.ndarray
    gap_post: np.ndarray
    gap_trajectory: np.ndarray
    att_value: float
    donor_weights_map: Dict[Any, float]

    @_model_validator(mode="after")
    def _populate_contract(self) -> "MicroSynthResults":
        if self.effects is not None:
            return self
        cf = _np.atleast_1d(_np.asarray(self.counterfactual_post, dtype=float))
        gap = _np.atleast_1d(_np.asarray(self.gap_post, dtype=float))
        inf = self.inference_detail
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen)
        set_("effects", _EffectsResults(
            att=float(self.att_value),
            att_std_err=(float(inf.se) if inf is not None and _np.isfinite(inf.se) else None)))
        # MicroSynth reports post-period series (observed = counterfactual + gap).
        set_("time_series", _TimeSeriesResults(
            observed_outcome=cf + gap, counterfactual_outcome=cf,
            estimated_gap=gap))
        set_("weights", _WeightsResults(donor_weights={
            str(k): float(v) for k, v in self.donor_weights_map.items()}))
        if inf is not None:
            ci = _np.asarray(getattr(inf, "ci", [_np.nan, _np.nan]), dtype=float)
            finite = ci.size == 2 and _np.all(_np.isfinite(ci))
            set_("inference", _InferenceResults(
                method=getattr(inf, "method", None),
                standard_error=(float(inf.se) if _np.isfinite(inf.se) else None),
                p_value=(float(inf.p_value) if _np.isfinite(getattr(inf, "p_value", _np.nan)) else None),
                ci_lower=(float(ci[0]) if finite else None),
                ci_upper=(float(ci[1]) if finite else None), details=inf))
        set_("fit_diagnostics", _FitDiagnosticsResults())
        set_("method_details", _MethodDetailsResults(method_name="MicroSynth"))
        return self
