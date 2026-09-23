"""Typed, NumPy-first result containers for generalized synthetic control.

GSYNTH ports Xu (2017): an interactive fixed effects model is fit to the
never-treated units, each treated unit's factor loadings come from its own
pre-treatment periods, and the untreated potential outcome follows from the two.
The layers below (inputs, cross-validation, fit, design, event study,
inference, results) keep that pipeline pluggable.
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


@dataclass(frozen=True)
class GSYNTHInputs:
    """Preprocessed staggered panel; ``setup`` is the only pandas touchpoint.

    Parameters
    ----------
    Y : np.ndarray
        Outcomes, shape ``(T, N)`` -- periods by units.
    D : np.ndarray
        Treatment indicator, same shape, absorbing in every column.
    X : np.ndarray
        Covariates, shape ``(T, N, p)``; ``p = 0`` when none were given.
    treated_index, control_index : np.ndarray
        Column positions of the ever-treated and never-treated units.
    adoption_index : np.ndarray
        Each treated unit's first treated period, as a position in
        ``time_labels``, aligned with ``treated_index``.
    unit_labels, time_labels : np.ndarray
        Panel labels, lengths ``N`` and ``T``.
    covariate_names : tuple of str
        Names of the covariate slices, length ``p``.
    """

    Y: np.ndarray
    D: np.ndarray
    X: np.ndarray
    treated_index: np.ndarray
    control_index: np.ndarray
    adoption_index: np.ndarray
    unit_labels: np.ndarray
    time_labels: np.ndarray
    covariate_names: Tuple[str, ...] = ()

    @property
    def T(self) -> int:
        """Number of periods."""
        return int(self.Y.shape[0])

    @property
    def N(self) -> int:
        """Number of units."""
        return int(self.Y.shape[1])

    @property
    def p(self) -> int:
        """Number of covariates."""
        return int(self.X.shape[2])

    @property
    def n_treated(self) -> int:
        """Number of ever-treated units."""
        return int(self.treated_index.size)

    @property
    def n_control(self) -> int:
        """Number of never-treated units."""
        return int(self.control_index.size)

    @property
    def min_pre_periods(self) -> int:
        """Shortest pre-treatment history among the treated units.

        This bounds the rank: a unit with ``T0`` pre-periods can identify at
        most ``T0`` loadings, and one fewer once an intercept is estimated
        alongside them.
        """
        return int(self.adoption_index.min())


@dataclass(frozen=True)
class GSYNTHCrossValidation:
    """Algorithm 1: leave-one-period-out selection of the factor count.

    Parameters
    ----------
    r_selected : int
        Rank minimizing the mean squared prediction error.
    mspe : dict
        ``{r: MSPE}`` over the ranks considered.
    n_scored : dict
        ``{r: held-out cells}``. Constant across ``r`` by construction, so the
        criterion is comparable; a rank that changed it would not be.
    """

    r_selected: int
    mspe: Dict[int, float]
    n_scored: Dict[int, int]


@dataclass(frozen=True)
class GSYNTHFit:
    """Steps 2 and 3: treated loadings and the imputed untreated outcome.

    Parameters
    ----------
    effect : np.ndarray
        ``Y - hat Y(0)`` for the treated units, shape ``(T, N_tr)``. Pre-period
        entries are a placebo diagnostic; post-period entries are the estimate.
    counterfactual : np.ndarray
        ``hat Y(0)`` for the treated units, same shape.
    att : float
        Mean effect over the treated post-treatment cells.
    n_post_cells : int
        How many cells that mean is taken over.
    loadings : np.ndarray
        Treated-unit loadings, shape ``(N_tr, r + 1)`` when ``force`` carries
        unit effects -- the trailing column is the unit effect, estimated
        jointly with the loadings off the pre-periods -- and ``(N_tr, r)``
        otherwise.
    unit_effects : np.ndarray
        That trailing column on its own, shape ``(N_tr,)``; zeros when additive
        unit effects are off.
    factors_used : np.ndarray
        The regressor matrix Step 2 projects onto: the estimated factors, with
        a column of ones appended when ``force`` carries unit effects.
    control_fit : ControlFit
        The Step 1 fit these were built from.
    """

    effect: np.ndarray
    counterfactual: np.ndarray
    att: float
    n_post_cells: int
    loadings: np.ndarray
    unit_effects: np.ndarray
    factors_used: np.ndarray
    control_fit: Any


@dataclass(frozen=True)
class GSYNTHDesign:
    """What the estimator settled on, and the fitted structure.

    Parameters
    ----------
    r_selected : int
        Factor count used.
    r_source : {"cv", "user"}
        Whether Algorithm 1 chose it or the caller fixed it.
    force : {"none", "unit", "time", "two-way"}
        Which additive effects were included.
    beta : np.ndarray
        Covariate coefficients, shape ``(p,)``.
    covariate_names : tuple of str
        Names aligned with ``beta``.
    dropped_covariates : tuple of str
        Covariates dropped as collinear with the fixed effects; their
        coefficients are reported as zero.
    factors : np.ndarray
        Estimated factors, shape ``(T, r)``, normalized ``F'F / T = I_r``.
    control_loadings : np.ndarray
        Never-treated loadings, shape ``(N_co, r)``.
    treated_loadings : np.ndarray
        Treated loadings, shape ``(N_tr, r)``.
    unit_effects : np.ndarray
        Treated unit effects, shape ``(N_tr,)``.
    period_effects : np.ndarray
        Period effects, shape ``(T,)``.
    grand_mean : float
        The fitted intercept.
    n_iterations : int
        Alternating-least-squares iterations in Step 1.
    cv : GSYNTHCrossValidation or None
        The selection record; ``None`` when the rank was fixed by the caller.
    """

    r_selected: int
    r_source: str
    force: str
    beta: np.ndarray
    covariate_names: Tuple[str, ...]
    dropped_covariates: Tuple[str, ...]
    factors: np.ndarray
    control_loadings: np.ndarray
    treated_loadings: np.ndarray
    unit_effects: np.ndarray
    period_effects: np.ndarray
    grand_mean: float
    n_iterations: int
    cv: Optional[GSYNTHCrossValidation] = None


@dataclass(frozen=True)
class GSYNTHEventStudy:
    """The effect path in time since adoption.

    Horizon ``0`` is each unit's first treated period and ``-1`` its last
    pre-treatment period, the convention STACKEDSC uses. Reference
    implementations in the gsynth/fect family number the first treated period
    ``1``, so their paths are this one shifted by one. With staggered adoption
    the units contributing to a horizon change with it, which ``n_units``
    records.

    Parameters
    ----------
    horizons : np.ndarray
        Periods relative to adoption.
    tau : np.ndarray
        Average effect at each horizon.
    n_units : np.ndarray
        Treated units observed at each horizon.
    """

    horizons: np.ndarray
    tau: np.ndarray
    n_units: np.ndarray


@dataclass(frozen=True)
class GSYNTHAggregate:
    """The treated group pooled, in calendar time.

    Parameters
    ----------
    att : float
        Mean effect over the treated post-treatment cells.
    n_post_cells : int
        How many cells that mean is taken over. With staggered adoption this is
        not ``N_tr`` times a common horizon, so it is carried explicitly.
    rmse_pre : float
        Root mean squared pre-treatment residual, pooled over treated units and
        their pre-periods.
    observed : np.ndarray
        Treated-average observed outcome, shape ``(T,)``.
    counterfactual : np.ndarray
        Treated-average imputed untreated outcome, same shape.
    intervention_time : Any
        The earliest adoption date, as a time label -- the point the pooled
        paths start to separate.
    """

    att: float
    n_post_cells: int
    rmse_pre: float
    observed: np.ndarray
    counterfactual: np.ndarray
    intervention_time: Any


@dataclass(frozen=True)
class GSYNTHUnitFit:
    """One treated unit's fit.

    Parameters
    ----------
    label : Any
        Unit label.
    adoption_time : Any
        First treated period, as a time label.
    n_pre, n_post : int
        Periods before and from adoption.
    att : float
        Mean post-treatment effect for this unit.
    prefit_rmse : float
        Root mean squared pre-treatment residual. A large value flags a unit
        whose pre-period path the factor model does not track, and whose
        ``att`` therefore carries less weight than its neighbors'.
    tau : np.ndarray
        This unit's effect path over all ``T`` periods.
    loading : np.ndarray
        Its factor loadings, shape ``(r,)``.
    unit_effect : float
        Its additive effect.
    """

    label: Any
    adoption_time: Any
    n_pre: int
    n_post: int
    att: float
    prefit_rmse: float
    tau: np.ndarray
    loading: np.ndarray
    unit_effect: float


@dataclass(frozen=True)
class GSYNTHInference:
    """Algorithm 2, the parametric bootstrap.

    Loop 1 builds a pool of prediction errors by treating a randomly chosen
    control as if it were treated and predicting it from a resampled control
    pool; Loop 2 rebuilds panels from the fitted values plus resampled errors
    and refits. Treated cells draw from the Loop 1 pool and control cells from
    the in-sample residuals, because the model fits a control unit better than
    a treated counterfactual and the two error distributions differ.

    Parameters
    ----------
    method : str
        ``"parametric bootstrap"``.
    n_bootstrap : int
        Draws in each loop.
    alpha : float
        Two-sided significance level.
    att, se, ci_lower, ci_upper, p_value : float
        Point estimate, bootstrap standard error, percentile interval, and the
        two-sided p-value of the null that the ATT is zero.
    tau, tau_se, tau_lower, tau_upper : np.ndarray
        The same, by horizon, aligned with the event study.
    horizons : np.ndarray
        Horizons the arrays are indexed by.
    n_failed : int
        Bootstrap draws discarded because the refit did not converge.
    """

    method: str = "parametric bootstrap"
    n_bootstrap: int = 0
    alpha: float = 0.05
    att: float = float("nan")
    se: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    p_value: float = float("nan")
    tau: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    tau_se: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    tau_lower: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    tau_upper: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    horizons: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    n_failed: int = 0


class GSYNTHResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.GSYNTH.fit`.

    An :class:`~mlsynth.config_models.EffectResult`. GSYNTH is a staggered
    estimator, so the standardized ``time_series`` carries the treated-average
    observed and imputed paths in calendar time, and ``effects.att`` is the
    aggregate over treated post-treatment cells. The factor-model detail lives
    on ``design``, the horizon path on ``event_study``, the per-unit breakdown
    on ``per_unit``, and the full bootstrap on ``inference_detail``.

    Parameters
    ----------
    inputs : GSYNTHInputs
        Preprocessed panel.
    design : GSYNTHDesign
        Rank, coefficients and fitted factor structure.
    aggregate : GSYNTHAggregate
        The pooled ATT and the treated-average paths in calendar time.
    event_study : GSYNTHEventStudy
        Effect by period since adoption.
    inference_detail : GSYNTHInference
        Bootstrap output.
    per_unit : dict
        ``{unit label: GSYNTHUnitFit}``.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: GSYNTHInputs
    design: GSYNTHDesign
    aggregate: GSYNTHAggregate
    event_study: GSYNTHEventStudy
    inference_detail: GSYNTHInference
    per_unit: Dict[Any, GSYNTHUnitFit] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def r(self) -> int:
        """Factor count used."""
        return self.design.r_selected

    @property
    def att(self) -> float:
        """Average effect on the treated."""
        return float(self.effects.att)

    @model_validator(mode="after")
    def _populate_contract(self) -> "GSYNTHResults":
        if self.effects is not None:
            return self
        inf = self.inference_detail
        agg = self.aggregate
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen)

        se = float(inf.se) if np.isfinite(inf.se) else None
        lo, hi = float(inf.ci_lower), float(inf.ci_upper)
        finite_ci = np.isfinite(lo) and np.isfinite(hi)
        pval = float(inf.p_value) if np.isfinite(inf.p_value) else None

        observed = np.asarray(agg.observed, dtype=float)
        cfact = np.asarray(agg.counterfactual, dtype=float)

        set_("effects", EffectsResults(
            att=float(agg.att), att_std_err=se,
            additional_effects={
                "n_post_cells": int(agg.n_post_cells),
                "n_treated": int(self.inputs.n_treated),
            }))
        set_("time_series", TimeSeriesResults(
            observed_outcome=observed,
            counterfactual_outcome=cfact,
            estimated_gap=observed - cfact,
            time_periods=np.asarray(self.inputs.time_labels),
            intervention_time=agg.intervention_time))
        set_("weights", WeightsResults(
            donor_weights={},
            summary_stats={
                "constraint": "interactive fixed effects on never-treated units "
                              "(no donor weights)"}))
        set_("fit_diagnostics", FitDiagnosticsResults(
            rmse_pre=float(agg.rmse_pre)))
        set_("inference", InferenceResults(
            method=inf.method if se is not None else None,
            standard_error=se,
            ci_lower=(lo if finite_ci else None),
            ci_upper=(hi if finite_ci else None),
            p_value=pval,
            confidence_level=1.0 - float(inf.alpha),
            details=inf))
        set_("method_details", MethodDetailsResults(
            method_name="GSYNTH", is_recommended=True,
            parameters_used={
                "r": int(self.design.r_selected),
                "r_source": self.design.r_source,
                "force": self.design.force,
                "covariates": list(self.design.covariate_names),
            }))
        return self


def event_study_from_effect(
    effect: np.ndarray, adoption_index: np.ndarray,
) -> GSYNTHEventStudy:
    """Average a ``(T, N_tr)`` effect matrix by period since adoption.

    Horizon ``0`` is each unit's first treated period, so pre-treatment
    horizons run ``-1, -2, ...``.
    """
    T = int(effect.shape[0])
    buckets: Dict[int, List[float]] = {}
    for j, t0 in enumerate(np.asarray(adoption_index, dtype=int)):
        for t in range(T):
            buckets.setdefault(t - int(t0), []).append(float(effect[t, j]))
    horizons = np.array(sorted(buckets), dtype=int)
    tau = np.array([float(np.mean(buckets[h])) for h in horizons], dtype=float)
    n = np.array([len(buckets[h]) for h in horizons], dtype=int)
    return GSYNTHEventStudy(horizons=horizons, tau=tau, n_units=n)


# Resolve forward references (module uses ``from __future__ import annotations``).
GSYNTHResults.model_rebuild()
