"""Staggered-adoption synthetic control for VanillaSC.

When ``dataprep`` returns a multi-cohort structure (several treated units adopting
at possibly different times), VanillaSC fits one synthetic control per treated
unit on the never-treated donor pool over that unit's own pre-treatment window,
following Cattaneo, Feng, Palomba and Titiunik (2025). The per-unit gaps are then
aggregated into the overall (unit-time) average treatment effect on the treated.

Point estimation only at this stage; the per-unit fits reuse the single-treated
:class:`~mlsynth.utils.bilevel.BilevelSCM` engine, so every weight constraint the
scalar estimator supports carries over unit by unit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..bilevel import BilevelSCM


@dataclass(frozen=True)
class StaggeredUnitFit:
    """The synthetic-control fit for one treated unit in a staggered design.

    The ``*_lower`` / ``*_upper`` arrays and ``att_ci_*`` carry the per-unit
    Cattaneo-Feng-Titiunik prediction intervals, populated only when SCPI
    inference is requested; they are ``None`` / ``NaN`` otherwise.
    """

    treated_unit_name: str
    adoption_time: Any
    att: float
    pre_periods: int
    post_periods: int
    donor_names: List[str]
    donor_weights: Dict[str, float]
    observed: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray
    time_labels: np.ndarray = field(default_factory=lambda: np.empty(0))
    tau_lower: Optional[np.ndarray] = None       # PI for the per-period effect
    tau_upper: Optional[np.ndarray] = None
    cf_lower: Optional[np.ndarray] = None        # PI for the counterfactual
    cf_upper: Optional[np.ndarray] = None
    att_ci_lower: float = float("nan")           # PI for the unit's average effect
    att_ci_upper: float = float("nan")


def _fit_one_unit(config, y: np.ndarray, Y0: np.ndarray, pre: int,
                  donor_names: List[str]):
    """Fit a single-treated synthetic control, outcome-only, via the engine."""
    engine = BilevelSCM(
        config.backend,
        canonical_v=config.canonical_v,
        seed=config.seed,
        augment=config.augment,
        ridge_lambda=config.ridge_lambda,
        residualize=config.residualize,
        maxiter=config.mscmt_maxiter,
        popsize=config.mscmt_popsize,
        prune_shady=config.mscmt_prune_shady,
        cv=config.penalized_cv,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = engine.fit(
            y[:pre], Y0[:pre],
            X1=None, X0=None, donor_names=donor_names, predictor_names=[],
        )
    return res


def _event_study(unit_fits: Dict[str, StaggeredUnitFit]) -> Dict[int, float]:
    """Average treatment effect by event time (``scpi`` ``effect='time'``).

    Event time ``ell`` is the ``ell``-th post-treatment period for each treated
    unit; the effect is averaged across the units observed at that event time,
    balanced over ``ell = 1 .. min_i T1_i`` (the event times present for every
    treated unit), matching the official package's convention.
    """
    if not unit_fits:
        return {}
    fits = list(unit_fits.values())
    min_post = min(f.post_periods for f in fits)
    out: Dict[int, float] = {}
    for ell in range(min_post):
        vals = [float(f.gap[f.pre_periods + ell]) for f in fits]
        out[ell + 1] = float(np.mean(vals))
    return out


def _event_study_intervals(config) -> Dict[int, Dict[str, Any]]:
    """Cross-unit (TSUA) prediction intervals via the clean-room engine.

    Returns, keyed by 1-indexed event time, the synthetic-prediction band
    ``synthetic_ci`` (directly comparable to ``scpi``'s ``CI_all_gaussian``) and
    the synthetic point. ``config.scpi_compat`` selects the ``1/iota`` (correct,
    default) vs ``1/iota**2`` (``scpi``-matching) in-sample scaling.
    """
    from .staggered_engine import staggered_pi_bands
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bands = staggered_pi_bands(
            config.df, outcome=config.outcome, unitid=config.unitid,
            time=config.time, treat=config.treat, effect="time",
            sims=config.scpi_sims, u_alpha=config.alpha, e_alpha=config.alpha,
            seed=config.seed, scpi_compat=config.scpi_compat,
        )
    out: Dict[int, Dict[str, Any]] = {}
    for k, lab in enumerate(list(bands["index"])):
        ell = int(lab[-1]) if isinstance(lab, tuple) else int(lab)
        out[ell] = {
            "synthetic_ci": (float(bands["lb"][k]), float(bands["ub"][k])),
            "insample_synthetic_ci": (float(bands["insample_lb"][k]),
                                      float(bands["insample_ub"][k])),
            "synthetic_point": float(bands["point"][k]),
        }
    return out


def _unit_scpi(config, y: np.ndarray, Y0: np.ndarray, pre: int,
               W: np.ndarray) -> Dict[str, Any]:
    """Per-unit CFT prediction bands, via the same engine the scalar path uses."""
    from .scpi import scpi_intervals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(
            y, Y0, pre, W, sims=config.scpi_sims,
            u_alpha=config.alpha, e_alpha=config.alpha,
            e_method=config.scpi_e_method, seed=config.seed,
        )
    return {
        "tau_lower": sc.lower, "tau_upper": sc.upper,
        "cf_lower": sc.cf_lower, "cf_upper": sc.cf_upper,
        "att_ci_lower": float(sc.metadata["att_lower"]),
        "att_ci_upper": float(sc.metadata["att_upper"]),
    }


def _aggregate_att_interval(unit_fits: Dict[str, StaggeredUnitFit],
                            overall_att: float, config) -> InferenceResults:
    """Pool the per-unit average-effect intervals into the overall ATT interval.

    Pointwise aggregation: the overall (unit-time) ATT is the post-weighted mean
    of the per-unit average effects, and -- treating the per-unit prediction
    errors as independent -- its half-width is the post-weighted quadrature of the
    per-unit half-widths. This is the pointwise predictand; uniform (simultaneous)
    coverage across post periods is a separate construction.
    """
    fits = [f for f in unit_fits.values() if np.isfinite(f.att_ci_lower)]
    if not fits or not np.isfinite(overall_att):
        return None
    ns = np.array([f.post_periods for f in fits], dtype=float)
    w = ns / ns.sum()
    halves = np.array([(f.att_ci_upper - f.att_ci_lower) / 2.0 for f in fits])
    overall_half = float(np.sqrt(np.sum((w * halves) ** 2)))
    return InferenceResults(
        ci_lower=overall_att - overall_half,
        ci_upper=overall_att + overall_half,
        confidence_level=1.0 - 2.0 * config.alpha,
        method=("staggered SCPI prediction intervals "
                "(Cattaneo, Feng, Palomba & Titiunik 2025)"),
        details={
            "aggregation": "pointwise, post-weighted quadrature of per-unit intervals",
            "per_unit_att_ci": {
                f.treated_unit_name: (f.att_ci_lower, f.att_ci_upper) for f in fits},
        },
    )


def run_vanillasc_staggered(config, prep: Dict[str, Any]) -> BaseEstimatorResults:
    """Fit per-unit synthetic controls under staggered adoption and aggregate."""
    cohorts = prep["cohorts"]
    time_labels = np.asarray(prep.get("time_labels"))

    mode = config.inference
    do_scpi = isinstance(mode, str) and mode.lower() == "scpi"

    unit_fits: Dict[str, StaggeredUnitFit] = {}
    post_gaps: List[np.ndarray] = []

    for adoption_time in sorted(cohorts.keys()):
        cohort = cohorts[adoption_time]
        y_mat = np.asarray(cohort["y"], dtype=float)
        if y_mat.ndim == 1:
            y_mat = y_mat[:, None]
        Y0 = np.asarray(cohort["donor_matrix"], dtype=float)
        pre = int(cohort["pre_periods"])
        donor_names = [str(d) for d in cohort["donor_names"]]
        treated_names = [str(u) for u in cohort["treated_units"]]

        for j, unit_name in enumerate(treated_names):
            y = y_mat[:, j]
            res = _fit_one_unit(config, y, Y0, pre, donor_names)
            counterfactual = res.counterfactual(Y0)
            gap = y - counterfactual
            post = gap[pre:]
            att = float(np.mean(post)) if post.size else float("nan")

            bands = {}
            if do_scpi and post.size:
                bands = _unit_scpi(config, y, Y0, pre, res.W)

            unit_fits[unit_name] = StaggeredUnitFit(
                treated_unit_name=unit_name,
                adoption_time=adoption_time,
                att=att,
                pre_periods=pre,
                post_periods=int(post.size),
                donor_names=donor_names,
                donor_weights=dict(getattr(res, "donor_weights", {})),
                observed=y,
                counterfactual=counterfactual,
                gap=gap,
                time_labels=time_labels,
                **bands,
            )
            if post.size:
                post_gaps.append(post)

    all_post = np.concatenate(post_gaps) if post_gaps else np.empty(0)
    overall_att = float(np.mean(all_post)) if all_post.size else float("nan")

    event_study = _event_study(unit_fits)

    # Cross-unit (TSUA) prediction intervals for the event-study series. The
    # engine returns the band on the synthetic prediction; the treatment-effect
    # interval is observed - synthetic (bounds reverse). Only meaningful with
    # more than one treated unit (otherwise the per-unit band already covers it).
    event_study_intervals = None
    if do_scpi and len(unit_fits) > 1 and event_study:
        es_bands = _event_study_intervals(config)
        event_study_intervals = {}
        for ell, eff in event_study.items():
            band = es_bands.get(ell)
            if band is None:
                continue
            s_lb, s_ub = band["synthetic_ci"]
            s_pt = band["synthetic_point"]
            obs_avg = eff + s_pt                       # observed = effect + synthetic
            event_study_intervals[ell] = {
                "effect": eff,
                "effect_ci": (obs_avg - s_ub, obs_avg - s_lb),
                "synthetic_ci": (s_lb, s_ub),
                "insample_synthetic_ci": band["insample_synthetic_ci"],
            }

    inference = _aggregate_att_interval(unit_fits, overall_att, config) if do_scpi else None

    return BaseEstimatorResults(
        effects=EffectsResults(
            att=None if np.isnan(overall_att) else overall_att,
            additional_effects={
                "aggregation": "unit-time",
                "n_treated_units": len(unit_fits),
                "per_unit_att": {n: f.att for n, f in unit_fits.items()},
            },
        ),
        time_series=TimeSeriesResults(),
        weights=WeightsResults(
            summary_stats={"constraint": "per-unit simplex (staggered adoption)"}),
        inference=inference,
        fit_diagnostics=FitDiagnosticsResults(),
        method_details=MethodDetailsResults(
            method_name="VanillaSC[staggered]",
            parameters_used={
                "n_cohorts": len(cohorts),
                "n_treated_units": len(unit_fits),
                "augment": config.augment,
                "donor_pool": "never-treated",
            },
        ),
        sub_method_results=unit_fits,
        additional_outputs={
            "adoption_times": sorted(cohorts.keys()),
            "event_study": event_study,
            "event_study_intervals": event_study_intervals,
            "per_unit_att": {n: f.att for n, f in unit_fits.items()},
        },
    )
