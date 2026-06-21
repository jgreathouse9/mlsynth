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
from typing import Any, Dict, List

import numpy as np

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..bilevel import BilevelSCM


@dataclass(frozen=True)
class StaggeredUnitFit:
    """The synthetic-control fit for one treated unit in a staggered design."""

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


def run_vanillasc_staggered(config, prep: Dict[str, Any]) -> BaseEstimatorResults:
    """Fit per-unit synthetic controls under staggered adoption and aggregate."""
    cohorts = prep["cohorts"]
    time_labels = np.asarray(prep.get("time_labels"))

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
            )
            if post.size:
                post_gaps.append(post)

    all_post = np.concatenate(post_gaps) if post_gaps else np.empty(0)
    overall_att = float(np.mean(all_post)) if all_post.size else float("nan")

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
        additional_outputs={"adoption_times": sorted(cohorts.keys())},
    )
