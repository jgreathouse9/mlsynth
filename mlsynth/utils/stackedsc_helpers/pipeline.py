"""The stacked-in-event-time fit: solve per cohort, average on the event clock.

Every treated unit in a cohort faces the same donor block and differs only in
its own pre-treatment target, so a cohort is one multiple-right-hand-side
program rather than N_g programs -- see
:func:`mlsynth.utils.bilevel.simplex.simplex_lstsq_batch`. The exception is a
donor predicate that actually binds: it gives each treated unit its own design
matrix, and the batching is forfeited. That is reported rather than hidden.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..bilevel import (bias_corrected_gaps, regression_v, simplex_lstsq,
                       simplex_lstsq_batch)
from .setup import aggregation_weights, build_cohorts, event_window
from .structures import (STACKEDSCResults, StackedDesign,
                         StackedEventStudy, StackedUnitFit)

_EPS = 1e-12


def _resolve_backend(backend: str, has_cov: bool) -> str:
    if backend != "auto":
        return backend
    return "regression" if has_cov else "outcome-only"


def _design(cohort, backend: str):
    """The matrices the weights are solved against, and the V that scales them.

    Outcome-only matching uses the pre-treatment path. With predictors, the
    columns are scaled by sqrt(V) so the lower-level problem is the ordinary
    least-squares one -- the same device the bilevel backends use.
    """
    t0 = cohort.t0_index
    if backend == "outcome-only" or cohort.X0 is None:
        return cohort.D[:t0 + 1], cohort.Y[:t0 + 1], None
    if backend == "regression":
        V = regression_v(cohort.X1[:, 0], cohort.X0,
                         cohort.Y[:t0 + 1, 0], cohort.D[:t0 + 1])
    else:
        V = np.full(cohort.X0.shape[0], 1.0 / cohort.X0.shape[0])
    root = np.sqrt(V)[:, None]
    return cohort.X0 * root, cohort.X1 * root, V


def _weights_for_cohort(cohort, backend, predicate):
    """(n_donors, n_units) weights. Batched unless a predicate binds."""
    A, B, V = _design(cohort, backend)
    if predicate is None:
        return simplex_lstsq_batch(A, B), V, True

    cols = []
    binds = False
    for j, unit in enumerate(cohort.units):
        keep = [k for k, d in enumerate(cohort.donors) if predicate(unit, d)]
        if not keep:
            raise ValueError(
                f"donor_predicate excludes every donor for treated unit "
                f"{unit!r}.")
        binds |= len(keep) < len(cohort.donors)
        w = np.zeros(len(cohort.donors))
        w[keep] = simplex_lstsq(A[:, keep], B[:, j])
        cols.append(w)
    return np.column_stack(cols), V, not binds


def run_stackedsc(config) -> BaseEstimatorResults:
    df = config.df
    unitid, time = config.unitid, config.time
    outcome, treat = config.outcome, config.treat

    cohorts, donors = build_cohorts(
        df, unitid, time, outcome, treat,
        normalize=config.normalize, covariates=config.covariates,
        covariate_windows=config.covariate_windows,
    )
    backend = _resolve_backend(config.backend, bool(config.covariates))
    grid = event_window(cohorts, config.n_lags, config.n_leads)

    all_units = [u for c in cohorts for u in c.units]
    gammas = aggregation_weights(df, unitid, config.agg_weights, all_units)
    gamma_of = dict(zip(all_units, gammas))

    per_unit: Dict[str, StackedUnitFit] = {}
    batched = True
    for cohort in cohorts:
        W, _V, was_batched = _weights_for_cohort(cohort, backend,
                                                 config.donor_predicate)
        batched &= was_batched
        gap = cohort.Y - cohort.D @ W                      # (T, n_units)
        if config.bias_correct:
            for j in range(len(cohort.units)):
                gap[:, j] = bias_corrected_gaps(
                    W[:, j], cohort.X1[:, j], cohort.X0,
                    cohort.Y[:, j], cohort.D,
                    ridge=config.bias_correct_ridge)
        e_all = np.arange(cohort.Y.shape[0]) - cohort.t0_index - 1
        take = np.isin(e_all, grid)
        for j, unit in enumerate(cohort.units):
            g = gap[take, j]
            pre = grid < 0
            per_unit[str(unit)] = StackedUnitFit(
                label=unit, adoption_time=cohort.adopt,
                base_period=cohort.base_period, horizons=grid, tau=g,
                donor_weights={str(d): float(W[k, j])
                               for k, d in enumerate(cohort.donors)
                               if abs(W[k, j]) > 1e-10},
                agg_weight=float(gamma_of[unit]),
                pre_rmse=float(np.sqrt(np.mean(g[pre] ** 2))) if pre.any()
                else float("nan"),
            )

    G = np.column_stack([per_unit[str(u)].tau for u in all_units])
    w = np.array([gamma_of[u] for u in all_units])
    tau = G @ w
    event = StackedEventStudy(horizons=grid, tau=tau,
                              n_units=np.full(len(grid), len(all_units)))

    post = grid >= 0
    att = float(np.mean(tau[post])) if post.any() else float("nan")
    design = StackedDesign(
        cohorts=[c.adopt for c in cohorts], n_treated=len(all_units),
        n_donors=len(donors), backend=backend, normalized=config.normalize,
        bias_corrected=config.bias_correct, batched=batched,
    )

    pooled: Dict[str, float] = {}
    for u in all_units:
        f = per_unit[str(u)]
        for d, v in f.donor_weights.items():
            pooled[d] = pooled.get(d, 0.0) + f.agg_weight * v

    return STACKEDSCResults(
        effects=EffectsResults(att=att, additional_effects={
            "event_study": {int(e): float(t) for e, t in zip(grid, tau)}}),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=float(np.sqrt(np.mean(tau[grid < 0] ** 2)))
            if (grid < 0).any() else None),
        time_series=TimeSeriesResults(
            estimated_gap=tau, time_periods=grid,
            intervention_time=0),
        weights=WeightsResults(
            donor_weights=pooled,
            summary_stats={"constraint": "simplex (non-negative, sum to 1)",
                           "aggregation": "gamma-weighted over treated units"}),
        method_details=MethodDetailsResults(
            method_name=f"STACKEDSC[{backend}]",
            parameters_used={
                "agg_weights": config.agg_weights,
                "normalize": config.normalize,
                "backend": backend,
                "bias_correct": config.bias_correct,
                "bias_correct_ridge": (float(config.bias_correct_ridge)
                                       if config.bias_correct else None),
                "covariates": config.covariates,
                "batched": batched,
                "cohorts": [c.adopt for c in cohorts],
            }),
        additional_outputs={"donor_names": donors},
        per_unit=per_unit, event_study=event, design=design,
    )
