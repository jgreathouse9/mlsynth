"""The stacked-in-event-time fit: solve per cohort, average on the event clock.

Every treated unit in a cohort faces the same donor block and differs only in
its own pre-treatment target. A donor predicate that binds is the exception: it
gives each unit its own design matrix, which the result reports as
``shared_donor_pool = False``.

The weights come from the primal active-set solver
(:func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`), the same one MEDSC,
SCD and COMPSC use. That choice is load-bearing on this design rather than a
matter of taste. A cohort has 5 to 10 pre-treatment periods against 39 donors,
so the Gram is rank deficient by 30 or more; measured on the Wiltshire panel a
first-order method never converged on it, with 20 of 39 leave-one-out columns
still improving at iteration 20,000 at every tolerance from 1e-11 down to 1e-6.
The active-set method solves the same program exactly in a bounded number of
pivots, and on that panel is 2.8x faster on the point estimate and roughly 130x
on the placebo layer.

Exactness pins the objective, not the answer. With fewer rows than donors the
optimum is a face, and two exact solvers land in different places on it -- on
the Wiltshire cohorts the active set and CLARABEL agree on the objective to
8.7e-6 while their per-county post-treatment paths differ by up to 2.1
percentage points and the population-weighted aggregate by 0.05. Read the
aggregate; ``docs/stackedsc.rst`` says so at more length.
"""

from __future__ import annotations

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
from ..bilevel import bias_corrected_gaps, regression_v
from ..bilevel.active_set import solve_simplex_qp
from .inference import placebo_inference
from .plotter import plot_stackedsc
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


def _allowed_donors(cohort, predicate):
    """Donor column indices available to each treated unit, and whether the
    predicate actually removes any."""
    full = list(range(len(cohort.donors)))
    if predicate is None:
        return [full] * len(cohort.units), False
    allowed, binds = [], False
    for unit in cohort.units:
        keep = [k for k, d in enumerate(cohort.donors) if predicate(unit, d)]
        if not keep:
            raise ValueError(
                f"donor_predicate excludes every donor for treated unit "
                f"{unit!r}.")
        binds |= len(keep) < len(full)
        allowed.append(keep)
    return allowed, binds


def _weights_for_cohort(cohort, A, B, allowed):
    """(n_donors, n_units) weights, one exact solve per treated unit.

    The design ``A`` is shared across a cohort unless a predicate binds, but
    the active-set solver takes one target at a time, so this is a loop either
    way. It is still the faster path: each solve terminates on a KKT
    certificate after a handful of pivots rather than running an iteration
    budget to exhaustion.
    """
    cols = []
    for j, keep in enumerate(allowed):
        w = np.zeros(len(cohort.donors))
        w[keep] = np.asarray(solve_simplex_qp(A[:, keep], B[:, j]),
                             dtype=float).ravel()
        cols.append(w)
    return np.column_stack(cols)


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
    shared_pool = True
    designs, pools = [], []
    for cohort in cohorts:
        A, B, _V = _design(cohort, backend)
        allowed, binds = _allowed_donors(cohort, config.donor_predicate)
        designs.append((A, B))
        pools.append(allowed)
        W = _weights_for_cohort(cohort, A, B, allowed)
        shared_pool &= not binds
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
        bias_corrected=config.bias_correct,
        shared_donor_pool=shared_pool,
    )

    pooled: Dict[str, float] = {}
    for u in all_units:
        f = per_unit[str(u)]
        for d, v in f.donor_weights.items():
            pooled[d] = pooled.get(d, 0.0) + f.agg_weight * v

    placebo = None
    inference = InferenceResults()
    if config.inference == "placebo":
        placebo = placebo_inference(cohorts, designs, pools, grid, tau,
                                    gamma_of, config)
        inference = InferenceResults(
            p_value=placebo.att_p_value,
            ci_lower=placebo.att_ci_lower, ci_upper=placebo.att_ci_upper,
            standard_error=placebo.att_se,
            confidence_level=placebo.confidence_level,
            method="placebo (sampled placebo averages)",
            details=placebo,
        )

    results = STACKEDSCResults(
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
                "shared_donor_pool": shared_pool,
                "cohorts": [c.adopt for c in cohorts],
                "inference": config.inference,
                "placebo_donor_pool": config.placebo_donor_pool,
                "n_placebo_samples": (int(config.n_placebo_samples)
                                      if config.inference == "placebo"
                                      else None),
            }),
        inference=inference,
        additional_outputs={"donor_names": donors},
        per_unit=per_unit, event_study=event, design=design, placebo=placebo,
    )
    if config.display_graphs or config.save:
        plot_stackedsc(results, save=config.save)
    return results
