"""End-to-end DRSC: fit the DR, solve for weights, test, assemble."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .dr import fit_cell, outcome_grid
from .inference import (covariance_kernel, integrated_effect, supremum_test,
                        weight_variance)
from .structures import ConditionalEffect, DRSCInputs, DRSCResults
from .weights import counterfactual, gram_and_cross, solve_weights
from ...config_models import MethodDetailsResults, WeightsResults
from ...exceptions import MlsynthEstimationError


def run_drsc(inputs: DRSCInputs, evaluation_points, link="probit", n_grid=32,
             grid_lo=0.10, grid_hi=0.90, focus_region=None, ridge=0.0,
             alpha=0.10, n_gp_draws=10_000, seed=1992) -> DRSCResults:
    """Fit DRSC and return the typed results."""
    from ...config_models import EffectsResults

    Lam, _ = __import__("mlsynth.utils.drsc_helpers.dr", fromlist=["x"]).link_functions(link)
    units = inputs.unit_names
    periods = list(inputs.pre_periods) + list(inputs.post_periods)

    pooled = np.concatenate([inputs.cells[(u, t)].y
                             for u in units for t in periods])
    grid = outcome_grid(pooled, n_grid, grid_lo, grid_hi)

    theta = {}
    for i, u in enumerate(units):
        for t in periods:
            cell = inputs.cells[(u, t)]
            theta[(i, t)] = fit_cell(cell.X, cell.y, grid, link)

    J = inputs.J
    G, c = gram_and_cross(theta, J, inputs.pre_periods)
    w = solve_weights(G, c, ridge)
    Vw = weight_variance(theta, inputs.cells, units, w, G, grid,
                         inputs.n_total, inputs.pre_periods, link, ridge)

    post = inputs.post_periods[0]
    th0 = counterfactual(theta, w, post)

    corridor = None
    if focus_region is not None:
        lo, hi = focus_region
        corridor = np.flatnonzero((grid >= lo) & (grid <= hi))
        if corridor.size == 0:
            raise MlsynthEstimationError(
                f"focus_region ({lo}, {hi}) contains no grid points; the "
                f"grid spans [{grid.min():.4g}, {grid.max():.4g}]. Widen the "
                "region or the quantile range.")

    effects: Dict[str, ConditionalEffect] = {}
    for label, point in evaluation_points.items():
        x = np.empty(len(inputs.covariates) + 1)
        x[0] = 1.0
        for j, name in enumerate(inputs.covariates):
            x[j + 1] = float(point[name])
        delta = Lam(theta[(0, post)] @ x) - Lam(th0 @ x)
        K = covariance_kernel(theta, inputs.cells, units, w, th0, post, x,
                              grid, inputs.n_total, Vw, link)
        stat, crit, p = supremum_test(delta, K, inputs.n_total, None, alpha,
                                      seed, n_gp_draws)
        f_hat, se, lcb = integrated_effect(delta, K, inputs.n_total, None, alpha)
        kw = {}
        if corridor is not None:
            sF, cF, pF = supremum_test(delta, K, inputs.n_total, corridor,
                                       alpha, seed, n_gp_draws)
            _, _, lF = integrated_effect(delta, K, inputs.n_total, corridor, alpha)
            kw = dict(t_stat_focused=sF, crit_value_focused=cF,
                      p_value_focused=pF, lower_bound_focused=lF)
        effects[label] = ConditionalEffect(
            label=label, x=x, delta=delta, f_hat=f_hat, std_error=se,
            lower_bound=lcb, t_stat=stat, crit_value=crit, p_value=p, **kw)

    return DRSCResults(
        conditional_effects=effects,
        outcome_grid=grid,
        gram_condition_number=float(np.linalg.cond(G)),
        n_negative_weights=int((w < 0).sum()),
        weights=WeightsResults(
            # the contract keys donor_weights by str; unit ids are often
            # integers (FIPS codes), so stringify here and keep the natively
            # typed mapping on `metadata` for callers that need to join back
            donor_weights={str(u): float(v)
                           for u, v in zip(inputs.donor_names, w)},
            summary_stats={"constraint": "sum to one (negative weights allowed)",
                           "sum": float(w.sum())}),
        method_details=MethodDetailsResults(
            method_name="DRSC",
            parameters_used={"link": link, "m_active": int(len(grid)),
                             "ridge": float(ridge), "alpha": float(alpha),
                             "J": int(J), "n": int(inputs.n_total)}),
        metadata={"donor_weights_native": dict(zip(inputs.donor_names,
                                                  (float(v) for v in w))),
                  "treated_unit": inputs.treated_unit_name,
                  "pre_periods": list(inputs.pre_periods),
                  "post_period": post,
                  "covariates": list(inputs.covariates)},
    )
