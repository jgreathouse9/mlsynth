"""Orchestration for ATEL, Steps 1-5 of Lee (2026).

1. Build the diversified weights, from a sieve basis in the covariates, in
   each unit's first outcome, or from deterministic sign columns
   (:mod:`.sieve`); ``weight_source`` picks which.
2. Estimate the factors as cross-sectional donor averages against those weights
   (:mod:`.factors`).
3. Fit the time-varying loading on the treated unit's pre-period by local linear
   regression, selecting the bandwidth by cross-validation if none is supplied
   (:mod:`.loadings`).
4. Carry the loading into the post-period and form the counterfactual.
5. Average the post-period gap against the localization kernel, and attach
   Theorem 1's variance (:mod:`.inference`).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm, t as student_t

from ...exceptions import MlsynthConfigError, MlsynthDataError
from .factors import diversified_factors
from .inference import atel_variance, pointwise_variance
from .loadings import (
    BANDWIDTH_GRID,
    cv_bandwidth,
    local_linear_loadings,
    post_period_kernel,
)
from .sieve import (
    basis_values,
    construct_weights,
    hadamard_weights,
    tile_unit_weights,
    weight_conditioning,
)
from .structures import ATELInputs

__all__ = ["run_atel"]


def build_weights(
    inputs: ATELInputs, n_factors: int, basis: str, weight_source: str
) -> np.ndarray:
    """Diversified weights for every unit, in the projection's block layout.

    ``"covariates"`` evaluates the sieve on each covariate at every unit-period
    (Fan and Liao 4.1); ``"initial"`` evaluates it once on the held-out first
    outcome and repeats that across periods (4.3); ``"hadamard"`` uses
    deterministic sign columns and no data (4.4).
    """
    n_periods = inputs.n_periods
    if weight_source == "covariates":
        return construct_weights(inputs.covariates, n_factors, basis)
    if weight_source == "initial":
        if inputs.initial_outcome is None:
            raise MlsynthDataError(
                "weight_source='initial' needs the held-out first outcome, "
                "which ingestion did not supply."
            )
        unit = basis_values(inputs.initial_outcome, n_factors, basis)
        return tile_unit_weights(unit, n_periods)
    if weight_source == "hadamard":
        unit = hadamard_weights(inputs.outcomes.shape[0], n_factors)
        return tile_unit_weights(unit, n_periods)
    raise MlsynthConfigError(f"Unknown weight_source {weight_source!r}.")


def run_atel(
    inputs: ATELInputs,
    n_factors: int,
    basis: str = "bspline",
    bandwidth: Optional[float] = None,
    alpha: float = 0.05,
    weight_source: str = "covariates",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the ATEL pipeline.

    Returns
    -------
    tuple of dict
        ``(estimates, diagnostics)``. ``estimates`` carries the localized
        estimate, its interval, the counterfactual path and the intermediate
        arrays; ``diagnostics`` records the bandwidth search and the windows.
    """
    Y = inputs.outcomes
    T0 = int(inputs.n_pre)
    T = inputs.n_periods
    T1 = T - T0
    treated, donors = Y[0], Y[1:]

    weights = build_weights(inputs, n_factors, basis, weight_source)
    factors = diversified_factors(donors, weights, n_factors)
    lambda_min = weight_conditioning(weights[1:], n_factors, T)
    factors_pre, factors_post = factors[:T0], factors[T0:]

    selected_by_cv = bandwidth is None
    if selected_by_cv:
        h = cv_bandwidth(treated[:T0], factors_pre)
    else:
        h = float(bandwidth)

    post_window = int(np.floor(T1 * h))
    if post_window < 1:
        raise MlsynthDataError(
            f"The localization window is floor(T1 * h) = floor({T1} * {h}) = 0, "
            f"so the kernel has no post-period window to average over. ATEL "
            f"needs at least two post-treatment periods."
        )
    pre_window = int(np.floor(T0 * h))

    at_grid_edge = selected_by_cv and h in (
        float(BANDWIDTH_GRID[0]),
        float(BANDWIDTH_GRID[-1]),
    )
    if at_grid_edge:
        warnings.warn(
            f"The cross-validated bandwidth {h} is an endpoint of the search "
            f"grid [{BANDWIDTH_GRID[0]}, {BANDWIDTH_GRID[-1]}], so the "
            f"criterion was still improving where the grid ran out. The "
            f"bandwidth is a corner solution and the smoothing window spans "
            f"{pre_window} of {T0} pre-periods; supply `bandwidth` explicitly "
            f"to pin it.",
            RuntimeWarning,
            stacklevel=3,
        )

    loadings = local_linear_loadings(treated, factors_pre, h)
    counterfactual_post = (loadings * factors_post.T).sum(axis=0)
    gap = treated[T0:] - counterfactual_post

    kernel = post_period_kernel(T0, T, post_window)
    point = float((gap * kernel).sum() / post_window)

    # ATEL is a donor-weighting estimator with time-varying weights. Substituting
    # the projection into the counterfactual,
    #   Yhat_1t = sum_j beta_jt F_tj
    #           = sum_i Y_it * [ (1/N) sum_j beta_jt W_it^(j) ],
    # so donor i carries the implied weight below at post-period t. The scalar
    # the result reports per donor is that path averaged against the kernel,
    # which is the weight the donor carries in the reported estimate.
    n_donors = donors.shape[0]
    implied = np.zeros((n_donors, T1))
    for j in range(n_factors):
        block_post = weights[1:, j * T : (j + 1) * T][:, T0:]
        implied += loadings[j][None, :] * block_post
    implied /= n_donors
    localized = (implied * kernel).sum(axis=1) / post_window

    variance, fitted_pre, residuals = atel_variance(treated, factors_pre,
                                                    factors_post, h)
    standard_error = float(np.sqrt(variance))
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    p_value = float(
        2.0 * student_t.cdf(-abs(point / standard_error), post_window - 1)
    )
    pointwise = np.sqrt(
        pointwise_variance(factors, loadings, residuals, h, T1)
    )

    estimates = {
        "atel": point,
        "standard_error": standard_error,
        "ci": (point - critical * standard_error, point + critical * standard_error),
        "p_value": p_value,
        "gap": gap,
        "counterfactual": np.concatenate([fitted_pre, counterfactual_post]),
        "observed": treated,
        "factors": factors,
        "loadings": loadings,
        "kernel_weights": kernel,
        "weights_matrix": weights,
        "implied_donor_weights": implied,
        "localized_donor_weights": localized,
        "pointwise_standard_errors": pointwise,
        "residuals_pre": residuals,
        "bandwidth": h,
    }
    diagnostics = {
        "bandwidth": h,
        "bandwidth_selected_by_cv": selected_by_cv,
        "bandwidth_at_grid_edge": bool(at_grid_edge),
        "bandwidth_grid": [float(g) for g in BANDWIDTH_GRID],
        "pre_window": pre_window,
        "post_window": post_window,
        "basis": basis,
        "n_factors": int(n_factors),
        "n_covariates": len(inputs.covariate_names),
        "weight_source": weight_source,
        "weight_lambda_min": float(lambda_min),
        "kernel_mass": float(kernel.sum() / post_window),
    }
    return estimates, diagnostics
