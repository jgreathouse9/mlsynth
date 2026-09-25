"""Helpers for the ATEL estimator (Lee 2026).

ATEL reports a kernel-weighted average effect localized at the adoption date.
The counterfactual comes from a time-varying factor model: the factors are
cross-sectional donor averages against sieve weights built from the covariates
(diversified projection, no eigendecomposition), and the loading moves with time,
fit by local linear regression on the pre-period and carried forward.
"""

from .config import ATELConfig
from .factors import diversified_factors
from .inference import atel_variance, pointwise_variance
from .loadings import (
    BANDWIDTH_GRID,
    cv_bandwidth,
    half_epanechnikov,
    local_linear_loadings,
    post_period_kernel,
)
from .pipeline import build_weights, run_atel
from .plotter import plot_atel
from .setup import prepare_atel_inputs
from .sieve import (
    basis_values,
    bspline_weights,
    construct_weights,
    hadamard_weights,
    poly_weights,
    tile_unit_weights,
    trig_weights,
    weight_conditioning,
)
from .structures import ATELInputs, ATELResults

__all__ = [
    "ATELConfig",
    "ATELInputs",
    "ATELResults",
    "BANDWIDTH_GRID",
    "atel_variance",
    "basis_values",
    "build_weights",
    "bspline_weights",
    "construct_weights",
    "cv_bandwidth",
    "diversified_factors",
    "hadamard_weights",
    "half_epanechnikov",
    "local_linear_loadings",
    "plot_atel",
    "pointwise_variance",
    "poly_weights",
    "post_period_kernel",
    "prepare_atel_inputs",
    "run_atel",
    "tile_unit_weights",
    "trig_weights",
    "weight_conditioning",
]
