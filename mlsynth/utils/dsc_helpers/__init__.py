"""Helper utilities for the Distributional Synthetic Control (DSC) estimator.

Implements Gunsilius (2023) DSC with the asymptotic theory of Zhang,
Zhang & Zhang (2026). DSC operates on micro-level panels: each
``(unit, time)`` cell carries multiple individual observations, and
weights are fit on the empirical quantile functions of those cells
rather than on aggregate means.

Module layout (one module per role):

* :mod:`.structures` -- :class:`DSCInputs`, :class:`DSCResults`,
  :class:`QTECurve` dataclasses.
* :mod:`.setup` -- micro-panel data preparation.
* :mod:`.quantiles` -- empirical quantile functions and Halton / Sobol
  / uniform quantile-grid samplers.
* :mod:`.weights` -- simplex-constrained least-squares solver for the
  per-pre-period weights.
* :mod:`.aggregation` -- :math:`\\lambda_t` weighting across pre-periods.
* :mod:`.pipeline` -- public dispatcher :func:`run_dsc` composing the
  four steps of Algorithm 1.
"""

from __future__ import annotations

from .aggregation import aggregate_period_weights, build_lambda_weights
from .pipeline import run_dsc
from .quantiles import (
    build_pseudo_sample_matrix,
    empirical_quantile,
    sample_quantile_grid,
)
from .setup import prepare_dsc_inputs
from .structures import DSCInputs, DSCResults, QTECurve
from .weights import solve_simplex_weights, wasserstein_loss_at_weights

__all__ = [
    "DSCInputs",
    "DSCResults",
    "QTECurve",
    "aggregate_period_weights",
    "build_lambda_weights",
    "build_pseudo_sample_matrix",
    "empirical_quantile",
    "prepare_dsc_inputs",
    "run_dsc",
    "sample_quantile_grid",
    "solve_simplex_weights",
    "wasserstein_loss_at_weights",
]
