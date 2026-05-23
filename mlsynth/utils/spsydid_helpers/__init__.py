"""Helper utilities for the Spatial Synthetic Difference-in-Differences estimator.

Implements Serenini & Masek (2024), *Spatial Synthetic
Difference-in-Differences*. SpSyDiD extends Arkhangelsky et al. (2021)
SDID with a spatial spillover term so the estimator can disentangle the
direct ATT on the treated units from the *indirect* effect on units
exposed via a spatial weight matrix :math:`W`.

Module layout:

* :mod:`.structures` -- :class:`SpSyDiDInputs`, :class:`SpSyDiDResults`
  frozen dataclasses.
* :mod:`.spatial` -- ``W`` validation + builders (k-NN, inverse-distance,
  contiguity).
* :mod:`.setup` -- long-panel data prep with auto-partition of donors
  into directly treated, spillover-exposed, and pure controls.
* :mod:`.weights` -- SDID unit / time weight QPs duplicated from
  ``sdid_helpers`` for behavioural isolation.
* :mod:`.pipeline` -- public dispatcher :func:`run_spsydid`.
"""

from __future__ import annotations

from .pipeline import run_spsydid
from .setup import prepare_spsydid_inputs
from .spatial import (
    contiguity_weights,
    inverse_distance_weights,
    knn_weights,
    row_standardize,
    validate_spatial_matrix,
)
from .structures import SpSyDiDInputs, SpSyDiDResults
from .weights import (
    compute_regularization,
    fit_time_weights,
    fit_unit_weights,
)

__all__ = [
    "SpSyDiDInputs",
    "SpSyDiDResults",
    "compute_regularization",
    "contiguity_weights",
    "fit_time_weights",
    "fit_unit_weights",
    "inverse_distance_weights",
    "knn_weights",
    "prepare_spsydid_inputs",
    "row_standardize",
    "run_spsydid",
    "validate_spatial_matrix",
]
