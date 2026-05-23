"""Helpers for the Imperfect Synthetic Controls (ISCM) estimator.

Implements Powell, D. (2026), *Imperfect Synthetic Controls* (Journal of
Applied Econometrics). ISCM relaxes the SCM's perfect-synthetic-control
assumption: it builds synthetic controls for every unit, identifies the
treatment effect even when the treated unit is outside the convex hull,
weights units by a data-driven fit metric, and uses Ibragimov-Muller
inference valid for small donor pools.

Module layout:

* :mod:`.structures` -- :class:`ISCMInputs`, :class:`ISCMResults`,
  :class:`ISCMInference`.
* :mod:`.setup` -- panel ingestion.
* :mod:`.weights` -- all-units synthetic-control weights (paper eq. 5).
* :mod:`.estimate` -- fit metric (eq. 14) and WLS effect (eq. 15).
* :mod:`.inference` -- Ibragimov-Muller randomization test (eq. 16).
* :mod:`.pipeline` -- :func:`run_iscm` dispatcher.
"""

from __future__ import annotations

from .estimate import fit_metric, residuals_and_exposure, weighted_att
from .inference import ibragimov_muller_inference
from .pipeline import run_iscm
from .setup import prepare_iscm_inputs
from .structures import ISCMInference, ISCMInputs, ISCMResults
from .weights import all_units_weights

__all__ = [
    "ISCMInference",
    "ISCMInputs",
    "ISCMResults",
    "all_units_weights",
    "fit_metric",
    "ibragimov_muller_inference",
    "prepare_iscm_inputs",
    "residuals_and_exposure",
    "run_iscm",
    "weighted_att",
]
