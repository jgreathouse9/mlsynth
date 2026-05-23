"""Helpers for the Continuous-Treatment Synthetic Control (CTSC) estimator.

Implements Powell, D. (2022), *Synthetic Control Estimation Beyond
Comparative Case Studies: Does the Minimum Wage Reduce Employment?*
(Journal of Business & Economic Statistics). The paper calls the method
"GSC"; mlsynth uses **CTSC** to avoid collision with Xu (2017)'s
Generalized Synthetic Control.

Module layout:

* :mod:`.structures` -- :class:`CTSCInputs`, :class:`CTSCResults`,
  :class:`CTSCInference`.
* :mod:`.setup` -- panel ingestion (continuous/multi-valued treatment).
* :mod:`.estimate` -- joint slopes + synthetic-control weights via
  block coordinate descent (paper eq. 5-7).
* :mod:`.inference` -- sign-flip Wald test (paper Section 4).
* :mod:`.pipeline` -- :func:`run_ctsc` dispatcher.
* :mod:`.simulation` -- the calibrated Monte-Carlo study (paper Table 1,
  Models 1-4).
"""

from __future__ import annotations

from .estimate import fit_ctsc
from .inference import sign_flip_wald_inference
from .pipeline import run_ctsc
from .setup import prepare_ctsc_inputs
from .simulation import (
    SimulationSummary,
    generate_model,
    run_simulation,
    twoway_fe_effect,
)
from .structures import CTSCInference, CTSCInputs, CTSCResults

__all__ = [
    "CTSCInference",
    "CTSCInputs",
    "CTSCResults",
    "SimulationSummary",
    "fit_ctsc",
    "generate_model",
    "prepare_ctsc_inputs",
    "run_ctsc",
    "run_simulation",
    "sign_flip_wald_inference",
    "twoway_fe_effect",
]
