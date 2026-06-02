"""Helpers for the SSC (Staggered Synthetic Control) estimator.

Cao, J., Lu, S. & Wu, H. (2026). *"Synthetic Control Inference for Staggered
Adoption."* The Econometrics Journal.

Models each unit's untreated outcome as an intercept plus a simplex synthetic
control on all other units (so not-yet-treated units are valid donors), jointly
estimates every unit x time treatment effect by GLS, aggregates to event-time /
overall ATT, and forms Andrews (2003) end-of-sample stability bands from
pre-treatment residual windows.

Module layout:

* :mod:`.weights` -- per-unit simplex synthetic-control weights (batch).
* :mod:`.estimation` -- selector tensor, the GLS effect estimator, aggregation
  and end-of-sample inference.
* :mod:`.structures` -- :class:`SSCInputs`, :class:`SSCBand`,
  :class:`SSCInference`, :class:`SSCResults`.
* :mod:`.setup` -- staggered-panel ingestion.
* :mod:`.pipeline` -- :func:`run_ssc` orchestration.
* :mod:`.plotter` -- event-study chart.
* :mod:`.simulation` -- staggered factor-model DGP.
* :mod:`.replication` -- Path-B replication of the paper's Section 3 study.
"""

from __future__ import annotations

from .pipeline import run_ssc
from .replication import replicate_guanajuato, run_ssc_simulation
from .setup import prepare_ssc_inputs
from .simulation import simulate_ssc_panel
from .structures import SSCBand, SSCInference, SSCInputs, SSCResults
from .weights import synthetic_control_batch

__all__ = [
    "SSCBand",
    "SSCInference",
    "SSCInputs",
    "SSCResults",
    "prepare_ssc_inputs",
    "replicate_guanajuato",
    "run_ssc",
    "run_ssc_simulation",
    "simulate_ssc_panel",
    "synthetic_control_batch",
]
