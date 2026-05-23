"""Helpers for the Synthetic Nearest Neighbors (SNN) estimator.

Implements Agarwal, Dahleh, Shah & Shen (2021), *Causal Matrix
Completion* (arXiv:2109.15154). SNN imputes missing matrix entries under
"missing not at random" (MNAR) patterns by finding a fully observed
anchor submatrix and running principal component regression -- a
generalisation of the Synthetic Interventions / synthetic-control PCR
machinery to arbitrary missingness. In the causal/panel setting it
imputes treated units' untreated potential outcomes and reports the ATT.

Module layout:

* :mod:`.completion` -- the matrix-completion engine (anchor finding +
  PCR), exposed as :func:`snn_complete`.
* :mod:`.structures` -- :class:`SNNInputs`, :class:`SNNResults`,
  :class:`SNNInference`.
* :mod:`.setup` -- panel ingestion.
* :mod:`.pipeline` -- :func:`run_snn` causal dispatcher.
"""

from __future__ import annotations

from .completion import snn_complete, snn_predict
from .pipeline import run_snn
from .setup import prepare_snn_inputs
from .structures import SNNInference, SNNInputs, SNNResults

__all__ = [
    "SNNInference",
    "SNNInputs",
    "SNNResults",
    "prepare_snn_inputs",
    "run_snn",
    "snn_complete",
    "snn_predict",
]
