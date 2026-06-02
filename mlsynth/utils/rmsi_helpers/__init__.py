"""Helpers for the RMSI (Robust Matrix estimation with Side Information) estimator.

Agarwal, A., Choi, J. & Yuan, M. (2026). *"Robust Matrix Estimation with Side
Information."* arXiv:2603.24833.

Imputes the treated counterfactual of a block-adoption causal panel by a
four-component sieve + nuclear-norm matrix estimator that exploits unit-level
(row) and time-level (column) covariates: each component is recovered by a
projection and a singular-value soft-threshold (Algorithm 1), and the
block-missing pattern is handled by the tall/wide recombination (Algorithm 3).

Module layout:

* :mod:`.core` -- sieve bases, projections, SVT, Algorithm 1 and Algorithm 3.
* :mod:`.structures` -- :class:`RMSIInputs`, :class:`RMSIResults`.
* :mod:`.setup` -- panel + side-information ingestion (block design enforced).
* :mod:`.pipeline` -- :func:`run_rmsi` orchestration.
* :mod:`.plotter` -- observed-vs-imputed chart.
* :mod:`.simulation` -- block causal DGP with side information.
"""

from __future__ import annotations

from .core import algorithm1, algorithm3, sieve_poly
from .pipeline import run_rmsi
from .replication import replicate_prop99, run_rmsi_simulation
from .setup import prepare_rmsi_inputs
from .simulation import simulate_rmsi_panel
from .structures import RMSIInputs, RMSIResults

__all__ = [
    "RMSIInputs",
    "RMSIResults",
    "algorithm1",
    "algorithm3",
    "prepare_rmsi_inputs",
    "replicate_prop99",
    "run_rmsi",
    "run_rmsi_simulation",
    "sieve_poly",
    "simulate_rmsi_panel",
]
