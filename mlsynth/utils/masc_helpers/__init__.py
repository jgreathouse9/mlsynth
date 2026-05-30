"""Helper modules for the MASC estimator.

See :mod:`mlsynth.estimators.masc` for the public estimator class.

Implements the Matching and Synthetic Control (MASC) estimator of
Kellogg, Mogstad, Pouliot & Torgovitsky (2021). Algorithmic structure
ported directly from Maxwell Kellogg's reference R package
(``masc/R/``), MIT-licensed.
"""

from __future__ import annotations

from .crossval import cross_validate
from .estimation import (
    analytic_phi,
    masc_combine,
    nearest_neighbor_weights,
    sc_simplex_weights,
)
from .orchestration import run_masc
from .plotter import plot_masc
from .setup import prepare_masc_inputs
from .structures import MASCFit, MASCInputs, MASCResults

__all__ = [
    "MASCFit",
    "MASCInputs",
    "MASCResults",
    "analytic_phi",
    "cross_validate",
    "masc_combine",
    "nearest_neighbor_weights",
    "plot_masc",
    "prepare_masc_inputs",
    "run_masc",
    "sc_simplex_weights",
]
