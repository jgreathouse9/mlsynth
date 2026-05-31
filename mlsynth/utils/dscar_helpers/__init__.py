"""Dynamic synthetic control helpers (Zheng & Chen 2024, JRSS-B 86(1)).

Estimator components for the Dynamic Synthetic Control (DSC) method
of Zheng & Chen (2024). DSC extends Abadie-Diamond-Hainmueller (2010)
synthetic control to settings with **time-varying confounders, spatial
dependence, and an auto-regressive outcome model**, by computing
**per-period weights** :math:`w_t^*` via empirical-likelihood
maximisation under per-period matching constraints (equations 2.7-2.9
of the paper).
"""

from __future__ import annotations

from .pipeline import run_dsc
from .setup import prepare_dsc_inputs
from .structures import DSCARFit, DSCARInputs, DSCARResults
from .weights import solve_dsc_weights

__all__ = [
    "DSCARFit",
    "DSCARInputs",
    "DSCARResults",
    "prepare_dsc_inputs",
    "run_dsc",
    "solve_dsc_weights",
]
