"""Helpers for the MC-NNM estimator (Athey et al. 2021).

Matrix Completion with Nuclear Norm Minimization: imputes the missing
(treated) entries of the outcome matrix by nuclear-norm-regularised
low-rank completion with unregularised two-way fixed effects (solved by
SOFT-IMPUTE, with the threshold chosen by cross-validation), then forms
treatment effects as observed minus imputed.

Module layout:

* :mod:`.completion` -- the SOFT-IMPUTE engine (:func:`mcnnm_fit`,
  :func:`mcnnm_cv`).
* :mod:`.structures` -- :class:`MCNNMInputs`, :class:`MCNNMResults`,
  :class:`MCNNMInference`.
* :mod:`.setup` -- panel ingestion.
* :mod:`.pipeline` -- :func:`run_mcnnm` causal dispatcher.
* :mod:`.plotter` -- observed-vs-counterfactual chart.
"""

from __future__ import annotations

from .completion import mcnnm_cv, mcnnm_fit
from .pipeline import run_mcnnm
from .setup import prepare_mcnnm_inputs
from .structures import MCNNMInference, MCNNMInputs, MCNNMResults

__all__ = [
    "MCNNMInference",
    "MCNNMInputs",
    "MCNNMResults",
    "mcnnm_cv",
    "mcnnm_fit",
    "prepare_mcnnm_inputs",
    "run_mcnnm",
]
