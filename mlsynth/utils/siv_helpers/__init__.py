"""Helper utilities for the Synthetic IV (SIV) estimator.

SIV implements Gulek and Vives-i-Bastida (2024) "Synthetic IV
Estimation in Panels", a two-stage estimator that combines per-unit
synthetic controls (to remove unobserved factor structure) with
2SLS (to handle treatment endogeneity given an instrument).
"""

from .ensemble import select_alpha
from .inference import asymptotic_ci, split_conformal_inference
from .projection import project_outcome_pre
from .setup import build_design_matrix, prepare_siv_inputs
from .structures import (
    SIVEstimate,
    SIVInference,
    SIVInputs,
    SIVResults,
    SIVWeights,
)
from .twosls import two_sls_just_identified
from .weights import assemble_weights, fit_synthetic_controls

__all__ = [
    "SIVEstimate",
    "SIVInference",
    "SIVInputs",
    "SIVResults",
    "SIVWeights",
    "assemble_weights",
    "asymptotic_ci",
    "build_design_matrix",
    "fit_synthetic_controls",
    "prepare_siv_inputs",
    "project_outcome_pre",
    "select_alpha",
    "split_conformal_inference",
    "two_sls_just_identified",
]
