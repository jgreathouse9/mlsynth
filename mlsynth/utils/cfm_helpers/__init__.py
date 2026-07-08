"""Helper utilities for the Causal Factor Model (CFM) estimator.

Implements Bai & Wang (2026), *"Causal Inference Using Factor Models"*.
CFM models both potential outcomes within one factor structure, letting the
treated unit's factor loadings break at the intervention date, and targets
the systematic causal effect for a single treated unit.
"""

from .config import CFMConfig
from .factors import ahn_horenstein, extract_cfm_factors
from .inference import (
    block_regression_variance,
    cfm_inference,
    factor_estimation_variance,
)
from .pipeline import chow_break_statistic, fit_systematic_effect
from .plotter import plot_cfm
from .setup import prepare_cfm_inputs
from .structures import CFMDesign, CFMInference, CFMInputs, CFMResults

__all__ = [
    "CFMConfig",
    "CFMDesign",
    "CFMInference",
    "CFMInputs",
    "CFMResults",
    "ahn_horenstein",
    "block_regression_variance",
    "cfm_inference",
    "chow_break_statistic",
    "extract_cfm_factors",
    "factor_estimation_variance",
    "fit_systematic_effect",
    "plot_cfm",
    "prepare_cfm_inputs",
]
