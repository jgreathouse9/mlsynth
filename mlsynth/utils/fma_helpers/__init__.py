"""Helper utilities for the Factor Model Approach (FMA) estimator.

Implements Li & Sonnier (2023), *"Statistical Inference for the
Factor Model Approach to Estimate Causal Effects in Quasi-
Experimental Settings"* (JMR 60(3):449-472).
"""

from .factors import extract_factors
from .fit import estimate_loading_and_counterfactual
from .inference import (
    asymptotic_inference,
    bootstrap_inference,
    placebo_inference,
)
from .plotter import plot_fma
from .setup import prepare_fma_inputs
from .structures import (
    FMADesign,
    FMAInference,
    FMAInputs,
    FMAResults,
)

__all__ = [
    "FMADesign",
    "FMAInference",
    "FMAInputs",
    "FMAResults",
    "asymptotic_inference",
    "bootstrap_inference",
    "estimate_loading_and_counterfactual",
    "extract_factors",
    "placebo_inference",
    "plot_fma",
    "prepare_fma_inputs",
]
