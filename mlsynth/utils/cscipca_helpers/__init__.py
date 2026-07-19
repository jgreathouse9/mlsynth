"""Helper utilities for the CSCIPCA estimator.

Implements Wang (2024), *"Counterfactual and Synthetic Control Method: Causal
Inference with Instrumented Principal Component Analysis"*. CSC-IPCA is a
factor-model counterfactual imputer whose factor loadings are a linear
projection of observed covariates (``Lambda_it = X_it Gamma``), estimated by
alternating least squares and validated period-by-period with moving-block
conformal inference.
"""

from .als import (
    als_estimate,
    counterfactual,
    normalize,
    solve_factors,
    solve_gamma,
)
from .config import CSCIPCAConfig
from .inference import cscipca_conformal
from .pipeline import CSCIPCAFit, fit_cscipca
from .plotter import plot_cscipca
from .setup import prepare_cscipca_inputs
from .structures import (
    CSCIPCADesign,
    CSCIPCAInference,
    CSCIPCAInputs,
    CSCIPCAResults,
)

__all__ = [
    "CSCIPCAConfig",
    "CSCIPCADesign",
    "CSCIPCAFit",
    "CSCIPCAInference",
    "CSCIPCAInputs",
    "CSCIPCAResults",
    "als_estimate",
    "counterfactual",
    "cscipca_conformal",
    "fit_cscipca",
    "normalize",
    "plot_cscipca",
    "prepare_cscipca_inputs",
    "solve_factors",
    "solve_gamma",
]
