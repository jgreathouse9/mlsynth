"""Helper package for the CMBSTS estimator (Causal Multivariate Bayesian
Structural Time Series; Menchetti and Bojinov 2022)."""

from .config import CMBSTSConfig
from .pipeline import run_cmbsts
from .plotter import plot_cmbsts
from .setup import prepare_cmbsts_inputs
from .structures import (
    CMBSTSInference,
    CMBSTSInputs,
    CMBSTSPosterior,
    CMBSTSResults,
)

__all__ = [
    "CMBSTSConfig",
    "CMBSTSInputs",
    "CMBSTSPosterior",
    "CMBSTSInference",
    "CMBSTSResults",
    "prepare_cmbsts_inputs",
    "run_cmbsts",
    "plot_cmbsts",
]
