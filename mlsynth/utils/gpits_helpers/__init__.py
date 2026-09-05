"""Helpers for the GPITS estimator (Cho 2026, arXiv:2608.20610).

    config.py     : GPITSConfig
    kernels.py    : covariance functions and the length-scale rule
    setup.py      : long DataFrame -> GPITSInputs (via dataprep)
    pipeline.py   : the Gaussian process, effects, and placebo checks
    plotter.py    : observed vs counterfactual, returns a Figure
    structures.py : frozen result containers
"""

from .config import GPITSConfig
from .kernels import getb_maxvar, kernel_gaussian, kernel_gaussian_periodic_linear
from .pipeline import fit_gpits, run_placebo, summarize_effects
from .plotter import plot_gpits, plot_gpits_panels
from .setup import prepare_gpits_inputs
from .structures import GPITSDesign, GPITSInputs, GPITSPlacebo, GPITSResults

__all__ = [
    "GPITSConfig", "GPITSDesign", "GPITSInputs", "GPITSPlacebo", "GPITSResults",
    "fit_gpits", "getb_maxvar", "kernel_gaussian",
    "kernel_gaussian_periodic_linear", "plot_gpits", "plot_gpits_panels", "prepare_gpits_inputs",
    "run_placebo", "summarize_effects",
]
