"""Helper utilities for the Nonlinear Synthetic Control (NSC) estimator.

NSC implements Tian (2023), *"The Synthetic Control Method with
Nonlinear Outcomes"* (arXiv:2306.01967): an affine-weight extension
of Abadie-Diamond-Hainmueller (2010) that drops the non-negativity
constraint, adds pairwise-distance-weighted L1 plus L2 penalties,
and cross-validates the eigenvalue-scaled tuning parameters on
``[0, 1]``.
"""

from .crossval import cv_select
from .inference import doudchenko_imbens_inference
from .optimization import (
    design_eigenvalues,
    fit_nsc,
    scale_a,
    scale_b,
    solve_nsc_weights,
)
from .plotter import plot_nsc
from .setup import prepare_nsc_inputs
from .structures import (
    NSCCVTrace,
    NSCDesign,
    NSCInference,
    NSCInputs,
    NSCResults,
)

__all__ = [
    "NSCCVTrace",
    "NSCDesign",
    "NSCInference",
    "NSCInputs",
    "NSCResults",
    "cv_select",
    "design_eigenvalues",
    "doudchenko_imbens_inference",
    "fit_nsc",
    "plot_nsc",
    "prepare_nsc_inputs",
    "scale_a",
    "scale_b",
    "solve_nsc_weights",
]
