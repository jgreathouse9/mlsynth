"""Helper utilities for the SPCD estimator.

Implements:

    Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
    "Synthetic Principal Component Design: Fast Covariate Balancing
    with Synthetic Controls." arXiv:2211.15241v1.

Module map (paper-grounded):

    formulation.py         : Eq. (2)         M = Y Y^T + alpha I + lambda 1 1^T
    spectral_init.py       : Algs 1 & 2      y^0 = sgn(smallest eigvec of M)
    iteration_spcd.py      : Eqs. (4)/(7)    SPCD update box
    iteration_norm_spcd.py : Eqs. (5)/(8)    NormSPCD update box
    weights_empirical.py   : Eq. (9)         Algorithm 2 final step
    weights_exact.py       : Eq. (6)         Algorithm 1 final step
    treatment_effect.py    : Algs 1 & 2 footer (minority flip, synthetic paths)
    orchestration.py       : end-to-end solve_spcd + holdout-aware wrapper

Inference and power (LEXSCM-style E/B holdout split):

    holdout.py             : pre-window split + holdout residuals
    inference.py           : moving-block conformal CI for the ATT
    power.py               : Monte Carlo MDE + detectability curve

Algorithms 3 and 4 of the paper (Appendix 3.2, page 20) are abstract
meta-versions of SPCD/NormSPCD used to prove Theorem 3 (global
convergence). They are not implemented here as separate code paths.
"""

from .formulation import build_iteration_matrix, validate_spcd_inputs
from .holdout import compute_holdout_residuals, split_pre_window
from .inference import SPCDConformalResult, compute_conformal_ci
from .iteration_norm_spcd import norm_spcd_step, run_norm_spcd_iteration
from .iteration_spcd import run_spcd_iteration, spcd_step
from .orchestration import solve_spcd, solve_spcd_with_holdout
from .plotter import plot_spcd_design
from .power import (
    SPCDPowerAnalysis,
    compute_detectability_curve,
    compute_mde,
)
from .results_assembly import build_summary
from .setup import prepare_spcd_inputs
from .spectral_init import spectral_initialization
from .structures import SPCDDesign, SPCDInputs, SPCDResults
from .treatment_effect import (
    apply_minority_flip,
    build_synthetic_paths,
    build_weight_groups,
    compute_att_and_fit,
)
from .weights_empirical import empirical_weights
from .weights_exact import exact_weights

__all__ = [
    "SPCDConformalResult",
    "SPCDDesign",
    "SPCDInputs",
    "SPCDPowerAnalysis",
    "SPCDResults",
    "apply_minority_flip",
    "build_iteration_matrix",
    "build_summary",
    "build_synthetic_paths",
    "build_weight_groups",
    "compute_att_and_fit",
    "compute_conformal_ci",
    "compute_detectability_curve",
    "compute_holdout_residuals",
    "compute_mde",
    "empirical_weights",
    "exact_weights",
    "norm_spcd_step",
    "plot_spcd_design",
    "prepare_spcd_inputs",
    "run_norm_spcd_iteration",
    "run_spcd_iteration",
    "solve_spcd",
    "solve_spcd_with_holdout",
    "spcd_step",
    "spectral_initialization",
    "split_pre_window",
    "validate_spcd_inputs",
]
