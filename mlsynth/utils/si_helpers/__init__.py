"""Synthetic Interventions (SI) helper package.

Implements the algorithmic pieces of:

    Agarwal, A., Shah, D., & Shen, D. (2026). "Synthetic Interventions:
    Extending Synthetic Controls to Multiple Treatments." Operations Research
    74(2):840-859.

Module layout (one responsibility each):

    structures.py    : frozen dataclass containers (inputs / arm / results)
    setup.py         : focal-unit + per-intervention donor-pool preparation
    estimation.py    : SI-PCR weights (eq. 10) and the bias-corrected fit
                       (Section 4.3, eqs. 12 and 14), built on shared HSVT
    orchestration.py : top-level solve_si over all intervention arms + CIs
    plotter.py       : observed-vs-counterfactual plot

SI-PCR reuses the HSVT primitives of the ClusterSC helpers
(:mod:`mlsynth.utils.clustersc_helpers.pcr.hsvt`).
"""

from .estimation import (
    bias_corrected_fit,
    resolve_rank,
    select_omega,
    si_pcr_weights,
)
from .orchestration import solve_si
from .setup import prepare_si_inputs
from .simulation import generate_low_rank_matrix, generate_low_rank_matrices
from .structures import SIArm, SIDonorPool, SIInputs, SIResults

__all__ = [
    "si_pcr_weights",
    "bias_corrected_fit",
    "select_omega",
    "resolve_rank",
    "prepare_si_inputs",
    "solve_si",
    "generate_low_rank_matrix",
    "generate_low_rank_matrices",
    "SIDonorPool",
    "SIInputs",
    "SIArm",
    "SIResults",
]
