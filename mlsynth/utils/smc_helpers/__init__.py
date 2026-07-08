"""Synthetic Matching Control (SMC) helper subpackage.

NumPy-first implementation of Zhu (2023): per-donor univariate matching plus a
Mallows/Cp box-``[0, 1]`` synthesis, solved exactly by an active-set box QP.
The weight computation matches the author's reference R implementation to
machine precision.
"""

from __future__ import annotations

from .estimation import SMCWeights, counterfactual, smc_weights
from .orchestration import run_smc
from .plotter import plot_smc
from .setup import prepare_smc_inputs
from .solver import solve_box_qp
from .structures import SMCFit, SMCInputs, SMCResults
from .vsearch import optimize_v

__all__ = [
    "SMCFit",
    "SMCInputs",
    "SMCResults",
    "SMCWeights",
    "counterfactual",
    "optimize_v",
    "plot_smc",
    "prepare_smc_inputs",
    "run_smc",
    "smc_weights",
    "solve_box_qp",
]
