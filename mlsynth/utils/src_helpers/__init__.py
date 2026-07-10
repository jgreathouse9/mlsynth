"""Synthetic Regressing Control (SRC) helper subpackage.

NumPy-first implementation of Zhu (2023): per-donor univariate matching plus a
Mallows/Cp box-``[0, 1]`` synthesis, solved exactly by an active-set box QP.
The weight computation matches the author's reference R implementation to
machine precision.
"""

from __future__ import annotations

from .estimation import SRCWeights, counterfactual, src_weights
from .orchestration import run_src
from .plotter import plot_src
from .setup import prepare_src_inputs
from .solver import solve_box_qp
from .structures import SRCFit, SRCInputs, SRCResults
from .vsearch import optimize_v

__all__ = [
    "SRCFit",
    "SRCInputs",
    "SRCResults",
    "SRCWeights",
    "counterfactual",
    "optimize_v",
    "plot_src",
    "prepare_src_inputs",
    "run_src",
    "src_weights",
    "solve_box_qp",
]
