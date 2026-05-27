"""MAREX (synthetic experimental design) helper package.

Implements the algorithmic pieces of:

    Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental Design."

Module layout (one responsibility each):

    structures.py    : frozen dataclass containers (study / cluster / global /
                       inference / results)
    setup.py         : long-panel -> design-ready arrays
    formulation.py   : cvxpy variables, constraints, and design objectives
    optimization.py  : exact MIQP and relaxed (post-hoc-discretised) solvers
    inference.py     : blank-period placebo / permutation inference
    orchestration.py : top-level solve_marex assembling frozen results
    plotter.py       : synthetic treated-vs-control plot
    simulation.py    : the paper's linear-factor DGP (Section 5)
"""

from .formulation import build_constraints, build_objective
from .inference import compute_inference
from .optimization import (
    post_hoc_discretize,
    solve_design,
    solve_design_relaxed,
)
from .orchestration import solve_marex
from .setup import MAREXPanel, prepare_marex_panel
from .simulation import MAREXSample, generate_marex_sample
from .structures import (
    MAREXClusterDesign,
    MAREXGlobalDesign,
    MAREXInference,
    MAREXResults,
    MAREXStudy,
)

__all__ = [
    "build_constraints",
    "build_objective",
    "solve_design",
    "solve_design_relaxed",
    "post_hoc_discretize",
    "compute_inference",
    "solve_marex",
    "prepare_marex_panel",
    "MAREXPanel",
    "generate_marex_sample",
    "MAREXSample",
    "MAREXStudy",
    "MAREXClusterDesign",
    "MAREXGlobalDesign",
    "MAREXInference",
    "MAREXResults",
]
