"""Helper modules for the Synthetic Historical Control (SHC) estimator.

Implements:

    Chen, Y.-T., Yang, J.-C., & Yang, T.-T. (2024). "Synthetic Historical
    Control for Policy Evaluation." SSRN 4995085.

Layout:

    setup.py         : prepare_shc_inputs (dataprep -> SHCInputs, time IndexSet)
    orchestration.py : solve_shc + summarize_effects (Section 2.3 pipeline)
    inference.py     : run_conformal_inference (footnote 21 permutation test)
    plotter.py       : plot_shc (observed vs counterfactual + conformal band)
    structures.py    : frozen dataclasses (SHCInputs/Design/Inference/Results)
    simulation.py    : Section 3.1 data-generating process
    monte_carlo.py   : MSE_pre / MSE_post(k) validation harness
"""

from __future__ import annotations

from .inference import run_conformal_inference
from .monte_carlo import monte_carlo_shc
from .orchestration import solve_shc, summarize_effects
from .setup import prepare_shc_inputs
from .simulation import simulate_shc_latent, simulate_shc_panel
from .structures import (
    SHCDesign,
    SHCInference,
    SHCInputs,
    SHCResults,
)

__all__ = [
    "SHCDesign",
    "SHCInference",
    "SHCInputs",
    "SHCResults",
    "monte_carlo_shc",
    "prepare_shc_inputs",
    "run_conformal_inference",
    "simulate_shc_latent",
    "simulate_shc_panel",
    "solve_shc",
    "summarize_effects",
]
