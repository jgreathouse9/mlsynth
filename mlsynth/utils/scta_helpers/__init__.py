"""Helper subpackage for Synthetic Control with Temporal Aggregation (SCTA).

NumPy-first engine: ``setup`` ingests the panel via ``dataprep``; ``pipeline``
builds the stacked ``[aggregate | disaggregated]`` matching design, weights it
with a fixed ``nu``-parameterised ``V``, and solves the simplex SC at the true
optimum (optionally ridge-augmented); ``structures`` holds the frozen inputs /
fit / results; ``plotter`` draws the counterfactual and the imbalance frontier.
"""

from .config import SCTAConfig
from .structures import SCTAInputs, SCTAFit, SCTAResults
from .setup import prepare_scta_inputs
from .pipeline import fit_one, run_scta
from .plotter import plot_scta

__all__ = [
    "SCTAConfig",
    "SCTAInputs", "SCTAFit", "SCTAResults",
    "prepare_scta_inputs", "fit_one", "run_scta", "plot_scta",
]
