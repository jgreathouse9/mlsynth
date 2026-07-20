"""Helper utilities for the MEDSC estimator.

Implements Mellace & Pasquini (2022), the causal-mediation synthetic control:
the treatment effect on a panel outcome is split into a direct effect (holding
the mediator at its treated post-treatment path, via a cross-world synthetic
control) and an indirect effect that runs through the mediator (total minus
direct). The total effect is an ordinary synthetic control; the direct effect
is re-estimated per post period on a donor pool that may add the units needed to
span the treated unit's post-treatment mediator values.
"""

from .config import MEDSCConfig
from .pipeline import run_medsc_core
from .plotter import plot_medsc
from .setup import prepare_medsc_inputs
from .structures import MEDSCInputs, MEDSCResults, MediationDecomposition

__all__ = [
    "MEDSCConfig",
    "MEDSCInputs",
    "MEDSCResults",
    "MediationDecomposition",
    "plot_medsc",
    "prepare_medsc_inputs",
    "run_medsc_core",
]
