"""Forward-Selected Synthetic Control (FSCM) helper subpackage.

NumPy-first port of Cerulli (2024): forward stepwise donor selection with
two-interval-time out-of-sample cross-validation to choose the donor count.
"""

from .structures import FSCMInputs, FSCMResults, FSCMSelectionPath
from .setup import derive_treatment, prepare_fscm_inputs
from .estimation import run_fscm
from .plotter import plot_fscm

__all__ = [
    "FSCMInputs",
    "FSCMResults",
    "FSCMSelectionPath",
    "derive_treatment",
    "prepare_fscm_inputs",
    "run_fscm",
    "plot_fscm",
]
