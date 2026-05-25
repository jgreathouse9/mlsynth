"""Helper subpackage for Synthetic Control with Multiple Outcomes (SCMO).

NumPy-first engine: a spec builds the matching matrix ``Z`` (matrix_builder),
the SC weights are solved on the simplex (solvers), and everything is indexed
through :class:`IndexSet`. The only DataFrame touchpoint is ``setup``.
"""

from .structures import (
    AVERAGED,
    CONCATENATED,
    MA,
    SEPARATE,
    SCMOInputs,
    SCMOMethodFit,
    SCMOResults,
)
from .matrix_builder import build_matching_matrix
from .solvers import simplex_weights
from .setup import prepare_scmo_inputs

__all__ = [
    "CONCATENATED", "AVERAGED", "SEPARATE", "MA",
    "SCMOInputs", "SCMOMethodFit", "SCMOResults",
    "build_matching_matrix", "simplex_weights", "prepare_scmo_inputs",
]
