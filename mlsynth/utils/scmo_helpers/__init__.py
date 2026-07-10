"""Helper subpackage for Synthetic Control with Multiple Outcomes (SCMO).

NumPy-first engine: a spec builds the matching matrix ``Z`` (matrix_builder),
SC weights are solved on the simplex (solvers), the weighting schemes
(concatenated / averaged / separate / MA) live in estimation, inference is
permutation/conformal, and everything is indexed through :class:`IndexSet`.
The only DataFrame touchpoint is ``setup``.
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
from .estimation import fit_scheme, model_average
from .inference import conformal_inference
from .results_assembly import assemble_scmo_results
from .orchestration import resolve_schemes, derive_treatment, build_spec, run_scmo
from .pcr_cv import rolling_origin_pcr_cv
from .plotter import plot_scmo

__all__ = [
    "CONCATENATED", "AVERAGED", "SEPARATE", "MA",
    "SCMOInputs", "SCMOMethodFit", "SCMOResults",
    "build_matching_matrix", "simplex_weights", "prepare_scmo_inputs",
    "fit_scheme", "model_average", "conformal_inference",
    "assemble_scmo_results", "resolve_schemes", "derive_treatment", "build_spec",
    "run_scmo", "rolling_origin_pcr_cv", "plot_scmo",
]
