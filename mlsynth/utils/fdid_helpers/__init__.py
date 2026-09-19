"""Helper subpackage for the Forward Difference-in-Differences estimator.

Mirrors the modern ``mlsynth`` estimator layout (cf. CLUSTERSC, PROXIMAL,
SYNDES): typed structures, data setup, the estimation core, analytical
inference, typed result assembly, and a plotting wrapper.
"""

from .structures import (
    DID,
    FDID,
    FDIDEventStudy,
    FDIDInputs,
    FDIDMethodFit,
    FDIDResults,
    FDIDStaggeredInputs,
    FDIDStaggeredResults,
    FDIDUnitFit,
)
from .setup import prepare_fdid_inputs, prepare_panel, prepare_staggered_inputs
from .estimation import did_from_mean, forward_did_select, forward_selection_path
from .staggered import (
    aggregate_variance,
    event_time_weights,
    fit_staggered,
    residual_cross_covariances,
)
from .inference import (
    block_mean_variance,
    did_inference,
    hac_lag,
    residual_autocovariances,
)
from .results_assembly import assemble_fdid_results
from .plotter import plot_fdid, plot_fdid_staggered

__all__ = [
    "DID",
    "FDID",
    "FDIDEventStudy",
    "FDIDInputs",
    "FDIDMethodFit",
    "FDIDResults",
    "FDIDStaggeredInputs",
    "FDIDStaggeredResults",
    "FDIDUnitFit",
    "prepare_fdid_inputs",
    "prepare_panel",
    "prepare_staggered_inputs",
    "did_from_mean",
    "forward_did_select",
    "forward_selection_path",
    "aggregate_variance",
    "event_time_weights",
    "fit_staggered",
    "residual_cross_covariances",
    "block_mean_variance",
    "did_inference",
    "hac_lag",
    "residual_autocovariances",
    "assemble_fdid_results",
    "plot_fdid",
    "plot_fdid_staggered",
]
