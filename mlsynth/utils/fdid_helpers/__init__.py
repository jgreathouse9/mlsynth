"""Helper subpackage for the Forward Difference-in-Differences estimator.

Mirrors the modern ``mlsynth`` estimator layout (cf. CLUSTERSC, PROXIMAL,
SYNDES): typed structures, data setup, the estimation core, analytical
inference, typed result assembly, and a plotting wrapper.
"""

from .structures import (
    ADID,
    DID,
    FDID,
    FDIDInputs,
    FDIDMethodFit,
    FDIDResults,
)
from .setup import prepare_fdid_inputs
from .estimation import adid_from_mean, did_from_mean, forward_did_select
from .inference import (
    adid_inference,
    block_mean_variance,
    did_inference,
    hac_lag,
    residual_autocovariances,
)
from .results_assembly import assemble_fdid_results
from .plotter import plot_fdid

__all__ = [
    "ADID",
    "DID",
    "FDID",
    "FDIDInputs",
    "FDIDMethodFit",
    "FDIDResults",
    "prepare_fdid_inputs",
    "adid_from_mean",
    "did_from_mean",
    "forward_did_select",
    "adid_inference",
    "block_mean_variance",
    "did_inference",
    "hac_lag",
    "residual_autocovariances",
    "assemble_fdid_results",
    "plot_fdid",
]
