"""Helper subpackage for the Forward Difference-in-Differences estimator.

Mirrors the modern ``mlsynth`` estimator layout (cf. CLUSTERSC, PROXIMAL,
SYNDES): typed structures, data setup, the estimation core, analytical
inference, typed result assembly, and a plotting wrapper.
"""

from .structures import (
    DID,
    FDID,
    FDIDInputs,
    FDIDMethodFit,
    FDIDResults,
)
from .setup import prepare_fdid_inputs
from .estimation import did_from_mean, forward_did_select
from .inference import did_inference
from .results_assembly import assemble_fdid_results
from .plotter import plot_fdid

__all__ = [
    "DID",
    "FDID",
    "FDIDInputs",
    "FDIDMethodFit",
    "FDIDResults",
    "prepare_fdid_inputs",
    "did_from_mean",
    "forward_did_select",
    "did_inference",
    "assemble_fdid_results",
    "plot_fdid",
]
