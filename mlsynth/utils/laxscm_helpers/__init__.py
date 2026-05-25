"""Helper subpackage for the Relaxed/penalized SCM engine (RESCM).

NumPy-first, IndexSet-indexed. A single convex synthetic-control program nests
a family of estimators selected by *name* (see :mod:`specs`):

* penalized branch -- classic ``SC``, ``LASSO``, ``RIDGE``, ``ENET``, and the
  L-infinity-norm SCM (``LINF`` / ``L1LINF``) of Wang, Xing & Ye (2025).
* relaxation branch -- ``RELAX_L2`` / ``RELAX_ENTROPY`` / ``RELAX_EL``, the
  SCM-relaxation of Liao, Shi & Zheng (2026).

The only DataFrame touchpoint is :func:`setup.prepare_rescm_inputs`.
"""

from .structures import (
    ELASTIC,
    RELAXED,
    RESCMInputs,
    RESCMMethodFit,
    RESCMResults,
)
from .specs import METHOD_SPECS, MethodSpec, normalize_method, resolve_specs
from .setup import derive_treatment, prepare_rescm_inputs
from .inference import ate_inference
from .estimation import run_rescm
from .results_assembly import assemble_rescm_results
from .plotter import plot_rescm

__all__ = [
    "ELASTIC", "RELAXED",
    "RESCMInputs", "RESCMMethodFit", "RESCMResults",
    "METHOD_SPECS", "MethodSpec", "normalize_method", "resolve_specs",
    "derive_treatment", "prepare_rescm_inputs",
    "ate_inference", "run_rescm", "assemble_rescm_results", "plot_rescm",
]
