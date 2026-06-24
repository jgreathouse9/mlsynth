"""GMM-SCE: the GMM Synthetic Control Estimator of Fry (2024), dispatched as a
method of :class:`mlsynth.ORTHSC` (``method="gmm_sce"``).

Control weights are estimated on the unit simplex by one-step GMM using the
instrument units as instruments, with an optional Andrews--Lu downward-testing
procedure to choose the control/instrument split.
"""
from .pipeline import run_gmm_sce
from .solver import gmm_sc_weights
from .selection import select_controls

__all__ = ["run_gmm_sce", "gmm_sc_weights", "select_controls"]
