"""Helpers for the Multivariate Square-root Lasso Synthetic Control (MSQRT).

Shen, Z., Song, X., & Abadie, A. (2025). *"Efficiently Learning Synthetic
Control Models for High-dimensional Disaggregated Data."* arXiv:2510.22828.

Stacks all treated units into one matrix regression ``Y = X Theta + E`` and
fits the donor-weight matrix by Multivariate Square-root Lasso (nuclear-norm
loss + element-wise L1), with the L1 penalty chosen by rolling-origin
cross-validation, then forms the ATT as the mean post-period gap.

Module layout:

* :mod:`.optimization` -- the cvxpy solve (:func:`fit_msqrt_weights`) and the
  rolling-origin penalty search (:func:`select_lambda_cv`).
* :mod:`.structures` -- :class:`MSQRTInputs`, :class:`MSQRTResults`.
* :mod:`.setup` -- panel ingestion (block design enforced).
* :mod:`.pipeline` -- :func:`run_msqrt` orchestration.
* :mod:`.plotter` -- observed-vs-synthetic chart.
* :mod:`.simulation` -- high-dimensional multiple-treated factor DGP.

Uncertainty quantification is delegated to :mod:`mlsynth.utils.scpi_helpers`
(Cattaneo, Feng, Palomba & Titiunik 2025).
"""

from __future__ import annotations

from .optimization import fit_msqrt_weights, select_lambda_cv
from .pipeline import run_msqrt
from .setup import prepare_msqrt_inputs
from .simulation import simulate_msqrt_panel
from .structures import MSQRTInputs, MSQRTResults

__all__ = [
    "MSQRTInputs",
    "MSQRTResults",
    "fit_msqrt_weights",
    "prepare_msqrt_inputs",
    "run_msqrt",
    "select_lambda_cv",
    "simulate_msqrt_panel",
]
