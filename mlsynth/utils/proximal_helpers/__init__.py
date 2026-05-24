"""Helper modules for the Proximal Inference (PROXIMAL) estimator.

Implements:

    Shi, X., Li, K., Miao, W., Hu, M., & Tchetgen Tchetgen, E. (2023).
    "Theory for Identification and Inference with Synthetic Controls: A
    Proximal Causal Inference Framework." arXiv:2108.13935.

    Liu, J., Tchetgen Tchetgen, E. J., & Varjao, C. (2023). "Proximal
    Causal Inference for Synthetic Control with Surrogates."
    arXiv:2308.09527.

    Park, C., & Tchetgen Tchetgen, E. J. (2025). "Single Proxy Synthetic
    Control." Journal of Causal Inference 13(1), 20230079.

Each estimator lives in its own subpackage so that new proximal methods
can be added as additional subpackages:

    pi/        : PI       -- donors-only two-proxy GMM (Shi et al.)
    pis/       : PIS      -- full-sample two-stage with surrogates (Liu et al.)
    pipost/    : PIPost   -- post-treatment-only surrogate variant (Liu et al.)
    spsc/      : SPSC     -- single-proxy ridge-GMM + conformal (Park & Tchetgen)

Shared infrastructure:

    structures.py       : PROXIMALInputs, ProximalMethodFit, PROXIMALResults
    setup.py            : prepare_proximal_inputs (dataprep + proxy prep + cleaning)
    inference.py        : bartlett kernel + HAC long-run variance (PI family)
    orchestration.py    : run_proximal (dispatch over the requested methods)
    plotter.py          : trajectories + gap overlay across methods
"""

from .inference import bartlett, hac
from .orchestration import run_proximal
from .pi import estimate_pi
from .pipost import estimate_pi_surrogate_post
from .pis import estimate_pi_surrogate
from .plotter import plot_proximal
from .setup import prepare_proximal_inputs
from .spsc import conformal_intervals, estimate_spsc
from .structures import (
    PI,
    PIPOST,
    PIS,
    SPSC,
    PROXIMALInputs,
    PROXIMALResults,
    ProximalMethodFit,
)

__all__ = [
    "PI",
    "PIPOST",
    "PIS",
    "SPSC",
    "PROXIMALInputs",
    "PROXIMALResults",
    "ProximalMethodFit",
    "bartlett",
    "conformal_intervals",
    "estimate_pi",
    "estimate_pi_surrogate",
    "estimate_pi_surrogate_post",
    "estimate_spsc",
    "hac",
    "plot_proximal",
    "prepare_proximal_inputs",
    "run_proximal",
]
