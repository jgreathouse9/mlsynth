"""Helper modules for the Proximal Inference (PROXIMAL) estimator.

Implements:

    Shi, X., Li, K., Miao, W., Hu, M., & Tchetgen Tchetgen, E. (2023).
    "Theory for Identification and Inference with Synthetic Controls: A
    Proximal Causal Inference Framework." arXiv:2108.13935.

    Liu, J., Tchetgen Tchetgen, E. J., & Varjao, C. (2023). "Proximal
    Causal Inference for Synthetic Control with Surrogates."
    arXiv:2308.09527.

PROXIMAL treats donor outcomes ``W`` as negative controls instrumented by
donor proxies ``Z0``, and (optionally) surrogate outcomes ``X``
instrumented by surrogate proxies ``Z1``. A pre-period IV fit imputes the
counterfactual (PI); surrogate stages refine the time-varying effect
(PIS, PIPost). Every method reports a GMM sandwich variance with a HAC
(Bartlett) middle, validated value-for-value against the authors'
reference code.

Layout:

    structures.py       : PROXIMALInputs, ProximalMethodFit, PROXIMALResults
    setup.py            : prepare_proximal_inputs (dataprep + proxy prep + cleaning)
    inference.py        : bartlett kernel + HAC long-run variance
    estimation.py       : estimate_pi / estimate_pi_surrogate / estimate_pi_surrogate_post
    orchestration.py    : run_proximal (drives the methods, builds fits)
    plotter.py          : trajectories + gap overlay across methods
"""

from .estimation import (
    estimate_pi,
    estimate_pi_surrogate,
    estimate_pi_surrogate_post,
)
from .inference import bartlett, hac
from .orchestration import run_proximal
from .plotter import plot_proximal
from .setup import prepare_proximal_inputs
from .structures import (
    PI,
    PIPOST,
    PIS,
    PROXIMALInputs,
    PROXIMALResults,
    ProximalMethodFit,
)

__all__ = [
    "PI",
    "PIPOST",
    "PIS",
    "PROXIMALInputs",
    "PROXIMALResults",
    "ProximalMethodFit",
    "bartlett",
    "estimate_pi",
    "estimate_pi_surrogate",
    "estimate_pi_surrogate_post",
    "hac",
    "plot_proximal",
    "prepare_proximal_inputs",
    "run_proximal",
]
