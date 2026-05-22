"""Helper utilities for the Cluster-based Synthetic Control (CLUSTERSC) estimator.

Implements the PCR-RSC + RPCA-SC family:

* Amjad, M., Shah, D., & Shen, D. (2018). "Robust synthetic control."
  *Journal of Machine Learning Research*, 19(22), 1-51.
* Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). "On Robustness of
  Principal Component Regression." *JASA*, 116(536), 1731-1745.
* Bayani, M. (2022). "Essays on Machine Learning Methods in Economics,"
  Chapter 1, CUNY Academic Works (Bayesian RSC).
* Candes, E., Li, X., Ma, Y., & Wright, J. (2011). "Robust principal
  component analysis?" *Journal of the ACM*, 58(3), 11. (PCP)
* Wang, Z., Li, X. P., So, H. C., & Liu, Z. (2023). "Robust PCA via
  non-convex half-quadratic regularization." *Signal Processing*,
  204, 108816. (HQF)
"""

from .pcr import run_pcr
from .plotter import plot_clustersc
from .rpca import run_rpca
from .setup import prepare_clustersc_inputs
from .structures import (
    CLUSTERSCInference,
    CLUSTERSCInputs,
    CLUSTERSCResults,
    MethodFit,
)

__all__ = [
    "CLUSTERSCInference",
    "CLUSTERSCInputs",
    "CLUSTERSCResults",
    "MethodFit",
    "plot_clustersc",
    "prepare_clustersc_inputs",
    "run_pcr",
    "run_rpca",
]
