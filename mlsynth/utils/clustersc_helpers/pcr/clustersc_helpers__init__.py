"""Helper utilities for the Cluster-based Synthetic Control (CLUSTERSC) estimator.

Implements the PCR-RSC + RPCA-SC family. Primary references:

* Rho, S., Tang, A., Bergam, N., Cummings, R., & Misra, V. (2025).
  "ClusterSC: Advancing Synthetic Control with Donor Selection."
  arXiv:2503.21629. (Algorithms 2-4 implemented in
  :mod:`.pcr`.)
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

from .pcr import (
    ClusterPartition,
    assign_target,
    cluster_donors,
    hsvt,
    run_pcr,
    select_rank,
    solve_bayesian,
    solve_ols,
    solve_simplex,
)
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
    "ClusterPartition",
    "MethodFit",
    "assign_target",
    "cluster_donors",
    "hsvt",
    "plot_clustersc",
    "prepare_clustersc_inputs",
    "run_pcr",
    "run_rpca",
    "select_rank",
    "solve_bayesian",
    "solve_ols",
    "solve_simplex",
]
