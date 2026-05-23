"""PCR-SC subpackage.

Paper: Rho, S., Tang, A., Bergam, N., Cummings, R., & Misra, V. (2025).
*ClusterSC: Advancing Synthetic Control with Donor Selection*. arXiv
preprint arXiv:2503.21629.

Module layout:

* :mod:`.hsvt` -- rank selection + Hard Singular Value Thresholding
  (Algorithm 2 Step 2).
* :mod:`.clustering` -- Algorithm 3 (SVD-feature k-means) and the
  target-matching step of Algorithm 4.
* :mod:`.frequentist` -- OLS weight solver (Algorithm 2 Step 3) with
  optional elastic-net regularisation (Appendix E).
* :mod:`.bayesian` -- Bayesian posterior weight solver (mlsynth
  extension; Bayani 2022 Ch. 1).
* :mod:`.convex` -- simplex-constrained weight solver (mlsynth extension
  combining HSVT denoising with Abadie-style convex weights).
* :mod:`.pipeline` -- public dispatcher :func:`run_pcr` composing the
  above into Algorithms 2/4.
"""

from __future__ import annotations

from .bayesian import solve_bayesian
from .clustering import ClusterPartition, assign_target, cluster_donors
from .convex import solve_simplex
from .frequentist import solve_ols
from .hsvt import hsvt, select_rank
from .inference import ShenInference, shen_inference
from .pipeline import run_pcr

__all__ = [
    "ClusterPartition",
    "ShenInference",
    "assign_target",
    "cluster_donors",
    "hsvt",
    "run_pcr",
    "select_rank",
    "shen_inference",
    "solve_bayesian",
    "solve_ols",
    "solve_simplex",
]
