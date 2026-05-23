"""RPCA-SC subpackage.

Primary paper: Bayani, M. (2021). *Robust PCA Synthetic Control.*
arXiv:2108.12542. (Also Chapter 1 of Bayani 2022, CUNY dissertation.)

Module layout (one module per stage of the paper's Algorithm 4):

* :mod:`.fpca` -- Step 1: Functional Principal Component Analysis
  (Li, Wang & Carroll 2010) on pre-period trajectories.
* :mod:`.clustering` -- Step 2: silhouette-driven :math:`k`-means
  (Hartigan & Wong 1979; Rousseeuw 1987) on the FPC scores; donor
  pool = same-cluster members of the treated unit.
* :mod:`.pcp` -- Step 3 variant: Principal Component Pursuit via ADMM
  (Candes, Li, Ma & Wright 2011).
* :mod:`.hqf` -- Step 3 variant: half-quadratic non-convex Robust PCA
  (Wang, Li, So & Liu 2023).
* :mod:`.weights` -- Step 4: non-negative least squares.
* :mod:`.pipeline` -- public dispatcher :func:`run_rpca` composing
  Steps 1-5.
"""

from __future__ import annotations

from .clustering import FPCACluster, assign_clusters
from .fpca import FPCAFeatures, compute_fpca_features
from .hqf import HQFResult, hqf_decompose
from .inference import CFTInference, cft_prediction_intervals
from .pcp import PCPResult, pcp_decompose
from .pipeline import run_rpca
from .tuning import CVResult, cv_hqf_rank, cv_pcp_lambda
from .weights import solve_nnls

__all__ = [
    "CFTInference",
    "CVResult",
    "FPCACluster",
    "FPCAFeatures",
    "HQFResult",
    "PCPResult",
    "assign_clusters",
    "cft_prediction_intervals",
    "compute_fpca_features",
    "cv_hqf_rank",
    "cv_pcp_lambda",
    "hqf_decompose",
    "pcp_decompose",
    "run_rpca",
    "solve_nnls",
]
