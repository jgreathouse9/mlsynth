"""Standalone RPCA-SC reference (Bayani 2021), for cross-validating CLUSTERSC.

A faithful, self-contained reproduction of Mani Bayani's dissertation code
(*Robust PCA Synthetic Control*): the donor pool is denoised by Robust PCA via
Principal Component Pursuit (Candes, Li, Ma & Wright 2011) into a low-rank ``L``
plus a sparse ``S``, and the treated unit's pre-period path is regressed onto the
low-rank donor components with non-negative weights; the counterfactual is the
weighted combination of the denoised donor trajectories.

This mirrors the author's ``RPCA_2.py`` (the PCP ADMM and the non-negative
least-squares fit) run on the West-Germany donor cluster his ``FPCA.R`` selects
(functional-PCA + k-means over the pre-1990 GDP curves). The cluster membership
is recorded here so the reference needs neither R nor the functional-PCA step;
mlsynth's CLUSTERSC reaches the same cluster through its own SVD-based clustering.

Deterministic (PCP is a fixed-point ADMM; the NNLS is convex), so the reference
is recomputed live rather than pinned to a captured constant.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

# West-Germany cluster from Bayani's FPCA.R (RPCA_2.py's own ``countries`` list),
# West Germany plus the eleven donors the functional-PCA k-means groups with it.
WEST_GERMANY_CLUSTER: List[str] = [
    "UK", "Austria", "Belgium", "Denmark", "France", "West Germany",
    "Italy", "Netherlands", "Norway", "Japan", "Australia", "New Zealand",
]
_TREATMENT_YEAR = 1990


def _shrink(matrix: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0.0)


def _singular_value_threshold(matrix: np.ndarray, tau: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ np.diag(_shrink(s, tau)) @ vt


def robust_pca(matrix: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """PCP robust-PCA: split ``matrix`` into low-rank ``L`` and sparse ``S``.

    Uses the author's self-tuned penalties (``mu`` from the mean absolute entry,
    ``lambda = 1/sqrt(max(n1, n2))``) and stopping rule.
    """
    n1, n2 = matrix.shape
    mu = n1 * n2 / (4.0 * np.sum(np.abs(matrix.reshape(-1))))
    lambd = 1.0 / np.sqrt(max(n1, n2))
    thresh = 1e-7 * np.linalg.norm(matrix)
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    dual = np.zeros_like(matrix)
    count = 0
    while np.linalg.norm(matrix - low_rank - sparse) > thresh and count < max_iter:
        low_rank = _singular_value_threshold(matrix - sparse + dual / mu, 1.0 / mu)
        sparse = _shrink(matrix - low_rank + dual / mu, lambd / mu)
        dual = dual + mu * (matrix - low_rank - sparse)
        count += 1
    return low_rank, sparse


def rpca_sc_west_germany(df: pd.DataFrame) -> Dict[str, object]:
    """Run Bayani's RPCA-SC for West Germany on a long GDP panel.

    Parameters
    ----------
    df : DataFrame
        Long panel with columns ``country``, ``year``, ``gdp`` covering the
        West-Germany cluster (the reference restricts to it).

    Returns
    -------
    dict with ``weights`` (donor -> weight), ``counterfactual`` (per-year path),
    ``years``, ``pre_rmse`` and ``att`` (mean post-treatment gap).
    """
    years = np.sort(df["year"].unique())
    wide = (df[df["country"].isin(WEST_GERMANY_CLUSTER)]
            .pivot(index="country", columns="year", values="gdp")
            .loc[WEST_GERMANY_CLUSTER])                       # fixed cluster order
    outcomes = wide.to_numpy(float)                           # 12 x T, WG at row 5
    treated_row = WEST_GERMANY_CLUSTER.index("West Germany")
    n_pre = int(np.sum(years < _TREATMENT_YEAR))

    donor_matrix = np.delete(outcomes, treated_row, axis=0)   # 11 donors x T
    low_rank, _ = robust_pca(donor_matrix)
    low_rank_pre = low_rank[:, :n_pre]
    treated_pre = outcomes[treated_row, :n_pre]

    weights = cp.Variable(donor_matrix.shape[0])
    cp.Problem(cp.Minimize(cp.sum_squares(treated_pre - weights @ low_rank_pre)),
               [weights >= 0]).solve()
    w = np.asarray(weights.value, dtype=float).flatten()
    counterfactual = low_rank.T @ w                           # denoised donors @ weights

    donors = [c for c in WEST_GERMANY_CLUSTER if c != "West Germany"]
    treated = outcomes[treated_row]
    pre_rmse = float(np.sqrt(np.mean((treated[:n_pre] - counterfactual[:n_pre]) ** 2)))
    att = float(np.mean((treated - counterfactual)[n_pre:]))
    return {
        "weights": {d: float(wi) for d, wi in zip(donors, w)},
        "counterfactual": counterfactual,
        "years": years,
        "pre_rmse": pre_rmse,
        "att": att,
    }
