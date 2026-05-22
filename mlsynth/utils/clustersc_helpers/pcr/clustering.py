"""Donor clustering for ClusterSC (Rho et al. 2025, Algorithm 3 & 4).

Implements the two paper-aligned steps:

* :func:`cluster_donors` -- Algorithm 3. SVD-truncate the (transposed)
  donor matrix at rank :math:`r`, build features
  :math:`\\widetilde{U} = U \\Sigma_r`, and run :math:`k`-means on
  the rows of :math:`\\widetilde{U}`. The number of clusters is either
  user-supplied or selected by the silhouette coefficient
  (Amjad, Shah, Shen 2018).
* :func:`assign_target` -- Algorithm 4 Step 2. Embed the treated unit
  into the right-singular-vector basis via :math:`\\tilde{u} = V_r^\\top x_0^-`
  and assign it to the nearest cluster centroid in
  :math:`\\widetilde{U}`-space.

Convention: ``X_pre`` here is shape ``(J, T0)`` -- one row per donor unit,
one column per pre-period -- to match the paper's notation
(:math:`X \\in \\mathbb{R}^{n \\times T}`). The orchestrator transposes
as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ....exceptions import MlsynthEstimationError


@dataclass(frozen=True)
class ClusterPartition:
    """Output of the paper's Algorithm 3."""

    labels: np.ndarray            # shape (J,), cluster id per donor
    centers: np.ndarray           # shape (k, r), centroid in Ũ-space
    V_r: np.ndarray               # shape (r, T0), truncated right singular vectors
    k: int                        # number of clusters chosen
    rank: int                     # truncation rank used to build features


def cluster_donors(
    donor_outcomes_pre: np.ndarray,
    rank: int,
    k_clusters: Optional[int] = None,
    k_max: int = 8,
    random_state: int = 0,
) -> ClusterPartition:
    """Run Algorithm 3 of Rho et al. (2025).

    Parameters
    ----------
    donor_outcomes_pre : np.ndarray
        Donor outcomes in the pre-period, shape ``(J, T0)`` (donors on rows).
    rank : int
        Truncation rank :math:`r` for the SVD feature map.
    k_clusters : int, optional
        Number of clusters. If ``None``, the silhouette coefficient picks
        :math:`k \\in [2, \\min(k_{\\max}, J-1)]`.
    k_max : int
        Upper bound for the silhouette search.
    random_state : int
        Seed forwarded to :class:`sklearn.cluster.KMeans` for reproducibility.

    Returns
    -------
    ClusterPartition
        Cluster labels, centroids in :math:`\\widetilde{U}`-space, and the
        rank-truncated right singular matrix :math:`V_r` needed for the
        target-assignment step.
    """
    if donor_outcomes_pre.ndim != 2:
        raise MlsynthEstimationError(
            "donor_outcomes_pre must be 2D (J, T0)."
        )
    J = donor_outcomes_pre.shape[0]
    if J < 2:
        raise MlsynthEstimationError(
            "Clustering requires at least two donors."
        )

    U, s, Vt = np.linalg.svd(donor_outcomes_pre, full_matrices=False)
    r = max(1, min(int(rank), s.size))
    U_r = U[:, :r]
    s_r = s[:r]
    Vt_r = Vt[:r, :]

    # Algorithm 3 step 1 (last line): Ũ = U Σ_r — features for k-means.
    U_tilde = U_r * s_r  # broadcasting scales each column by σ_i

    # Silhouette-driven k unless caller fixes it.
    if k_clusters is None:
        upper = max(2, min(k_max, J - 1))
        if upper < 2:
            k_chosen = 1
        else:
            best_k, best_score = 2, -np.inf
            for k in range(2, upper + 1):
                km = KMeans(
                    n_clusters=k, n_init=10, random_state=random_state,
                ).fit(U_tilde)
                if len(np.unique(km.labels_)) < 2:
                    continue
                sc = silhouette_score(U_tilde, km.labels_)
                if sc > best_score:
                    best_k, best_score = k, sc
            k_chosen = best_k
    else:
        k_chosen = int(k_clusters)
        if k_chosen < 1:
            raise MlsynthEstimationError("k_clusters must be >= 1.")
        k_chosen = min(k_chosen, J)

    if k_chosen == 1:
        labels = np.zeros(J, dtype=int)
        centers = U_tilde.mean(axis=0, keepdims=True)
    else:
        km = KMeans(
            n_clusters=k_chosen, n_init=10, random_state=random_state,
        ).fit(U_tilde)
        labels = km.labels_.astype(int)
        centers = km.cluster_centers_

    return ClusterPartition(
        labels=labels,
        centers=centers,
        V_r=Vt_r,
        k=k_chosen,
        rank=r,
    )


def assign_target(
    target_outcome_pre: np.ndarray,
    partition: ClusterPartition,
) -> Tuple[int, np.ndarray]:
    """Algorithm 4 Step 2: assign the target to its matching cluster.

    Computes :math:`\\tilde{u} = V_r^\\top x_0^-` and picks the centroid
    minimising :math:`\\| c_\\ell - \\tilde{u} \\|_2`.

    Returns
    -------
    cluster_id : int
        Index of the nearest cluster.
    target_embedding : np.ndarray
        :math:`\\tilde{u}`, shape ``(r,)``.
    """
    x = np.asarray(target_outcome_pre, dtype=float).flatten()
    if x.size != partition.V_r.shape[1]:
        raise MlsynthEstimationError(
            f"target_outcome_pre length {x.size} does not match T0="
            f"{partition.V_r.shape[1]} used for clustering."
        )
    u_tilde = partition.V_r @ x  # (r,)
    diffs = partition.centers - u_tilde[np.newaxis, :]
    cluster_id = int(np.argmin(np.linalg.norm(diffs, axis=1)))
    return cluster_id, u_tilde
