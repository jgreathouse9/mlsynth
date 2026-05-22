"""K-means clustering on FPC scores for RPCA-SC.

Implements Step 2 of Bayani (2021): apply Hartigan-Wong style
:math:`k`-means to the standardised FPC scores (treated unit included
in the panel), then select the donor pool as the other members of the
cluster containing the treated unit.

The number of clusters is either user-supplied or chosen by the
silhouette coefficient (Rousseeuw 1987) over
:math:`k \\in [2, k_{\\max}]`, matching Bayani's recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ....exceptions import MlsynthEstimationError


@dataclass(frozen=True)
class FPCACluster:
    """Output of :func:`assign_clusters`."""

    labels: np.ndarray          # (n_units,) cluster id per row
    treated_cluster: int        # cluster the treated unit belongs to
    donor_indices: np.ndarray   # rows (excluding treated) in same cluster
    k: int                      # number of clusters used


def assign_clusters(
    scores: np.ndarray,
    treated_row: int,
    k_clusters: Optional[int] = None,
    k_max: int = 8,
    random_state: int = 0,
) -> FPCACluster:
    """Cluster units by their FPC scores and pick the treated unit's cluster.

    Parameters
    ----------
    scores : np.ndarray
        Standardised FPC scores, shape ``(n_units, rank)``. The treated
        unit is row ``treated_row``.
    treated_row : int
        Index of the treated unit's row in ``scores``.
    k_clusters : int, optional
        Number of clusters. If ``None``, the silhouette coefficient
        selects :math:`k \\in [2, k_{\\max}]`.
    k_max : int
        Upper bound for the silhouette search.
    random_state : int
        Seed for :class:`sklearn.cluster.KMeans` initialisation.

    Returns
    -------
    FPCACluster
        Cluster labels, the treated unit's cluster id, the indices of
        the donor pool (cluster members minus the treated row), and
        the number of clusters used.
    """
    if scores.ndim != 2:
        raise MlsynthEstimationError("scores must be 2D (n_units, rank).")
    n_units = scores.shape[0]
    if not (0 <= treated_row < n_units):
        raise MlsynthEstimationError(
            f"treated_row {treated_row} out of range for n_units={n_units}."
        )

    # Degenerate cases: no usable features → everyone in one cluster.
    if scores.shape[1] == 0 or n_units <= 1:
        labels = np.zeros(n_units, dtype=int)
        donor_mask = np.arange(n_units) != treated_row
        return FPCACluster(
            labels=labels,
            treated_cluster=0,
            donor_indices=np.where(donor_mask)[0],
            k=1,
        )

    if k_clusters is None:
        upper = max(2, min(k_max, n_units - 1))
        if upper < 2:
            k_chosen = 1
        else:
            best_k, best_score = 2, -np.inf
            for k in range(2, upper + 1):
                km = KMeans(
                    n_clusters=k, n_init=10, random_state=random_state,
                    init="k-means++",
                ).fit(scores)
                if len(np.unique(km.labels_)) < 2:
                    continue
                sc = silhouette_score(scores, km.labels_)
                if sc > best_score:
                    best_k, best_score = k, sc
            k_chosen = best_k
    else:
        k_chosen = int(k_clusters)
        if k_chosen < 1:
            raise MlsynthEstimationError("k_clusters must be >= 1.")
        k_chosen = min(k_chosen, n_units)

    if k_chosen == 1:
        labels = np.zeros(n_units, dtype=int)
    else:
        labels = KMeans(
            n_clusters=k_chosen, n_init=10, random_state=random_state,
            init="k-means++",
        ).fit(scores).labels_.astype(int)

    treated_cluster = int(labels[treated_row])
    donor_idx = np.where((labels == treated_cluster) & (np.arange(n_units) != treated_row))[0]
    return FPCACluster(
        labels=labels,
        treated_cluster=treated_cluster,
        donor_indices=donor_idx,
        k=k_chosen,
    )
