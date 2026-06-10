"""Spillover/interference conflict graph for LEXSCM treated-unit selection.

Implements the two exclusion criteria of Vives-i-Bastida (2022, "Synthetic
Experimental Design for a UBI pilot study"), which extend Abadie & Zhao (2021):

* **"No interference" (treatment criterion)** -- treated units may not belong to
  the same cluster (the paper: "the same province"). In Stage 1 this restricts
  the admissible treated ``m``-tuples to **independent sets** of the conflict
  graph.
* **"Exclusion restriction" (control criterion)** -- a treated unit's
  spillover-neighbours may not serve as its donors (the paper: controls "can not
  be within the same labour market ... as the treated units"). In Stage 2 this
  drops ``N(S)`` from the donor pool.

Both criteria read off **one** boolean conflict matrix ``A`` (``A[i, j]`` True iff
units ``i`` and ``j`` interfere), which this module builds from either a cluster
assignment, an adjacency/spillover matrix, or both. The matrix is **aligned to
the IndexSet** -- the single source of truth for unit identity (rows/columns are
in ``unit_index.labels`` order) -- so every downstream consumer indexes it the
same way as ``G`` and the candidate indices.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError


def _cluster_conflict(labels: np.ndarray, cluster_of: Mapping[Any, Any]) -> np.ndarray:
    """``A[i, j] = (cluster(i) == cluster(j))`` off-diagonal (clique per cluster).

    ``labels`` are the IndexSet labels in order; ``cluster_of`` maps each label to
    its cluster. A unit with a missing/``NaN`` cluster conflicts with no one.
    """
    J = len(labels)
    clusters = []
    for lab in labels:
        if lab not in cluster_of:
            raise MlsynthDataError(
                f"Unit {lab!r} has no cluster value in the cluster column."
            )
        clusters.append(cluster_of[lab])
    # Encode clusters as integers; NaN/None -> a unique sentinel that matches no one.
    codes = np.empty(J, dtype=object)
    for i, c in enumerate(clusters):
        codes[i] = None if (c is None or (isinstance(c, float) and np.isnan(c))) else c
    A = np.zeros((J, J), dtype=bool)
    for i in range(J):
        if codes[i] is None:
            continue
        for j in range(i + 1, J):
            if codes[j] is not None and codes[i] == codes[j]:
                A[i, j] = A[j, i] = True
    return A


def _adjacency_conflict(labels: np.ndarray, adjacency: Any, threshold: float) -> np.ndarray:
    """``A[i, j] = (adjacency[i, j] > threshold)`` off-diagonal, aligned to ``labels``.

    ``adjacency`` may be a square ``(J, J)`` array in IndexSet order, or a
    ``pd.DataFrame`` indexed/columned by unit labels (reindexed to ``labels``).
    The matrix is symmetrised (``A or A.T``) so direction does not matter.
    """
    J = len(labels)
    if isinstance(adjacency, pd.DataFrame):
        missing = [l for l in labels if l not in adjacency.index or l not in adjacency.columns]
        if missing:
            raise MlsynthDataError(
                f"Adjacency DataFrame is missing rows/columns for units: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        M = adjacency.reindex(index=labels, columns=labels).to_numpy(dtype=float)
    else:
        M = np.asarray(adjacency, dtype=float)
        if M.shape != (J, J):
            raise MlsynthConfigError(
                f"Adjacency matrix has shape {M.shape}, expected ({J}, {J}) to align "
                f"with the {J} units. Pass a pandas DataFrame indexed by unit id to "
                f"avoid relying on ordering."
            )
    if not np.all(np.isfinite(M)):
        raise MlsynthDataError("Adjacency matrix contains non-finite entries.")
    A = M > threshold
    A = A | A.T                       # symmetrise: spillover is mutual
    np.fill_diagonal(A, False)
    return A


def build_conflict_matrix(
    unit_index: Any,
    *,
    cluster_of: Optional[Mapping[Any, Any]] = None,
    adjacency: Optional[Any] = None,
    spillover_threshold: float = 0.0,
) -> Optional[np.ndarray]:
    """Build the ``(J, J)`` boolean conflict matrix aligned to ``unit_index``.

    Parameters
    ----------
    unit_index : IndexSet
        The source of truth for unit identity; ``unit_index.labels`` fixes the
        row/column order of the returned matrix.
    cluster_of : mapping label -> cluster, optional
        Cluster assignment (e.g. from ``cluster_col``). Two units conflict iff
        they share a (non-missing) cluster.
    adjacency : array or DataFrame, optional
        Spillover/adjacency matrix; two units conflict iff their entry exceeds
        ``spillover_threshold``.
    spillover_threshold : float
        Threshold for the adjacency mode.

    Returns
    -------
    np.ndarray of bool, shape (J, J), or None
        Symmetric, zero-diagonal conflict matrix. If both inputs are given the
        two graphs are combined with logical OR. Returns ``None`` when neither
        input is supplied (no spillover constraint).
    """
    labels = np.asarray(unit_index.labels)
    J = len(labels)
    A: Optional[np.ndarray] = None
    if cluster_of is not None:
        A = _cluster_conflict(labels, cluster_of)
    if adjacency is not None:
        Aa = _adjacency_conflict(labels, adjacency, spillover_threshold)
        A = Aa if A is None else (A | Aa)
    if A is None:
        return None
    np.fill_diagonal(A, False)
    return A


def neighbours(conflict: np.ndarray, idx) -> np.ndarray:
    """Indices that conflict with ANY unit in ``idx`` (the spillover set ``N(S)``).

    Excludes the members of ``idx`` themselves; returns a sorted int array.
    """
    idx = np.asarray(list(idx), dtype=int)
    if idx.size == 0:
        return np.empty(0, dtype=int)
    nbr = np.any(conflict[idx, :], axis=0)
    nbr[idx] = False
    return np.flatnonzero(nbr)


def is_independent(conflict: Optional[np.ndarray], idx) -> bool:
    """True iff no two members of ``idx`` share a conflict edge (an independent set)."""
    if conflict is None:
        return True
    idx = np.asarray(list(idx), dtype=int)
    if idx.size < 2:
        return True
    sub = conflict[np.ix_(idx, idx)]
    return not bool(np.any(np.triu(sub, k=1)))


def greedy_independent_set_size(conflict: Optional[np.ndarray], candidate_idx) -> int:
    """Size of a greedy minimum-degree independent set among ``candidate_idx``.

    A **lower bound** on the maximum independent set (greedy can undercount), so
    it is sound as a *sufficient* feasibility signal: if it reaches ``m`` a
    conflict-free ``m``-tuple certainly exists.
    """
    if conflict is None:
        return len(np.asarray(list(candidate_idx)))
    cand = np.asarray(list(candidate_idx), dtype=int)
    sub = conflict[np.ix_(cand, cand)]
    remaining = list(range(len(cand)))
    chosen = 0
    while remaining:
        deg = sub[np.ix_(remaining, remaining)].sum(axis=1)
        pick = remaining[int(np.argmin(deg))]
        chosen += 1
        # drop pick and all its neighbours within the remaining set
        remaining = [r for r in remaining if r != pick and not sub[pick, r]]
    return chosen


def max_independent_set_size_at_least(conflict: Optional[np.ndarray], candidate_idx, m: int) -> bool:
    """Cheap sufficient feasibility check: can a size-``m`` independent set exist
    among ``candidate_idx``? (Greedy lower bound; a ``False`` is only advisory.)
    """
    if conflict is None:
        return True
    cand = np.asarray(list(candidate_idx), dtype=int)
    if len(cand) < m:
        return False
    return greedy_independent_set_size(conflict, cand) >= m
