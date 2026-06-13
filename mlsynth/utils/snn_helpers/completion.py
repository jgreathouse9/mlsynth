"""Core matrix-completion engine for Synthetic Nearest Neighbors (SNN).

Agarwal, A., Dahleh, M., Shah, D. & Shen, D. (2021). *"Causal Matrix
Completion."* arXiv:2109.15154.

SNN imputes a missing entry :math:`(i, j)` of a partially observed matrix
by (1) finding **anchor rows and columns** -- a fully observed submatrix
:math:`S` whose rows are observed in column :math:`j` and whose columns
are observed in row :math:`i` -- and (2) running **principal component
regression** (PCR): truncate the SVD of :math:`S`, regress row
:math:`i`'s anchor-column values on :math:`S` to learn weights
:math:`\\beta`, and apply them to column :math:`j`'s anchor-row values
(paper Algorithm 1).

It generalises the Synthetic Interventions / synthetic-control PCR
machinery to arbitrary "missing not at random" (MNAR) patterns, because
the anchor submatrix is found per entry rather than assuming a fixed
treated/donor block. The reference implementation
(github.com/deshen24/syntheticNN) uses a NetworkX maximum-biclique
search to find anchors; this implementation uses a dependency-free greedy
search for the largest fully observed submatrix.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_EPS = 1e-12


def _spectral_rank(s: np.ndarray, energy: float = 0.95) -> int:
    """Smallest rank whose singular values capture ``energy`` of the spectrum."""
    if s.size == 0:
        return 0
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(cum, energy) + 1)


def _universal_rank(s: np.ndarray, shape: Tuple[int, int]) -> int:
    """Donoho & Gavish (2014) optimal hard-threshold rank for square-ish noise."""
    m, n = shape
    if min(m, n) == 0:
        return 0
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    thresh = omega * np.median(s) if s.size else 0.0
    r = int((s > thresh).sum())
    return max(r, 1)


def _pcr(
    S: np.ndarray, q: np.ndarray, x: np.ndarray,
    *, max_rank: Optional[int], spectral_energy: float, universal: bool,
) -> Tuple[float, np.ndarray, float]:
    """Principal component regression for one synthetic neighbour.

    Parameters
    ----------
    S : np.ndarray
        Fully observed anchor submatrix, shape ``(|AR|, |AC|)``.
    q : np.ndarray
        Row ``i``'s values on the anchor columns, shape ``(|AC|,)``.
    x : np.ndarray
        Column ``j``'s values on the anchor rows, shape ``(|AR|,)``.
    max_rank, spectral_energy, universal :
        Rank-selection controls.

    Returns
    -------
    prediction : float
        Imputed value ``<x, beta>``.
    beta : np.ndarray
        Regression weights over the anchor rows, shape ``(|AR|,)``.
    train_error : float
        Mean squared reconstruction error of ``q`` on the anchor columns.
    """
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    if max_rank is not None:
        r = min(max_rank, sv.size)
    elif universal:
        r = _universal_rank(sv, S.shape)
    else:
        r = _spectral_rank(sv, spectral_energy)
    r = max(r, 1)
    Ur, svr, Vtr = U[:, :r], sv[:r], Vt[:r]
    # beta = U_r diag(1/sv_r) V_r^T q  (weights over anchor rows)
    beta = Ur @ ((Vtr @ q) / np.maximum(svr, _EPS))
    prediction = float(x @ beta)
    train_error = float(np.mean((S.T @ beta - q) ** 2))
    return prediction, beta, train_error


def _find_anchors(
    mask: np.ndarray, i: int, j: int, *, min_anchor: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy search for a large fully observed submatrix (anchor block).

    Anchor rows are drawn from ``NR(j) = {a : mask[a, j]}`` and anchor
    columns from ``NC(i) = {b : mask[i, b]}``; the returned block
    ``AR x AC`` is fully observed. The greedy rule repeatedly drops the
    row or column with the most missing entries until the block is
    complete, favouring a large, roughly square submatrix.
    """
    rows = np.where(mask[:, j] > 0)[0]
    rows = rows[rows != i]
    cols = np.where(mask[i, :] > 0)[0]
    cols = cols[cols != j]
    if rows.size == 0 or cols.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    rows = list(rows)
    cols = list(cols)
    while rows and cols:
        block = mask[np.ix_(rows, cols)]
        if block.all():
            break
        row_missing = (block == 0).sum(axis=1)      # per-row missing count
        col_missing = (block == 0).sum(axis=0)      # per-col missing count
        # Drop whichever single row/col removes the most missing cells,
        # breaking ties toward keeping the block square.
        if row_missing.max() >= col_missing.max():
            rows.pop(int(np.argmax(row_missing)))
        else:
            cols.pop(int(np.argmax(col_missing)))
    if len(rows) < min_anchor or len(cols) < min_anchor:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def snn_predict(
    X: np.ndarray, mask: np.ndarray, i: int, j: int,
    *, n_neighbors: int = 1, max_rank: Optional[int] = None,
    spectral_energy: float = 0.95, universal: bool = False,
    random_state: int = 0,
) -> Tuple[float, bool]:
    """Impute entry ``(i, j)`` of ``X`` via SNN. Returns (value, feasible)."""
    AR, AC = _find_anchors(mask, i, j)
    if AR.size == 0 or AC.size == 0:
        return np.nan, False

    # Split anchor rows into n_neighbors disjoint groups and average.
    rng = np.random.default_rng(random_state)
    order = rng.permutation(AR.size)
    n_groups = max(1, min(n_neighbors, AR.size))
    groups = np.array_split(order, n_groups)

    preds = []
    for g in groups:
        ar = AR[g]
        if ar.size == 0:
            continue
        S = X[np.ix_(ar, AC)]
        q = X[i, AC]
        x = X[ar, j]
        pred, _, _ = _pcr(
            S, q, x, max_rank=max_rank,
            spectral_energy=spectral_energy, universal=universal,
        )
        if np.isfinite(pred):
            preds.append(pred)
    if not preds:
        return np.nan, False
    return float(np.mean(preds)), True


def snn_donor_weights(
    X: np.ndarray, mask: np.ndarray, i: int,
    *, n_neighbors: int = 1, max_rank: Optional[int] = None,
    spectral_energy: float = 0.95, universal: bool = False,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Effective PCR donor weights for treated unit ``i``.

    For a treated unit, every missing (post-treatment) cell shares the same
    anchor rows (the donor units) and anchor columns (the pre-periods), so
    a single weight vector :math:`\\beta` over the donors reproduces the
    imputed counterfactual: :math:`\\widehat Y_{it}(0) = \\sum_j \\beta_j
    Y_{jt}`. Returns ``(donor_indices, weights)``; the weights are the
    (unconstrained) PCR coefficients -- they need not be non-negative nor
    sum to one. Returns empty arrays if no anchor block exists.
    """
    # Find anchors using any missing column of row i (all share the same AR/AC).
    missing_cols = np.where(mask[i] == 0)[0]
    if missing_cols.size == 0:
        return np.array([], dtype=int), np.array([])
    AR, AC = _find_anchors(mask, i, int(missing_cols[0]))
    if AR.size == 0 or AC.size == 0:
        return np.array([], dtype=int), np.array([])

    rng = np.random.default_rng(random_state)
    order = rng.permutation(AR.size)
    n_groups = max(1, min(n_neighbors, AR.size))
    groups = np.array_split(order, n_groups)

    weights = np.zeros(AR.size)
    q = X[i, AC]
    for g in groups:
        if g.size == 0:
            continue
        ar = AR[g]
        S = X[np.ix_(ar, AC)]
        _, beta, _ = _pcr(S, q, X[ar, missing_cols[0]], max_rank=max_rank,
                          spectral_energy=spectral_energy, universal=universal)
        weights[g] = beta / n_groups   # averaged across neighbour groups
    return AR, weights


def snn_complete(
    X: np.ndarray,
    *,
    n_neighbors: int = 1,
    max_rank: Optional[int] = None,
    spectral_energy: float = 0.95,
    universal: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete a matrix with missing entries marked as ``NaN`` via SNN.

    Parameters
    ----------
    X : np.ndarray
        Partially observed matrix; missing entries are ``NaN``.
    n_neighbors : int
        Number of synthetic neighbours (anchor-row groups) to average.
    max_rank : int, optional
        Fixed PCR truncation rank; overrides the spectral/universal rule.
    spectral_energy : float
        Energy threshold for spectral rank selection (when ``max_rank`` and
        ``universal`` are unset).
    universal : bool
        Use the Donoho-Gavish universal hard threshold for the rank.
    min_value, max_value : float, optional
        Clip imputed values to this range.
    random_state : int
        Seed for the anchor-row splitting.

    Returns
    -------
    completed : np.ndarray
        Matrix with missing entries imputed (NaN where infeasible).
    feasible : np.ndarray
        Boolean mask, ``True`` where an imputation was produced.
    """
    X = np.array(X, dtype=float)
    mask = (~np.isnan(X)).astype(float)
    completed = X.copy()
    feasible = mask.astype(bool).copy()

    missing = np.argwhere(mask == 0)
    for i, j in missing:
        val, ok = snn_predict(
            X, mask, int(i), int(j), n_neighbors=n_neighbors,
            max_rank=max_rank, spectral_energy=spectral_energy,
            universal=universal, random_state=random_state,
        )
        if ok and np.isfinite(val):
            if min_value is not None:
                val = max(val, min_value)
            if max_value is not None:
                val = min(val, max_value)
            completed[i, j] = val
            feasible[i, j] = True
        else:
            feasible[i, j] = False
    return completed, feasible
