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
the anchor submatrix is found per entry instead of assuming a fixed
treated/donor block. The reference implementation
(github.com/deshen24/syntheticNN) uses a NetworkX maximum-biclique
search to find anchors; this implementation uses a dependency-free greedy
search for the largest fully observed submatrix.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..pcr import hsvt, pcr_weights, spectral_rank, usvt_rank

_EPS = 1e-12


def _spectral_rank(s: np.ndarray, energy: float = 0.95) -> int:
    """Smallest rank whose singular values capture ``energy`` of the spectrum.

    Thin wrapper over the shared kernel (:func:`mlsynth.utils.pcr.spectral_rank`).
    """
    return spectral_rank(s, energy)


def _universal_rank(s: np.ndarray, shape: Tuple[int, int]) -> int:
    """Donoho & Gavish (2014) optimal hard-threshold rank for square-ish noise.

    Thin wrapper over the shared kernel (:func:`mlsynth.utils.pcr.usvt_rank`),
    evaluated at the canonical ``min/max`` aspect ratio.
    """
    m, n = shape
    if min(m, n) == 0:
        return 0
    return usvt_rank(s, min(m, n) / max(m, n))


def _span_error(S: np.ndarray, q: np.ndarray, beta: np.ndarray) -> float:
    """Linear span statistic: the normalized reconstruction error of ``q``.

    :math:`\\|S^\\top \\beta - q\\|^2 / \\|q\\|^2`, the empirical counterpart of
    the paper's Assumption 3 (the target row's factor lies in the span of the
    anchor rows'). Matches ``_train_error`` in ``deshen24/syntheticNN``. A
    target row that is exactly a linear combination of the anchor rows scores
    zero; the reference calls the entry feasible at or below 0.1.

    ``q = 0`` leaves the ratio undefined; the numerator is then zero too, so 0
    is returned -- a zero target row is reconstructed exactly by any weights.
    """
    denom = float(q @ q)
    if denom <= _EPS:
        return 0.0
    resid = S.T @ beta - q
    return float((resid @ resid) / denom)


def _subspace_stat(Vt_r: np.ndarray, x: np.ndarray) -> float:
    """Subspace inclusion statistic for the target column.

    :math:`\\|(I - V^\\top V) x\\|^2 / \\|x\\|^2` for :math:`V` the retained
    right singular directions of :math:`S^\\top`, i.e. the share of the target
    column's energy lying outside the subspace the weights were fit on. This is
    the empirical counterpart of Assumption 7 (the target column's factor lies
    in the span of the anchor columns'), and SI's Assumption 8 (post-treatment
    generalizability) is the same condition stated for a fixed column set.
    Matches ``_subspace_inclusion`` in ``deshen24/syntheticNN``.

    ``x = 0`` leaves the ratio undefined; the numerator is then zero too, so 0
    is returned -- the zero vector lies in every subspace.
    """
    denom = float(x @ x)
    if denom <= _EPS:
        return 0.0
    resid = x - Vt_r.T @ (Vt_r @ x)
    return float((resid @ resid) / denom)


def _pcr(
    S: np.ndarray, q: np.ndarray, x: np.ndarray,
    *, max_rank: Optional[int], spectral_energy: float, universal: bool,
    diagnostics: bool = False,
) -> Tuple[float, np.ndarray, float, float]:
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
    diagnostics : bool
        Also compute the two span statistics. Off by default: the subspace
        statistic needs the retained right singular vectors, a second
        decomposition of the anchor block that the jackknife would pay for on
        every re-fit without using the result.

    Returns
    -------
    prediction : float
        Imputed value ``<x, beta>``.
    beta : np.ndarray
        Regression weights over the anchor rows, shape ``(|AR|,)``.
    span_error : float
        Linear span statistic (:func:`_span_error`), ``nan`` when
        ``diagnostics`` is False.
    subspace_stat : float
        Subspace inclusion statistic (:func:`_subspace_stat`), ``nan`` when
        ``diagnostics`` is False.
    """
    sv = np.linalg.svd(S, compute_uv=False)
    if max_rank is not None:
        r = min(max_rank, sv.size)
    elif universal:
        r = _universal_rank(sv, S.shape)
    else:
        r = _spectral_rank(sv, spectral_energy)
    r = max(r, 1)
    # beta = U_r diag(1/sv_r) V_r^T q  (weights over anchor rows): the shared
    # PCR kernel applied to S^T regresses q onto the anchor-row subspace.
    beta = pcr_weights(S.T, q, r)
    prediction = float(x @ beta)
    if not diagnostics:
        return prediction, beta, float("nan"), float("nan")
    _, _, _, Vt_r = hsvt(S.T, r)
    return prediction, beta, _span_error(S, q, beta), _subspace_stat(Vt_r, x)


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
        # Drop whichever single row/col is emptiest, comparing *shares* and not
        # counts. A row's missing count is out of len(cols) and a column's is
        # out of len(rows), so comparing the raw counts makes the longer side
        # always look worse: on a mask with more rows than columns the search
        # strips every column and returns nothing with the rows untouched. On
        # scattered MNAR masks that lost the cross on 55% of missing entries,
        # where the exact maximum-biclique search never failed. Normalizing
        # takes that to 0% and raises the mean block min-dimension from 2.86
        # to 5.36 (exact: 5.94). Ties break toward dropping a row.
        if row_missing.max() / len(cols) >= col_missing.max() / len(rows):
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
    random_state: int = 0, return_diagnostics: bool = False,
):
    """Impute entry ``(i, j)`` of ``X`` via SNN.

    Returns ``(value, feasible)``, or ``(value, feasible, diagnostics)`` when
    ``return_diagnostics`` is set. The diagnostics dict carries the two span
    statistics -- ``span_error`` (:func:`_span_error`) and ``subspace_stat``
    (:func:`_subspace_stat`) -- averaged over the synthetic neighbours, and
    ``nan`` for both when no anchor cross exists.

    ``feasible`` reports whether an estimate could be formed at all (an anchor
    cross existed and the value was finite). It does not fold in the span
    statistics: those are reported for the caller to act on, and an entry
    failing them is still imputed.
    """
    AR, AC = _find_anchors(mask, i, j)
    if AR.size == 0 or AC.size == 0:
        if return_diagnostics:
            return np.nan, False, {"span_error": float("nan"),
                                   "subspace_stat": float("nan")}
        return np.nan, False

    # Split anchor rows into n_neighbors disjoint groups and average.
    rng = np.random.default_rng(random_state)
    order = rng.permutation(AR.size)
    n_groups = max(1, min(n_neighbors, AR.size))
    groups = np.array_split(order, n_groups)

    preds, spans, subs = [], [], []
    for g in groups:
        ar = AR[g]
        if ar.size == 0:
            continue
        S = X[np.ix_(ar, AC)]
        q = X[i, AC]
        x = X[ar, j]
        pred, _, span, sub = _pcr(
            S, q, x, max_rank=max_rank,
            spectral_energy=spectral_energy, universal=universal,
            diagnostics=return_diagnostics,
        )
        if np.isfinite(pred):
            preds.append(pred)
            spans.append(span)
            subs.append(sub)
    if not preds:
        if return_diagnostics:
            return np.nan, False, {"span_error": float("nan"),
                                   "subspace_stat": float("nan")}
        return np.nan, False
    value = float(np.mean(preds))
    if return_diagnostics:
        # One statistic per synthetic neighbour; the entry's reading is their
        # mean, matching how the neighbours' predictions are combined.
        return value, True, {"span_error": float(np.mean(spans)),
                             "subspace_stat": float(np.mean(subs))}
    return value, True


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
        _, beta, _, _ = _pcr(S, q, X[ar, missing_cols[0]], max_rank=max_rank,
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
    return_diagnostics: bool = False,
):
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
    return_diagnostics : bool
        Also return the two span statistics per imputed entry.

    Returns
    -------
    completed : np.ndarray
        Matrix with missing entries imputed (NaN where infeasible).
    feasible : np.ndarray
        Boolean mask, ``True`` where an imputation was produced.
    span_error, subspace_stat : np.ndarray
        Only when ``return_diagnostics`` is set: the linear span and subspace
        inclusion statistics at each imputed entry, ``NaN`` at observed entries
        and at entries no anchor cross reached.
    """
    X = np.array(X, dtype=float)
    mask = (~np.isnan(X)).astype(float)
    completed = X.copy()
    feasible = mask.astype(bool).copy()
    span_error = np.full(X.shape, np.nan)
    subspace_stat = np.full(X.shape, np.nan)

    missing = np.argwhere(mask == 0)
    for i, j in missing:
        out = snn_predict(
            X, mask, int(i), int(j), n_neighbors=n_neighbors,
            max_rank=max_rank, spectral_energy=spectral_energy,
            universal=universal, random_state=random_state,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            val, ok, diag = out
            span_error[i, j] = diag["span_error"]
            subspace_stat[i, j] = diag["subspace_stat"]
        else:
            val, ok = out
        if ok and np.isfinite(val):
            if min_value is not None:
                val = max(val, min_value)
            if max_value is not None:
                val = min(val, max_value)
            completed[i, j] = val
            feasible[i, j] = True
        else:
            feasible[i, j] = False
    if return_diagnostics:
        return completed, feasible, span_error, subspace_stat
    return completed, feasible
