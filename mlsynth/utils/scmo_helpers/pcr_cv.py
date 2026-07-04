"""Rolling-origin cross-validation for the PCR metric weights (delta).

The concatenated PCR scheme (mRSC; [Amjad2019]_) weights each metric block of
the standardized matching matrix by ``sqrt(delta_k)`` before the HSVT + PCR
step. The paper gives no closed form for ``delta`` -- it is a cross-validation
hyperparameter, equal by default. This module chooses the auxiliary-metric
weight by *rolling-origin* (expanding-window, forward-chaining) cross-validation
that minimizes out-of-sample pre-treatment MSE of the primary outcome: for each
origin we fit the weights on the earliest pre-periods and score the next
``horizon`` held-out pre-period(s), all before treatment, so nothing leaks from
the post-period.

Everything is pure NumPy and deterministic (no RNG), operating on the arrays in
:class:`SCMOInputs`; the primary outcome is scored on the raw panel ``Y`` while
the weights are fit on the standardized matching matrix ``Z`` -- the same
train/predict split the point estimate uses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pcr.core import pcr_weights
from ..pcr.rank import select_rank

_DEFAULT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def metric_ids(predictor_labels) -> np.ndarray:
    """Metric (outcome) identity of each matching-matrix column.

    Labels are ``name`` (single period) or ``name@period`` (stacked periods);
    the metric is the ``name`` part, matching the spec's outcome column.
    """
    return np.asarray([str(l).split("@")[0] for l in predictor_labels])


def metric_order(ids: np.ndarray) -> List[str]:
    """Distinct metrics in order of first appearance (primary outcome first)."""
    seen: List[str] = []
    for m in ids:
        if m not in seen:
            seen.append(m)
    return seen


def column_scale(metric_weights: List[float], ids: np.ndarray,
                 order: List[str]) -> np.ndarray:
    """Per-column ``sqrt(delta)`` scale vector from per-metric weights."""
    w_by_metric = {m: float(metric_weights[i]) for i, m in enumerate(order)}
    return np.sqrt(np.asarray([w_by_metric[m] for m in ids], dtype=float))


def _pcr_predict(Z_scaled: np.ndarray, treated_idx: int, donor_idx: np.ndarray,
                 train_cols: np.ndarray, Y_donors_test: np.ndarray,
                 pcr_rank: Optional[int], pcr_cumvar: float) -> np.ndarray:
    """Fit PCR weights on the training columns, predict the primary outcome."""
    design = Z_scaled[np.ix_(donor_idx, train_cols)].T        # (P_train, J)
    target = Z_scaled[treated_idx, train_cols]                # (P_train,)
    if pcr_rank is not None:
        r = select_rank(design, method="fixed", r=pcr_rank)
    else:
        r = select_rank(design, method="cumvar", cumvar_threshold=pcr_cumvar)
    w = pcr_weights(design, target, r)
    return w @ Y_donors_test                                  # (horizon,)


def rolling_origin_pcr_cv(
    inputs,
    grid: Optional[List[float]] = None,
    horizon: int = 1,
    min_train: Optional[int] = None,
    pcr_rank: Optional[int] = None,
    pcr_cumvar: float = 0.95,
) -> Tuple[List[float], List[str], Dict[float, float]]:
    """Pick the auxiliary-metric weight by rolling-origin CV.

    Returns
    -------
    best_weights : list of float
        Per-metric weights in ``metric_order`` (primary fixed at 1.0, the
        auxiliaries share the selected weight).
    order : list of str
        The metric order (primary outcome first).
    mse_table : dict
        ``{candidate auxiliary weight: out-of-sample MSE}``.
    """
    ids = metric_ids(inputs.predictor_labels)
    order = metric_order(ids)
    K = len(order)

    grid = list(_DEFAULT_GRID if grid is None else grid)
    if 1.0 not in grid:                       # equal weighting is always a candidate
        grid = grid + [1.0]

    if K < 2:                                 # no auxiliary metric -> nothing to tune
        return [1.0] * K, order, {1.0: 0.0}

    # chronological pre-periods and their columns in Y
    col_period = inputs.col_period
    pre_labels = list(dict.fromkeys(col_period.tolist()))
    y_cols = [int(inputs.time_index.get_index([p])[0]) for p in pre_labels]
    chrono = np.argsort(y_cols)
    periods_sorted = [pre_labels[i] for i in chrono]
    ycols_sorted = [y_cols[i] for i in chrono]
    P = len(periods_sorted)

    if min_train is None:
        min_train = max(2, inputs.T0 // 2)
    origins = list(range(min_train, P - horizon + 1))
    if not origins:
        from ...exceptions import MlsynthEstimationError
        raise MlsynthEstimationError(
            f"Not enough pre-periods for rolling-origin CV: need > "
            f"min_train + horizon ({min_train} + {horizon}), have {P}.")

    Y = inputs.Y
    mse_table: Dict[float, float] = {}
    for g in grid:
        weights = [1.0] + [g] * (K - 1)                       # primary anchored
        scale = column_scale(weights, ids, order)
        Z_scaled = inputs.Z * scale[None, :]
        sse = 0.0
        n = 0
        for o in origins:
            train_periods = set(periods_sorted[:o])
            train_cols = np.where(np.isin(col_period, list(train_periods)))[0]
            test_ycols = ycols_sorted[o:o + horizon]
            pred = _pcr_predict(
                Z_scaled, inputs.treated_idx, inputs.donor_idx, train_cols,
                Y[np.ix_(inputs.donor_idx, test_ycols)], pcr_rank, pcr_cumvar)
            truth = Y[inputs.treated_idx, test_ycols]
            sse += float(np.sum((truth - pred) ** 2))
            n += len(test_ycols)
        mse_table[g] = sse / n

    best_g = min(mse_table, key=lambda k: (mse_table[k], k))
    best_weights = [1.0] + [best_g] * (K - 1)
    return best_weights, order, mse_table
