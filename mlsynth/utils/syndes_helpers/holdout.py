"""Holdout (train/validate) design selection for SYNDES.

The vanilla SYNDES MIP ranks treated sets by the *in-sample* pre-period
contrast, which can reward a set whose tight balance is transient co-movement
that will not persist. The holdout selector guards against that overfitting:

1. split the pre-period into a leading training block (``1 - holdout_frac`` of
   the periods) and a held-out validation tail (``holdout_frac``);
2. solve the SYNDES candidate pool on the training block only
   (:func:`mlsynth.utils.syndes_helpers.optimization.solve_synthetic_design_pool`);
3. score each candidate's *out-of-sample* contrast error on the validation tail
   -- ``sqrt(mean((Y_val @ contrast_weights)**2))`` -- and rank ascending.

The rank-1 (smallest OOS error) design is the winner; downstream power and
inference are computed exactly as in the in-sample path. This mirrors the
train/validate selection used by LEXSCM and the MAREX blank-period split, but on
the SYNDES contrast vector.

Note the OOS error uses the *training-learned* weights applied to the validation
periods: it measures how well a design selected on the training block continues
to balance out of sample, which is the quantity overfitting inflates.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
from .optimization import solve_synthetic_design_pool


def split_pre(
    Y_pre: np.ndarray, holdout_frac: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split the pre-period matrix into a training block and a validation tail.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix, shape ``(T_pre, N)`` (rows are periods).
    holdout_frac : float
        Fraction of the pre-period (the tail) held out for validation; must be
        in the open interval ``(0, 1)``.

    Returns
    -------
    (Y_train, Y_val, n_train) : tuple
        The leading ``n_train`` rows, the trailing validation rows, and the
        training-block length. Both blocks are guaranteed at least one row.

    Raises
    ------
    MlsynthConfigError
        If ``holdout_frac`` is outside ``(0, 1)`` or fewer than two pre-periods
        are available (nothing to split).
    """
    if not (0.0 < float(holdout_frac) < 1.0):
        raise MlsynthConfigError(
            f"holdout_frac must lie in the open interval (0, 1); got {holdout_frac!r}."
        )
    Y = np.asarray(Y_pre, dtype=float)
    T = Y.shape[0]
    if T < 2:
        raise MlsynthConfigError(
            "holdout selection needs at least 2 pre-treatment periods to split; "
            f"got {T}."
        )
    n_val = int(round(float(holdout_frac) * T))
    n_val = min(max(n_val, 1), T - 1)        # >=1 each side
    n_train = T - n_val
    return Y[:n_train], Y[n_train:], n_train


def oos_contrast_rmse(design: Any, Y_val: np.ndarray) -> float:
    """Out-of-sample contrast RMSE of a design on a validation block.

    Applies the design's (training-learned) contrast weights to the validation
    periods and returns ``sqrt(mean((Y_val @ contrast_weights)**2))`` -- the
    held-out analogue of ``design.pre_fit_rmse``. Returns ``inf`` when the design
    carries no contrast vector (so it never wins the argmin).
    """
    c = getattr(design, "contrast_weights", None)
    if c is None:
        return float("inf")
    c = np.asarray(c, dtype=float).reshape(-1)
    g = np.asarray(Y_val, dtype=float) @ c
    return float(np.sqrt(np.mean(g ** 2)))


def select_by_holdout(
    Y_pre: np.ndarray,
    *,
    holdout_frac: float,
    top_K: int,
    **solve_kw: Any,
) -> Tuple[List[Any], List[float]]:
    """Rank the SYNDES candidate pool by out-of-sample (holdout) contrast error.

    Solves the ``top_K`` candidate pool on the training block and re-ranks it by
    validation-tail OOS error (ascending). The rank-1 design is the holdout
    winner.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix, shape ``(T_pre, N)``.
    holdout_frac : float
        Validation-tail fraction in ``(0, 1)`` (see :func:`split_pre`).
    top_K : int
        Candidate-pool size handed to
        :func:`~mlsynth.utils.syndes_helpers.optimization.solve_synthetic_design_pool`.
    **solve_kw
        Remaining solver arguments (``K``, ``mode``, ``lam``, ``solver``,
        ``verbose``, ``unit_index``, ``costs``, ``budget``, ``gap_limit``,
        ``time_limit``) forwarded to the pool solver. ``Y`` and ``top_K`` are
        supplied here and must not appear in ``solve_kw``.

    Returns
    -------
    (ranked_designs, oos_errors) : tuple
        Designs ordered by ascending OOS error, and the matching OOS errors
        (same order). Ties keep the solver's in-sample order (stable sort).
    """
    Y_train, Y_val, _ = split_pre(Y_pre, holdout_frac)
    designs = solve_synthetic_design_pool(Y=Y_train, top_K=top_K, **solve_kw)
    oos = [oos_contrast_rmse(d, Y_val) for d in designs]
    order = sorted(range(len(designs)), key=lambda i: (oos[i], i))   # stable
    ranked = [designs[i] for i in order]
    oos_sorted = [oos[i] for i in order]
    return ranked, oos_sorted
