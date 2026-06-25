"""Placebo-distribution inference for SCUL (reference ``CreatePlaceboDistribution`` / ``PValue``).

Each donor unit is treated, in turn, as a fake-treated target and fit with SCUL
on the remaining pool. The treatment statistic is the unit-free ratio of the
post- to pre-treatment root mean squared gap (Abadie, Diamond & Hainmueller
2010); the p-value is the rank of the real treated unit's ratio in the placebo
distribution. Placebo units whose pre-treatment fit is poor (Cohen's D above the
threshold) are trimmed, per the paper's quality-control recommendation.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from .estimate import _PLACEBO_TOL, fit_scul


def _rmspe_ratio(gap: np.ndarray, T0: int) -> float:
    pre = np.sqrt(np.mean(gap[:T0] ** 2))
    post = np.sqrt(np.mean(gap[T0:] ** 2))
    return float(post / pre) if pre > 0 else np.inf


def placebo_pvalue(
    inputs, treated_gap: np.ndarray, *,
    number_initial_periods: int, training_post_length: int, cv_option: str,
    cohensd_threshold: float,
) -> Tuple[float, int]:
    """Placebo p-value for the treated unit's effect.

    Returns ``(p_value, n_placebo_kept)``. Each donor unit is re-fit as a
    fake-treated target against the pool of columns it does not own; placebos
    with a pre-fit worse than ``cohensd_threshold`` are dropped.
    """
    T0 = inputs.T0
    col_unit = inputs.col_unit
    treated_ratio = _rmspe_ratio(treated_gap, T0)

    ratios = []
    for j, unit in enumerate(inputs.donor_names):
        keep_cols = col_unit != unit
        if keep_cols.sum() < 2:                       # need a pool to fit against
            continue
        target = inputs.donor_outcome[:, j]
        pool = inputs.donor_matrix[:, keep_cols]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_scul(
                    target, pool, T0,
                    number_initial_periods=number_initial_periods,
                    training_post_length=training_post_length,
                    cv_option=cv_option, tol=_PLACEBO_TOL,
                )
        except (ValueError, np.linalg.LinAlgError):    # pragma: no cover - degenerate placebo
            continue
        if fit["cohens_d"] > cohensd_threshold:        # trim poor pre-fit placebos
            continue
        ratios.append(_rmspe_ratio(target - fit["counterfactual"], T0))

    ratios = np.asarray(ratios, dtype=float)
    n = ratios.size
    if n == 0:                                          # pragma: no cover - no admissible placebo
        return float("nan"), 0
    # rank of the treated ratio (more extreme => smaller p-value), reference PValue.
    p = float((np.sum(ratios >= treated_ratio) + 1) / (n + 1))
    return p, int(n)
