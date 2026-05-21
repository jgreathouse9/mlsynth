"""Data preparation for Partially Pooled SCM.

Identifies treated units and their adoption times, never-treated
controls, and stacks (L, N) pre-treatment donor windows for each
treated unit.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance
from .structures import PPSCMInputs


def prepare_ppscm_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    L: Optional[int] = None,
    K: Optional[int] = None,
    demean: bool = False,
) -> PPSCMInputs:
    """Build PPSCM inputs from a staggered-adoption panel.

    Parameters
    ----------
    df, outcome, treat, unitid, time : standard mlsynth panel inputs.
    L : int or None
        Number of pre-treatment lags to use, common to every treated
        unit. ``None`` defaults to ``min_j (T_j - 1)``.
    K : int or None
        Maximum event-time horizon. ``None`` defaults to
        ``min_j (T - T_j)`` so every treated unit contributes at every
        modelled horizon.
    demean : bool
        If True, subtract each unit's pre-treatment mean from its full
        outcome series before stacking. This is the paper's "intercept
        shift" extension.

    Returns
    -------
    PPSCMInputs
        Stacked pre and post donor windows aligned per treated unit.
    """

    balance(df, unitid, time)

    Ywide = (
        df.pivot(index=time, columns=unitid, values=outcome)
        .sort_index()
    )
    Twide = (
        df.pivot(index=time, columns=unitid, values=treat)
        .sort_index()
        .reindex(columns=Ywide.columns)
        .fillna(0)
        .astype(int)
    )
    time_labels = np.asarray(Ywide.index.tolist())
    unit_labels = list(Ywide.columns)
    Y = Ywide.to_numpy(dtype=float)  # (T, n_total)
    T_total, n_total = Y.shape

    # Identify treated units and their (1-based) adoption-period indices.
    treated_idx: list[int] = []
    adoption_periods: list[int] = []
    for k, u in enumerate(unit_labels):
        treated_positions = np.where(Twide.iloc[:, k].to_numpy() == 1)[0]
        if treated_positions.size > 0:
            treated_idx.append(k)
            adoption_periods.append(int(treated_positions[0]) + 1)

    if not treated_idx:
        raise MlsynthDataError(
            "PPSCM requires at least one treated unit; none found in the panel."
        )

    donor_idx = [k for k in range(n_total) if k not in set(treated_idx)]
    if not donor_idx:
        raise MlsynthDataError(
            "PPSCM requires at least one never-treated donor unit."
        )

    adoption_arr = np.asarray(adoption_periods, dtype=int)
    J = len(treated_idx)
    N = len(donor_idx)

    # Sensible L and K defaults.
    L_eff = int(L) if L is not None else int(adoption_arr.min() - 1)
    K_eff = int(K) if K is not None else int(T_total - adoption_arr.max())
    if L_eff < 1:
        raise MlsynthDataError(
            f"L = {L_eff} is too small; need L >= 1. The earliest cohort "
            f"adopts at period {int(adoption_arr.min())}, leaving at most "
            f"{int(adoption_arr.min() - 1)} pre-treatment lags."
        )
    if K_eff < 0:
        raise MlsynthDataError(
            f"K = {K_eff} is negative; no post-treatment horizons fit "
            "inside the panel."
        )

    # Optional demeaning per unit on its pre-treatment slice.
    Y_work = Y.copy()
    if demean:
        for j, t_idx in zip(treated_idx, adoption_arr):
            pre_slice = slice(0, int(t_idx) - 1)  # 0-based pre window
            mean_val = Y_work[pre_slice, j].mean()
            Y_work[:, j] -= mean_val
        for d in donor_idx:
            # Demean donors by their average over the longest pre window
            # we will use (the earliest-treated unit's pre window).
            earliest_pre_end = int(adoption_arr.min()) - 1
            mean_val = Y_work[:earliest_pre_end, d].mean()
            Y_work[:, d] -= mean_val

    # Pre-treatment stacks.
    Y_treated_pre = np.zeros((L_eff, J))
    Y_donors_pre = np.zeros((L_eff, N, J))
    Y_treated_post = np.zeros((K_eff + 1, J))
    Y_donors_post = np.zeros((K_eff + 1, N, J))
    for col, (t_unit, T_j) in enumerate(zip(treated_idx, adoption_arr)):
        # 1-based T_j means treatment starts at 0-based index T_j - 1.
        # Pre window: positions T_j - 1 - L to T_j - 2 (inclusive).
        pre_start = int(T_j) - 1 - L_eff
        pre_end = int(T_j) - 1  # exclusive
        if pre_start < 0:
            raise MlsynthDataError(
                f"Treated unit {unit_labels[t_unit]} adopts at period {T_j}, "
                f"but L = {L_eff} requires at least {L_eff} pre-treatment "
                "lags."
            )
        Y_treated_pre[:, col] = Y_work[pre_start:pre_end, t_unit]
        for r, d in enumerate(donor_idx):
            Y_donors_pre[:, r, col] = Y_work[pre_start:pre_end, d]

        # Post window: positions T_j - 1, T_j, ..., T_j - 1 + K.
        post_start = int(T_j) - 1
        post_end = post_start + K_eff + 1
        if post_end > T_total:
            raise MlsynthDataError(
                f"Treated unit {unit_labels[t_unit]} (adopts at {T_j}) "
                f"does not have {K_eff + 1} post-treatment observations."
            )
        Y_treated_post[:, col] = Y_work[post_start:post_end, t_unit]
        for r, d in enumerate(donor_idx):
            Y_donors_post[:, r, col] = Y_work[post_start:post_end, d]

    return PPSCMInputs(
        Y_treated_pre=Y_treated_pre,
        Y_donors_pre=Y_donors_pre,
        Y_treated_post=Y_treated_post,
        Y_donors_post=Y_donors_post,
        L=L_eff,
        K=K_eff,
        J=J,
        N=N,
        treated_unit_names=[unit_labels[k] for k in treated_idx],
        donor_names=[unit_labels[k] for k in donor_idx],
        adoption_periods=adoption_arr,
        time_labels=time_labels,
        Ywide=Ywide,
        outcome=outcome,
    )
