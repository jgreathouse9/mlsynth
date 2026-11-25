"""Panel ingestion for the MSQRT estimator (block, multiple treated units).

Reads a long-format panel through :func:`mlsynth.utils.datautils.balance`
and :func:`mlsynth.utils.datautils.dataprep` and splits it into the stacked
``Y = X Theta + E`` block matrices.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import MSQRTInputs


def _adoption_starts(prep) -> dict:
    """Each treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit
    is treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def prepare_msqrt_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    weight_col: Optional[str] = None,
) -> MSQRTInputs:
    """Build the stacked ``Y = X Theta + E`` block matrices from a long panel.

    MSQRT (Shen, Song & Abadie 2025) assumes a block design: every
    ever-treated unit adopts at the same period. Treated units form the columns
    of ``Y``; never-treated units form the columns of ``X``.

    ``dataprep`` supplies the outcome matrix, the time labels and each treated
    unit's adoption period. ``balance`` and ``dataprep`` refuse a missing or
    non-finite outcome, an absent or duplicate unit-period, a non-binary
    treatment and a treatment that switches off after it starts. MSQRT's own
    checks (two pre-periods, two never-treated donors, one adoption period,
    no unit treated in the first period) stay on top.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    # dataprep's binary check skips NaN, so a missing treatment cell would be
    # read as untreated; the old pivot refused it as a missing cell.
    if df[treat].isna().any():
        raise MlsynthDataError("Treatment column contains missing values.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice. A single treated unit
    # with fewer than two donors is refused below with MSQRT's own message.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, allow_no_donors=True, keys=keys)

    unit_names = sorted(np.asarray(prep.Ywide.columns))
    # Built from the labels one by one, as the old setup's sorted list was, so
    # datetime periods stay an object array of Timestamps.
    time_labels = np.array(list(prep.time_labels))
    T = time_labels.size
    # Unit by period, C-ordered as the old pivot was: the slices below keep
    # the layout the solver's sums were computed on.
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    starts = _adoption_starts(prep)

    # Adoption period: first time any unit is treated. dataprep refuses
    # treatment in the first period only when one unit is treated.
    T0 = min(starts.values())
    if T0 == 0:
        raise MlsynthDataError("Some unit is treated at the earliest period.")
    if T0 < 2:
        raise MlsynthDataError("MSQRT needs at least 2 pre-treatment periods.")

    treated_idx = np.array([i for i, u in enumerate(unit_names) if u in starts], dtype=np.int64)
    control_idx = np.array([i for i, u in enumerate(unit_names) if u not in starts],
                           dtype=np.int64)
    if control_idx.size < 2:
        raise MlsynthDataError("MSQRT needs at least 2 never-treated donor units.")

    # Block-design check: all treated units share one adoption period.
    if len(set(starts.values())) > 1:
        raise MlsynthDataError(
            "MSQRT assumes a block design (all treated units adopt at the same "
            "period). Staggered adoption detected -- use SDID, SequentialSDID, "
            "PPSCM, or MCNNM instead."
        )

    treated_names = [unit_names[i] for i in treated_idx]
    control_names = [unit_names[i] for i in control_idx]

    # Optional per-treated-unit size weights for the weighted aggregate ATT.
    # The column is taken to be unit-constant (first value per unit).
    unit_weights = None
    if weight_col is not None:
        if weight_col not in df.columns:
            raise MlsynthDataError(f"weight_col {weight_col!r} missing from df.")
        wmap = (df.drop_duplicates(subset=[unitid])
                  .set_index(unitid)[weight_col])
        unit_weights = {}
        for u in treated_names:
            val = wmap.get(u)
            if val is None or pd.isna(val):
                raise MlsynthDataError(
                    f"weight_col {weight_col!r} has no value for treated unit {u!r}.")
            fv = float(val)
            if fv < 0:
                raise MlsynthDataError(f"weight_col {weight_col!r} is negative for {u!r}.")
            unit_weights[str(u)] = fv
        if sum(unit_weights.values()) <= 0:
            raise MlsynthDataError(
                f"weight_col {weight_col!r} sums to zero across treated units.")

    Y_pre = Y[np.ix_(treated_idx, np.arange(T0))].T        # (T0, m)
    Y_post = Y[np.ix_(treated_idx, np.arange(T0, T))].T    # (T_post, m)
    X_pre = Y[np.ix_(control_idx, np.arange(T0))].T        # (T0, n)
    X_post = Y[np.ix_(control_idx, np.arange(T0, T))].T    # (T_post, n)

    return MSQRTInputs(
        Y_pre=Y_pre, Y_post=Y_post, X_pre=X_pre, X_post=X_post,
        treated_names=treated_names, control_names=control_names,
        time_labels=time_labels, T0=int(T0), unit_weights=unit_weights,
    )
