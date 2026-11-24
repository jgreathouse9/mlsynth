"""Panel ingestion for the MC-NNM estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import MCNNMInputs


def _adoption_starts(prep) -> dict:
    """Each treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit is
    treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def prepare_mcnnm_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> MCNNMInputs:
    """Build :class:`MCNNMInputs` from a long panel through ``datautils.dataprep``.

    ``dataprep`` supplies the outcome matrix, the time labels and each treated
    unit's adoption period. ``balance`` and ``dataprep`` refuse a missing or
    non-finite outcome, an unbalanced panel, a duplicate unit-period, a
    non-binary treatment and a treatment that switches off after it starts.
    MC-NNM is fitted here to one-time and staggered adoption, the assignment
    patterns of Athey, Bayati, Doudchenko, Imbens and Khosravi (JASA 116, 2021,
    section 2), so the treatment matrix ``D`` is rebuilt from the adoption
    periods.

    Observed entries (``mask == 1``) are the control units and the treated
    units' pre-treatment periods; the treated post-treatment cells are the
    missing entries MC-NNM imputes.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)
    unit_names = sorted(prep.Ywide.columns)
    N = len(unit_names)
    if N < 3:
        raise MlsynthDataError("MC-NNM needs at least 3 units.")

    time_labels = np.asarray(prep.time_labels)
    T = int(time_labels.size)
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    starts = _adoption_starts(prep)
    D = np.zeros((N, T), dtype=float)
    for i, name in enumerate(unit_names):
        if name in starts:
            D[i, starts[name]:] = 1.0

    # dataprep refuses treatment in the first period and a panel with no
    # treated cell, so T0 >= 1 and some unit is treated.
    T0 = min(starts.values())
    treated_idx = np.where(D.max(axis=1) == 1)[0]
    if N - treated_idx.size < 2:
        raise MlsynthDataError("MC-NNM needs at least 2 control units.")

    mask = (D == 0).astype(float)   # observed where untreated

    return MCNNMInputs(
        Y=Y, mask=mask, D=D, treated_idx=treated_idx, T0=T0,
        unit_names=list(unit_names), time_labels=time_labels,
    )
