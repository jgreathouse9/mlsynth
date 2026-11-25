"""Panel ingestion for the SNN estimator.

Builds the ``(N, T)`` outcome and treatment matrices from a long panel through
:func:`mlsynth.utils.datautils.dataprep`. SNN treats the treated
post-treatment cells as missing and imputes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import SNNInputs


def _adoption_starts(prep) -> dict:
    """Each treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit is
    treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def prepare_snn_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SNNInputs:
    """Build :class:`SNNInputs` from a long panel through ``datautils.dataprep``.

    ``dataprep`` supplies the outcome matrix, the time labels and each treated
    unit's adoption period. ``balance`` and ``dataprep`` refuse a missing or
    non-finite outcome, an absent or duplicate unit-period, a non-binary
    treatment, treatment in the first period, a panel with no treated cell and
    a treatment that switches off after it starts. A treated unit has
    ``treat == 1`` from its adoption period onward, with one adoption period or
    staggered ones, and the treatment matrix ``D`` is rebuilt from the adoption
    periods; SNN imputes those cells' untreated potential outcomes. SNN's own
    minimums (3 units, 2 never-treated donors) are checked on top.
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
        raise MlsynthDataError("SNN needs at least 3 units.")

    time_labels = np.asarray(prep.time_labels)
    T = int(time_labels.size)
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    starts = _adoption_starts(prep)
    D = np.zeros((N, T), dtype=float)
    for i, name in enumerate(unit_names):
        if name in starts:
            D[i, starts[name]:] = 1.0

    # dataprep refuses treatment in the first period and a panel with no
    # treated cell, so 1 <= T0 < T and some unit is treated.
    T0 = min(starts.values())
    treated_idx = np.where(D.max(axis=1) == 1)[0]
    if N - treated_idx.size < 2:
        raise MlsynthDataError("SNN needs at least 2 control (donor) units.")

    return SNNInputs(
        Y=Y, D=D, treated_idx=treated_idx, T0=T0,
        unit_names=list(unit_names), time_labels=time_labels,
    )
