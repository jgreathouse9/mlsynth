"""Panel ingestion for the SSC estimator (staggered adoption).

Reads a long-format panel into the dense ``(N, T)`` outcome and treatment
matrices SSC operates on, through :func:`mlsynth.utils.datautils.balance`
and :func:`mlsynth.utils.datautils.dataprep`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import SSCInputs


def _adoption_starts(prep) -> dict:
    """Each treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit
    is treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def prepare_ssc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SSCInputs:
    """Build :class:`SSCInputs` from a long panel through ``balance`` and ``dataprep``.

    SSC (Cao, Lu & Wu 2026) targets staggered adoption with a long
    pre-period: ``T0`` is the number of clean periods before *any* unit is
    treated, and all units -- including not-yet-treated ones -- serve as donors.
    The treatment indicator must be absorbing (once 1, stays 1).

    ``dataprep`` supplies the outcome matrix, the time labels and each treated
    unit's adoption period; the treatment matrix ``D`` and the adoption vector
    are built from the adoption periods. ``balance`` and ``dataprep`` refuse a
    missing or non-finite outcome, an absent or duplicate unit-period, a
    non-binary treatment and a treatment that switches off after it starts.
    SSC's own checks (three units, two clean pre-periods, no unit treated in
    the first period, no missing treatment value) stay on top.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    # dataprep's binary check skips NaN, so a missing treatment cell would be
    # read as untreated; the old pivot refused it as a missing cell.
    if df[treat].isna().any():
        raise MlsynthDataError("Treatment column contains missing values.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    if keys.n_units < 3:
        raise MlsynthDataError("SSC needs at least 3 units (each unit is "
                               "predicted from the others).")
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)

    unit_names = sorted(np.asarray(prep.Ywide.columns))
    # Built from the labels one by one, as the old setup's sorted list was, so
    # datetime periods stay an object array of Timestamps.
    time_labels = np.array(list(prep.time_labels))
    T1 = int(time_labels.size)
    N = len(unit_names)
    # SSC's Y is unit x time; the C-order copy keeps the row-major layout the
    # old pivot produced.
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    starts = _adoption_starts(prep)
    D = np.zeros((N, T1), dtype=int)
    adoption = np.full(N, -1, dtype=int)
    for i, name in enumerate(unit_names):
        if name in starts:
            D[i, starts[name]:] = 1
            adoption[i] = starts[name]

    # First period with any treatment. dataprep refuses treatment in the
    # first period only when one unit is treated.
    T0 = min(starts.values())
    if T0 == 0:
        raise MlsynthDataError("Some unit is treated at the earliest period; "
                               "SSC needs clean pre-treatment periods.")
    if T0 < 2:
        raise MlsynthDataError("SSC needs at least 2 clean pre-treatment periods.")
    # Note: end-of-sample inference additionally needs T0 > S (so at least one
    # pre-treatment placebo window exists). When T0 <= S the point estimates are
    # still computed but the bands are NaN (matching the reference).

    treated_idx = np.where(adoption >= 0)[0]

    return SSCInputs(
        Y=Y, D=D, T0=T0, unit_names=list(unit_names), time_labels=time_labels,
        treated_idx=treated_idx, adoption=adoption,
    )
