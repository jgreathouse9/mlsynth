"""Panel ingestion for the ISCM estimator.

Reads a long-format panel into the dense ``(N, T)`` outcome and treatment
matrices ISCM operates on, through :func:`mlsynth.utils.datautils.balance`
and :func:`mlsynth.utils.datautils.dataprep`. ISCM builds a synthetic
control for *every* unit, so -- unlike single-treated SCM -- there is no
treated / donor split at this stage; all units are retained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import ISCMInputs


def _adoption_starts(prep) -> dict:
    """Each treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit
    is treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def prepare_iscm_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> ISCMInputs:
    """Build :class:`ISCMInputs` from a long panel through ``balance`` and ``dataprep``.

    ``dataprep`` supplies the outcome matrix, the time labels and each
    treated unit's adoption period, and the treatment matrix ``D`` is built
    from the adoption periods. ``balance`` and ``dataprep`` refuse a missing
    or non-finite outcome, an absent or duplicate unit-period, a non-binary
    treatment and a treatment that switches off after it starts.

    Parameters
    ----------
    df : pd.DataFrame
        Balanced long panel; one row per ``(unit, time)``.
    outcome, treat, unitid, time : str
        Column names. A treated unit has ``treat == 1`` from its adoption
        period to the end of the panel. Powell (2026) states ISCM for a
        single adoption date; with staggered adoption ``T0`` is the first
        period any unit is treated, every later adopter is treated in ``D``
        from its own start, and all ever-treated units are in
        ``treated_idx``.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    # dataprep refuses a missing treatment cell as well; ISCM refuses it here,
    # before balance scans the keys, with the message its old pivot gave.
    if df[treat].isna().any():
        raise MlsynthDataError("Treatment column contains missing values.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    if keys.n_units < 3:
        raise MlsynthDataError(
            "ISCM needs at least 3 units (a synthetic control is built for "
            "every unit from the others)."
        )
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)

    unit_names = sorted(np.asarray(prep.Ywide.columns))
    # Built from the labels one by one, as the old setup's sorted list was, so
    # datetime periods stay an object array of Timestamps and a 32-bit
    # period column stays 32-bit (``.array`` keeps the column's dtype).
    time_labels = np.array(list(prep.time_labels.array))
    T = int(time_labels.size)
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    starts = _adoption_starts(prep)
    D = np.zeros((len(unit_names), T), dtype=float)
    for i, name in enumerate(unit_names):
        if name in starts:
            D[i, starts[name]:] = 1.0

    # Adoption period: first time any unit is treated. dataprep refuses
    # treatment in the first period only when one unit is treated.
    T0 = min(starts.values())
    if T0 == 0:
        raise MlsynthDataError("Some unit is treated at the earliest period.")
    if T0 < 2:
        raise MlsynthDataError(
            "ISCM requires at least 2 pre-treatment periods (and assumes a "
            "large pre-period for its asymptotics)."
        )

    treated_idx = np.where(D[:, T0:].max(axis=1) == 1)[0]

    return ISCMInputs(
        Y=Y,
        D=D,
        T0=T0,
        unit_names=list(unit_names),
        time_labels=time_labels,
        treated_idx=treated_idx,
    )
