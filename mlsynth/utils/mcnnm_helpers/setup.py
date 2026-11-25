"""Panel ingestion for the MC-NNM estimator.

MC-NNM fits on the observed set and imputes every cell outside it (Athey,
Bayati, Doudchenko, Imbens and Khosravi, JASA 116, 2021, eq. 4.3). A missing
outcome cell is kept (``missing_outcomes="keep"``) and left out of the mask
like a treated cell. A missing outcome of a treated unit after its adoption is
refused (:func:`_refuse_missing_treated_post`).
"""

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


def _refuse_missing_treated_post(Y_nan: np.ndarray, D: np.ndarray, unit_names,
                                 time_labels, outcome: str) -> None:
    """Raise if a treated unit's outcome is missing at or after its adoption.

    MC-NNM's effect at a treated cell is the realized outcome minus the
    imputed untreated outcome. Athey et al. (2021, section 2) observe a
    realized outcome in every cell, and ``MCPanel`` completes the matrix with
    no effect estimand, so neither defines the effect at a treated cell whose
    outcome is missing; the cell is refused and not averaged over or dropped.
    The first such cell is the first in time order, then unit order.
    """
    bad = (D > 0) & Y_nan
    if not bad.any():
        return
    t, i = np.argwhere(bad.T)[0]
    raise MlsynthDataError(
        f"Missing outcome for a treated unit after its adoption: {int(bad.sum())} treated "
        f"post-treatment cell(s) of '{outcome}' are NaN, the first at treated unit "
        f"'{unit_names[i]}', time {time_labels[t]}. MC-NNM's effect at a treated cell is the "
        "realized outcome minus the imputed untreated outcome, and MC-NNM (Athey, Bayati, "
        "Doudchenko, Imbens and Khosravi 2021) does not define it when the realized outcome "
        "is missing. Drop or fill that unit-period before fitting; a missing cell anywhere "
        "else is imputed."
    )


def prepare_mcnnm_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> MCNNMInputs:
    """Build :class:`MCNNMInputs` from a long panel through ``datautils.dataprep``.

    ``dataprep`` supplies the outcome matrix, the time labels and each treated
    unit's adoption period. ``balance`` and ``dataprep`` refuse an infinite
    outcome, an unbalanced panel, a duplicate unit-period, a missing or
    non-binary treatment and a treatment that switches off after it starts.
    A missing (NaN) outcome stays ``NaN`` in ``Y`` and is left out of the mask,
    except on a treated unit at or after its adoption, which is refused.
    MC-NNM is fitted here to one-time and staggered adoption, the assignment
    patterns of Athey, Bayati, Doudchenko, Imbens and Khosravi (JASA 116, 2021,
    section 2), so the treatment matrix ``D`` is rebuilt from the adoption
    periods.

    Observed entries (``mask == 1``) are the untreated cells whose outcome is
    present: the control units and the treated units' pre-treatment periods.
    The treated post-treatment cells and the missing cells are the entries
    MC-NNM imputes.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys,
                    missing_outcomes="keep")
    unit_names = sorted(prep.Ywide.columns)
    N = len(unit_names)
    if N < 3:
        raise MlsynthDataError("MC-NNM needs at least 3 units.")

    time_labels = np.asarray(prep.time_labels)
    T = int(time_labels.size)
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    observed = prep.observed[:, prep.Ywide.columns.get_indexer(unit_names)].T
    starts = _adoption_starts(prep)
    D = np.zeros((N, T), dtype=float)
    for i, name in enumerate(unit_names):
        if name in starts:
            D[i, starts[name]:] = 1.0
    _refuse_missing_treated_post(~observed, D, unit_names, time_labels, outcome)

    # dataprep refuses treatment in the first period and a panel with no
    # treated cell, so T0 >= 1 and some unit is treated.
    T0 = min(starts.values())
    treated_idx = np.where(D.max(axis=1) == 1)[0]
    if N - treated_idx.size < 2:
        raise MlsynthDataError("MC-NNM needs at least 2 control units.")

    mask = ((D == 0) & observed).astype(float)   # observed where untreated and present

    return MCNNMInputs(
        Y=Y, mask=mask, D=D, treated_idx=treated_idx, T0=T0,
        unit_names=list(unit_names), time_labels=time_labels,
    )
