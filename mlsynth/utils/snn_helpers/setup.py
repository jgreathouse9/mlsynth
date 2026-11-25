"""Panel ingestion for the SNN estimator.

Builds the ``(N, T)`` outcome and treatment matrices from a long panel through
:func:`mlsynth.utils.datautils.dataprep`. SNN treats the treated
post-treatment cells as missing and imputes them.

SNN is a matrix-completion method (Agarwal, Dahleh, Shah and Shen 2021): it
imputes any missing entry from a fully observed anchor block, so a missing
outcome cell in the panel is kept (``missing_outcomes="keep"``) and its
``observed`` mask is passed on to the completion. A missing outcome of a treated
unit after its adoption is refused (:func:`_refuse_missing_treated_post`).
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


def _refuse_missing_treated_post(Y_nan: np.ndarray, D: np.ndarray, unit_names,
                                 time_labels, outcome: str) -> None:
    """Raise if a treated unit's outcome is missing at or after its adoption.

    SNN's effect at a treated cell is the observed outcome minus the imputed
    ``Y(0)``. Agarwal et al. (2021) and the reference implementation
    (``deshen24/syntheticNN``) impute missing ``Y(0)`` entries and define no
    effect at a cell whose treated outcome is missing, so the cell is refused
    and not averaged over or dropped. The first such cell is the first in time
    order, then unit order.
    """
    bad = (D > 0) & Y_nan
    if not bad.any():
        return
    t, i = np.argwhere(bad.T)[0]
    raise MlsynthDataError(
        f"Missing outcome for a treated unit after its adoption: {int(bad.sum())} treated "
        f"post-treatment cell(s) of '{outcome}' are NaN, the first at treated unit "
        f"'{unit_names[i]}', time {time_labels[t]}. SNN's effect at a treated cell is the "
        "observed outcome minus the imputed untreated outcome, and SNN (Agarwal, Dahleh, "
        "Shah and Shen 2021) does not define it when the observed outcome is missing. "
        "Drop or fill that unit-period before fitting; a missing cell anywhere else is "
        "imputed."
    )


def prepare_snn_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SNNInputs:
    """Build :class:`SNNInputs` from a long panel through ``datautils.dataprep``.

    ``dataprep`` supplies the outcome matrix, the time labels and each treated
    unit's adoption period. A missing (NaN) outcome cell is kept and marked
    ``False`` in ``observed``; SNN imputes it like any other missing entry,
    except on a treated unit at or after its adoption, which is refused.
    ``balance`` and ``dataprep`` refuse an infinite outcome, an absent or
    duplicate unit-period, a missing or non-binary treatment, treatment in the
    first period, a panel with no treated cell and a treatment that switches
    off after it starts. A treated unit has
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
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys,
                    missing_outcomes="keep")
    unit_names = sorted(prep.Ywide.columns)
    N = len(unit_names)
    if N < 3:
        raise MlsynthDataError("SNN needs at least 3 units.")

    time_labels = np.asarray(prep.time_labels)
    T = int(time_labels.size)
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    observed = np.ascontiguousarray(
        prep.observed[:, prep.Ywide.columns.get_indexer(unit_names)].T)
    starts = _adoption_starts(prep)
    D = np.zeros((N, T), dtype=float)
    for i, name in enumerate(unit_names):
        if name in starts:
            D[i, starts[name]:] = 1.0
    _refuse_missing_treated_post(~observed, D, unit_names, time_labels, outcome)

    # dataprep refuses treatment in the first period and a panel with no
    # treated cell, so 1 <= T0 < T and some unit is treated.
    T0 = min(starts.values())
    treated_idx = np.where(D.max(axis=1) == 1)[0]
    if N - treated_idx.size < 2:
        raise MlsynthDataError("SNN needs at least 2 control (donor) units.")

    return SNNInputs(
        Y=Y, D=D, treated_idx=treated_idx, T0=T0,
        unit_names=list(unit_names), time_labels=time_labels, observed=observed,
    )
