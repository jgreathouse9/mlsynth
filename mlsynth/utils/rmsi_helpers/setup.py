"""Panel + side-information ingestion for the RMSI estimator.

The outcome matrix, the time labels and each treated unit's adoption period
come from :func:`mlsynth.utils.datautils.balance` and
:func:`mlsynth.utils.datautils.dataprep`; the side information is the unit and
period means of the covariate columns.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import RMSIInputs


def _adoption_starts(prep) -> dict:
    """Each treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit
    is treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def prepare_rmsi_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    unit_covariates: Optional[List[str]] = None,
    time_covariates: Optional[List[str]] = None,
) -> RMSIInputs:
    """Pivot a long panel into block matrices and extract side information.

    RMSI (Agarwal, Choi & Yuan 2026) assumes a **block** design: every
    ever-treated unit adopts at the same period ``T0``. ``unit_covariates`` are
    columns (approximately) constant within a unit -> the row feature matrix
    ``X`` (one row per unit, time-averaged); ``time_covariates`` are columns
    constant within a period -> the column feature matrix ``Z`` (one row per
    period, unit-averaged). Either may be empty, in which case the corresponding
    projection captures only the mean.

    ``dataprep`` supplies the outcome matrix, the time labels and each treated
    unit's adoption period, and the treatment matrix ``D`` is built from the
    adoption periods. ``balance`` and ``dataprep`` refuse a missing or
    non-finite outcome, an absent or duplicate unit-period, a non-binary
    treatment and a treatment that switches off after it starts. RMSI's own
    checks (three units, two pre-periods, two never-treated controls, one
    adoption period, no unit treated in the first period) stay on top.
    """
    unit_covariates = list(unit_covariates or [])
    time_covariates = list(time_covariates or [])
    required = {outcome, treat, unitid, time, *unit_covariates, *time_covariates}
    missing = required - set(df.columns)
    if missing:
        raise MlsynthDataError(f"Missing columns: {', '.join(sorted(missing))}.")
    # dataprep's binary check skips NaN, so a missing treatment cell would be
    # read as untreated; the old pivot refused it as a missing cell.
    if df[treat].isna().any():
        raise MlsynthDataError("Treatment column contains missing values.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    N = keys.n_units
    if N < 3:
        raise MlsynthDataError("RMSI needs at least 3 units.")
    # With three units a single treated unit has two donors, so dataprep's
    # no-donor refusal cannot fire; a block with too few controls is refused
    # below with RMSI's own message.
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)

    unit_names = sorted(np.asarray(prep.Ywide.columns))
    # Built from the index's array one label at a time, as the old setup's
    # sorted list was: datetime periods stay an object array of Timestamps and
    # 32-bit periods stay 32-bit (iterating the index itself yields Python
    # scalars, which widen them).
    time_labels = np.array(list(prep.time_labels.array))
    T = int(time_labels.size)
    # RMSI's Y is unit x time; the C-order copy keeps the row-major layout
    # the old pivot produced.
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    starts = _adoption_starts(prep)
    D = np.zeros((N, T), dtype=int)
    for i, name in enumerate(unit_names):
        if name in starts:
            D[i, starts[name]:] = 1

    # Adoption period: first time any unit is treated. dataprep refuses
    # treatment in the first period only when one unit is treated.
    T0 = min(starts.values())
    if T0 == 0:
        raise MlsynthDataError("Some unit is treated at the earliest period.")
    if T0 < 2:
        raise MlsynthDataError("RMSI needs at least 2 pre-treatment periods.")

    treated_idx = np.where(D.max(axis=1) == 1)[0]
    control_idx = np.where(D.max(axis=1) == 0)[0]
    if control_idx.size < 2:
        raise MlsynthDataError("RMSI needs at least 2 never-treated control units.")

    # Block-design check: all treated units share one adoption period.
    if len(set(starts.values())) > 1:
        raise MlsynthDataError(
            "RMSI assumes a block design (all treated units adopt at the same "
            "period). Staggered adoption detected -- use SDID, SequentialSDID, "
            "PPSCM, SSC, or MCNNM instead."
        )

    # Side information is not an outcome: a covariate may have missing cells,
    # which the means skip, so it is not read through dataprep, which refuses
    # them. These are group means, not pivots.
    # Row features X (one row per unit): unit-level mean of each unit covariate.
    if unit_covariates:
        X = np.column_stack([
            df.groupby(unitid, observed=True)[c].mean().loc[unit_names].to_numpy(dtype=float)
            for c in unit_covariates])
    else:
        X = np.zeros((N, 0))
    # Column features Z (one row per period): period-level mean of each time cov.
    if time_covariates:
        Z = np.column_stack([
            df.groupby(time, observed=True)[c].mean().loc[time_labels].to_numpy(dtype=float)
            for c in time_covariates])
    else:
        Z = np.zeros((T, 0))

    return RMSIInputs(
        Y=Y, D=D, X=X, Z=Z, T0=T0, treated_idx=treated_idx,
        control_idx=control_idx, unit_names=list(unit_names),
        time_labels=time_labels, unit_covariates=unit_covariates,
        time_covariates=time_covariates,
    )
