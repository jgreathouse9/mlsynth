"""Panel ingestion for the SPOTSYNTH estimator (single treated unit).

Reads a long-format panel through :func:`mlsynth.utils.datautils.balance` and
:func:`mlsynth.utils.datautils.dataprep` into the treated series and the donor
matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import CohortPanel
from .structures import SpotSynthInputs


def prepare_spotsynth_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> SpotSynthInputs:
    """Build the treated series and the donor matrix from a long panel.

    SPOTSYNTH targets the classic single-treated-unit synthetic-control design:
    one unit adopts the intervention at a common time ``T0`` and every other unit
    is a candidate donor to be screened for spillover contamination.

    ``dataprep`` supplies the outcome matrix, the time labels, the treated unit
    and its adoption period. ``balance`` and ``dataprep`` refuse a missing or
    non-finite outcome, an absent or duplicate unit-period, a non-binary or
    missing treatment value, a treatment that switches off after it starts
    and a single treated unit treated in the first period. SPOTSYNTH's own
    checks (three units, one treated unit, three pre-periods) stay on top.
    """
    required = {outcome, treat, unitid, time}
    missing = required - set(df.columns)
    if missing:
        raise MlsynthDataError(f"Missing columns: {', '.join(sorted(missing))}.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    if keys.n_units < 3:
        raise MlsynthDataError("SPOTSYNTH needs a treated unit and at least 2 donors.")
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)
    if isinstance(prep, CohortPanel):
        n_treated = sum(len(c.treated_units) for c in prep.cohorts.values())
        raise MlsynthDataError(
            "SPOTSYNTH expects a single treated unit. Found "
            f"{n_treated}; restrict the panel or aggregate first."
        )

    unit_names = sorted(np.asarray(prep.Ywide.columns))
    # Built from the index's array one label at a time, as the old setup's
    # sorted list was: datetime periods stay an object array of Timestamps and
    # 32-bit periods stay 32-bit (iterating the index itself yields Python
    # scalars, which widen them).
    time_labels = np.array(list(prep.time_labels.array))
    # Unit by period and C-ordered, as the old pivot was, so the treated row
    # is contiguous and the donor block keeps its layout.
    Y = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    ti = unit_names.index(prep.treated_unit_name)

    # dataprep refuses a single treated unit with no pre-period.
    T0 = int(prep.pre_periods)
    if T0 < 3:
        raise MlsynthDataError("SPOTSYNTH needs at least 3 pre-intervention periods.")

    donor_rows = [i for i in range(len(unit_names)) if i != ti]
    y = Y[ti]
    D = Y[donor_rows].T  # (T, n_donors)
    donor_names = [unit_names[i] for i in donor_rows]
    return SpotSynthInputs(
        y=y, D=D, T0=T0, donor_names=donor_names,
        treated_name=unit_names[ti], time_labels=time_labels,
    )
