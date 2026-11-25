"""Ingestion for ROLLDID: long panel -> per-unit series + cohort structure.

The rolling-DiD transformation is unit-level, so we need each unit's full
outcome series, its treatment **cohort** (the first period in which its
treatment indicator turns on, ``g``), and the set of never-treated units. The
design is *common timing* when every treated unit shares one cohort, and
*staggered* otherwise. The panel is read through
:func:`mlsynth.utils.datautils.balance` and
:func:`mlsynth.utils.datautils.dataprep`.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

from ..datautils import balance, dataprep
from ..prepared import SinglePanel


def _adoption_starts(prep) -> Dict[Any, int]:
    """Each treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit
    is treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


def rolldid_setup(
    df: pd.DataFrame,
    unit_id: str,
    time_id: str,
    outcome: str,
    treat: str,
) -> Dict[str, Any]:
    """Resolve the panel into the inputs the rolling-DiD pipeline needs.

    ``balance`` refuses a duplicate or absent unit-period. ``dataprep``
    supplies the outcome matrix, the time labels and each treated unit's
    adoption period, and refuses a missing or non-finite outcome anywhere in
    the panel, a treatment that switches off after it starts, and a single
    treated unit treated from the first period.

    Returns a dict with ``Ywide`` (``time x unit``), ``cohort_of`` (``{unit: g}``
    for treated units), ``never`` (never-treated unit labels), ``treated`` (all
    eventually-treated labels), ``times`` (sorted time labels), and ``design``
    (``"common"`` / ``"staggered"``).

    Raises
    ------
    MlsynthConfigError
        If a required column is missing.
    MlsynthDataError
        If the panel is unbalanced or has a duplicate unit-period, an outcome
        is missing or not finite, the treatment indicator is not 0/1, there are
        no treated units, no never-treated controls, or treatment switches off
        (the indicator must be absorbing once on).
    """
    for col in (unit_id, time_id, outcome, treat):
        if col not in df.columns:
            raise MlsynthConfigError(f"column {col!r} not found in df.")

    w = df[treat]
    if not np.isin(w.dropna().unique(), [0, 1]).all():
        raise MlsynthDataError(f"treatment indicator {treat!r} must be 0/1.")
    if not (w == 1).any():
        raise MlsynthDataError("no treated units found in the treatment indicator.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being taken again. A panel with no
    # never-treated unit is ROLLDID's own refusal, below.
    keys = balance(df, unit_id, time_id)
    prep = dataprep(df, unit_id, time_id, outcome, treat, allow_no_donors=True,
                    keys=keys)

    # The pipeline's sums depend on the memory layout. This is the layout the
    # old df.pivot gave under pandas 2.x (the version CI runs): column-major
    # values, so each unit's series is contiguous. dataprep's matrix is
    # row-major.
    Ywide = pd.DataFrame(np.asfortranarray(prep.Ywide.to_numpy()),
                         index=prep.Ywide.index, columns=prep.Ywide.columns)
    times = Ywide.index.to_numpy()

    # Per-unit cohort g = first period the indicator is on, in the panel's unit
    # order; never-treated units have none.
    starts = _adoption_starts(prep)
    cohort_of: Dict[Any, Any] = {unit: Ywide.index[starts[unit]]
                                 for unit in Ywide.columns if unit in starts}

    treated = sorted(cohort_of, key=lambda u: str(u))
    never = [u for u in Ywide.columns if u not in cohort_of]
    if not never:
        raise MlsynthDataError(
            "no never-treated control units found; ROLLDID needs a never-treated "
            "group as the comparison.")

    cohorts = sorted(set(cohort_of.values()))
    design = "common" if len(cohorts) == 1 else "staggered"

    return {
        "Ywide": Ywide,
        "cohort_of": cohort_of,
        "never": never,
        "treated": treated,
        "times": times,
        "cohorts": cohorts,
        "design": design,
    }
