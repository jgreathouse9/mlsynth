"""Long-DataFrame -> NumPy boundary for RESCM (the only pandas touchpoint).

The panel is read through :func:`mlsynth.utils.datautils.balance` and
:func:`mlsynth.utils.datautils.dataprep`.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..fast_scm_helpers.structure import IndexSet
from .structures import RESCMInputs


def derive_treatment(df: pd.DataFrame, unitid: str, time: str, treat: str) -> Tuple[Any, Any]:
    """Read the single treated unit and its first treated period from ``treat``."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(f"No treated rows (treat == 1) found in '{treat}'.")
    treated_units = pd.unique(treated_rows[unitid])
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"RESCM expects exactly one treated unit; found {len(treated_units)}."
        )
    treated_unit = treated_units[0]
    intervention_time = treated_rows.loc[treated_rows[unitid] == treated_unit, time].min()
    return treated_unit, intervention_time


def prepare_rescm_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treat: str,
) -> RESCMInputs:
    """Read the panel through ``balance`` and ``dataprep``, build ``IndexSet``\\ s, split pre/post.

    ``balance`` refuses a duplicate or absent unit-period. ``dataprep``
    supplies the outcome matrix, the time labels, the donor order and the
    pre-period count, and refuses a missing or non-finite outcome anywhere in
    the panel, a non-binary treatment and a treatment that switches off after
    it starts.

    Returns
    -------
    RESCMInputs
        Pure-NumPy container: treated vector ``y`` (T,), donor matrix ``X``
        (T, N), ``T0``, and unit/time :class:`IndexSet`\\ s.

    Raises
    ------
    MlsynthDataError
        If the panel is unbalanced or has a duplicate unit-period, an outcome
        value is missing or not finite, the treatment is not binary or switches
        off, there is not exactly one treated unit, or there are fewer than two
        pre-treatment periods.
    """
    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being taken again.
    keys = balance(df, unitid, time)
    treated_unit, intervention_time = derive_treatment(df, unitid, time, treat)
    # A lone treated unit with no donor reached the engine before, so dataprep
    # is asked not to refuse it here either.
    prep = dataprep(df, unitid, time, outcome, treat, allow_no_donors=True,
                    keys=keys)

    times = np.asarray(prep.time_labels)
    time_index = IndexSet.from_labels(times)

    wide = prep.Ywide
    donors = list(prep.donor_names)
    unit_index = IndexSet.from_labels(donors)

    # The solvers' sums depend on the memory layout. These are the layouts the
    # old df.pivot gave under pandas 2.x (the version CI runs): a contiguous
    # treated series and a column-major donor matrix. dataprep's matrix is
    # row-major.
    y = np.ascontiguousarray(wide[treated_unit].to_numpy(dtype=float))
    X = np.asfortranarray(wide[donors].to_numpy(dtype=float))
    T0 = int(prep.pre_periods)
    if T0 < 2:
        raise MlsynthDataError("RESCM needs at least two pre-treatment periods.")

    return RESCMInputs(
        unit_index=unit_index,
        time_index=time_index,
        y=y,
        X=X,
        T0=T0,
        treated_label=treated_unit,
        metadata={"outcome": outcome, "intervention_time": intervention_time},
    )
