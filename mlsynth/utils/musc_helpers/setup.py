"""Long-DataFrame -> NumPy boundary for MUSC (the only pandas touchpoint).

The estimator reads the panel once through
:func:`mlsynth.utils.datautils.balance` and
:func:`mlsynth.utils.datautils.dataprep` (:func:`read_musc_panel`), and each
cohort's inputs are built from that matrix (:func:`cohort_inputs`), a
multi-unit cohort collapsed to its mean on the matrix itself.
:func:`prepare_musc_inputs`, which is given the treated unit and has no
treatment column, reads its panel through
:func:`mlsynth.utils.datautils.geoex_dataprep`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep, geoex_dataprep
from ..fast_scm_helpers.structure import IndexSet
from .structures import MUSCInputs


def _synthetic_cohort_label(treated_units: Sequence[Any]) -> str:
    """Build a stable, panel-safe label for a collapsed cohort."""
    return "_musc_cohort__" + "__".join(str(u) for u in treated_units)


def _inputs_from_wide(
    wide: pd.DataFrame,
    *,
    outcome: str,
    treated_unit: Any,
    intervention_time: Any,
    unitid: str = "",
) -> MUSCInputs:
    """Build :class:`MUSCInputs` from a ``(time x unit)`` outcome matrix.

    Units are the matrix's columns and periods its rows, in their order.
    """
    # Labels one by one from the axes' arrays, as the old pd.unique lists were,
    # so datetime periods stay datetime64.
    units = list(wide.columns.to_numpy())
    if treated_unit not in units:
        raise MlsynthDataError(
            f"treated_unit {treated_unit!r} not found in column '{unitid}'."
        )
    unit_index = IndexSet.from_labels(units)
    treated_idx = int(unit_index.get_index([treated_unit])[0])
    donor_idx = np.asarray(
        [i for i in range(len(units)) if i != treated_idx], dtype=int
    )
    if donor_idx.size < 3:
        raise MlsynthDataError(
            "MUSC requires at least 3 donor units (N - 1 >= 3); got "
            f"{donor_idx.size}."
        )

    times = list(wide.index.to_numpy())
    time_index = IndexSet.from_labels(times)
    # The old pivot's reindexed frame was one C-ordered (T, N) block, and
    # MUSC's Y is its transpose.
    Y = np.ascontiguousarray(wide.to_numpy(dtype=float)).T        # (N, T)

    if intervention_time not in times:
        raise MlsynthDataError(
            f"intervention_time {intervention_time!r} not in time column."
        )
    T0 = int(time_index.get_index([intervention_time])[0])
    if T0 < 2:
        raise MlsynthDataError(
            f"MUSC requires at least 2 pre-treatment periods; got T0={T0}."
        )
    # T0 indexes one of the periods, so at least one post-period remains.

    return MUSCInputs(
        unit_index=unit_index,
        time_index=time_index,
        treated_idx=int(treated_idx),
        donor_idx=donor_idx,
        Y=Y,
        T0=T0,
        metadata={
            "outcome": outcome,
            "treated_unit": treated_unit,
            "intervention_time": intervention_time,
        },
    )


def read_musc_panel(
    df: pd.DataFrame, *, unitid: str, time: str, outcome: str, treat: str,
) -> pd.DataFrame:
    """The panel's ``(time x unit)`` outcome matrix, read through ``dataprep``.

    ``balance`` refuses a duplicate or absent unit-period. ``dataprep`` refuses
    a missing or non-finite outcome anywhere in the panel, a non-binary
    treatment, a treatment that switches off after it starts, and one treated
    unit treated from the first period. Units and periods are sorted.
    """
    # balance()'s key scan is reused by dataprep instead of being taken again.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, allow_no_donors=True,
                    keys=keys)
    return prep.Ywide


def cohort_inputs(
    wide: pd.DataFrame,
    *,
    outcome: str,
    treated_units: Sequence[Any],
    intervention_time: Any,
    other_treated_units: Sequence[Any] = (),
    collapse: bool = False,
    unitid: str = "",
) -> MUSCInputs:
    """Build one cohort's :class:`MUSCInputs` from the panel's outcome matrix.

    Without ``collapse`` the cohort is a single unit and the matrix is used as
    it is. With ``collapse`` the cohort's units are replaced by one synthetic
    unit whose outcome is their mean in each period (the uniform-weight version
    of Bottmer et al. 2024 Appendix D.1), labelled by
    :func:`_synthetic_cohort_label` and placed after the donors, and the
    units of ``other_treated_units`` are dropped so only never-treated units
    serve as donors.
    """
    if not collapse:
        return _inputs_from_wide(wide, outcome=outcome, treated_unit=treated_units[0],
                                 intervention_time=intervention_time, unitid=unitid)
    label = _synthetic_cohort_label(treated_units)
    if label in set(wide.columns):
        raise MlsynthDataError(
            f"synthetic_label {label!r} collides with an existing unit label "
            "in the panel."
        )
    dropped = set(treated_units) | set(other_treated_units)
    donors = [u for u in wide.columns if u not in dropped]
    synthetic = wide[list(treated_units)].mean(axis=1)
    collapsed = pd.concat([wide[donors], synthetic.rename(label)], axis=1)
    return _inputs_from_wide(collapsed, outcome=outcome, treated_unit=label,
                             intervention_time=intervention_time, unitid=unitid)


def prepare_musc_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treated_unit: Any,
    intervention_time: Any,
) -> MUSCInputs:
    """Read the long panel through ``geoex_dataprep`` and build the ``MUSCInputs``.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel (one row per unit-period).
    unitid, time, outcome : str
        Column names for the unit id, period index, and outcome.
    treated_unit : Any
        Label of the treated unit.
    intervention_time : Any
        First treated period; pre-period is ``time < intervention_time``.

    Returns
    -------
    MUSCInputs
        Pure-NumPy container for the MUSC engine, units and periods sorted.

    Raises
    ------
    MlsynthDataError
        If the panel is unbalanced or has a duplicate unit-period, an outcome is
        missing or not finite, the treated unit or the intervention period is
        not in the panel, there are fewer than 3 donors, or fewer than 2
        pre-treatment periods.
    """
    wide = geoex_dataprep(df, unitid, time, outcome)["Ywide"]
    return _inputs_from_wide(wide, outcome=outcome, treated_unit=treated_unit,
                             intervention_time=intervention_time, unitid=unitid)
