"""Input preparation for the Synthetic Interventions (SI) estimator.

Builds an :class:`~mlsynth.utils.si_helpers.structures.SIInputs` from a long
panel: the focal target unit (flagged by ``treat``) plus one donor pool per
alternative intervention column in ``inters``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import SIDonorPool, SIInputs


def prepare_si_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
    inters: List[str],
) -> SIInputs:
    """Prepare focal-unit data and per-intervention donor pools.

    Parameters
    ----------
    df : pandas.DataFrame
        Balanced long panel.
    outcome, unitid, time, treat : str
        Column names. ``treat`` flags the focal target unit's treatment timing
        (defining the common pre-period ``T0``).
    inters : list of str
        Binary indicator columns; for each, the units flagged ``1`` form that
        intervention's donor pool.

    Returns
    -------
    SIInputs
    """
    prepped = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepped and "pre_periods" not in prepped:
        raise MlsynthConfigError(
            "SI expects a single focal treated unit; the 'treat' column yielded "
            "multiple adoption cohorts. Flag exactly one focal unit."
        )

    Ywide = prepped["Ywide"]
    y_target = np.asarray(prepped["y"], dtype=float)
    T0 = int(prepped["pre_periods"])
    treated_unit_name = prepped["treated_unit_name"]
    time_labels = np.asarray(Ywide.index.to_numpy())

    pools = {}
    for col in inters:
        donor_units = set(df.loc[df[col] == 1, unitid])
        donor_units.discard(treated_unit_name)
        donor_names = [c for c in Ywide.columns if c in donor_units]
        if not donor_names:
            raise MlsynthDataError(
                f"Intervention '{col}' has no donor units (no unit with "
                f"'{col}' == 1 other than the focal unit)."
            )
        matrix = Ywide[donor_names].to_numpy(dtype=float)
        pools[col] = SIDonorPool(
            name=col, matrix=matrix, names=[str(n) for n in donor_names]
        )

    return SIInputs(
        treated_unit_name=treated_unit_name,
        y_target=y_target,
        T0=T0,
        time_labels=time_labels,
        pools=pools,
    )
