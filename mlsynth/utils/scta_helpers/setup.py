"""Long-DataFrame -> NumPy boundary for SCTA (the only pandas touchpoint).

Wraps the canonical :func:`mlsynth.utils.datautils.dataprep` contract
(``Ywide`` / ``y`` / ``donor_matrix`` / ``pre_periods``) into the NumPy-only
:class:`SCTAInputs` consumed by the engine.
"""

from __future__ import annotations

import pandas as pd

from ...exceptions import MlsynthConfigError
from ..datautils import dataprep
from .structures import SCTAInputs


def prepare_scta_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treat: str,
    block_length: int,
) -> SCTAInputs:
    """Ingest the panel via ``dataprep`` and assemble :class:`SCTAInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced high-frequency panel (one row per unit-period).
    unitid, time, outcome, treat : str
        Column names for unit id, period, outcome, and treatment indicator.
    block_length : int
        Aggregation block length ``K``. At least one whole block must fit in the
        pre-period (``T0 >= K``); the leading ``floor(T0 / K)`` complete blocks
        are aggregated and every disaggregated pre-period is retained, so a
        ragged tail (``T0`` not a multiple of ``K``) is kept disaggregated --
        matching the paper's Texas construction (6 whole years + 3 spare months).

    Returns
    -------
    SCTAInputs
        Pure-NumPy container for the estimation engine.

    Raises
    ------
    MlsynthConfigError
        If the pre-period is shorter than one block (``T0 < block_length``).
    """
    prep = dataprep(df, unitid, time, outcome, treat)
    T0 = int(prep["pre_periods"])
    if T0 < block_length:
        raise MlsynthConfigError(
            f"Pre-period length T0={T0} is shorter than block_length="
            f"{block_length}; need at least one whole block to aggregate."
        )
    return SCTAInputs(
        treated_name=prep["treated_unit_name"],
        donor_names=list(prep["donor_names"]),
        y=prep["y"].astype(float).ravel(),
        donor_matrix=prep["donor_matrix"].astype(float),
        T0=T0,
        block_length=int(block_length),
        time_labels=list(prep["time_labels"]),
        metadata={"outcome": outcome},
    )
