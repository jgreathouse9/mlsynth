"""Long-DataFrame -> NumPy boundary for SCUL (the only pandas touchpoint).

Wraps :func:`mlsynth.utils.datautils.dataprep` and expands the donor pool with
any extra ``donor_variables`` to build SCUL's wide, multi-type donor matrix.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import SCULInputs


def prepare_scul_inputs(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treat: str,
    donor_variables: Optional[List[str]] = None,
) -> SCULInputs:
    """Ingest the panel via ``dataprep`` and assemble :class:`SCULInputs`.

    The donor pool is the donor units' ``outcome`` series, optionally widened by
    one block of all donor units per name in ``donor_variables`` (the
    high-dimensional, multi-type pool). The treated unit's own predictors are
    never placed in the pool.

    Raises
    ------
    MlsynthConfigError
        If a requested ``donor_variables`` column is absent from ``df``.
    MlsynthDataError
        If a donor variable has missing values over the donor/time panel.
    """
    prep = dataprep(df, unitid, time, outcome, treat)
    donor_names = list(prep["donor_names"])
    time_labels = list(prep["time_labels"])
    y = prep["y"].astype(float).ravel()
    donor_outcome = prep["donor_matrix"].astype(float)        # (T, J) outcome block

    blocks = [donor_outcome]
    col_unit = list(donor_names)
    col_variable = [outcome] * len(donor_names)

    for var in (donor_variables or []):
        if var not in df.columns:
            raise MlsynthConfigError(
                f"donor_variables column '{var}' not found in the panel."
            )
        wide = df.pivot(index=time, columns=unitid, values=var)
        try:
            block = wide.loc[time_labels, donor_names].to_numpy(float)
        except KeyError as exc:  # pragma: no cover - defensive (donor/time misalignment)
            raise MlsynthDataError(
                f"donor variable '{var}' is not aligned with the donor/time panel."
            ) from exc
        if not np.isfinite(block).all():
            raise MlsynthDataError(
                f"donor variable '{var}' has missing values over the donor panel."
            )
        blocks.append(block)
        col_unit.extend(donor_names)
        col_variable.extend([var] * len(donor_names))

    donor_matrix = np.hstack(blocks)
    return SCULInputs(
        treated_name=prep["treated_unit_name"],
        donor_names=donor_names,
        y=y,
        donor_matrix=donor_matrix,
        col_unit=np.asarray(col_unit, dtype=object),
        col_variable=col_variable,
        donor_outcome=donor_outcome,
        T0=int(prep["pre_periods"]),
        time_labels=time_labels,
        metadata={"outcome": outcome, "donor_variables": list(donor_variables or [])},
    )
