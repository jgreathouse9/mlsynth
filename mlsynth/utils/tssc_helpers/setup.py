"""Data preparation for TSSC.

Wraps :func:`mlsynth.utils.datautils.dataprep` into the typed
:class:`TSSCInputs` container. No demeaning or scaling is applied -- the
SC-class regressions operate on the raw outcome levels (the intercept
variants absorb level differences directly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import TSSCInputs


def prepare_tssc_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
) -> TSSCInputs:
    """Pivot a long panel into the typed inputs the TSSC pipeline expects."""

    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "TSSC supports a single treated unit; the input panel appears "
            "to contain multiple treatment cohorts."
        )

    y = np.asarray(prepared["y"], dtype=float).ravel()
    donor_matrix = np.asarray(prepared["donor_matrix"], dtype=float)
    if donor_matrix.ndim != 2:
        raise MlsynthDataError(
            f"TSSC requires a 2D donor matrix; got shape {donor_matrix.shape}."
        )

    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    T2 = int(prepared["post_periods"])
    n_donors = donor_matrix.shape[1]

    if n_donors < 1:
        raise MlsynthDataError("TSSC requires at least one donor unit.")
    if T0 < 2:
        raise MlsynthDataError("TSSC requires at least two pre-treatment periods.")

    return TSSCInputs(
        y=y,
        donor_matrix=donor_matrix,
        donor_names=list(prepared["donor_names"]),
        T0=T0,
        T2=T2,
        T=T,
        time_labels=np.asarray(prepared["time_labels"]),
        treated_unit_name=prepared["treated_unit_name"],
    )
