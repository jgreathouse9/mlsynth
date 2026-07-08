"""Assemble the ``(N, T, K)`` compositional panel for PROPSC.

Each proportion is pivoted through :func:`mlsynth.utils.datautils.dataprep`
(the canonical ingestion), and the ``K`` wide frames are stacked into one array
with the ``propsdid`` row convention: controls first, treated last. The
treatment block (``N0`` controls, ``T0`` pre-periods) is derived once and
required to be a single simultaneous-adoption block, matching the estimator's
identifying design.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import PropscInputs


def prepare_propsc_inputs(
    df: pd.DataFrame, outcomes: List[str], treat: str, unitid: str, time: str,
    target: str,
) -> PropscInputs:
    """Build a :class:`PropscInputs` from a long compositional panel.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form balanced panel.
    outcomes : list of str
        The ``K`` proportion columns, in the desired slice order.
    treat, unitid, time : str
        Column names for the treatment indicator, unit id, and time.
    target : str
        Proportion whose effect drives the flat accessors.

    Returns
    -------
    PropscInputs
    """
    balance(df, unitid, time)

    # Canonical time/unit axes from the first proportion's pivot.
    prep0 = dataprep(df, unitid, time, outcomes[0], treat, allow_no_donors=True)
    Ywide0 = prep0["Ywide"]                      # (T, N) frame
    time_labels = np.asarray(prep0["time_labels"])
    units = list(Ywide0.columns)

    # Treatment matrix aligned to the same (time x unit) axes.
    W = (df.pivot(index=time, columns=unitid, values=treat)
           .reindex(index=Ywide0.index, columns=units))
    if W.isna().any().any():  # pragma: no cover - balance() guards imbalance first
        raise MlsynthDataError(
            "Treatment indicator has gaps after pivoting; panel must be "
            "balanced with a defined treatment status for every unit-time.")
    Wv = W.to_numpy()
    if not np.isin(Wv, (0, 1)).all():  # pragma: no cover - dataprep guards 0/1
        raise MlsynthDataError("Treatment indicator must be 0/1.")

    treated_mask = Wv.max(axis=0) > 0          # units ever treated
    if not treated_mask.any():  # pragma: no cover - dataprep guards variation
        raise MlsynthDataError("No treated units: treatment never switches on.")
    treated_periods = np.where(Wv.max(axis=1) > 0)[0]
    T0 = int(treated_periods.min())            # leading periods with no treatment
    if T0 == 0:
        raise MlsynthDataError("Treatment starts at t=0; no pre-treatment periods.")

    # Require a single simultaneous-adoption block (the estimator's design):
    # every ever-treated unit switches on together at T0 and stays on.
    control_cols = ~treated_mask
    if Wv[:T0, treated_mask].any() or not Wv[T0:, treated_mask].all():
        raise MlsynthDataError(
            "Treatment adoption is not simultaneous; PROPSC requires all "
            "treated units to adopt in the same period with no reversals.")

    # Row order: controls first (sorted), treated last (sorted).
    units_arr = np.asarray(units, dtype=object)
    donor_labels = list(units_arr[control_cols])
    treated_labels = list(units_arr[treated_mask])
    order = donor_labels + treated_labels
    N0 = len(donor_labels)

    # Stack each proportion's (N, T) matrix in the shared row/column order.
    slices = []
    for c in outcomes:
        prep = dataprep(df, unitid, time, c, treat, allow_no_donors=True)
        Yc = prep["Ywide"].reindex(index=Ywide0.index, columns=order)
        if Yc.isna().any().any():
            raise MlsynthDataError(f"Missing values in outcome {c!r} after pivot.")
        slices.append(Yc.to_numpy().T)          # (N, T)
    Y = np.stack(slices, axis=2)                 # (N, T, K)

    return PropscInputs(
        Y=Y, N0=N0, T0=T0, outcomes=list(outcomes),
        donor_labels=donor_labels, treated_labels=treated_labels,
        time_labels=time_labels, target_index=outcomes.index(target),
    )
