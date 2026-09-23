"""Assemble the ``(J, T, K)`` compositional panel for COMPSC.

Each share column is pivoted through :func:`mlsynth.utils.datautils.dataprep`
(the canonical ingestion) and the ``K`` wide frames are stacked with the
paper's row convention: the treated unit first, donors after. Rows are closed
to sum to one -- compositions are scale-free, so unnormalized counts are
accepted -- and Assumption 1 (strict positivity) is enforced, optionally after
the standard zero-replacement device.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import CompscInputs


def prepare_compsc_inputs(
    df: pd.DataFrame, outcomes: List[str], treat: str, unitid: str, time: str,
    target: str, baseline: Optional[str] = None, zero_replacement: float = 1e-6,
) -> CompscInputs:
    """Build a :class:`CompscInputs` from a long compositional panel.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form balanced panel.
    outcomes : list of str
        The ``K`` share columns, in the desired slice order.
    treat, unitid, time : str
        Column names for the treatment indicator, unit id, and time.
    target : str
        Share whose effect drives the flat accessors.
    baseline : str, optional
        Reference category for ``geometry="alr"``; defaults to the last outcome.
    zero_replacement : float
        Constant added to zero shares before closure. ``0`` rejects zeros.

    Returns
    -------
    CompscInputs

    Raises
    ------
    MlsynthDataError
        If the panel has no donors, more than one treated unit, no
        pre-treatment periods, negative shares, or zeros that cannot be
        repaired.
    """
    keys = balance(df, unitid, time)

    prep0 = dataprep(df, unitid, time, outcomes[0], treat, allow_no_donors=True,
                     keys=keys)
    Ywide0 = prep0["Ywide"]                       # (T, N) frame
    time_labels = np.asarray(prep0["time_labels"])
    units = list(Ywide0.columns)

    W = (df.pivot(index=time, columns=unitid, values=treat)
           .reindex(index=Ywide0.index, columns=units))
    if W.isna().any().any():  # pragma: no cover - balance() guards imbalance first
        raise MlsynthDataError(
            "Treatment indicator has gaps after pivoting; panel must be "
            "balanced with a defined treatment status for every unit-time.")
    Wv = W.to_numpy()
    if not np.isin(Wv, (0, 1)).all():  # pragma: no cover - dataprep guards 0/1
        raise MlsynthDataError("Treatment indicator must be 0/1.")

    treated_mask = Wv.max(axis=0) > 0
    n_treated = int(treated_mask.sum())
    if n_treated == 0:  # pragma: no cover - dataprep guards treatment variation
        raise MlsynthDataError("No treated units: treatment never switches on.")
    if n_treated > 1:
        raise MlsynthDataError(
            f"COMPSC requires a single treated unit; found {n_treated}. The "
            "estimator is built for one aggregate unit receiving the "
            "intervention (use PROPSC for a simultaneous treated block).")
    if treated_mask.all():
        raise MlsynthDataError("No donor units: every unit is treated.")

    treated_periods = np.where(Wv.max(axis=1) > 0)[0]
    T0 = int(treated_periods.min())
    if T0 == 0:  # pragma: no cover - dataprep rejects a zero-length pre-period first
        raise MlsynthDataError("Treatment starts at t=0; no pre-treatment periods.")

    units_arr = np.asarray(units, dtype=object)
    treated_label = units_arr[treated_mask][0]
    donor_labels = list(units_arr[~treated_mask])
    order = [treated_label] + donor_labels

    slices = []
    for c in outcomes:
        prep = dataprep(df, unitid, time, c, treat, allow_no_donors=True)
        Yc = prep["Ywide"].reindex(index=Ywide0.index, columns=order)
        if Yc.isna().any().any():
            raise MlsynthDataError(f"Missing values in outcome {c!r} after pivot.")
        slices.append(Yc.to_numpy().T)           # (J, T)
    A = np.stack(slices, axis=2)                  # (J, T, K)

    A = _validate_and_close(A, outcomes, order, zero_replacement)

    return CompscInputs(
        P=A, T0=T0, outcomes=list(outcomes), treated_label=treated_label,
        donor_labels=donor_labels, time_labels=time_labels,
        target_index=outcomes.index(target),
        baseline_index=outcomes.index(baseline if baseline is not None
                                      else outcomes[-1]),
    )


def _validate_and_close(A: np.ndarray, outcomes, order, zero_replacement: float):
    """Enforce Assumption 1 and renormalize each row to the simplex."""
    if not np.isfinite(A).all():
        raise MlsynthDataError("Composition contains non-finite values.")
    if (A < 0).any():
        j, t, k = np.argwhere(A < 0)[0]
        raise MlsynthDataError(
            f"Shares must be non-negative; unit {order[j]!r} at period index "
            f"{t} has a negative value in {outcomes[k]!r}.")

    totals = A.sum(axis=-1)
    if (totals <= 0).any():
        j, t = np.argwhere(totals <= 0)[0]
        raise MlsynthDataError(
            f"Composition for unit {order[j]!r} at period index {t} sums to "
            "zero; every unit-period must have a positive total.")

    if (A == 0).any():
        if zero_replacement <= 0:
            j, t, k = np.argwhere(A == 0)[0]
            raise MlsynthDataError(
                f"Composition has a zero share ({outcomes[k]!r} for unit "
                f"{order[j]!r} at period index {t}) and zero_replacement is "
                "disabled; log-ratios are undefined (Assumption 1). Set "
                "zero_replacement > 0 or drop the unit.")
        A = np.where(A == 0, zero_replacement, A)

    from .pipeline import closure
    return closure(A)
