"""Panel ingestion for the GSYNTH estimator.

Outcomes come through :func:`mlsynth.utils.datautils.dataprep`, which supplies
the wide ``time x unit`` matrix, the sorted period labels and the adoption
cohorts. The treatment indicator and any covariates are aligned to that same
column and row order, so the arrays the estimator sees cannot disagree with each
other about which column is which unit.

Three structural requirements go beyond what ``dataprep`` enforces, and each is
checked here with its own message:

* some unit is never treated -- the factor space comes from those units alone,
  so a design where everyone is eventually treated has nothing to fit on;
* adoption is absorbing -- Step 2 projects onto one contiguous pre-period per
  treated unit, which an on-and-off treatment does not define;
* every treated unit has at least one pre-treatment period, since a unit that is
  treated from the first period has no history to recover a loading from.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import GSYNTHInputs


def _wide(df: pd.DataFrame, column: str, unit_col: str,
          time_col: str) -> pd.DataFrame:
    """Pivot ``column`` to ``time x unit``, rejecting any unobserved cell."""
    wide = df.pivot(index=time_col, columns=unit_col, values=column).sort_index()
    if wide.isna().to_numpy().any():
        raise MlsynthDataError(
            f"Column {column!r} is missing for some unit-period cells; GSYNTH "
            f"needs a fully observed panel."
        )
    return wide


def _aligned(wide: pd.DataFrame, units: Sequence, times: Sequence) -> np.ndarray:
    """Put a pivoted frame in the outcome panel's row and column order."""
    return wide.reindex(index=list(times), columns=list(units)).to_numpy(dtype=float)


def prepare_gsynth_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[List[str]] = None,
) -> GSYNTHInputs:
    """Assemble the arrays Xu (2017) Steps 1-3 run on.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel with one row per ``(unit, period)``.
    outcome, treat, unitid, time : str
        Column names.
    covariates : list of str, optional
        Time-varying covariate columns.

    Returns
    -------
    GSYNTHInputs

    Raises
    ------
    MlsynthDataError
        If a named column is absent, the panel is unbalanced, no unit is ever
        treated, no unit is never treated, adoption is not absorbing, or a
        treated unit has no pre-treatment period.
    """
    required = [outcome, treat, unitid, time] + list(covariates or [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Required column(s) {missing} not found in the panel."
        )

    balance(df, unitid, time)

    # The treatment structure is validated before ``dataprep`` runs, so a
    # design GSC cannot serve is reported in GSC's own terms. ``dataprep`` also
    # rejects non-absorbing treatment, and reaching it first would answer a
    # different question than the one the caller asked.
    dwide = _wide(df, treat, unitid, time)
    labels, D_raw = list(dwide.columns), dwide.to_numpy()
    if not np.isin(D_raw, (0, 1)).all():
        raise MlsynthDataError(
            f"Treatment column {treat!r} must be 0/1; GSYNTH reads it as an "
            f"adoption indicator."
        )
    D_raw = D_raw.astype(int)

    if D_raw.max() == 0:
        raise MlsynthDataError(
            "No treated units found: the treatment indicator is zero "
            "everywhere."
        )
    if (D_raw.max(axis=0) > 0).all():
        raise MlsynthDataError(
            "GSYNTH estimates the factor space from never-treated units, and "
            "every unit in this panel is eventually treated. Use an estimator "
            "that identifies the factors off not-yet-treated units instead."
        )
    falls = np.diff(D_raw, axis=0) < 0
    if falls.any():
        bad = [labels[j] for j in np.flatnonzero(falls.any(axis=0))]
        raise MlsynthDataError(
            f"GSYNTH needs absorbing adoption -- treatment that turns on and "
            f"stays on -- because each treated unit's loadings are fit on one "
            f"contiguous pre-treatment block. Treatment reverses for {bad}."
        )
    early = [labels[j] for j in np.flatnonzero(D_raw[0] > 0)]
    if early:
        raise MlsynthDataError(
            f"Every treated unit needs at least one pre-treatment period to "
            f"recover its factor loadings from; {early} are treated from the "
            f"first period."
        )

    prep = dataprep(df, unitid, time, outcome, treat)
    ywide = prep["Ywide"]
    units, times = list(ywide.columns), list(ywide.index)
    Y = ywide.to_numpy(dtype=float)

    D = _aligned(dwide, units, times).astype(int)
    treated_index = np.flatnonzero(D.max(axis=0) > 0)
    control_index = np.flatnonzero(D.max(axis=0) == 0)
    adoption_index = np.array([int(np.argmax(D[:, j] > 0)) for j in treated_index],
                              dtype=int)

    names = tuple(covariates or ())
    if names:
        X = np.stack([_aligned(_wide(df, c, unitid, time), units, times)
                      for c in names], axis=2)
    else:
        X = np.empty((Y.shape[0], Y.shape[1], 0))

    return GSYNTHInputs(
        Y=Y, D=D, X=X,
        treated_index=treated_index,
        control_index=control_index,
        adoption_index=adoption_index,
        unit_labels=np.asarray(units, dtype=object),
        time_labels=np.asarray(times),
        covariate_names=names,
    )
