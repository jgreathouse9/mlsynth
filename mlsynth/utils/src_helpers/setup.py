"""Long-DataFrame -> NumPy boundary for SRC (the only pandas touchpoint).

The outcome matrix, the treated unit, the time labels and the adoption period
come from :func:`mlsynth.utils.datautils.balance` and
:func:`mlsynth.utils.datautils.dataprep`, and each covariate panel from its own
``dataprep`` call (see :func:`_covariate_panel`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .structures import SRCInputs


def _covariate_panel(df: pd.DataFrame, unitid: str, time: str, treat: str,
                     column: str, keys) -> pd.DataFrame:
    """One covariate as a period-by-unit frame, each unit's gaps filled.

    The covariate is read as the outcome of its own ``dataprep`` call on the
    same panel, so its axes are the outcome matrix's. A covariate need not be
    observed in every period, and SRC fills each unit's gaps forward and then
    backward before it averages over the covariate's window (the Basque sector
    shares are observed in a handful of years), so a NaN cell is kept
    (``missing_outcomes="keep"``); an infinite value is refused under the
    covariate's name.
    """
    try:
        prep = dataprep(df, unitid, time, column, treat, keys=keys,
                        missing_outcomes="keep")
    except MlsynthDataError as exc:
        raise MlsynthDataError(
            f"Covariate {column!r} cannot be read; SRC needs every covariate "
            f"finite where it is observed. {exc}"
        ) from exc
    return prep.Ywide.ffill().bfill()


def prepare_src_inputs(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
    covariate_windows: Optional[Dict[str, Tuple[Any, Any]]] = None,
    fit_window: Optional[Tuple[Any, Any]] = None,
) -> SRCInputs:
    """Build SRC's ``(Y_treated, Y_donors, T0)`` matrices through ``dataprep``.

    Single treated unit only: the treated unit is the one with any ``treat == 1``
    row, the donor pool is every other unit, in the order units first appear in
    ``df``. Optional ``covariates`` are aggregated over their window
    (``covariate_windows``, else the whole pre-period). ``fit_window`` restricts
    the pre-treatment periods used as outcome matching rows.

    ``balance`` and ``dataprep`` refuse a missing or non-finite outcome, an
    absent or duplicate unit-period, a non-binary treatment and a treatment
    that switches off after it starts. SRC's own checks (one treated unit, two
    pre-periods, the covariate columns and windows, the fit window) stay on top.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by every dataprep call instead of being built again.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)
    if not isinstance(prep, SinglePanel):
        n_treated = sum(len(c.treated_units) for c in prep.cohorts.values())
        raise MlsynthDataError(
            f"SRC supports a single treated unit; found {n_treated}."
        )
    treated_label = prep.treated_unit_name
    # dataprep orders the donors by label; the old setup listed them in the
    # order units first appear in the frame, which downstream code reads
    # positionally.
    donors = [u for u in df[unitid].unique() if u != treated_label]

    time_index = prep.time_labels.to_numpy()
    T0 = int(prep.pre_periods)
    intervention_time = prep.time_labels[T0]
    # The fit's sums depend on the memory layout. These are the layouts the old
    # df.pivot gave under pandas 2.x (the version CI runs): a contiguous
    # treated series and a column-major donor matrix.
    Y_treated = np.ascontiguousarray(prep.Ywide[treated_label].to_numpy(dtype=float))
    Y_donors = np.asfortranarray(prep.Ywide[donors].to_numpy(dtype=float))

    treatment_period = T0 + 1
    T = int(time_index.size)
    T1 = T - T0
    if T0 < 2:
        raise MlsynthDataError(
            f"SRC needs at least 2 pre-treatment periods; got T0={T0}."
        )

    pre_mask = time_index < intervention_time
    windows = dict(covariate_windows or {})
    cov_treated: Optional[np.ndarray] = None
    cov_donors: Optional[np.ndarray] = None
    cov_names: List[Any] = list(covariates or [])
    if cov_names:
        treated_vals, donor_vals = [], []
        for c in cov_names:
            if c not in df.columns:
                raise MlsynthDataError(f"Covariate {c!r} not found in df.columns.")
            cwide = _covariate_panel(df, unitid, time, treat, c, keys)
            block = cwide[[treated_label, *donors]]
            if block.isna().any().any():
                raise MlsynthDataError(
                    f"Covariate '{c}' is entirely missing for some unit; cannot match."
                )
            if c in windows:
                start, end = windows[c]
                mask = (time_index >= start) & (time_index <= end)
                if not mask.any():
                    raise MlsynthDataError(
                        f"Covariate window {windows[c]!r} for '{c}' selects no periods."
                    )
            else:
                mask = pre_mask
            treated_vals.append(float(cwide.loc[mask, treated_label].mean()))
            donor_vals.append(cwide.loc[mask, donors].mean().to_numpy(dtype=float))
        cov_treated = np.asarray(treated_vals, dtype=float)       # (P,)
        cov_donors = np.vstack(donor_vals)                        # (P, J)

    # Outcome matching rows: all pre-periods, or a restricted fit window (SSR).
    pre_positions = np.where(pre_mask)[0]
    if fit_window is not None:
        start, end = fit_window
        fit_idx = np.array(
            [i for i in pre_positions if start <= time_index[i] <= end], dtype=int)
        if fit_idx.size < 2:
            raise MlsynthDataError(
                f"SRC fit_window {fit_window!r} selects fewer than 2 pre-treatment "
                "periods."
            )
    else:
        fit_idx = pre_positions.astype(int)

    return SRCInputs(
        Y_treated=Y_treated,
        Y_donors=Y_donors,
        treated_label=treated_label,
        donor_labels=tuple(donors),
        time_index=time_index,
        intervention_time=intervention_time,
        treatment_period=treatment_period,
        T=T, T0=T0, T1=T1, J=len(donors),
        cov_treated=cov_treated,
        cov_donors=cov_donors,
        covariate_names=tuple(cov_names),
        fit_idx=fit_idx,
    )
