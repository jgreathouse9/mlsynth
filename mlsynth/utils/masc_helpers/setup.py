"""Read a long panel into the matrices the MASC estimator consumes.

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
from .structures import MASCInputs


def _covariate_panel(df: pd.DataFrame, unitid: str, time: str, treat: str,
                     column: str, keys) -> pd.DataFrame:
    """One covariate as a period-by-unit frame, each unit's gaps filled.

    The covariate is read as the outcome of its own ``dataprep`` call on the
    same panel, so its axes are the outcome matrix's. A covariate need not be
    observed in every period, and MASC fills each unit's gaps forward and then
    backward (the Basque sector shares are observed in a handful of years), so
    a NaN cell is kept (``missing_outcomes="keep"``); an infinite value is
    refused under the covariate's name.
    """
    try:
        prep = dataprep(df, unitid, time, column, treat, keys=keys,
                        missing_outcomes="keep")
    except MlsynthDataError as exc:
        raise MlsynthDataError(
            f"Covariate {column!r} cannot be read; MASC needs every covariate "
            f"finite where it is observed. {exc}"
        ) from exc
    return prep.Ywide.ffill().bfill()


def prepare_masc_inputs(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
    covariate_windows: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> MASCInputs:
    """Build MASC's ``(Y_treated, Y_donors, treatment_period)`` through ``dataprep``.

    Single-treated-unit only for v1. The treated unit is the panel
    unit with any ``treat == 1`` row; the donor pool is every other
    unit, in the order units first appear in ``df``. ``treatment_period``
    is the 1-indexed position of the first treated period (matches the R
    reference's ``treatment`` argument).

    ``balance`` and ``dataprep`` refuse a missing or non-finite outcome, an
    absent or duplicate unit-period, a non-binary treatment and a treatment
    that switches off after it starts. MASC's own checks (one treated unit,
    two pre-periods, the covariate columns) stay on top.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)
    if not isinstance(prep, SinglePanel):
        n_treated = sum(len(c.treated_units) for c in prep.cohorts.values())
        raise MlsynthDataError(
            "MASC currently supports a single treated unit; found "
            f"{n_treated}."
        )
    treated_label = prep.treated_unit_name
    # dataprep orders the donors by label; the old setup listed them in the
    # order units first appear in the frame, and the weights keep that order.
    never_treated = [u for u in df[unitid].unique() if u != treated_label]

    time_index = prep.time_labels.to_numpy()
    T0 = int(prep.pre_periods)
    intervention_time = prep.time_labels[T0]
    # The fit's sums depend on the memory layout. These are the layouts the old
    # df.pivot gave under pandas 2.x (the version CI runs): a contiguous
    # treated series and a column-major donor matrix. dataprep's matrix is
    # row-major, which would move the weights by about 1e-12.
    Y_treated = np.ascontiguousarray(prep.Ywide[treated_label].to_numpy(dtype=float))
    Y_donors = np.asfortranarray(prep.Ywide[never_treated].to_numpy(dtype=float))

    treatment_period = T0 + 1
    T = int(time_index.size)
    T1 = T - T0
    if T0 < 2:
        raise MlsynthDataError(
            f"MASC needs at least 2 pre-treatment periods; got T0={T0}."
        )

    # Covariate panels (optional). Stored as ``(T, P)`` for the treated unit
    # and ``(T, J, P)`` for donors so CV folds can aggregate within their own
    # pre-window. ``covariate_windows`` (optional) overrides the aggregation
    # range per covariate -- defaults to "1..fold_treatment - 1" inside
    # downstream callers.
    cov_treated_panel: Optional[np.ndarray] = None
    cov_donors_panel: Optional[np.ndarray] = None
    if covariates:
        treated_stacks: List[np.ndarray] = []
        donor_stacks: List[np.ndarray] = []
        for c in covariates:
            if c not in df.columns:
                raise MlsynthDataError(f"Covariate {c!r} not found in df.columns.")
            cwide = _covariate_panel(df, unitid, time, treat, c, keys)
            if cwide[[treated_label, *never_treated]].isna().any().any():
                raise MlsynthDataError(
                    f"Covariate {c!r} is missing for some unit-period after "
                    "forward/back fill; cannot aggregate."
                )
            treated_stacks.append(
                cwide[treated_label].to_numpy(dtype=float)
            )
            donor_stacks.append(
                cwide[list(never_treated)].to_numpy(dtype=float)
            )
        cov_treated_panel = np.column_stack(treated_stacks)         # (T, P)
        cov_donors_panel = np.stack(donor_stacks, axis=-1)          # (T, J, P)

    return MASCInputs(
        Y_treated=Y_treated,
        Y_donors=Y_donors,
        treated_label=treated_label,
        donor_labels=tuple(never_treated),
        time_index=time_index,
        intervention_time=intervention_time,
        treatment_period=treatment_period,
        T=T,
        T0=T0,
        T1=T1,
        J=Y_donors.shape[1],
        cov_treated_panel=cov_treated_panel,
        cov_donors_panel=cov_donors_panel,
        covariate_names=tuple(covariates or ()),
        covariate_windows=dict(covariate_windows or {}),
    )
