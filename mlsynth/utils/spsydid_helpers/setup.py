"""Micro-panel data preparation for SpSyDiD.

Converts a long-format panel + a spatial weight matrix into the
:class:`SpSyDiDInputs` container expected by :func:`run_spsydid`. The
donor pool is auto-partitioned into three classes following Serenini
& Masek (2024):

* **Directly treated** -- units with :math:`D_{it} = 1` for some t.
* **Indirectly treated (spillover-exposed)** -- units with
  :math:`D_{it} = 0` for all t but :math:`(WD)_{it} > 0` for some t,
  i.e. they have at least one spatial neighbour who is treated.
* **Pure controls** -- units with :math:`D = 0` and :math:`(WD) = 0`
  for all t. Only these are used to fit the SDID unit / time weights.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .spatial import row_standardize, validate_spatial_matrix
from .structures import SpSyDiDInputs


_SPILLOVER_TOL = 1e-12


def prepare_spsydid_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    spatial_matrix: np.ndarray,
    unit_order: Optional[Sequence] = None,
    row_standardize_spatial: bool = True,
) -> SpSyDiDInputs:
    """Pivot a long-format panel into :class:`SpSyDiDInputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Balanced long panel with columns ``unitid``, ``time``, ``outcome``,
        ``treat``.
    outcome, treat, unitid, time : str
        Column names.
    spatial_matrix : np.ndarray
        Square ``(N, N)`` spatial weight matrix. Rows / columns must be
        ordered consistently with the units in the panel (use
        ``unit_order`` to fix the ordering; otherwise sorted unique
        ``unitid`` values are used).
    unit_order : sequence, optional
        Canonical ordering of unit ids that matches the rows / columns
        of ``spatial_matrix``. If ``None`` (default), units are ordered
        by ``sorted(df[unitid].unique())``.
    row_standardize_spatial : bool
        If True (default), row-standardise ``W`` before storing it.
        Skip when the caller has already standardised.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(
                f"Required column {col!r} missing from input DataFrame."
            )
    if df[outcome].isna().any():
        raise MlsynthDataError("Outcome column contains NaN values.")
    if df[treat].isna().any():
        raise MlsynthDataError("Treatment column contains NaN values.")

    if unit_order is None:
        unit_names: List = sorted(df[unitid].unique())
    else:
        unit_names = list(unit_order)
        unit_set_panel = set(df[unitid].unique())
        if set(unit_names) != unit_set_panel:
            raise MlsynthDataError(
                "unit_order does not match the set of unit ids in the panel."
            )
    N = len(unit_names)

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)

    # Wide pivots: outcome and treatment.
    outcome_wide = df.pivot(index=unitid, columns=time, values=outcome)
    treat_wide = df.pivot(index=unitid, columns=time, values=treat)
    # A missing (unit, time) cell leaves a NaN in the pivot (the shape stays
    # (N, T) because the index / columns come from the same unique values),
    # so detect imbalance by checking for those gaps rather than by shape.
    if (
        outcome_wide.isna().to_numpy().any()
        or treat_wide.isna().to_numpy().any()
    ):
        raise MlsynthDataError(
            "Panel is not balanced -- some (unit, time) cells are missing."
        )
    outcome_wide = outcome_wide.loc[unit_names, time_labels].to_numpy(dtype=float)
    treat_wide = treat_wide.loc[unit_names, time_labels].to_numpy(dtype=float)

    # Treatment must be a 0/1 indicator: non-binary values silently corrupt
    # the partition (``treat > 0``) and push the exposure term (WD) outside
    # the [0, 1] range the spillover coefficient is defined against.
    unique_treat = np.unique(treat_wide)
    if not np.all(np.isin(unique_treat, (0.0, 1.0))):
        raise MlsynthDataError(
            "Treatment column must be a binary 0/1 indicator; found values "
            f"{unique_treat.tolist()}."
        )

    # Validate + (optionally) row-standardise W.
    W = validate_spatial_matrix(spatial_matrix, n_units=N)
    if row_standardize_spatial:
        W = row_standardize(W, warn_isolated=True)

    # Spillover exposure (WD)_it = sum_j w_ij * D_jt.
    exposure = W @ treat_wide   # shape (N, T)

    # Auto-detect partition.
    direct_mask = (treat_wide > 0).any(axis=1)
    spillover_mask = (~direct_mask) & ((exposure > _SPILLOVER_TOL).any(axis=1))
    pure_mask = ~(direct_mask | spillover_mask)

    direct_idx = np.where(direct_mask)[0]
    spillover_idx = np.where(spillover_mask)[0]
    pure_idx = np.where(pure_mask)[0]

    if direct_idx.size == 0:
        raise MlsynthDataError("No directly treated units found (all D == 0).")
    if pure_idx.size == 0:
        raise MlsynthDataError(
            "No pure controls left after partitioning -- every donor is "
            "exposed via the spatial matrix. SpSyDiD is not identified in "
            "this setting; consider widening the donor pool or restricting W."
        )

    # T0: largest pre-period such that no unit has treat==1 at any t' <= T0.
    any_treated_at_t = (treat_wide > 0).any(axis=0)
    if any_treated_at_t[0]:
        raise MlsynthDataError(
            "Some unit is treated at the earliest period (no pre-period)."
        )
    T0 = int(np.argmax(any_treated_at_t)) if any_treated_at_t.any() else T
    # Defensive guards: by construction these are unreachable. ``T0 == 0``
    # is already caught by the ``any_treated_at_t[0]`` check above, and
    # ``T0 == T`` (no treated period) is impossible because the empty
    # ``direct_idx`` guard fires first. Retained for safety against future
    # refactors of the partition logic.
    if T0 < 1:  # pragma: no cover
        raise MlsynthDataError("SpSyDiD requires at least one pre-period.")
    if T - T0 < 1:  # pragma: no cover
        raise MlsynthDataError("SpSyDiD requires at least one post-period.")

    return SpSyDiDInputs(
        outcome_matrix=outcome_wide,
        treatment_matrix=treat_wide,
        spatial_matrix=W,
        exposure_matrix=exposure,
        unit_names=unit_names,
        time_labels=time_labels,
        T=T,
        T0=T0,
        direct_indices=direct_idx,
        spillover_indices=spillover_idx,
        pure_control_indices=pure_idx,
    )
