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

The outcome matrix, the time labels and each directly treated unit's adoption
period come from :func:`mlsynth.utils.datautils.balance` and
:func:`mlsynth.utils.datautils.dataprep`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from ..prepared import SinglePanel
from .spatial import row_standardize, validate_spatial_matrix
from .structures import SpSyDiDInputs


_SPILLOVER_TOL = 1e-12


def _adoption_starts(prep) -> dict:
    """Each directly treated unit's first treated period, as an index into the time axis.

    ``dataprep`` has already refused a treatment that switches off, so a unit
    is treated from its start to the end of the panel.
    """
    if isinstance(prep, SinglePanel):
        return {prep.treated_unit_name: int(prep.pre_periods)}
    return {unit: int(cohort.pre_periods)
            for cohort in prep.cohorts.values() for unit in cohort.treated_units}


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
    """Build :class:`SpSyDiDInputs` from a long panel through ``balance`` and ``dataprep``.

    ``dataprep`` supplies the outcome matrix, the time labels and each
    directly treated unit's adoption period, and the treatment matrix is built
    from the adoption periods. ``balance`` and ``dataprep`` refuse a missing
    or non-finite outcome, an absent or duplicate unit-period, a non-binary
    or missing treatment value, a treatment that switches off after it
    starts and a single treated unit treated in the first period. SpSyDiD's
    own checks (``unit_order``, the spatial matrix, a pure control, no unit
    treated in the first period) stay on top. Staggered
    adoption is accepted: ``T0`` is the first period any unit is treated.

    Parameters
    ----------
    df : pd.DataFrame
        Balanced long panel with columns ``unitid``, ``time``, ``outcome``,
        ``treat``; ``treat`` is a 0/1 indicator that stays 1 once it starts.
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

    if unit_order is not None:
        unit_names = list(unit_order)
        unit_set_panel = set(df[unitid].unique())
        if set(unit_names) != unit_set_panel:
            raise MlsynthDataError(
                "unit_order does not match the set of unit ids in the panel."
            )

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # reused by dataprep instead of being built twice.
    keys = balance(df, unitid, time)
    prep = dataprep(df, unitid, time, outcome, treat, keys=keys)
    if unit_order is None:
        unit_names = sorted(np.asarray(prep.Ywide.columns))
    N = len(unit_names)

    # Built from the index's array one label at a time, as the old setup's
    # sorted list was: datetime periods stay an object array of Timestamps and
    # 32-bit periods stay 32-bit (iterating the index itself yields Python
    # scalars, which widen them).
    time_labels = np.array(list(prep.time_labels.array))
    T = int(time_labels.size)

    # Unit by period and C-ordered, as the old pivot was: the exposure W D and
    # the regression sum over these layouts.
    outcome_wide = np.ascontiguousarray(prep.Ywide[unit_names].to_numpy(dtype=float).T)
    starts = _adoption_starts(prep)
    treat_wide = np.zeros((N, T), dtype=float)
    for i, name in enumerate(unit_names):
        if name in starts:
            treat_wide[i, starts[name]:] = 1.0

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

    if pure_idx.size == 0:
        raise MlsynthDataError(
            "No pure controls left after partitioning -- every donor is "
            "exposed via the spatial matrix. SpSyDiD is not identified in "
            "this setting; consider widening the donor pool or restricting W."
        )

    # T0: the first period any unit is treated. dataprep refuses treatment in
    # the first period only when one unit is treated.
    T0 = min(starts.values())
    if T0 == 0:
        raise MlsynthDataError(
            "Some unit is treated at the earliest period (no pre-period)."
        )

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
