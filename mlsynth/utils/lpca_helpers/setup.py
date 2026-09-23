"""Panel ingestion for the LPCA estimator."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import LPCAInputs


def resolve_match_periods(
    n_periods: int, match_periods: Optional[int], match_fraction: float
) -> int:
    """Number of leading periods reserved for matching."""
    if match_periods is not None:
        return int(match_periods)
    return int(round(n_periods * match_fraction))


def resolve_neighbours(n_units: int, n_neighbors: Optional[int]) -> int:
    """Neighbourhood size ``K``, defaulting to the paper's ``round(N^(2/3))``."""
    if n_neighbors is not None:
        return int(n_neighbors)
    return int(round(n_units ** (2 / 3)))


def prepare_lpca_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    *,
    match_periods: Optional[int] = None,
    match_fraction: float = 0.5,
) -> LPCAInputs:
    """Pivot a long panel into :class:`LPCAInputs`.

    Ingestion goes through :func:`mlsynth.utils.datautils.dataprep`. The treated
    units' post-treatment cells become the missing entries local PCA imputes.

    Raises
    ------
    MlsynthDataError
        If the panel is unbalanced, has fewer than three units, is treated in
        the first period, or is split so that the PCA block holds no
        pre-treatment period.
    """
    prepped = dataprep(
        df,
        unit_id_column_name=unitid,
        time_period_column_name=time,
        outcome_column_name=outcome,
        treatment_indicator_column_name=treat,
    )
    wide = prepped["Ywide"]
    if wide.isna().to_numpy().any():
        raise MlsynthDataError(
            "Panel is unbalanced (missing unit-time cells); LPCA needs a "
            "rectangular outcome matrix."
        )

    unit_names = list(wide.columns)
    time_labels = np.asarray(wide.index)
    Y = np.array(wide.to_numpy(dtype=float).T, copy=True)   # (N, T)
    n_units, n_periods = Y.shape
    if n_units < 3:
        raise MlsynthDataError(
            f"LPCA needs at least 3 units; got {n_units}. The local SVD is "
            "taken over a neighbourhood, which a two-unit panel cannot form."
        )

    treatment = df.pivot(index=time, columns=unitid, values=treat)
    D = treatment.loc[time_labels, unit_names].to_numpy(dtype=float).T
    if np.isnan(D).any():
        raise MlsynthDataError("Treatment indicator has missing unit-time cells.")
    if D[:, 0].max() == 1:  # pragma: no cover - dataprep rejects this first
        # Kept as a local invariant: the block split below indexes off T0, and
        # a T0 of zero would silently produce an empty pre-period.
        raise MlsynthDataError("Some unit is treated in the earliest period.")
    treated_idx = np.flatnonzero(D.max(axis=1) == 1)
    if treated_idx.size == 0:  # pragma: no cover - dataprep rejects this first
        raise MlsynthDataError("No treated unit-periods found (all treat == 0).")

    T0 = int(np.argmax(D.max(axis=0) == 1))
    resolved = resolve_match_periods(n_periods, match_periods, match_fraction)
    if resolved < 1 or resolved >= n_periods:
        raise MlsynthDataError(
            f"match_periods ({resolved}) must be between 1 and {n_periods - 1}; "
            "the PCA block cannot be empty."
        )
    if resolved >= T0:
        raise MlsynthDataError(
            f"match_periods ({resolved}) leaves no pre-treatment period in the "
            f"PCA block (treatment starts at period index {T0}). Local PCA "
            "predicts only the periods held out from matching, so a shorter "
            "matching block is needed for the fit to be checkable."
        )

    Y_masked = Y.copy()
    for unit in treated_idx:
        Y_masked[unit, D[unit] == 1] = np.nan

    return LPCAInputs(
        Y=Y, Y_masked=Y_masked, D=D, treated_idx=treated_idx, T0=T0,
        match_periods=resolved, unit_names=unit_names, time_labels=time_labels,
    )
