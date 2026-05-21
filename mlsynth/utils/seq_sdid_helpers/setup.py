"""Cohort-level aggregation for Sequential SDiD.

Reuses :func:`mlsynth.utils.datautils.dataprep` to identify treated /
never-treated units and their adoption periods, then aggregates the
unit-level outcomes into the cohort-level ``Y_{a,t}`` panel that
Algorithm 1 consumes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import balance, dataprep
from .structures import SeqSDIDInputs


_NEVER_TREATED_SENTINEL = np.iinfo(np.int64).max


def _label_to_index(time_labels: np.ndarray, label: Any) -> int:
    """1-based index of ``label`` within ``time_labels``."""
    positions = np.where(time_labels == label)[0]
    if positions.size == 0:
        raise MlsynthDataError(
            f"Adoption time {label!r} not found in time axis."
        )
    return int(positions[0]) + 1


def prepare_seq_sdid_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    a_min: Optional[int] = None,
    a_max: Optional[int] = None,
    K: Optional[int] = None,
) -> SeqSDIDInputs:
    """Aggregate the panel into cohort-level outcomes ``Y_{a,t}`` and shares.

    Parameters
    ----------
    df, outcome, treat, unitid, time : standard mlsynth panel inputs.
    a_min, a_max : int, optional
        Earliest / latest treated cohort (1-based time index) to estimate.
        Default: span all treated cohorts in the data.
    K : int, optional
        Maximum event-time horizon. Default: ``T - a_max`` so every
        estimable effect fits inside the panel.
    """

    balance(df, unitid, time)
    prep: Dict[str, Any] = dataprep(df, unitid, time, outcome, treat)

    Ywide = prep["Ywide"]
    time_labels = np.asarray(prep["time_labels"])
    unit_labels = list(Ywide.columns)
    Y_units = np.asarray(Ywide.to_numpy(), dtype=float)  # (T, N_units)
    T, N_units = Y_units.shape

    # Build a unit -> adoption_label mapping (or sentinel for never-treated).
    treat_wide = (
        df.pivot(index=time, columns=unitid, values=treat)
        .reindex(index=time_labels, columns=unit_labels)
        .astype(int)
    )
    unit_adoption: Dict[Any, Any] = {}
    for unit_id in unit_labels:
        col = treat_wide[unit_id].to_numpy()
        treated_positions = np.where(col == 1)[0]
        if treated_positions.size == 0:
            unit_adoption[unit_id] = None
        else:
            unit_adoption[unit_id] = time_labels[int(treated_positions[0])]

    # Cohort -> list of unit labels.
    cohort_groups: Dict[Any, list] = {}
    for unit_id, adoption_label in unit_adoption.items():
        cohort_groups.setdefault(adoption_label, []).append(unit_id)

    treated_labels = sorted(label for label in cohort_groups if label is not None)
    if not treated_labels:
        raise MlsynthDataError(
            "No treated units found; Sequential SDiD requires at least one cohort."
        )

    # Assemble Y_agg and pi. Treated cohorts first (in adoption order), then
    # never-treated (if any) at the end.
    cohort_labels: list = []
    cohort_periods: list = []
    cohort_sizes: list = []
    Y_agg_cols: list = []
    for label in treated_labels:
        units = cohort_groups[label]
        cohort_idx = [unit_labels.index(u) for u in units]
        Y_agg_cols.append(Y_units[:, cohort_idx].mean(axis=1))
        cohort_labels.append(label)
        cohort_periods.append(_label_to_index(time_labels, label))
        cohort_sizes.append(len(units))

    if None in cohort_groups:
        units = cohort_groups[None]
        cohort_idx = [unit_labels.index(u) for u in units]
        Y_agg_cols.append(Y_units[:, cohort_idx].mean(axis=1))
        cohort_labels.append("never_treated")
        cohort_periods.append(_NEVER_TREATED_SENTINEL)
        cohort_sizes.append(len(units))

    Y_agg = np.column_stack(Y_agg_cols)  # (T, A)
    cohort_periods_arr = np.asarray(cohort_periods, dtype=np.int64)
    cohort_sizes_arr = np.asarray(cohort_sizes, dtype=int)
    pi = cohort_sizes_arr / cohort_sizes_arr.sum()

    treated_cohort_indices = np.where(
        cohort_periods_arr != _NEVER_TREATED_SENTINEL
    )[0]
    if treated_cohort_indices.size == 0:
        raise MlsynthDataError(
            "Sequential SDiD requires at least one treated cohort."
        )

    treated_periods = cohort_periods_arr[treated_cohort_indices]
    earliest = int(treated_periods.min())
    latest = int(treated_periods.max())

    a_min_final = int(a_min) if a_min is not None else earliest
    a_max_final = int(a_max) if a_max is not None else latest
    if a_min_final > a_max_final:
        raise MlsynthDataError(
            f"a_min ({a_min_final}) must be <= a_max ({a_max_final})."
        )

    # Default K is T - a_max (largest horizon that fits in the panel).
    K_final = int(K) if K is not None else (T - a_max_final)
    if K_final < 0:
        raise MlsynthDataError(
            f"K must be non-negative; got K = {K_final} with a_max = {a_max_final} "
            f"and T = {T}."
        )
    if a_max_final + K_final > T:
        raise MlsynthDataError(
            f"a_max + K = {a_max_final + K_final} exceeds T = {T}. Reduce K "
            "or a_max."
        )

    return SeqSDIDInputs(
        Y_agg=Y_agg,
        pi=pi,
        cohort_periods=cohort_periods_arr,
        cohort_labels=cohort_labels,
        treated_cohort_indices=treated_cohort_indices,
        time_labels=time_labels,
        n_units=int(cohort_sizes_arr.sum()),
        a_min=a_min_final,
        a_max=a_max_final,
        K=K_final,
    )
