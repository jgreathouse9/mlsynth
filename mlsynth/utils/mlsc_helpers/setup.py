"""Two-DataFrame data preparation for mlSC.

Wraps ``datautils.dataprep`` once on each panel and assembles the matrices
the rest of the pipeline consumes. Validation enforces alignment between the
aggregate and disaggregate panels: matching treatment timing, complete
``agg_id`` coverage, and population weights that sum to 1 within each
aggregate (or default to uniform).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError, MlsynthConfigError
from ..datautils import balance, dataprep
from .structures import MLSCInputs


def _build_population_vector(
    df_disagg: pd.DataFrame,
    unitid_disagg: str,
    agg_id: str,
    weight_col: Optional[str],
    disagg_labels: List,
    disagg_to_agg_labels: List,
) -> np.ndarray:
    """Construct ``v_sc`` aligned to ``disagg_labels``, normalized within agg."""

    unit_to_agg = (
        df_disagg[[unitid_disagg, agg_id]]
        .drop_duplicates()
        .set_index(unitid_disagg)[agg_id]
        .to_dict()
    )

    if weight_col is None:
        # Uniform weights 1 / C_s within each aggregate.
        from collections import Counter

        counts = Counter(unit_to_agg[u] for u in disagg_labels)
        v = np.asarray(
            [1.0 / counts[unit_to_agg[u]] for u in disagg_labels], dtype=float
        )
        return v

    unit_to_weight = (
        df_disagg[[unitid_disagg, weight_col]]
        .drop_duplicates()
        .set_index(unitid_disagg)[weight_col]
        .to_dict()
    )

    raw = np.asarray([float(unit_to_weight[u]) for u in disagg_labels], dtype=float)
    if not np.all(np.isfinite(raw)):
        raise MlsynthDataError(
            f"Non-finite population weights found in '{weight_col}'."
        )
    if np.any(raw < 0):
        raise MlsynthDataError(
            f"Negative population weights in '{weight_col}' are not allowed."
        )

    # Normalize within each aggregate to sum to 1.
    out = np.zeros_like(raw)
    agg_arr = np.asarray(disagg_to_agg_labels)
    for label in np.unique(agg_arr):
        mask = agg_arr == label
        total = raw[mask].sum()
        if total <= 0:
            raise MlsynthDataError(
                f"Population weights sum to zero within aggregate '{label}'."
            )
        out[mask] = raw[mask] / total
    return out


def prepare_mlsc_inputs(
    df_agg: pd.DataFrame,
    df_disagg: pd.DataFrame,
    outcome: str,
    time: str,
    treat: str,
    unitid_agg: str,
    unitid_disagg: str,
    agg_id: str,
    weight_col: Optional[str],
) -> MLSCInputs:
    """Validate, pivot, and stack the two-level panel for mlSC.

    Parameters
    ----------
    df_agg, df_disagg : pd.DataFrame
        Long-form aggregate and disaggregate panels.
    outcome, time, treat : str
        Column names shared by both panels.
    unitid_agg, unitid_disagg, agg_id, weight_col : str
        Aggregate and disaggregate identifier columns, the disaggregate->
        aggregate mapping column, and an optional population-weights column.

    Returns
    -------
    MLSCInputs
        Pre-processed matrices, weight vector, block-index map, and metadata.
    """

    # Balance both panels independently.
    balance(df_agg, unitid_agg, time)
    balance(df_disagg, unitid_disagg, time)

    # ``dataprep`` on the aggregate panel: standard single-treated path.
    prep_agg = dataprep(df_agg, unitid_agg, time, outcome, treat)
    if "cohorts" in prep_agg:
        raise MlsynthDataError(
            "mlSC requires a single treated aggregate; the aggregate panel "
            "contains multiple treatment cohorts."
        )
    T = int(prep_agg["total_periods"])
    T0 = int(prep_agg["pre_periods"])
    treated_agg_label = prep_agg["treated_unit_name"]
    Y_agg_treated = np.asarray(prep_agg["y"], dtype=float)
    time_labels = np.asarray(prep_agg["time_labels"])

    # ``dataprep`` on the disaggregate panel: returns one of two shapes.
    # If only one disaggregate unit inside the treated aggregate carries
    # treat=1, ``dataprep`` takes the single-treated path. If C > 1 children
    # share the same treatment timing, it takes the cohorts path and bundles
    # them as a single cohort. Both shapes give us everything we need; we
    # just have to unify them.
    prep_disagg = dataprep(df_disagg, unitid_disagg, time, outcome, treat)

    if "cohorts" in prep_disagg:
        cohorts = prep_disagg["cohorts"]
        if len(cohorts) != 1:
            raise MlsynthDataError(
                "mlSC requires all treated disaggregate units to share the "
                f"same treatment start period; found {len(cohorts)} distinct "
                "cohorts in the disaggregate panel."
            )
        (cohort_start_time, cohort), = cohorts.items()
        treated_disagg_labels = list(cohort["treated_units"])
        disagg_donor_labels = list(cohort["donor_names"])
        X_disagg_full = np.asarray(cohort["donor_matrix"], dtype=float)
        T_disagg = int(cohort["total_periods"])
        T0_disagg = int(cohort["pre_periods"])
    else:
        treated_disagg_labels = [prep_disagg["treated_unit_name"]]
        disagg_donor_labels = list(prep_disagg["donor_names"])
        X_disagg_full = np.asarray(prep_disagg["donor_matrix"], dtype=float)
        T_disagg = int(prep_disagg["total_periods"])
        T0_disagg = int(prep_disagg["pre_periods"])

    if T_disagg != T or T0_disagg != T0:
        raise MlsynthDataError(
            "Treatment timing differs between aggregate and disaggregate "
            f"panels: T_agg={T}, T0_agg={T0}; T_disagg={T_disagg}, "
            f"T0_disagg={T0_disagg}."
        )

    # Verify every treated disaggregate unit lives inside the treated
    # aggregate (this catches mismatched agg_id / treat assignments).
    disagg_unit_to_agg = (
        df_disagg[[unitid_disagg, agg_id]]
        .drop_duplicates()
        .set_index(unitid_disagg)[agg_id]
        .to_dict()
    )
    parent_aggs = {disagg_unit_to_agg[u] for u in treated_disagg_labels}
    if parent_aggs != {treated_agg_label}:
        raise MlsynthDataError(
            "Treated disaggregate units must all live inside the treated "
            f"aggregate '{treated_agg_label}'. Parents found: "
            f"{sorted(parent_aggs)}."
        )

    # The Ywide returned by dataprep is the full disaggregate outcome matrix
    # (treated children included). We use it for the Appendix-G variance
    # decomposition, which needs every aggregate.
    Ywide_disagg = prep_disagg["Ywide"]

    # Aggregate-block ordering. Control aggregates first (sorted), treated
    # aggregate last; the latter only matters for the variance-decomposition
    # bookkeeping in ``Y_disagg_pre_full`` below.
    all_agg_labels_sorted = sorted(set(df_agg[unitid_agg].unique()))
    control_agg_labels = [a for a in all_agg_labels_sorted if a != treated_agg_label]
    full_agg_labels = control_agg_labels + [treated_agg_label]
    agg_label_to_idx_full = {a: i for i, a in enumerate(full_agg_labels)}
    treated_agg_idx_full = agg_label_to_idx_full[treated_agg_label]

    # X_disagg from dataprep already excludes the treated children. Reorder
    # its columns so each aggregate block is contiguous, matching the
    # block-diagonal layout of the penalty matrix Q.
    donor_order = sorted(
        range(len(disagg_donor_labels)),
        key=lambda i: (
            agg_label_to_idx_full[disagg_unit_to_agg[disagg_donor_labels[i]]],
            disagg_donor_labels[i],
        ),
    )
    X_disagg = X_disagg_full[:, donor_order]
    disagg_labels_ctrl = [disagg_donor_labels[i] for i in donor_order]
    disagg_to_agg_full_for_ctrl = np.asarray(
        [agg_label_to_idx_full[disagg_unit_to_agg[u]] for u in disagg_labels_ctrl],
        dtype=int,
    )

    # Densify the control-block index (0..S-1 over the surviving aggregates).
    surviving = sorted(set(disagg_to_agg_full_for_ctrl.tolist()))
    full_to_ctrl_idx = {f: i for i, f in enumerate(surviving)}
    disagg_to_agg = np.asarray(
        [full_to_ctrl_idx[f] for f in disagg_to_agg_full_for_ctrl], dtype=int
    )
    agg_labels_ctrl = [full_agg_labels[f] for f in surviving]

    disagg_to_agg_labels_ctrl = [
        full_agg_labels[full_idx] for full_idx in disagg_to_agg_full_for_ctrl
    ]
    v_population = _build_population_vector(
        df_disagg=df_disagg,
        unitid_disagg=unitid_disagg,
        agg_id=agg_id,
        weight_col=weight_col,
        disagg_labels=disagg_labels_ctrl,
        disagg_to_agg_labels=disagg_to_agg_labels_ctrl,
    )

    # Full (T0, M_full) pre-treatment matrix INCLUDING the treated children;
    # needed by the Appendix-G variance decomposition which averages variance
    # estimates across non-treated aggregates and so must know which columns
    # belong to which aggregate.
    Y_disagg_full = np.asarray(Ywide_disagg.to_numpy(), dtype=float)
    disagg_to_agg_full = np.asarray(
        [agg_label_to_idx_full[disagg_unit_to_agg[u]] for u in Ywide_disagg.columns],
        dtype=int,
    )

    if T0 < 2:
        raise MlsynthDataError("mlSC requires at least two pre-treatment periods.")
    if X_disagg.shape[1] < 1:
        raise MlsynthDataError("mlSC requires at least one disaggregate control unit.")

    return MLSCInputs(
        Y_agg_treated=Y_agg_treated,
        X_disagg=X_disagg,
        v_population=v_population,
        disagg_to_agg=disagg_to_agg,
        agg_labels=agg_labels_ctrl,
        disagg_labels=disagg_labels_ctrl,
        Y_disagg_pre_full=Y_disagg_full[:T0, :],
        disagg_to_agg_full=disagg_to_agg_full,
        treated_agg_idx_full=treated_agg_idx_full,
        T=T,
        T0=T0,
        treated_unit_name=str(treated_agg_label),
        time_labels=time_labels,
        Ywide_agg=prep_agg["Ywide"],
        outcome=outcome,
    )
