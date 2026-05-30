"""High-level fitters that glue setup + estimation + inference together."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .estimation import att_for_unit, solve_musc_qp
from .inference import (
    normal_ci_from_variance,
    randomization_ci,
    unbiased_variance,
)
from .setup import prepare_musc_inputs
from .structures import (
    MUSC,
    SC,
    MUSCCohortFit,
    MUSCInference,
    MUSCInputs,
    MUSCMultiCohortResults,
    MUSCResults,
    MUSCVariantFit,
)


# ---------------------------------------------------------------------------
# Treatment derivation (the only DataFrame touchpoint in orchestration)
# ---------------------------------------------------------------------------

def derive_treatment(
    df: pd.DataFrame, unitid: str, time: str, treat: str,
) -> Tuple[Any, Any]:
    """Identify a single treated unit and its first treated period.

    Kept for back-compatibility with single-unit callers. Returns a
    plain ``(treated_unit, intervention_time)`` tuple. Raises if more
    than one treated unit is found; use :func:`derive_treatment_cohorts`
    for the multi-treated / staggered case.
    """
    treated_rows = df.loc[df[treat] == 1, [unitid, time]]
    if treated_rows.empty:
        raise MlsynthDataError(
            f"No rows with {treat}=1 found; MUSC requires at least one "
            "treated unit with a sharp intervention."
        )
    treated_units = treated_rows[unitid].unique()
    if treated_units.size != 1:
        raise MlsynthDataError(
            "derive_treatment was called on a panel with multiple treated "
            "units; use derive_treatment_cohorts for the multi-treated / "
            f"staggered case (found {treated_units.size} treated units)."
        )
    treated_unit = treated_units[0]
    intervention_time = treated_rows[time].min()
    return treated_unit, intervention_time


def derive_treatment_cohorts(
    df: pd.DataFrame, unitid: str, time: str, treat: str,
) -> List[Tuple[Tuple[Any, ...], Any]]:
    """Group treated units into cohorts by intervention period.

    A *cohort* is the set of units that share a common first treated
    period. The result is a list of ``(treated_units, intervention_time)``
    tuples sorted by intervention time, where ``treated_units`` is a
    tuple of one or more unit labels.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel with a 0/1 treatment indicator.
    unitid, time, treat : str
        Column names.

    Returns
    -------
    list of (tuple of unit labels, intervention time)
        One entry per cohort; the cohort with the earliest intervention
        time comes first.

    Raises
    ------
    MlsynthDataError
        If no treated unit is found.
    """
    treated_rows = df.loc[df[treat] == 1, [unitid, time]]
    if treated_rows.empty:
        raise MlsynthDataError(
            f"No rows with {treat}=1 found; MUSC requires at least one "
            "treated unit with a sharp intervention."
        )
    first_treated = (
        treated_rows.groupby(unitid)[time].min().to_dict()
    )
    cohort_map: Dict[Any, List[Any]] = {}
    for unit, intervention_time in first_treated.items():
        cohort_map.setdefault(intervention_time, []).append(unit)
    cohorts = [
        (tuple(sorted(units, key=str)), intervention_time)
        for intervention_time, units in sorted(
            cohort_map.items(), key=lambda kv: kv[0]
        )
    ]
    return cohorts


def collapse_cohort(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treat: str,
    treated_units: Sequence[Any],
    intervention_time: Any,
    synthetic_label: Any,
    other_treated_units: Sequence[Any] = (),
) -> pd.DataFrame:
    """Collapse a cohort of treated units into a single synthetic row.

    Implements the uniform-treated-weight version of Bottmer et al.
    (2024) Appendix D.1: with the constraint ``M_{k, j, t} = 1 / N_T``
    for ``j`` in the treated set, the K-row formulation's per-row
    objective reduces to a single-row objective on a synthetic unit
    whose outcome is the within-period mean of the treated units'
    outcomes. We construct that synthetic unit here and drop the
    constituent treated rows from the panel; the resulting DataFrame
    is then passed to :func:`prepare_musc_inputs` unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Original long panel.
    unitid, time, outcome, treat : str
        Column names for the unit id, period, outcome, and 0/1 treatment
        indicator.
    treated_units : sequence of unit labels
        Units belonging to this cohort.
    intervention_time : Any
        First treated period for this cohort.
    synthetic_label : Any
        Label assigned to the synthetic cohort row in the returned
        DataFrame (must not collide with any existing unit label).
    other_treated_units : sequence of unit labels, default ()
        Treated units belonging to *other* cohorts that should be
        excluded from the donor pool. Mirrors ``dataprep``'s cohort
        donor-pool convention -- only never-treated units serve as
        donors for any cohort.

    Returns
    -------
    pd.DataFrame
        Modified long panel containing the synthetic cohort row plus
        all non-treated units. The synthetic row's ``treat`` column is
        1 on/after ``intervention_time`` and 0 otherwise.
    """
    treated_set = set(treated_units)
    other_set = set(other_treated_units)
    if synthetic_label in set(df[unitid].unique()):
        raise MlsynthDataError(
            f"synthetic_label {synthetic_label!r} collides with an "
            "existing unit label in the panel."
        )

    cohort_rows = df.loc[df[unitid].isin(treated_set), [time, outcome]].copy()
    averaged = (
        cohort_rows.groupby(time, as_index=False)[outcome].mean()
    )
    averaged[unitid] = synthetic_label
    averaged[treat] = (averaged[time] >= intervention_time).astype(int)

    # Strip cohort + other-cohort treated rows, keep never-treated donors.
    keep_mask = ~df[unitid].isin(treated_set | other_set)
    donor_rows = df.loc[keep_mask].copy()
    return pd.concat([donor_rows, averaged], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-variant assembly
# ---------------------------------------------------------------------------

def _assemble_variant_fit(
    name: str, inputs: MUSCInputs, M: np.ndarray,
) -> MUSCVariantFit:
    """Build a :class:`MUSCVariantFit` from a solved weight matrix."""
    Y_full = inputs.Y.T                                         # (T, N)
    counterfactual, gap, att, pre_rmse = att_for_unit(
        M, Y_full, inputs.treated_idx, inputs.T0
    )

    # Canonical SC sign on the treated unit's donor weights.
    treated_row = M[inputs.treated_idx, 1:]                     # length N
    donor_weights_arr = -treated_row[inputs.donor_idx]          # in [0, 1]
    donor_weights_arr = np.where(
        np.abs(donor_weights_arr) < 1e-10, 0.0, donor_weights_arr
    )
    donor_weights = {
        lbl: float(w)
        for lbl, w in zip(inputs.donor_labels, donor_weights_arr)
    }

    col_sums = M[:, 1:].sum(axis=0)
    col_sum_residual = float(np.abs(col_sums).max())

    return MUSCVariantFit(
        name=name,
        M=M,
        weights_on_treated=donor_weights_arr,
        intercept=float(M[inputs.treated_idx, 0]),
        counterfactual=counterfactual,
        gap=gap,
        att=att,
        pre_rmse=pre_rmse,
        column_sum_residual=col_sum_residual,
        donor_weights=donor_weights,
    )


# ---------------------------------------------------------------------------
# Main fit pipeline
# ---------------------------------------------------------------------------

def run_musc(
    inputs: MUSCInputs,
    *,
    alpha: float = 0.05,
    run_inference: bool = True,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> MUSCResults:
    """Fit MUSC + SC, compute Prop 1 variance and the randomization CI.

    Parameters
    ----------
    inputs : MUSCInputs
        Preprocessed panel.
    alpha : float, default 0.05
        Two-sided significance level for the CIs.
    run_inference : bool, default True
        When False, skip both the unbiased-variance computation and
        the placebo refits (useful for quick exploratory fits and
        large-``N`` panels where the placebo loop is the bottleneck).
    solver : str, optional
        Forwarded to cvxpy.
    verbose : bool
        Forwarded to cvxpy.

    Returns
    -------
    MUSCResults
    """
    Y_pre = inputs.Y_pre                                        # (T_pre, N)

    M_sc, _ = solve_musc_qp(
        Y_pre, column_balance=False, solver=solver, verbose=verbose
    )
    M_musc, _ = solve_musc_qp(
        Y_pre, column_balance=True, solver=solver, verbose=verbose
    )

    fits = {
        SC: _assemble_variant_fit(SC, inputs, M_sc),
        MUSC: _assemble_variant_fit(MUSC, inputs, M_musc),
    }
    primary = fits[MUSC]

    if run_inference and inputs.N >= 4:
        # Use Y at the first post-treatment period for the variance
        # estimator -- matches the paper's convention (Lemma 3 and the
        # MATLAB reference operate at a single treated period at a
        # time).
        y_first_post = inputs.Y[:, inputs.T0]
        variance = unbiased_variance(M_musc, y_first_post)
        se = float(np.sqrt(variance)) if variance >= 0 else float("nan")
        ci_normal = normal_ci_from_variance(primary.att, variance, alpha=alpha)
        ci_rand, placebo_atts = randomization_ci(
            inputs.Y.T,                                         # (T, N)
            treated_idx=inputs.treated_idx,
            T0=inputs.T0,
            column_balance=True,
            att_observed=primary.att,
            alpha=alpha,
            solver=solver,
        )
        inference = MUSCInference(
            variance=variance,
            se=se,
            ci_normal=ci_normal,
            ci_randomization=ci_rand,
            placebo_atts=placebo_atts,
            alpha=alpha,
        )
    else:
        inference = MUSCInference(
            variance=float("nan"),
            se=float("nan"),
            ci_normal=(float("nan"), float("nan")),
            ci_randomization=(float("nan"), float("nan")),
            placebo_atts=np.array([]),
            alpha=alpha,
        )

    return MUSCResults(
        inputs=inputs,
        fits=fits,
        inference=inference,
        selected_variant=MUSC,
        metadata={
            "alpha": alpha,
            "run_inference": run_inference,
            "solver": solver or "CLARABEL",
        },
    )


# ---------------------------------------------------------------------------
# Multi-cohort dispatcher (Appendix D.1, uniform-treated-weight version)
# ---------------------------------------------------------------------------

def _synthetic_cohort_label(treated_units: Sequence[Any]) -> str:
    """Build a stable, panel-safe label for a collapsed cohort."""
    return "_musc_cohort__" + "__".join(str(u) for u in treated_units)


def run_musc_cohorts(
    df: pd.DataFrame,
    *,
    unitid: str,
    time: str,
    outcome: str,
    treat: str,
    cohorts: List[Tuple[Tuple[Any, ...], Any]],
    alpha: float = 0.05,
    run_inference: bool = True,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> MUSCMultiCohortResults:
    """Fit per-cohort MUSC on a multi-treated / staggered panel.

    For each cohort the constituent treated units are collapsed to
    their within-period mean (uniform-treated-weight version of
    Bottmer et al. 2024 Appendix D.1), and single-unit MUSC is fitted
    on the resulting panel against a shared pool of never-treated
    donors -- the same convention used by ``dataprep`` and the
    staggered-adoption estimators elsewhere in :mod:`mlsynth`.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel.
    unitid, time, outcome, treat : str
        Column names.
    cohorts : list of (tuple of unit labels, intervention time)
        As returned by :func:`derive_treatment_cohorts`.
    alpha, run_inference, solver, verbose
        Forwarded to :func:`run_musc` for each cohort.

    Returns
    -------
    MUSCMultiCohortResults
    """
    if not cohorts:
        raise MlsynthDataError("No cohorts supplied to run_musc_cohorts.")

    # All treated units across all cohorts: removed from every cohort's
    # donor pool so the same never-treated donors fit each cohort.
    all_treated: List[Any] = []
    for treated_units, _ in cohorts:
        all_treated.extend(treated_units)

    cohort_fits: Dict[Any, MUSCCohortFit] = {}
    for treated_units, intervention_time in cohorts:
        synthetic_label = _synthetic_cohort_label(treated_units)
        others = [u for u in all_treated if u not in set(treated_units)]
        df_collapsed = collapse_cohort(
            df,
            unitid=unitid, time=time, outcome=outcome, treat=treat,
            treated_units=treated_units,
            intervention_time=intervention_time,
            synthetic_label=synthetic_label,
            other_treated_units=others,
        )
        inputs = prepare_musc_inputs(
            df_collapsed,
            unitid=unitid, time=time, outcome=outcome,
            treated_unit=synthetic_label,
            intervention_time=intervention_time,
        )
        results = run_musc(
            inputs,
            alpha=alpha,
            run_inference=run_inference,
            solver=solver,
            verbose=verbose,
        )
        cohort_fits[intervention_time] = MUSCCohortFit(
            intervention_time=intervention_time,
            treated_units=tuple(treated_units),
            results=results,
        )

    return MUSCMultiCohortResults(
        cohort_fits=cohort_fits,
        metadata={
            "alpha": alpha,
            "run_inference": run_inference,
            "solver": solver or "CLARABEL",
            "n_cohorts": len(cohort_fits),
        },
    )
