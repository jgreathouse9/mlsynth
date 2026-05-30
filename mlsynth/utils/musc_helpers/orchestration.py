"""High-level fitters that glue setup + estimation + inference together."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .estimation import att_for_unit, solve_musc_qp
from .inference import (
    normal_ci_from_variance,
    randomization_ci,
    unbiased_variance,
)
from .structures import (
    MUSC,
    SC,
    MUSCInference,
    MUSCInputs,
    MUSCResults,
    MUSCVariantFit,
)


# ---------------------------------------------------------------------------
# Treatment derivation (the only DataFrame touchpoint in orchestration)
# ---------------------------------------------------------------------------

def derive_treatment(
    df: pd.DataFrame, unitid: str, time: str, treat: str,
) -> Tuple[Any, Any]:
    """Identify the single treated unit and the first treated period.

    Parameters
    ----------
    df : pd.DataFrame
        Long balanced panel.
    unitid, time, treat : str
        Column names for the unit id, period index, and 0/1 treatment
        indicator.

    Returns
    -------
    (treated_unit, intervention_time) : (Any, Any)
        The treated unit's label and the first period with
        ``treat == 1`` for that unit.
    """
    treated_rows = df.loc[df[treat] == 1, [unitid, time]]
    if treated_rows.empty:
        raise MlsynthDataError(
            f"No rows with {treat}=1 found; MUSC requires a single "
            "treated unit with a sharp intervention."
        )
    treated_units = treated_rows[unitid].unique()
    if treated_units.size != 1:
        raise MlsynthDataError(
            f"MUSC currently supports exactly one treated unit; got "
            f"{treated_units.size}: {treated_units.tolist()}."
        )
    treated_unit = treated_units[0]
    intervention_time = treated_rows[time].min()
    return treated_unit, intervention_time


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
