"""Abadie-style placebo (in-space) permutation inference for HSC.

Each donor is reassigned to the treated slot (with the original treated unit
demoted to the donor pool), HSC is refit, and the absolute post-period mean
effect is recorded. The permutation p-value compares the real treated unit's
absolute ATT to the placebo distribution. This is the standard synthetic
control inference; it is optional and off by default because it refits HSC
(including the rho cross-validation) once per donor.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .orchestration import solve_hsc, summarize_effects
from .structures import HSCDesign, HSCInference, HSCInputs


def placebo_inference(
    inputs: HSCInputs,
    design: HSCDesign,
    rho_grid: Sequence[float],
    n_splits: int = 3,
    ridge: object = 1e-6,
    forecaster: str = "arima110",
    solver: Optional[object] = None,
    max_placebo: Optional[int] = None,
) -> HSCInference:
    """Run the placebo permutation test for the treated unit's ATT."""

    q = inputs.q
    treated_te = inputs.Y_post - design.counterfactual_post
    treated_att = float(np.mean(treated_te)) if treated_te.size else 0.0

    Y = inputs.donor_matrix
    y0 = inputs.y_target
    N = Y.shape[1]
    donor_idx = list(range(N if max_placebo is None else min(N, max_placebo)))

    placebo_atts = []
    for j in donor_idx:
        others = [k for k in range(N) if k != j]
        placebo_donors = np.column_stack([y0] + [Y[:, k] for k in others])
        placebo_inputs = HSCInputs(
            y_target=Y[:, j],
            donor_matrix=placebo_donors,
            T=inputs.T,
            T0=inputs.T0,
            treated_unit_name=inputs.donor_names[j],
            donor_names=[inputs.treated_unit_name]
            + [inputs.donor_names[k] for k in others],
            time_labels=inputs.time_labels,
            q=q,
        )
        try:
            pl_design = solve_hsc(
                placebo_inputs, q, rho_grid, n_splits, ridge, forecaster, solver
            )
            _, _, pl_te = summarize_effects(placebo_inputs, pl_design)
            placebo_atts.append(abs(float(np.mean(pl_te))))
        except Exception:  # pragma: no cover - skip a degenerate placebo
            continue

    placebo_atts = np.asarray(placebo_atts, dtype=float)
    n_pl = placebo_atts.size
    if n_pl == 0:
        p_value = float("nan")
    else:
        p_value = float((np.sum(placebo_atts >= abs(treated_att)) + 1) / (n_pl + 1))

    return HSCInference(
        p_value=p_value,
        att=treated_att,
        placebo_atts=placebo_atts,
        n_placebo=n_pl,
    )
