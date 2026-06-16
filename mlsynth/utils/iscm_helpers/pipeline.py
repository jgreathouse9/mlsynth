"""Orchestration for the ISCM estimator (Powell 2026).

Pipeline:

1. Build synthetic controls for every unit (:func:`all_units_weights`).
2. Form SC residuals and treatment exposure; compute the per-unit fit
   metric :math:`a_i` (:mod:`.estimate`).
3. Estimate the ATT by the :math:`a_i`-weighted least-squares regression
   pooled over all units (paper eq. 15).
4. Optionally run Ibragimov-Muller inference over the per-unit estimates.
"""

from __future__ import annotations

import numpy as np

from .estimate import fit_metric, residuals_and_exposure, weighted_att
from .inference import ibragimov_muller_inference
from ..results_helpers import make_weights_results
from ...config_models import EffectsResults
from .structures import ISCMInputs, ISCMResults
from .weights import all_units_weights


def run_iscm(
    inputs: ISCMInputs,
    *,
    inference: bool = True,
    null_value: float = 0.0,
    alpha_level: float = 0.05,
    n_draws: int = 10000,
    random_state: int = 0,
) -> ISCMResults:
    """Run ISCM and assemble :class:`ISCMResults`.

    Parameters
    ----------
    inputs : ISCMInputs
        Preprocessed panel.
    inference : bool
        If True, run Ibragimov-Muller inference. Default True.
    null_value : float
        Null effect :math:`\\alpha_0` for the test.
    alpha_level : float
        Two-sided level for the confidence interval.
    n_draws : int
        Number of Rademacher sign-flip draws.
    random_state : int
        RNG seed for the randomization test.
    """
    Y, D, T0 = inputs.Y, inputs.D, inputs.T0

    W = all_units_weights(Y, T0)
    R, E = residuals_and_exposure(Y, D, W)
    a = fit_metric(R, Y, T0)
    att, unit_att, contribution = weighted_att(R, E, a, T0)

    inf = None
    if inference:
        inf = ibragimov_muller_inference(
            att=att, unit_att=unit_att, contribution=contribution,
            N=inputs.N, null_value=null_value, alpha_level=alpha_level,
            n_draws=n_draws, random_state=random_state,
        )

    n_contributing = int(np.isfinite(unit_att).sum())
    metadata = {
        "N": inputs.N,
        "T0": T0,
        "T_post": inputs.n_post,
        "n_treated": int(inputs.treated_idx.size),
        "n_contributing": n_contributing,
        "treated_fit_metric": [float(a[i]) for i in inputs.treated_idx],
    }

    # Donor weights for the treated unit(s): their synthetic-control rows.
    names = inputs.unit_names
    per_unit = {}
    for i in inputs.treated_idx:
        per_unit[str(names[i])] = {str(names[j]): float(W[i, j])
                                   for j in range(inputs.N)
                                   if j != i and abs(W[i, j]) > 1e-10}
    if len(per_unit) == 1:
        donor_weights = next(iter(per_unit.values()))
    else:
        donor_weights = {}
        for d in {dn for u in per_unit.values() for dn in u}:
            donor_weights[d] = float(np.mean([u.get(d, 0.0)
                                              for u in per_unit.values()]))
    extra = {"note": ("ISCM builds a synthetic control for every unit; these "
                      "are the treated unit's SC weights. The effect is "
                      "identified across all contributing units (see "
                      "contribution / fit_metric).")}
    if len(per_unit) > 1:
        extra["per_unit_donor_weights"] = per_unit
    weights_res = make_weights_results(
        donor_weights, constraint="simplex (non-negative, sum to 1)", extra=extra,
    )

    return ISCMResults(
        inputs=inputs,
        effects=EffectsResults(att=att),
        unit_weight_matrix=W,
        fit_metric=a,
        unit_att=unit_att,
        contribution=contribution,
        residuals=R,
        exposure=E,
        weights=weights_res,
        inference_detail=inf,
        metadata=metadata,
    )
