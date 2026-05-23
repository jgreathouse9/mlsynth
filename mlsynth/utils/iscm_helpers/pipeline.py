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

    return ISCMResults(
        inputs=inputs,
        att=att,
        weights=W,
        fit_metric=a,
        unit_att=unit_att,
        contribution=contribution,
        residuals=R,
        exposure=E,
        inference=inf,
        metadata=metadata,
    )
