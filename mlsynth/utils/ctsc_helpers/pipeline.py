"""Orchestration for the CTSC estimator (Powell 2022)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .estimate import fit_ctsc
from .inference import sign_flip_wald_inference
from .structures import CTSCInputs, CTSCResults


def run_ctsc(
    inputs: CTSCInputs,
    *,
    use_fit_weights: bool = True,
    inference: bool = True,
    null_value: Optional[np.ndarray] = None,
    n_draws: int = 2000,
    random_state: int = 0,
) -> CTSCResults:
    """Run CTSC and assemble :class:`CTSCResults`.

    Parameters
    ----------
    inputs : CTSCInputs
        Preprocessed panel.
    use_fit_weights : bool
        Use the two-step per-unit fit weights :math:`\\Omega_i` (eq. 6).
    inference : bool
        Run the sign-flip Wald test of ``H0: alpha^AE = null_value``.
    null_value : np.ndarray, optional
        Per-variable null for the average effect (default zeros).
    n_draws : int
        Rademacher draws for the randomization test.
    random_state : int
        RNG seed.
    """
    fit = fit_ctsc(
        inputs.Y, inputs.D,
        population_weights=inputs.population_weights,
        use_fit_weights=use_fit_weights,
    )

    inf = None
    if inference:
        inf = sign_flip_wald_inference(
            inputs.Y, inputs.D, pi=inputs.population_weights,
            omega=fit["omega"], null_value=null_value,
            n_draws=n_draws, random_state=random_state,
        )

    metadata = {
        "n": inputs.n,
        "T": inputs.T,
        "K": inputs.K,
        "treatment_names": list(inputs.treatment_names),
        "used_fit_weights": use_fit_weights,
    }
    return CTSCResults(
        inputs=inputs,
        average_effect=fit["average_effect"],
        unit_effects=fit["alpha"],
        weights=fit["weights"],
        fit_metric=fit["omega"],
        objective=fit["objective"],
        inference=inf,
        metadata=metadata,
    )
