"""CSC-IPCA point estimation (Wang 2024, Sec. 3, four-step imputation).

1. Estimate the factors ``F`` and control mapping ``Gamma_ctrl`` by ALS on the
   control panel over the whole period.
2. Re-estimate the treated mapping ``Gamma_tr`` on the treated unit's
   pre-treatment periods, holding ``F`` fixed.
3. Normalize ``(Gamma_tr, F)`` to the identifiable rotation.
4. Impute ``hat Y_t(0) = (X_t Gamma_tr) F_t`` for the treated unit over every
   period; the effect is ``Y_t - hat Y_t(0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...exceptions import MlsynthEstimationError
from .als import als_estimate, counterfactual, normalize, solve_gamma
from .structures import CSCIPCAInputs


@dataclass(frozen=True)
class CSCIPCAFit:
    """Point-estimation output of :func:`fit_cscipca`."""

    gamma: np.ndarray
    factors: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray
    tau: np.ndarray
    att: float
    n_iter: int
    converged: bool
    pre_rmse: float


def fit_cscipca(
    inputs: CSCIPCAInputs, n_factors: int, max_iter: int, tol: float
) -> CSCIPCAFit:
    """Run the four-step CSC-IPCA imputation for a single treated unit."""
    K = int(n_factors)
    T, T0 = inputs.T, inputs.T0
    Yc = inputs.control_outcomes.T            # (N_co, T)
    Xc = inputs.control_covariates            # (N_co, T, L)
    y = inputs.treated_outcome                # (T,)
    Xt = inputs.treated_covariates            # (T, L)

    if K > min(Yc.shape):
        raise MlsynthEstimationError(
            f"n_factors={K} exceeds min(N_control, T)={min(Yc.shape)}; "
            "reduce n_factors."
        )

    # Step 1: factors + control mapping by ALS on the whole control panel.
    F, _gamma_ctrl, n_iter, converged = als_estimate(Yc, Xc, K, max_iter, tol)

    # Step 2: treated mapping on the pre-period, F fixed.
    gamma_tr = solve_gamma(y[None, :T0], Xt[None, :T0, :], F[:, :T0], K)

    # Step 3: identifiable normalization (counterfactual invariant to it).
    gamma_norm, F_norm = normalize(gamma_tr, F)

    # Step 4: impute the untreated path and form the effect.
    yhat = counterfactual(Xt[None], gamma_norm, F_norm)[0]     # (T,)
    gap = y - yhat
    tau = gap[T0:]
    att = float(np.mean(tau)) if tau.size else float("nan")
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2))) if T0 else float("nan")

    return CSCIPCAFit(
        gamma=gamma_norm,
        factors=F_norm,
        counterfactual=yhat,
        gap=gap,
        tau=tau,
        att=att,
        n_iter=int(n_iter),
        converged=bool(converged),
        pre_rmse=pre_rmse,
    )
