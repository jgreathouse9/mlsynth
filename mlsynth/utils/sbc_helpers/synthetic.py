"""Step 3 of the SBC procedure: synthetic control on the cyclical residuals.

From Shi, Xi, Xie (2025), Eq. (3):

    (w_hat_2, ..., w_hat_{N+1}) = argmin sum_{t <= T_0} (c_hat_{1, t}
                                                        - sum_i w_i c_hat_{i, t})^2

The paper's default is the simplex form (non-negative weights summing to
one) inherited from Abadie et al. (2010). The unrestricted vertical-
regression form (intercept + free weights) is also offered, mirroring
the Doudchenko-Imbens (2016) variant the paper discusses in Section 2.1.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError


def solve_sbc_weights(
    cycles_treated: np.ndarray,
    cycles_donors: np.ndarray,
    weights_mode: str = "simplex",
) -> Tuple[np.ndarray, Optional[float]]:
    """Fit donor weights on the pre-treatment cycles.

    Parameters
    ----------
    cycles_treated : np.ndarray
        Length-``T_eff`` treated cyclical series over the *effective*
        pre-treatment window (i.e., rows where the Hamilton filter
        produced a non-NaN cycle).
    cycles_donors : np.ndarray
        Shape ``(T_eff, n_donors)`` cyclical residuals for the donors over
        the same window.
    weights_mode : {"simplex", "unrestricted"}
        * ``"simplex"`` — non-negative, sum to one (paper default,
          Eq. (3)). No intercept (cycles are mean-zero by construction).
        * ``"unrestricted"`` — intercept + free coefficients, the
          vertical-regression form discussed in Section 2.1. Useful when
          comparing against Doudchenko-Imbens (2016)-style benchmarks.

    Returns
    -------
    weights : np.ndarray
        Length-``n_donors`` weight vector.
    intercept : float or None
        Fitted intercept for ``unrestricted``; ``None`` otherwise.
    """

    if weights_mode not in {"simplex", "unrestricted"}:
        raise MlsynthEstimationError(
            f"Unknown weights_mode '{weights_mode}'. "
            "Use 'simplex' or 'unrestricted'."
        )

    cycles_treated = np.asarray(cycles_treated, dtype=float)
    cycles_donors = np.asarray(cycles_donors, dtype=float)

    T_eff, n_donors = cycles_donors.shape
    if cycles_treated.shape[0] != T_eff:
        raise MlsynthEstimationError(
            "cycles_treated and cycles_donors must share their first "
            f"dimension; got {cycles_treated.shape[0]} vs {T_eff}."
        )

    if weights_mode == "simplex":
        w = cp.Variable(n_donors, nonneg=True)
        objective = cp.Minimize(
            cp.sum_squares(cycles_treated - cycles_donors @ w)
        )
        constraints = [cp.sum(w) == 1]
        cp.Problem(objective, constraints).solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            raise MlsynthEstimationError(
                "Simplex SCM on cycles failed to converge."
            )
        return np.asarray(w.value, dtype=float), None

    # Unrestricted: closed-form OLS with intercept.
    X = np.column_stack([np.ones(T_eff), cycles_donors])
    coefs, *_ = np.linalg.lstsq(X, cycles_treated, rcond=None)
    intercept = float(coefs[0])
    weights = coefs[1:]
    return weights, intercept
