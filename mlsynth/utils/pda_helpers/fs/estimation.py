"""Forward-selected PDA estimation (Shi & Huang 2023).

Greedy forward selection of control units: at each step add the donor whose
inclusion maximizes the pre-treatment OLS ``R^2`` (equivalently minimizes the
residual sum of squares). The number of selected units ``R`` is chosen by the
modified BIC of Wang, Li & Tsai (2009),

    R_hat = argmin_r  log( sigma_hat^2(U_r) ) + log(log N) * r * log(T1) / T1,

where ``sigma_hat^2(U_r)`` is the pre-period residual variance of OLS on the
``r``-donor set. The counterfactual is the OLS extrapolation on ``U_{R_hat}``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _ols_ssr(y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, float]:
    """OLS coefficients (incl. intercept col in Z) and residual sum of squares."""
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    return coef, float(resid @ resid)


def forward_select(
    y: np.ndarray, X: np.ndarray, T0: int, max_R: Optional[int] = None,
) -> Tuple[List[int], np.ndarray, float, np.ndarray]:
    """Forward-select donors, choose ``R`` by modified BIC, refit OLS.

    Returns ``(selected_indices, beta_full, intercept, counterfactual)`` where
    ``beta_full`` is an ``N``-vector with zeros off the selected support.
    """
    y_pre, X_pre = y[:T0], X[:T0]
    N = X.shape[1]
    cap = max_R if max_R is not None else min(N, max(1, T0 - 2))

    selected: List[int] = []
    remaining = list(range(N))
    path_ssr: List[float] = []
    while len(selected) < cap and remaining:
        best_j, best_ssr = None, np.inf
        for j in remaining:
            cols = [0] + [k + 1 for k in selected + [j]]   # intercept + chosen donors
            Z = np.column_stack([np.ones(T0), X_pre[:, selected + [j]]])
            _, ssr = _ols_ssr(y_pre, Z)
            if ssr < best_ssr:
                best_ssr, best_j = ssr, j
        selected.append(best_j)
        remaining.remove(best_j)
        path_ssr.append(best_ssr)

    # modified BIC over the greedy path
    log_logN = np.log(np.log(max(N, 3)))
    bic = [np.log(path_ssr[r] / T0) + log_logN * (r + 1) * np.log(T0) / T0
           for r in range(len(path_ssr))]
    R_hat = int(np.argmin(bic)) + 1
    chosen = selected[:R_hat]

    Z = np.column_stack([np.ones(T0), X_pre[:, chosen]])
    coef, _ = _ols_ssr(y_pre, Z)
    intercept = float(coef[0])
    beta_full = np.zeros(N)
    beta_full[chosen] = coef[1:]
    counterfactual = X @ beta_full + intercept
    return chosen, beta_full, intercept, counterfactual
