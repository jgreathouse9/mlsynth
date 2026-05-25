"""Forward-selected PDA estimation (Shi & Huang 2023).

Greedy forward selection of control units: at each step add the donor whose
inclusion maximizes the pre-treatment OLS ``R^2`` (equivalently minimizes the
residual variance ``sigma^2 = mean(e^2)``). Selection **stops** as soon as the
modified information criterion

    IC(r) = log( sigma^2(U_r) ) + B * r,   B = log(log N) * log(T1) / T1

stops decreasing (the stopping rule of Wang, Li & Tsai used in the authors'
``fsPDA`` R package), starting from the intercept-only ``IC = log(var(y1))``.
The counterfactual is the OLS extrapolation on the selected set. This is a
direct port of ``est.fsPDA.R``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _ols_sigma2(y: np.ndarray, Z: np.ndarray) -> float:
    """OLS (pinv) residual variance ``mean(e^2)`` for design ``Z`` (incl. intercept)."""
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    return float(np.mean(resid ** 2))


def forward_select(
    y: np.ndarray, X: np.ndarray, T0: int,
) -> Tuple[List[int], np.ndarray, float, np.ndarray]:
    """Forward-select donors with the stop-on-increase IC rule, then refit OLS.

    Returns ``(selected_indices, beta_full, intercept, counterfactual)`` where
    ``beta_full`` is an ``N``-vector with zeros off the selected support.
    """
    y_pre, X_pre = y[:T0], X[:T0]
    N = X.shape[1]
    B = np.log(np.log(max(N, 3))) * np.log(T0) / T0
    IC = float(np.log(np.var(y_pre, ddof=1)))

    selected: List[int] = []
    remaining = list(range(N))
    for _ in range(T0):
        if not remaining:
            break
        best_j, best_s2 = None, np.inf
        for j in remaining:
            cols = selected + [j]
            Z = np.column_stack([np.ones(T0), X_pre[:, cols]])
            s2 = _ols_sigma2(y_pre, Z)
            if s2 < best_s2:
                best_s2, best_j = s2, j
        IC_new = np.log(best_s2) + B * (len(selected) + 1)
        if IC_new < IC:                       # accept and continue
            IC = IC_new
            selected.append(best_j)
            remaining.remove(best_j)
        else:                                 # stop at first non-improvement
            break

    if not selected:                          # degenerate: intercept only
        intercept = float(np.mean(y_pre))
        beta_full = np.zeros(N)
        return [], beta_full, intercept, np.full(X.shape[0], intercept)

    Z = np.column_stack([np.ones(T0), X_pre[:, selected]])
    coef, *_ = np.linalg.lstsq(Z, y_pre, rcond=None)
    intercept = float(coef[0])
    beta_full = np.zeros(N)
    beta_full[selected] = coef[1:]
    counterfactual = X @ beta_full + intercept
    return selected, beta_full, intercept, counterfactual
