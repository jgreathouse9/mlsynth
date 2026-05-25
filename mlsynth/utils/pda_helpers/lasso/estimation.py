"""L1/LASSO PDA estimation (Li & Bell 2017).

Selects control units and estimates coefficients by LASSO on the pre-period,
with the penalty chosen by cross-validation, then predicts the treated unit's
counterfactual out-of-sample.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LassoCV


def fit_lasso(
    y: np.ndarray, X: np.ndarray, T0: int, cv: int = 5, random_state: int = 0,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit LASSO PDA; return ``(beta, intercept, counterfactual, support_mask)``."""
    y_pre, X_pre = y[:T0], X[:T0]
    n_splits = min(cv, T0 - 1)
    model = LassoCV(cv=max(2, n_splits), fit_intercept=True, max_iter=100000, random_state=random_state)
    model.fit(X_pre, y_pre)
    beta = np.asarray(model.coef_, dtype=float)
    intercept = float(model.intercept_)
    counterfactual = X @ beta + intercept
    support = np.abs(beta) > 1e-10
    return beta, intercept, counterfactual, support
