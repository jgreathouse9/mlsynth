"""L1/LASSO PDA estimation (Li & Bell 2017).

Selects control units and estimates coefficients by LASSO on the pre-period,
with the penalty chosen by cross-validation, then predicts the treated unit's
counterfactual out-of-sample.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import Lasso, LassoCV

from ....exceptions import MlsynthConfigError


def _glmnet_scale(X_pre: np.ndarray) -> np.ndarray:
    """Column scaling ``glmnet(standardize = TRUE, intercept = FALSE)`` applies.

    glmnet standardises internally and returns coefficients on the original
    scale. With no intercept it divides each column by its population standard
    deviation and does *not* centre first -- checked against ``glmnet`` 4.1.8,
    where this convention reproduces its coefficients to 5e-09 while centring, or
    scaling by the root mean square, is wrong in the third decimal.

    A column with no spread is left alone, since dividing it by zero would send
    the whole path to ``nan``.
    """
    scale = np.std(X_pre, axis=0)
    return np.where(scale > 0, scale, 1.0)


def fit_lasso(
    y: np.ndarray, X: np.ndarray, T0: int, cv: int = 5, random_state: int = 0,
    alpha: Optional[float] = None, intercept: bool = True,
    standardize: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit LASSO PDA; return ``(beta, intercept, counterfactual, support_mask)``.

    ``alpha`` fixes the penalty (skipping cross-validation); ``None`` (default)
    selects it by :class:`~sklearn.linear_model.LassoCV`. The fixed-penalty path
    is used by the prediction-interval bootstrap, which reuses ``lambda_{T0}``.

    ``intercept`` defaults to ``True``, the cross-validated path's convention.
    ``standardize`` scales the donors the way ``glmnet`` does before penalising
    them and returns coefficients on the original scale; it is off by default and
    on for Shi and Huang's rule. Their call is
    ``glmnet(..., standardize = TRUE, intercept = FALSE)``, so
    :func:`lasso_mbic_alpha` pairs with ``intercept=False, standardize=True`` --
    the penalty and the criterion have to see the same fit.
    """
    y_pre, X_pre = y[:T0], X[:T0]
    scale = _glmnet_scale(X_pre) if standardize else np.ones(X.shape[1])
    if alpha is None:
        n_splits = min(cv, T0 - 1)
        model = LassoCV(cv=max(2, n_splits), fit_intercept=intercept,
                        max_iter=100000, random_state=random_state)
    else:
        model = Lasso(alpha=float(alpha), fit_intercept=intercept,
                      max_iter=200000, tol=1e-12)
    model.fit(X_pre / scale, y_pre)
    beta = np.asarray(model.coef_, dtype=float) / scale
    intercept_hat = float(model.intercept_)
    counterfactual = X @ beta + intercept_hat
    support = np.abs(beta) > 1e-10
    return beta, intercept_hat, counterfactual, support


def lasso_cv_alpha(
    y: np.ndarray, X: np.ndarray, T0: int, cv: int = 5, random_state: int = 0,
) -> float:
    """The cross-validated LASSO penalty for the pre-period (the ``alpha_`` that
    :func:`fit_lasso` selects).

    Exposed so the prediction-interval bootstrap can refit at a *fixed* penalty
    (Jiang et al. 2025 reuse ``lambda_{T0}`` across bootstrap samples rather than
    re-cross-validating each one)."""
    y_pre, X_pre = y[:T0], X[:T0]
    n_splits = min(cv, T0 - 1)
    model = LassoCV(cv=max(2, n_splits), fit_intercept=True, max_iter=100000,
                    random_state=random_state)
    model.fit(X_pre, y_pre)
    return float(model.alpha_)


#: The penalty grid ``lasso.BIC`` searches (``seq(0.01, 1, by = 0.01)``).
MBIC_GRID = np.arange(0.01, 1.0 + 1e-9, 0.01)

#: The constant ``H`` in the modified BIC; ``lasso.BIC``'s default.
MBIC_CONST = 2.0


def _mbic_fit(y: np.ndarray, X: np.ndarray, T0: int, alpha: float) -> np.ndarray:
    """LASSO coefficients at ``alpha`` under their glmnet conventions."""
    beta, _, _, _ = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                              standardize=True)
    return beta


def mbic_criterion(
    y: np.ndarray, X: np.ndarray, T0: int, alpha: float,
    const: float = MBIC_CONST,
) -> float:
    """Shi and Huang's modified BIC at one penalty.

    ``log(sigma^2) + const * log(log N) * log(T0) / T0 * k``, with ``sigma^2`` the
    pre-period mean squared residual and ``k`` the number of selected donors.
    """
    beta = _mbic_fit(y, X, T0, alpha)
    resid = y[:T0] - X[:T0] @ beta
    mse = float(np.mean(resid ** 2))
    k = int(np.count_nonzero(np.abs(beta) > 1e-10))
    n_donors = X.shape[1]
    # log(log N) is undefined at one donor and negative below e; the charge per
    # donor is meant to be non-negative, so it is floored at zero.
    loglog = max(np.log(np.log(n_donors)) if n_donors > 1 else 0.0, 0.0)
    penalty = const * loglog * np.log(T0) / T0 * k
    return float(np.log(max(mse, np.finfo(float).tiny)) + penalty)


def lasso_mbic_alpha(
    y: np.ndarray, X: np.ndarray, T0: int, *,
    const: float = MBIC_CONST, grid: Optional[np.ndarray] = None,
) -> float:
    """The penalty Shi and Huang's modified BIC selects on the pre-period.

    Reproduces ``fsPDA``'s ``lasso.BIC``: fit the LASSO path without an
    intercept, score each penalty by :func:`mbic_criterion`, and keep the
    minimiser. Pair it with ``fit_lasso(..., intercept=False)``, which is the
    convention their ``glmnet`` call uses.

    Parameters
    ----------
    const : float
        ``H`` in the criterion. Their default is 2; the paper describes tuning it
        "to allow Lasso to take in more variables", and raising it selects fewer
        donors.
    grid : np.ndarray, optional
        Penalties to search. Defaults to :data:`MBIC_GRID`, their
        ``seq(0.01, 1, by = 0.01)``.

    Raises
    ------
    MlsynthConfigError
        If ``const`` is not positive, ``T0`` is not positive, or the grid is
        empty or contains a non-positive penalty.
    """
    if not float(const) > 0:
        raise MlsynthConfigError(
            f"the modified BIC constant must be positive; got const={const}.")
    if int(T0) <= 0:
        raise MlsynthConfigError(
            f"the pre-period must contain at least one period; got T0={T0}.")
    penalties = MBIC_GRID if grid is None else np.asarray(grid, dtype=float).reshape(-1)
    if penalties.size == 0:
        raise MlsynthConfigError("the modified BIC penalty grid is empty.")
    if np.any(penalties <= 0):
        raise MlsynthConfigError(
            "every penalty in the modified BIC grid must be positive; a zero "
            "penalty is ordinary least squares, not a point on the LASSO path.")

    scores = [mbic_criterion(y, X, int(T0), a, const=float(const))
              for a in penalties]
    return float(penalties[int(np.argmin(scores))])
