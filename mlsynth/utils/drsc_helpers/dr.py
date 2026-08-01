"""Distribution regression: the per-cell, per-grid-point binary regression.

Wied eq. (4). For each cell and each grid point ``y``, regress ``1{Y <= y}`` on
the design ``X`` under the link ``Lambda``. The resulting coefficient function
``theta(y)`` is the object parallel trends is imposed on -- which is what keeps
the counterfactual inside the model class and makes the weight program linear.

Solved by Newton-Raphson with a step-halving line search on the log-likelihood,
which is the reference implementation's method and is stable on the degenerate
grid points (all-zero or all-one indicators) that occur at the ends of the grid.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import logistic, norm

from ...exceptions import MlsynthEstimationError

#: Below this the indicator is treated as degenerate and the fit short-circuits
#: to a saturated intercept rather than diverging.
_SATURATED = 8.0
_EPS = 1e-12


def link_functions(link: str):
    """``(Lambda, lambda)`` -- the CDF and its derivative."""
    if link == "probit":
        return norm.cdf, norm.pdf
    if link == "logit":
        return logistic.cdf, logistic.pdf
    raise MlsynthEstimationError(  # pragma: no cover - config restricts this
        f"unknown link {link!r}; expected 'probit' or 'logit'.")


def fit_cell(X: np.ndarray, y: np.ndarray, grid: np.ndarray, link: str = "probit",
             max_iter: int = 50, tol: float = 1e-8, ridge: float = 1e-8):
    """Distribution-regression coefficients for one cell.

    Parameters
    ----------
    X : numpy.ndarray
        Shape ``(n, k)`` design, intercept in column 0.
    y : numpy.ndarray
        Shape ``(n,)`` outcomes.
    grid : numpy.ndarray
        Shape ``(m,)`` outcome grid.
    link : {'probit', 'logit'}
    max_iter, tol, ridge
        Newton controls. ``ridge`` is added to the Hessian diagonal only to
        keep the solve well posed; it does not penalise the estimate.

    Returns
    -------
    numpy.ndarray
        Shape ``(m, k)`` coefficient function, one row per grid point.
    """
    Lam, lam = link_functions(link)
    n, k = X.shape
    theta = np.zeros((len(grid), k), dtype=np.float64)
    for j, cut in enumerate(grid):
        z = (y <= cut).astype(np.float64)
        s = z.mean()
        if s <= 0.0 or s >= 1.0:
            # Degenerate indicator: the CDF is 0 or 1 for every x here, which a
            # finite coefficient cannot express. Saturate rather than diverge.
            theta[j, 0] = _SATURATED if s >= 1.0 else -_SATURATED
            continue
        beta = np.zeros(k)
        beta[0] = norm.ppf(np.clip(s, 1e-3, 1 - 1e-3))
        for _ in range(max_iter):
            eta = X @ beta
            P = np.clip(Lam(eta), _EPS, 1 - _EPS)
            d = lam(eta)
            grad = X.T @ (d * (z - P) / (P * (1 - P)))
            wts = d * d / (P * (1 - P))
            H = (X * wts[:, None]).T @ X + ridge * np.eye(k)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError as exc:
                raise MlsynthEstimationError(
                    "the distribution-regression design is rank deficient at "
                    f"grid point {j}; drop collinear covariates.") from exc
            ll0 = float(np.sum(z * np.log(P) + (1 - z) * np.log(1 - P)))
            s_step, improved = 1.0, False
            for _ in range(20):
                cand = beta + s_step * step
                P2 = np.clip(Lam(X @ cand), _EPS, 1 - _EPS)
                if float(np.sum(z * np.log(P2) + (1 - z) * np.log(1 - P2))) >= ll0 - 1e-12:
                    beta, improved = cand, True
                    break
                s_step *= 0.5
            if not improved or np.max(np.abs(s_step * step)) < tol:
                break
        theta[j] = beta
    return theta


def outcome_grid(y: np.ndarray, n_grid: int, lo: float, hi: float) -> np.ndarray:
    """Quantile grid of the pooled outcome, ties collapsed.

    Collapsing ties matters: with a discrete-ish outcome (wages reported to the
    cent) several requested quantiles land on the same value, and fitting the
    same regression twice would double-count that point in the weight
    objective. The paper's 38 requested points reduce to 32 active ones.
    """
    q = np.quantile(np.asarray(y, dtype=np.float64), np.linspace(lo, hi, n_grid))
    grid = np.unique(np.round(q, 8))
    if grid.size < 2:
        raise MlsynthEstimationError(
            "the outcome grid collapsed to fewer than two distinct points; the "
            "outcome has almost no variation over the requested quantile "
            "range.")
    return grid
