"""Original HCW best-subset panel data approach (Hsiao, Ching & Wan 2012).

The counterfactual for the treated unit is an *unrestricted* OLS regression
(with intercept) of its pre-treatment outcome on a **best-subset**-selected set
of control series (eq. 17), with no simplex / non-negativity constraint. The
subset is chosen in two steps (Section 5): for each model size the best subset
by residual sum of squares, then the size minimising a model-selection
criterion (AICc by default, also AIC / BIC). This is a direct port of the
``pampe`` R package (``leaps::regsubsets`` best-subset + AICc + ``lm`` with
intercept), and reproduces HCW (2012) Table XVI value-for-value.

Best-subset selection is combinatorial. Like ``pampe``'s ``nvmax`` argument, the
search ranges over model sizes up to ``nvmax`` (default: all controls); for a
large donor pool the caller must either cap ``nvmax`` or pre-restrict the pool
(as HCW did, limiting Hong Kong to ten candidate economies), otherwise a
translated error is raised pointing to the scalable ``fs`` / ``LASSO`` / ``l2``
variants.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import List, Optional, Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError

# Guard on the exhaustive search: refuse pools whose subset count would make the
# best-subset enumeration intractable (≈ the regime where pampe's leaps would
# also strain). Callers should cap ``nvmax`` or use fs / LASSO / l2 instead.
_MAX_SUBSETS = 2_000_000

_CRITERIA = ("AICc", "AIC", "BIC")


def info_criterion(rss: float, n: int, n_regressors: int, criterion: str) -> float:
    """Model-selection criterion for an OLS fit, in the ``pampe`` convention.

    Parameters
    ----------
    rss : float
        Residual sum of squares of the pre-period OLS fit.
    n : int
        Number of pre-treatment observations.
    n_regressors : int
        Number of regressors including the intercept (i.e. donors + 1).
    criterion : {"AICc", "AIC", "BIC"}
        Which criterion. ``pampe`` (and HCW Table XVI) count the error variance
        as a parameter, so the penalised parameter count is
        ``K = n_regressors + 1`` (donors + intercept + variance).

    Returns
    -------
    float
        The criterion value (lower is better). Uses the concentrated Gaussian
        form ``n * log(rss / n)`` (no ``n log 2*pi + n`` constant), matching the
        magnitudes HCW report (e.g. AICc = -171.771 for the Table XVI model).
    """
    if criterion not in _CRITERIA:
        raise MlsynthEstimationError(
            f"Unknown HCW criterion {criterion!r}; use one of {_CRITERIA}."
        )
    if rss <= 0 or n <= 0:
        return -np.inf
    K = n_regressors + 1                         # + error variance (pampe count)
    ll_term = n * np.log(rss / n)
    if criterion == "AIC":
        return float(ll_term + 2 * K)
    if criterion == "BIC":
        return float(ll_term + K * np.log(n))
    # AICc: the small-sample-corrected AIC.
    if n - K - 1 <= 0:                            # correction undefined; +inf
        return float("inf")
    return float(ll_term + 2 * K + 2 * K * (K + 1) / (n - K - 1))


def _rss(y: np.ndarray, Z: np.ndarray) -> float:
    """OLS residual sum of squares for design ``Z`` (intercept included)."""
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ coef
    return float(e @ e)


def _resolve_nvmax(nvmax: Optional[int], N: int, T0: int) -> int:
    """Largest model size to search: bounded by donors and the OLS df."""
    cap = min(N, max(T0 - 2, 0))                 # need n > params for a fit
    if nvmax is None:
        return cap
    if nvmax < 1:
        raise MlsynthEstimationError("HCW nvmax must be a positive integer.")
    return min(nvmax, cap)


def best_subset_select(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    *,
    criterion: str = "AICc",
    nvmax: Optional[int] = None,
) -> List[int]:
    """Best-subset donor selection by ``criterion`` (HCW Section 5 / ``pampe``).

    For every model size ``r = 0, 1, ..., nvmax`` the subset of ``r`` donors with
    the smallest pre-period RSS is found by exhaustive enumeration, and the size
    (and subset) minimising ``criterion`` is returned.

    Returns the selected donor column indices (a possibly empty list).
    """
    if criterion not in _CRITERIA:
        raise MlsynthEstimationError(
            f"Unknown HCW criterion {criterion!r}; use one of {_CRITERIA}."
        )
    y_pre = np.asarray(y, dtype=float)[:T0]
    X_pre = np.asarray(X, dtype=float)[:T0]
    N = X_pre.shape[1]
    r_max = _resolve_nvmax(nvmax, N, T0)

    total = sum(comb(N, r) for r in range(0, r_max + 1))
    if total > _MAX_SUBSETS:
        raise MlsynthEstimationError(
            f"HCW best-subset search over {N} donors up to size {r_max} needs "
            f"{total:,} OLS fits (> {_MAX_SUBSETS:,}). Cap 'hcw_nvmax', restrict "
            "the donor pool (as HCW limited Hong Kong to ten economies), or use "
            "the scalable 'fs' / 'LASSO' / 'l2' PDA variants instead."
        )

    ones = np.ones((T0, 1))
    best_idx: List[int] = []
    # Intercept-only baseline (no donors).
    best_ic = info_criterion(_rss(y_pre, ones), T0, 1, criterion)
    for r in range(1, r_max + 1):
        for combo in combinations(range(N), r):
            Z = np.column_stack([ones, X_pre[:, combo]])
            ic = info_criterion(_rss(y_pre, Z), T0, r + 1, criterion)
            if ic < best_ic:
                best_ic, best_idx = ic, list(combo)
    return best_idx


def fit_hcw(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    *,
    criterion: str = "AICc",
    nvmax: Optional[int] = None,
) -> Tuple[List[int], np.ndarray, float, np.ndarray]:
    """HCW best-subset fit: select donors, refit OLS, extrapolate counterfactual.

    Returns ``(selected_indices, beta_full, intercept, counterfactual)`` where
    ``beta_full`` is an ``N``-vector with zeros off the selected support and the
    counterfactual is the OLS extrapolation ``X @ beta_full + intercept`` over
    all periods.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = X.shape[1]
    selected = best_subset_select(y, X, T0, criterion=criterion, nvmax=nvmax)

    beta_full = np.zeros(N)
    if not selected:                             # intercept-only counterfactual
        intercept_val = float(np.mean(y[:T0]))
        return [], beta_full, intercept_val, np.full(X.shape[0], intercept_val)

    Z = np.column_stack([np.ones(T0), X[:T0, selected]])
    coef, *_ = np.linalg.lstsq(Z, y[:T0], rcond=None)
    intercept_val = float(coef[0])
    beta_full[selected] = coef[1:]
    counterfactual = X @ beta_full + intercept_val
    return selected, beta_full, intercept_val, counterfactual
