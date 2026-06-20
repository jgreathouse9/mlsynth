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


def _gram(y_pre: np.ndarray, X_pre: np.ndarray):
    """Pre-period sufficient statistics for OLS-by-subset.

    Returns ``(G, Zty, yty)`` for the augmented design ``Z = [1, X_pre]``:
    ``G = Z'Z`` (``(N+1, N+1)``), ``Zty = Z'y`` (``(N+1,)``) and ``yty = y'y``.
    Computed once; every subset's RSS then comes from a small submatrix solve,
    so the best-subset search costs no SVD and is independent of ``T0`` per
    subset.
    """
    Z = np.column_stack([np.ones(len(y_pre)), X_pre])
    return Z.T @ Z, Z.T @ y_pre, float(y_pre @ y_pre)


def _subset_rss(G: np.ndarray, Zty: np.ndarray, yty: float, cols) -> float:
    """RSS of OLS on the intercept + donor columns ``cols`` via the normal eqs.

    ``RSS = y'y - (Z_S'y)' (Z_S'Z_S)^{-1} (Z_S'y)`` over the submatrix indexed by
    the intercept (column 0) and ``cols`` (shifted by +1). Falls back to a
    least-norm solve when the submatrix is singular (collinear donors), so the
    result matches the lstsq reference on rank-deficient subsets too.
    """
    idx = [0]
    idx.extend(c + 1 for c in cols)
    Gs = G[np.ix_(idx, idx)]
    bs = Zty[idx]
    try:
        sol = np.linalg.solve(Gs, bs)
    except np.linalg.LinAlgError:                # collinear subset -> min-norm
        sol, *_ = np.linalg.lstsq(Gs, bs, rcond=None)
    return yty - float(bs @ sol)


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
            f"HCW best-subset over {N} donors up to size {r_max} has a "
            f"worst-case {total:,} subsets (> {_MAX_SUBSETS:,}). Branch-and-bound "
            "prunes most of these, but the worst case is not guaranteed, so the "
            "pool is refused. Cap 'hcw_nvmax', restrict the donor pool (as HCW "
            "limited Hong Kong to ten economies), or use the scalable 'fs' / "
            "'LASSO' / 'l2' PDA variants instead."
        )

    # Precompute the Gram sufficient statistics once; each subset's RSS is then
    # an O(r^3) submatrix solve rather than an O(T0 r^2) SVD.
    G, Zty, yty = _gram(y_pre, X_pre)
    return _best_subset_bnb(G, Zty, yty, N, T0, r_max, criterion)


def _best_subset_exhaustive(G, Zty, yty, N, n, r_max, criterion) -> List[int]:
    """Brute-force best subset: score every subset up to ``r_max`` by criterion.

    The reference search -- correct by construction and the oracle that
    :func:`_best_subset_bnb` is validated against.
    """
    best_idx: List[int] = []
    best_ic = info_criterion(_subset_rss(G, Zty, yty, ()), n, 1, criterion)
    for r in range(1, r_max + 1):
        for combo in combinations(range(N), r):
            ic = info_criterion(_subset_rss(G, Zty, yty, combo), n, r + 1, criterion)
            if ic < best_ic:
                best_ic, best_idx = ic, list(combo)
    return best_idx


def _best_subset_bnb(G, Zty, yty, N, n, r_max, criterion) -> List[int]:
    """Best subset by branch-and-bound (Furnival-Wilson style pruning).

    Donors are ordered strongest-first (smallest univariate RSS) so the
    incumbent tightens early. At each node the criterion is lower-bounded over
    the whole subtree by the smallest achievable RSS (include *all* remaining
    donors -- RSS is monotone in the variable set) paired with the smallest
    feasible penalty (stop now); if that bound is no better than the incumbent,
    the subtree is pruned. The bound is a true lower bound, so the returned
    optimum is identical to :func:`_best_subset_exhaustive`.
    """
    uni = [_subset_rss(G, Zty, yty, (j,)) for j in range(N)]
    order = sorted(range(N), key=lambda j: uni[j])

    best = {"ic": info_criterion(_subset_rss(G, Zty, yty, ()), n, 1, criterion),
            "idx": []}

    def recurse(chosen: List[int], pos: int) -> None:
        k = len(chosen)
        if k >= 1:
            ic = info_criterion(_subset_rss(G, Zty, yty, chosen), n, k + 1, criterion)
            if ic < best["ic"]:
                best["ic"], best["idx"] = ic, list(chosen)
        if k >= r_max or pos >= N:
            return
        remaining = order[pos:]
        # Lower bound over the subtree: include every remaining donor (minimum
        # RSS) at the smallest feasible model size (minimum penalty).
        rss_lb = _subset_rss(G, Zty, yty, chosen + remaining)
        if info_criterion(rss_lb, n, max(k, 1) + 1, criterion) >= best["ic"]:
            return
        for p in range(pos, N):
            recurse(chosen + [order[p]], p + 1)

    recurse([], 0)
    return best["idx"]


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
