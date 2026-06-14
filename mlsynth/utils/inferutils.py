"""General-purpose inference utilities for synthetic-control estimators.

These tools operate on a treated outcome series and a donor matrix and are
deliberately agnostic to *how* the synthetic-control weights are produced, so
any estimator (or backend) that yields ℓ2-consistent weights can reuse them.
This is why they live at the shared ``utils/`` level rather than inside a single
estimator's helper package.

Currently provides :func:`debiased_sc_ttest`, the Chernozhukov, Wuthrich & Zhu
(2025, arXiv:1812.10820) debiased synthetic-control *t*-test for the ATT.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm as _norm
from scipy.stats import t as _t

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

WeightFn = Callable[[np.ndarray], np.ndarray]


def _outcome_only_simplex(y: np.ndarray, Y0: np.ndarray) -> np.ndarray:
    """The canonical SC weights of CWZ eq. (6): the outcome-only simplex fit.

    Solves ``min_w ||Y0 w - y||^2  s.t.  1'w = 1, w >= 0`` — the same program as
    ``scinference``'s ``estimators.R::sc`` (``limSolve::lsei``). Used as the
    default per-fold solver when the caller supplies no ``weight_fn``.
    """
    import cvxpy as cp

    J = Y0.shape[1]
    w = cp.Variable(J)
    cp.Problem(
        cp.Minimize(cp.sum_squares(Y0 @ w - y)),
        [cp.sum(w) == 1, w >= 0],
    ).solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000)
    if w.value is None:  # pragma: no cover - defensive: OSQP non-convergence
        raise MlsynthEstimationError(
            "outcome-only simplex SC failed to solve in debiased_sc_ttest"
        )
    return np.asarray(w.value).ravel()


def debiased_sc_ttest(
    y: np.ndarray,
    Y0: np.ndarray,
    T0: int,
    T1: int,
    K: int = 3,
    alpha: float = 0.1,
    weight_fn: Optional[WeightFn] = None,
) -> Dict[str, Any]:
    r"""Debiased synthetic-control *t*-test for the ATT (CWZ 2025).

    Implements the K-fold cross-fitting debiasing and self-normalized
    *t*-statistic of Chernozhukov, Wuthrich & Zhu (2025), a faithful port of the
    authors' ``scinference`` package (``ttest.R::sc.cf``). The pre-period is split
    into ``K`` consecutive blocks of length ``r = min(floor(T0/K), T1)``; for each
    block ``H_k`` the weights are refit on the block's complement and

    .. math::

       \widehat{\tau}_k = \operatorname{mean}_{t > T_0}(y_t - \mathbf{Y}_{0,t}
       \widehat{\mathbf{w}}_{(k)}) - \operatorname{mean}_{t \in H_k}(y_t -
       \mathbf{Y}_{0,t}\widehat{\mathbf{w}}_{(k)}),

    where the second (held-out pre-period) term estimates and removes the SC bias.
    The debiased ATT is :math:`\widehat{\tau} = K^{-1}\sum_k \widehat{\tau}_k`,
    the self-normalized statistic
    :math:`\sqrt{K}(\widehat{\tau}-\tau)/\widehat{\sigma}` with
    :math:`\widehat{\sigma} = \sqrt{1 + Kr/T_1}\,\operatorname{sd}(\widehat{\tau}_k)`
    is asymptotically ``t_{K-1}``, and the CI is
    :math:`\widehat{\tau} \pm t_{K-1}(1-\alpha/2)\,\widehat{\sigma}/\sqrt{K}`.

    Following ``scinference``, the ``K`` blocks are the *last* ``K*r``
    pre-treatment periods (offset ``T0 - K*r``).

    Parameters
    ----------
    y : ndarray, shape (T,)
        Treated-unit outcomes, time-ordered, ``T = T0 + T1``.
    Y0 : ndarray, shape (T, J)
        Donor outcomes (one column per donor), time-ordered.
    T0, T1 : int
        Number of pre- and post-treatment periods.
    K : int, default 3
        Number of cross-fitting folds; must be ``>= 2``.
    alpha : float, default 0.1
        Significance level for the two-sided confidence interval.
    weight_fn : callable, optional
        ``weight_fn(pre_keep_idx) -> w`` returning the ``(J,)`` SC weights fit on
        the given pre-period row indices (the complement of each held-out block).
        Defaults to the outcome-only simplex (matches the reference). A covariate
        backend passes a closure that refits on those pre-years plus all post.

    Returns
    -------
    dict
        ``att``, ``se``, ``tstat``, ``dof`` (``=K-1``), ``ci_lower``,
        ``ci_upper``, ``tau_k`` ((K,) array), ``K``, ``r``, ``alpha``.

    Raises
    ------
    MlsynthConfigError
        If ``K < 2`` or the block length ``r < 1`` (``K`` too large for ``T0`` or
        ``T1 < 1``).
    MlsynthDataError
        If ``y``/``Y0`` lengths are inconsistent with ``T0 + T1``.
    MlsynthEstimationError
        If ``weight_fn`` returns weights of the wrong shape, or the default
        solver fails.
    """
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise MlsynthConfigError(f"K must be an integer >= 2; got {K!r}.")

    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    if Y0.ndim != 2:
        raise MlsynthDataError("Y0 must be a 2-D (T, J) donor matrix.")
    T = T0 + T1
    if y.shape[0] != T:
        raise MlsynthDataError(
            f"len(y)={y.shape[0]} must equal T0+T1={T}."
        )
    if Y0.shape[0] != T:
        raise MlsynthDataError(
            f"Y0 has {Y0.shape[0]} rows; must equal T0+T1={T}."
        )

    r = min(T0 // K, T1)
    if r < 1:
        raise MlsynthConfigError(
            f"block length r=min(T0//K, T1)={r} < 1; need T0 >= K (got T0={T0}, "
            f"K={K}) and T1 >= 1 (got T1={T1})."
        )

    J = Y0.shape[1]
    if weight_fn is None:
        weight_fn = lambda idx: _outcome_only_simplex(y[idx], Y0[idx])

    y_pre, Y0_pre = y[:T0], Y0[:T0]
    y_post, Y0_post = y[T0:], Y0[T0:]
    offset = T0 - r * K

    tau = np.empty(K)
    for k in range(K):
        block = np.arange(offset + k * r, offset + k * r + r)
        keep = np.setdiff1d(np.arange(T0), block)
        w = np.asarray(weight_fn(keep), dtype=float).ravel()
        if w.shape != (J,):
            raise MlsynthEstimationError(
                f"weight_fn returned weights of shape {w.shape}; expected ({J},)."
            )
        post_gap = float(np.mean(y_post - Y0_post @ w))
        block_gap = float(np.mean(y_pre[block] - Y0_pre[block] @ w))
        tau[k] = post_gap - block_gap

    att = float(tau.mean())
    se = float(np.sqrt(1.0 + (K * r) / T1) * tau.std(ddof=1) / np.sqrt(K))
    tstat = att / se if se > 0 else np.inf * np.sign(att)
    crit = float(_t.ppf(1 - alpha / 2, K - 1))
    return {
        "att": att,
        "se": se,
        "tstat": tstat,
        "dof": K - 1,
        "ci_lower": att - crit * se,
        "ci_upper": att + crit * se,
        "tau_k": tau,
        "K": K,
        "r": r,
        "alpha": alpha,
    }


def rae(c0: float, K: int, alpha: float = 0.1) -> float:
    r"""Relative asymptotic efficiency of the K-fold debiased-SC CI (CWZ eq. 14).

    The ratio of the limiting expected CI length as :math:`K \to \infty` to its
    length at finite ``K``, as a function only of ``alpha``, ``K`` and
    ``c0 = T0/T1``. Reproduces the paper's Table 1. Used to guide the choice of
    ``K`` (higher ``K`` -> higher RAE / shorter intervals, traded against
    coverage accuracy).
    """
    K = int(K)
    if K < 2:
        raise MlsynthConfigError("rae requires K >= 2.")
    z = float(_norm.ppf(1 - alpha / 2))
    tq = float(_t.ppf(1 - alpha / 2, K - 1))
    if c0 < 1:
        g = float(K)
    elif c0 <= K:
        g = K / c0
    else:
        g = 1.0
    num = z * np.sqrt(min(1.0 / c0, 1.0)) * np.sqrt(1.0 + c0)
    gamma_ratio = float(np.exp(gammaln(K / 2) - gammaln((K - 1) / 2)))
    denom = (tq * (1.0 / (np.sqrt(K) * np.sqrt(K - 1)))
             * np.sqrt(1.0 + min(c0, K)) * np.sqrt(g) * np.sqrt(2.0) * gamma_ratio)
    return float(num / denom)


def _ar1(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation (AR(1) slope) of the demeaned residual series."""
    r = np.asarray(residuals, dtype=float).ravel()
    r = r - r.mean()
    if r.size < 2:
        return 0.0
    denom = float(np.dot(r[:-1], r[:-1]))
    if denom == 0.0:
        return 0.0
    return float(np.dot(r[1:], r[:-1]) / denom)


def select_K(
    T0: int,
    T1: int,
    residuals: np.ndarray,
    alpha: float = 0.1,
    rho_threshold: float = 0.5,
    block_min: int = 6,
    rae_target: float = 0.85,
    K_cap: int = 10,
) -> tuple:
    """Automatic choice of K for the debiased SC t-test (CWZ Section 3.2).

    A pragmatic operationalization of the paper's guidance: K=3 is the robust
    benchmark for small/moderate ``T0``; low persistence in the SC prediction
    errors bumps to K=4 for efficiency; and a larger ``T0`` (room for blocks of
    at least ``block_min`` periods) lets ``K`` climb to the smallest value whose
    RAE meets ``rae_target``. Persistence is gauged by an AR(1) fit to the SC
    pre-period residuals. Returns ``(K, info)``.

    For deliberate / publication work, choose ``K`` explicitly per Section 3.2;
    this is a sensible default, not a substitute for application-based judgement.
    """
    rho = _ar1(residuals)
    feasible = [K for K in range(2, K_cap + 1) if K <= T0 and T0 // K >= block_min]
    relaxed = False
    if not feasible:                       # T0 too small for block_min: relax it
        relaxed = True
        feasible = [K for K in range(2, min(K_cap, T0) + 1) if T0 // K >= 1] or [2]
    Kmin, Kmax = min(feasible), max(feasible)
    base = 4 if rho < rho_threshold else 3
    base = min(max(base, Kmin), Kmax)
    c0 = T0 / T1
    if relaxed:
        K = base
    else:
        climb = [K for K in feasible if K >= base and rae(c0, K, alpha) >= rae_target]
        K = min(climb) if climb else base
    info = {
        "rho_hat": float(rho), "K": int(K), "base": int(base),
        "K_feasible": (Kmin, Kmax), "rae": float(rae(c0, K, alpha)),
        "block_min": block_min, "relaxed": relaxed,
    }
    return int(K), info
