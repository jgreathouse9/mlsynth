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
