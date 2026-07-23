"""General-purpose inference utilities for synthetic-control estimators.

These tools operate on a treated outcome series and a donor matrix and are
deliberately agnostic to *how* the synthetic-control weights are produced, so
any estimator (or backend) that yields ℓ2-consistent weights can reuse them.
This is why they live at the shared ``utils/`` level rather than inside a single
estimator's helper package.

Provides:

* :func:`debiased_sc_ttest`, the Chernozhukov, Wuthrich & Zhu (2025,
  arXiv:1812.10820) debiased synthetic-control *t*-test for the ATT;
* :func:`pda_prediction_intervals`, the Jiang, Li, Shen & Zhou (2025, *J. Appl.
  Econometrics*) bootstrap prediction intervals for the panel-data approach.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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

# A refit callback: given a bootstrap treated series (length T), refit the PDA
# point estimator on its pre-period and return ``(counterfactual, support)`` --
# the full-length predicted untreated outcome and the active-support column
# indices used for the studentization sandwich.
PDARefitFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


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


def split_conformal_quantile(residuals, alpha: float = 0.05) -> float:
    r"""Split-conformal prediction-band half-width (Chernozhukov, Wuthrich & Zhu 2021).

    Returns ``q``, the constant half-width of the symmetric prediction band
    ``counterfactual +/- q``: the ``ceil((n+1)(1-alpha))``-th order statistic of
    the absolute pre-period residuals (gaps). Under exchangeability of the
    residuals this band has finite-sample :math:`(1-\alpha)` coverage. When
    ``n < ceil(1/alpha) - 1`` the required order statistic does not exist and
    ``q`` is ``+inf`` (an uninformative band).

    This is the constant-width "split" construction used by R ``Synth``'s
    ``synth_inference(method = "conformal")`` (Hainmueller's j-hai/Synth), as
    distinct from the test-inversion conformal band (which widens over the
    post-period).

    Parameters
    ----------
    residuals : array-like
        Pre-treatment gaps (treated minus synthetic), one per pre-period.
    alpha : float
        Miscoverage level in ``(0, 1)``; the band targets ``1 - alpha`` coverage.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = r.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(r[k - 1]) if k <= n else float("inf")


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


# ---------------------------------------------------------------------------
# Jiang, Li, Shen & Zhou (2025) prediction intervals for the panel data approach
# ---------------------------------------------------------------------------

def _dwb_bandwidth(T0: int) -> int:
    """Dependent-wild-bootstrap kernel bandwidth ``floor(2 * T0**0.235)`` (Sec 2.3)."""
    return max(1, int(np.floor(2.0 * T0 ** 0.235)))


def _hac_bandwidth(T0: int) -> int:
    """HAC variance truncation lag ``floor(T0**0.24)`` (Sec 2.3)."""
    return max(1, int(np.floor(T0 ** 0.24)))


def _dwb_factor(T0: int, ell: int) -> np.ndarray:
    """Cholesky-like factor ``L`` of the Bartlett DWB covariance ``Sigma``.

    ``Sigma[s, t] = a((s - t) / ell)`` with the Bartlett kernel
    ``a(z) = max(1 - |z|, 0)`` (Condition 2.1). Correlated multipliers are then
    ``u = L z`` for standard-normal ``z`` -- each unit-variance and
    ``ell``-dependent. The factor is from an eigen-decomposition (clipping tiny
    negative eigenvalues), so it is robust even when ``Sigma`` is only PSD.
    """
    idx = np.arange(T0)
    d = np.abs(idx[:, None] - idx[None, :]) / float(ell)
    Sigma = np.clip(1.0 - d, 0.0, None)
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w)


def _hac_matrix(g: np.ndarray, K: int) -> np.ndarray:
    """Newey-West (Bartlett) HAC matrix of a mean-zero-able ``(T0, q)`` series."""
    T0 = g.shape[0]
    gc = g - g.mean(axis=0, keepdims=True)
    phi = (gc.T @ gc) / T0
    for l in range(1, min(K, T0 - 1) + 1):
        gl = (gc[l:].T @ gc[:-l]) / T0
        w = 1.0 - l / (K + 1.0)
        phi += w * (gl + gl.T)
    return phi


def _prediction_variance(
    X_pre_Q: np.ndarray, resid_pre: np.ndarray, X_post_Q: np.ndarray, K: int,
) -> Optional[np.ndarray]:
    """Per-post-period prediction variance ``V_t`` (post-selection OLS HAC sandwich).

    ``V_t = (1/T0) x_{t,Q}' S^{-1} Phi S^{-1} x_{t,Q}`` with
    ``S = (1/T0) sum_s x_{s,Q} x_{s,Q}'`` and ``Phi`` the HAC matrix of
    ``x_{s,Q} e_s``. Returns ``None`` when the support is empty or ``S`` is
    singular (``|Q| >= T0`` or collinear), signalling the ``sigma^2``-only
    fallback.
    """
    T0, q = X_pre_Q.shape
    if q == 0 or q >= T0:
        return None
    S = (X_pre_Q.T @ X_pre_Q) / T0
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:  # pragma: no cover - collinear support
        return None
    if not np.all(np.isfinite(Sinv)):  # pragma: no cover - near-singular support
        return None
    phi = _hac_matrix(X_pre_Q * resid_pre[:, None], K)
    M = Sinv @ phi @ Sinv
    return np.einsum("ti,ij,tj->t", X_post_Q, M, X_post_Q) / T0


def pda_prediction_intervals(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    *,
    counterfactual: np.ndarray,
    support: Sequence[int],
    refit: PDARefitFn,
    alpha: float = 0.05,
    n_boot: int = 999,
    dependent: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Bootstrap prediction intervals for the panel data approach (Jiang 2025).

    A faithful, estimator-agnostic implementation of Algorithm 2.1 of Jiang, Li,
    Shen & Zhou (2025): for each post-treatment period it constructs prediction
    intervals for the untreated potential outcome :math:`Y_t` and the treatment
    effect :math:`\Delta_t = y_t - \widehat Y_t`, using a dependent wild
    bootstrap for the pre-period prediction error and a simple residual bootstrap
    for the out-of-sample error. The bootstrap statistic is self-normalized by
    :math:`\sqrt{\widehat V_t + \widehat\sigma^2}`, where :math:`\widehat V_t` is
    the post-selection OLS HAC sandwich variance of the prediction and
    :math:`\widehat\sigma^2` the residual variance.

    Both the equal-tailed (``eq``) and symmetric (``sy``) intervals of the paper
    are returned, and the ``studentization`` field reports whether the sandwich
    variance was usable (``"sandwich"``) or the method fell back to
    :math:`\widehat\sigma^2` alone (``"sigma2"``; e.g. an empty or rank-deficient
    support, as for the dense L2-relaxation in high dimensions).

    Parameters
    ----------
    y : ndarray, shape (T,)
        Treated-unit outcomes, ``T = T0 + T1``.
    X : ndarray, shape (T, p)
        Control-unit outcomes / covariates (the donor design).
    T0 : int
        Number of pre-treatment periods.
    counterfactual : ndarray, shape (T,)
        The point estimator's fitted untreated outcome :math:`\widehat Y` over
        all ``T`` periods (the interval centre).
    support : sequence of int
        Column indices of ``X`` in the estimator's active set, used for the
        sandwich variance. Empty or rank-deficient triggers the ``sigma^2``
        fallback.
    refit : callable
        ``refit(y_boot) -> (cf_boot, support_boot)`` refits the point estimator
        on ``y_boot[:T0]`` and returns the full-length counterfactual and the
        bootstrap support indices.
    alpha : float, default 0.05
        Significance level (intervals have nominal coverage ``1 - alpha``).
    n_boot : int, default 999
        Number of bootstrap replications (``>= 2``).
    dependent : bool, default True
        Use the dependent wild bootstrap for the pre-period error (Bartlett
        multipliers). ``False`` uses ordinary i.i.d. standard-normal multipliers
        (Remark 2.2, valid when the errors are independent).
    seed : int, optional
        Seed for the bootstrap RNG.

    Returns
    -------
    dict
        ``alpha``, ``n_boot``, ``post_periods`` (``T1``), ``studentization``
        (``"sandwich"`` | ``"sigma2"``), ``se`` ((T1,) :math:`\sqrt{V_t +
        \sigma^2}`), and two blocks ``effect`` (for :math:`\Delta_t`) and
        ``counterfactual`` (for :math:`Y_t`), each a dict of ``point``,
        ``eq_lower``/``eq_upper`` (equal-tailed) and ``sy_lower``/``sy_upper``
        (symmetric), all ``(T1,)`` arrays.

    Raises
    ------
    MlsynthConfigError
        If ``n_boot < 2``, ``alpha`` is out of ``(0, 1)``, or ``T1 = T - T0 < 1``.
    MlsynthDataError
        If array shapes are inconsistent.
    """
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise MlsynthConfigError(f"n_boot must be an integer >= 2; got {n_boot!r}.")
    if not (0.0 < alpha < 1.0):
        raise MlsynthConfigError(f"alpha must be in (0, 1); got {alpha!r}.")

    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    counterfactual = np.asarray(counterfactual, dtype=float).ravel()
    if X.ndim != 2:
        raise MlsynthDataError("X must be a 2-D (T, p) control matrix.")
    T = y.shape[0]
    if X.shape[0] != T or counterfactual.shape[0] != T:
        raise MlsynthDataError(
            f"y, X, counterfactual must share length T; got {y.shape[0]}, "
            f"{X.shape[0]}, {counterfactual.shape[0]}."
        )
    T1 = T - T0
    if T1 < 1:
        raise MlsynthConfigError(f"need at least one post period; T0={T0}, T={T}.")

    support = np.asarray(list(support), dtype=int)
    rng = np.random.default_rng(seed)

    K = _hac_bandwidth(T0)
    ell = _dwb_bandwidth(T0)
    L = _dwb_factor(T0, ell) if dependent else None

    resid_pre = y[:T0] - counterfactual[:T0]
    sigma2 = float(np.mean(resid_pre ** 2))

    # Original-sample studentization scale.
    s2 = sigma2
    Vt = _prediction_variance(
        X[:T0][:, support], resid_pre, X[T0:][:, support], K
    ) if support.size else None
    studentization = "sandwich" if Vt is not None else "sigma2"
    se = np.sqrt((Vt if Vt is not None else 0.0) + s2)

    # Bootstrap.
    resid_centered = resid_pre - resid_pre.mean()
    S = np.empty((n_boot, T1))
    for b in range(n_boot):
        if dependent:
            e_pre = (L @ rng.standard_normal(T0)) * resid_pre
        else:
            e_pre = rng.standard_normal(T0) * resid_pre
        e_post = rng.choice(resid_centered, size=T1, replace=True)
        y_star = counterfactual.copy()
        y_star[:T0] += e_pre
        y_star[T0:] += e_post
        cf_star, supp_star = refit(y_star)
        cf_star = np.asarray(cf_star, dtype=float).ravel()
        supp_star = np.asarray(list(supp_star), dtype=int)
        resid_star_pre = y_star[:T0] - cf_star[:T0]
        s2_star = float(np.mean(resid_star_pre ** 2))
        Vt_star = _prediction_variance(
            X[:T0][:, supp_star], resid_star_pre, X[T0:][:, supp_star], K
        ) if supp_star.size else None
        se_star = np.sqrt((Vt_star if Vt_star is not None else 0.0) + s2_star)
        e_star_post = y_star[T0:] - cf_star[T0:]
        S[b] = e_star_post / np.where(se_star > 0, se_star, np.inf)

    # Quantiles per post-period.
    xi_lo = np.quantile(S, alpha / 2.0, axis=0)
    xi_hi = np.quantile(S, 1.0 - alpha / 2.0, axis=0)
    zeta = np.quantile(np.abs(S), 1.0 - alpha, axis=0)

    Y_hat = counterfactual[T0:]
    delta = y[T0:] - Y_hat

    cf_block = {
        "point": Y_hat,
        "eq_lower": Y_hat + xi_lo * se,
        "eq_upper": Y_hat + xi_hi * se,
        "sy_lower": Y_hat - zeta * se,
        "sy_upper": Y_hat + zeta * se,
    }
    # Delta_t = y_t - Y_t, so a (lo, hi) band on Y_t maps to (-hi, -lo) on Delta.
    eff_block = {
        "point": delta,
        "eq_lower": delta - xi_hi * se,
        "eq_upper": delta - xi_lo * se,
        "sy_lower": delta - zeta * se,
        "sy_upper": delta + zeta * se,
    }
    return {
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "post_periods": int(T1),
        "studentization": studentization,
        "se": se if np.ndim(se) else np.full(T1, float(se)),
        "effect": eff_block,
        "counterfactual": cf_block,
    }
