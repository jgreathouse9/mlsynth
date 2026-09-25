"""Inference procedures for FMA (Li & Sonnier 2023).

Four procedures live here and can run in any combination:

* :func:`asymptotic_inference` -- Theorem 3.1 (stationary) /
  Theorem 3.3 (non-stationary) normal CI for the ATT.
* :func:`bootstrap_inference` -- Web Appendix F residual bootstrap
  for per-period ``ATT_t`` CIs. Uses the pre-period residuals as the
  bootstrap distribution and refits the loading on each draw, so the
  CIs reflect the joint variability of the loading estimate and the
  idiosyncratic shock at time ``t``.
* :func:`percentile_t_inference` -- Wang, Racine & Wang (2025)
  studentized bootstrap for the average ATT. Draws the treated unit's
  bootstrap errors from ``N(0, sigma_tr^2)`` and inverts the order
  statistics of the studentized statistic, so the interval is not
  symmetric around the point estimate.
* :func:`placebo_inference` -- Web Appendix G placebo test where
  every control is treated as a pseudo-treated unit in turn. Returns
  the pointwise quantile band across the placebo ATT curves.

The two bootstraps answer different questions. :func:`bootstrap_inference`
gives a band around each post-period ``ATT_t``; :func:`percentile_t_inference`
gives one interval for the post-period average. They also differ in where the
treated unit's error distribution comes from: the Web Appendix F procedure
resamples that unit's own estimated residuals, while the percentile-t
procedure draws Gaussian errors with the estimated variance ``sigma_tr^2``,
which is what lets it stay valid when the treated and control idiosyncratic
variances differ.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from .factors import extract_factors
from .fit import estimate_loading_and_counterfactual


# ---------------------------------------------------------------------------
# Asymptotic (Theorem 3.1)
# ---------------------------------------------------------------------------

def asymptotic_inference(
    treated_outcome: np.ndarray,
    counterfactual: np.ndarray,
    factors_with_const: np.ndarray,
    residual_variance: float,
    T0: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """Theorem 3.1 normal CI for the ATT.

    Returns
    -------
    se_att, lower, upper, p_value
    """

    T = treated_outcome.shape[0]
    T2 = T - T0
    if T2 <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    gap = treated_outcome - counterfactual
    att = float(np.mean(gap[T0:]))

    F_pre = factors_with_const[:T0]
    F_post = factors_with_const[T0:]
    F_post_mean = F_post.mean(axis=0).reshape(-1, 1)
    # Ψ̂ = (X' X / T₁)⁻¹ (the population second-moment matrix's inverse),
    # NOT (X' X)⁻¹. Web Appendix A's Ω_1 = σ_tr² · φ · C' E[F_s F_s'] C
    # with C = E[F_s F_s']⁻¹ E[F_t]; plugging in sample analogues:
    #   Ω̂_1 = σ̂_tr² · (T₂/T₁) · F̄_post' (X'X/T₁)⁻¹ F̄_post.
    XtX_normalised = (F_pre.T @ F_pre) / max(T0, 1)
    try:
        psi_hat = np.linalg.inv(XtX_normalised)
    except np.linalg.LinAlgError:
        psi_hat = np.linalg.pinv(XtX_normalised)

    # Omega_hat = Omega1 + Omega2; both terms scale with σ_tr² (the
    # residual-variance estimate).
    omega1 = (T2 / max(T0, 1)) * float(residual_variance) * float(
        (F_post_mean.T @ psi_hat @ F_post_mean).item()
    )
    omega2 = float(residual_variance)
    omega_total = omega1 + omega2

    se_att = float(np.sqrt(max(omega_total, 0.0)) / np.sqrt(T2))
    if not np.isfinite(se_att) or se_att <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower = att - z * se_att
    upper = att + z * se_att
    p_value = 2.0 * (1.0 - float(norm.cdf(abs(att) / se_att)))
    return se_att, lower, upper, p_value


# ---------------------------------------------------------------------------
# Bootstrap (Web Appendix F)
# ---------------------------------------------------------------------------

def bootstrap_inference(
    treated_outcome: np.ndarray,
    factors: np.ndarray,
    counterfactual: np.ndarray,
    T0: int,
    alpha: float = 0.05,
    n_replicates: int = 1000,
    seed: int = 0,
) -> dict:
    """Web Appendix F residual bootstrap for per-period ATT_t CIs.

    Procedure
    ---------
    1. Compute pre-period residuals ``e_hat_1t = y_1t - F_aug_t' lambda_hat``
       for ``t = 1, ..., T0``.
    2. For each bootstrap draw b = 1, ..., B:
       a. Sample ``e*_1t`` from {e_hat} with replacement for every t.
       b. Form ``y*_1t = F_aug_t' lambda_hat + e*_1t``.
       c. Re-fit the loading on the bootstrap pre-period.
       d. Compute ``Delta*_1t = y*_1t - F_aug_t' lambda*_hat`` for t > T0.
    3. The (1 - alpha) CI for Delta_1t is
       ``[Delta_hat_1t - q_{1 - alpha/2}, Delta_hat_1t - q_{alpha/2}]``,
       with quantiles taken across bootstrap replicates of Delta*.
    """
    T = treated_outcome.shape[0]
    T2 = T - T0
    if T2 <= 0 or T0 < 2:
        return {
            "lower": np.asarray([], dtype=float),
            "upper": np.asarray([], dtype=float),
            "replicates": np.asarray([], dtype=float),
            "n_replicates": 0,
        }

    ones = np.ones((T, 1))
    F_aug = np.concatenate([ones, factors], axis=1)
    F_aug_pre = F_aug[:T0]
    F_aug_post = F_aug[T0:]

    # Loading from the observed treated pre-period
    XtX_pre = F_aug_pre.T @ F_aug_pre
    diag_mean = float(np.trace(XtX_pre) / max(F_aug_pre.shape[1], 1))
    XtX_reg = XtX_pre + max(diag_mean, 1.0) * 1e-10 * np.eye(F_aug_pre.shape[1])
    XtX_inv = np.linalg.inv(XtX_reg)

    lambda_hat = XtX_inv @ (F_aug_pre.T @ treated_outcome[:T0])
    counterfactual_post = F_aug_post @ lambda_hat
    delta_hat = treated_outcome[T0:] - counterfactual_post

    # Pre-period residuals -- bootstrap pool.
    residuals = treated_outcome[:T0] - F_aug_pre @ lambda_hat
    if residuals.size == 0:
        return {
            "lower": np.asarray([], dtype=float),
            "upper": np.asarray([], dtype=float),
            "replicates": np.asarray([], dtype=float),
            "n_replicates": 0,
        }

    rng = np.random.default_rng(seed)
    replicates = np.empty((n_replicates, T2), dtype=float)

    Pmat = XtX_inv @ F_aug_pre.T       # (r+1, T0) -- avoids re-solving each draw
    Hmat = F_aug_post @ Pmat            # (T2, T0) -- maps pre-resids -> post-CF
    Imat = np.eye(T2)
    for b in range(n_replicates):
        # Bootstrap residuals for the full T-period sequence; only the
        # pre-period subset affects lambda*_hat (Step 2c) while the
        # post-period subset enters as the y* component (Step 2a, b).
        e_star_pre = rng.choice(residuals, size=T0, replace=True)
        e_star_post = rng.choice(residuals, size=T2, replace=True)
        # lambda*_hat = (XtX)^{-1} X'_pre (F_pre lambda_hat + e_star_pre)
        #             = lambda_hat + (XtX)^{-1} X'_pre e_star_pre
        # so the post-period CF perturbation is H_mat @ e_star_pre.
        # Delta* = (F_post lambda_hat + e_star_post) - F_post lambda*_hat
        #        = e_star_post - H_mat @ e_star_pre
        replicates[b] = e_star_post - Hmat @ e_star_pre

    # CI for ATT_t = delta_hat[t] - quantile(replicates[:, t])
    q_lower = np.quantile(replicates, 1.0 - alpha / 2.0, axis=0)
    q_upper = np.quantile(replicates, alpha / 2.0, axis=0)
    lower = delta_hat - q_lower
    upper = delta_hat - q_upper

    return {
        "lower": lower,
        "upper": upper,
        "replicates": replicates,
        "n_replicates": int(n_replicates),
    }


# ---------------------------------------------------------------------------
# Percentile-t bootstrap (Wang, Racine & Wang 2025)
# ---------------------------------------------------------------------------

def robust_omega(
    factors_with_const: np.ndarray,
    T0: int,
    T2: int,
    residuals_pre: np.ndarray,
) -> Tuple[float, float, float]:
    r"""Appendix A.1's :math:`\hat\Omega`, the variance of the average ATT.

    Wang, Racine & Wang (2025, Appendix A.1) write the large-sample variance
    of :math:`\sqrt{T_2}(\widehat{ATT} - ATT)` as
    :math:`\Omega = \Omega_1 + \Omega_2` with

    .. math::

       \Omega_1 = \varphi\, \eta' \Psi \eta, \qquad
       \Psi = [E(f_t f_t')]^{-1} V [E(f_t f_t')]^{-1}, \qquad
       V = E(u_{0t}^2 f_t f_t'),

    :math:`\varphi = \lim T_2 / T_1`, :math:`\eta = E(f_t)` and
    :math:`\Omega_2 = E(u_{0t}^2)`. The sample analogues use the pre-period
    for the second moments and the post-period mean for :math:`\eta`:

    .. math::

       \hat\Omega_1 = \frac{T_2}{T_1}\, \hat\eta'
           \hat A^{-1} \hat V \hat A^{-1} \hat\eta, \qquad
       \hat A = T_1^{-1} \sum_{t \le T_1} \hat f_t \hat f_t', \qquad
       \hat V = T_1^{-1} \sum_{t \le T_1} \hat u_{0t}^2
           \hat f_t \hat f_t',

    and :math:`\hat\Omega_2 = T_1^{-1} \sum_{t \le T_1} \hat u_{0t}^2
    \equiv \hat\sigma^2_{tr}`.

    Writing :math:`c = \hat A^{-1} \hat\eta` and :math:`g_t = \hat f_t' c`
    collapses the sandwich to a weighted mean of squares, since
    :math:`\hat\eta' \hat A^{-1} (\hat f_t \hat f_t') \hat A^{-1} \hat\eta
    = g_t^2`:

    .. math::

       \hat\Omega_1 = \frac{T_2}{T_1} \cdot
           T_1^{-1} \sum_{t \le T_1} \hat u_{0t}^2 g_t^2 .

    This is the form implemented, and it makes the relationship to the
    Theorem 3.1 variance in :func:`asymptotic_inference` explicit. When
    :math:`\hat u_{0t}^2` is constant at :math:`\hat\sigma^2`,
    :math:`T_1^{-1}\sum g_t^2 = c' \hat A c = \hat\eta' \hat A^{-1}
    \hat\eta` and :math:`\hat\Omega_1` becomes
    :math:`\hat\sigma^2 (T_2/T_1) \hat\eta' \hat A^{-1} \hat\eta`, which is
    what that function computes. The sandwich is therefore the
    heteroskedasticity-robust generalisation of it: the two agree when the
    treated unit's error variance is constant over the pre-period and
    diverge when the large squared residuals sit at periods whose factor
    values matter most for the post-period projection.

    Parameters
    ----------
    factors_with_const : np.ndarray
        ``(T, r + 1)`` factor matrix with the leading constant column,
        as returned by
        :func:`~mlsynth.utils.fma_helpers.fit.estimate_loading_and_counterfactual`.
    T0 : int
        Pre-treatment periods (the paper's ``T1``).
    T2 : int
        Post-treatment periods.
    residuals_pre : np.ndarray
        Pre-period residuals :math:`\hat u_{0t}`, shape ``(..., T0)``. A
        leading axis is broadcast over, which is what the bootstrap uses to
        evaluate :math:`\hat\Omega^*` for every draw at once.

    Returns
    -------
    omega, omega1, omega2
        Scalars when ``residuals_pre`` is one-dimensional; arrays over the
        leading axis otherwise.
    """
    F_pre = factors_with_const[:T0]
    F_post = factors_with_const[T0:]
    eta = F_post.mean(axis=0)
    A = (F_pre.T @ F_pre) / max(T0, 1)
    try:
        c = np.linalg.solve(A, eta)
    except np.linalg.LinAlgError:
        c = np.linalg.pinv(A) @ eta
    g2 = (F_pre @ c) ** 2                       # (T0,)

    squared = np.asarray(residuals_pre, dtype=float) ** 2
    omega2 = squared.mean(axis=-1)
    omega1 = (T2 / max(T0, 1)) * (squared * g2).mean(axis=-1)
    omega = omega1 + omega2
    if np.ndim(omega) == 0:
        return float(omega), float(omega1), float(omega2)
    return omega, omega1, omega2


def percentile_t_inference(
    treated_outcome: np.ndarray,
    counterfactual: np.ndarray,
    factors_with_const: np.ndarray,
    T0: int,
    alpha: float = 0.05,
    n_replicates: int = 1000,
    seed: int = 0,
) -> dict:
    r"""Wang, Racine & Wang (2025) studentized bootstrap for the average ATT.

    Li & Sonnier's normal interval is correctly sized asymptotically but
    under-covers when the pre-period is short: at :math:`T_1 = 10` the
    paper's Table 1 reports 80.0-80.7% coverage for a nominal 95% interval.
    The fix is to take the critical values from a bootstrap distribution of
    the studentized statistic instead of from the normal table. Because the
    treated unit's bootstrap errors are drawn from
    :math:`N(0, \hat\sigma^2_{tr})`, estimated on that unit alone, the
    procedure does not assume the treated and control units share an error
    variance -- the assumption behind the Xu (2017) bootstrap, which the
    paper measures failing badly (46.8-78.7% coverage) when the variance
    ratio reaches 10.

    Procedure (Section 3.1)
    -----------------------
    Step 1. From the fitted model take :math:`\hat\lambda_0`,
    :math:`\widehat{ATT}`, :math:`\hat\Omega` (Appendix A.1, via
    :func:`robust_omega`) and
    :math:`\hat\sigma^2_{tr} = T_1^{-1} \sum_{t \le T_1} \hat u_{0t}^2`,
    where :math:`\hat u_{0t} = y_{0t} - \hat f_t' \hat\lambda_0`.

    Step 2. For each of ``M`` draws, sample
    :math:`u^*_{0t} \sim N(0, \hat\sigma^2_{tr})` for
    :math:`t = 1, \dots, T` and form
    :math:`y^*_{0t} = \hat f_t' \hat\lambda_0 + u^*_{0t}`. Refit the loading
    on the bootstrap pre-period,
    :math:`\hat\lambda^*_0 = (\sum_{t \le T_1} \hat f_t \hat f_t')^{-1}
    \sum_{t \le T_1} \hat f_t y^*_{0t}`, and take (Equation 7)

    .. math::

       \widehat{ATT}^* = \widehat{ATT}
         - T_2^{-1} \sum_{t > T_1} [\hat f_t' \hat\lambda^*_0
                                    - \hat f_t' \hat\lambda_0]
         + T_2^{-1} \sum_{t > T_1} u^*_{0t},

    then the studentized statistic (Equation 8)
    :math:`\hat S^* = \sqrt{T_2}(\widehat{ATT}^* - \widehat{ATT}) /
    \sqrt{\hat\Omega^*}`, with :math:`\hat\Omega^*` from Appendix A.2:
    :math:`\hat\Omega_1` and :math:`\hat\Omega_2` recomputed with
    :math:`\hat u^*_{0t} = y^*_{0t} - \hat f_t' \hat\lambda^*_0`. The
    factors are held fixed at :math:`f^*_t = \hat f_t`, so
    :math:`\hat A` and :math:`\hat\eta` do not change across draws.

    Step 3. Sort the statistics and invert (Equation 10):

    .. math::

       \Big[\widehat{ATT} - \hat S^*_{((1 - \alpha/2)M)}
              \sqrt{\hat\Omega / T_2},\;
            \widehat{ATT} - \hat S^*_{(\alpha M / 2)}
              \sqrt{\hat\Omega / T_2}\Big].

    The upper bound subtracts the lower order statistic; the interval is
    asymmetric around :math:`\widehat{ATT}` whenever the bootstrap
    distribution is skewed.

    Implementation
    --------------
    Subtracting :math:`y^*_{0t} = \hat f_t' \hat\lambda_0 + u^*_{0t}` from
    the refit gives
    :math:`\hat\lambda^*_0 - \hat\lambda_0 = (F'_{pre} F_{pre})^{-1}
    F'_{pre} u^*_{pre}`, so the draws enter linearly and
    :math:`\hat\lambda_0` itself never appears in the statistic. Both
    :math:`\widehat{ATT}^* - \widehat{ATT}` and
    :math:`\hat u^*_{0t} = u^*_{0t} - \hat f_t'(\hat\lambda^*_0 -
    \hat\lambda_0)` are then linear maps of the draw matrix, and all ``M``
    replicates are evaluated in one pass with no loop over draws.

    :math:`\hat\sigma^2_{tr}` is the uncorrected mean square that Step 1
    specifies, not the degrees-of-freedom-corrected ``residual_variance``
    that :func:`asymptotic_inference` is given.

    Its value then cancels. The numerator of :math:`\hat S^*` is linear in
    the draws and :math:`\hat\Omega^*` is quadratic in them, so the ratio
    is free of their scale and the returned interval is the same whatever
    variance the errors are drawn from; :math:`\hat\sigma^2_{tr}` enters
    the result only through the check that it is non-zero. That
    cancellation is also what makes the procedure indifferent to the
    treated/control variance ratio which breaks the Xu (2017) interval:
    Xu's is a percentile interval on :math:`\widehat{ATT}^*` itself, whose
    width carries the scale of whatever residuals it resampled.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    counterfactual : np.ndarray
        Fitted untreated path :math:`\hat f_t' \hat\lambda_0` over all
        ``T`` periods, so that the reported ``ATT`` and the residuals come
        from the same fit the estimator reports.
    factors_with_const : np.ndarray
        ``(T, r + 1)`` factor matrix with the constant column.
    T0 : int
        Pre-treatment periods.
    alpha : float
        Two-sided level; the interval has nominal coverage ``1 - alpha``.
    n_replicates : int
        Bootstrap draws ``M``. The paper uses 1,000.
    seed : int
        Seed for the Gaussian draws.

    Returns
    -------
    dict
        ``se_att`` (:math:`\sqrt{\hat\Omega / T_2}`, the robust standard
        error), ``lower``, ``upper``, ``p_value``, ``omega``, ``omega1``,
        ``omega2``, ``statistics`` (the ``M`` draws of :math:`\hat S^*`)
        and ``n_replicates`` (how many were finite). Every scalar is NaN
        and ``statistics`` empty when the interval is not computable.
    """
    empty = {
        "se_att": float("nan"),
        "lower": float("nan"),
        "upper": float("nan"),
        "p_value": float("nan"),
        "omega": float("nan"),
        "omega1": float("nan"),
        "omega2": float("nan"),
        "statistics": np.asarray([], dtype=float),
        "n_replicates": 0,
    }

    T = int(treated_outcome.shape[0])
    T2 = T - int(T0)
    if T2 <= 0 or int(T0) <= 0:
        return dict(empty)

    T0 = int(T0)
    n_columns = int(np.asarray(factors_with_const).shape[1])
    if T0 <= n_columns:
        # The residual maker I - F_pre (F'F)^-1 F' has rank T0 - n_columns, so
        # at T0 <= n_columns every bootstrap residual is identically zero and
        # Omega* is zero for every draw. The observed residuals are zero to
        # floating point too, which makes a numerical test on sigma^2_tr
        # unreliable -- the fit leaves 1e-32, not 0 -- so the check is on the
        # design.
        warnings.warn(
            "FMA percentile-t bootstrap: the pre-period has no residual "
            f"degrees of freedom (T0 = {T0} against {n_columns} regressors), "
            "so the treated unit's residual variance is not estimable and the "
            "studentized statistic is undefined. Returning NaN bounds. Fit "
            "fewer factors or use a longer pre-period.",
            UserWarning,
            stacklevel=2,
        )
        return dict(empty)
    gap = np.asarray(treated_outcome, dtype=float) - np.asarray(
        counterfactual, dtype=float
    )
    att = float(np.mean(gap[T0:]))
    residuals_pre = gap[:T0]

    # Step 1: sigma^2_tr and Omega_hat.
    sigma2_tr = float(np.mean(residuals_pre ** 2))
    omega, omega1, omega2 = robust_omega(
        factors_with_const=factors_with_const, T0=T0, T2=T2,
        residuals_pre=residuals_pre,
    )
    if not np.isfinite(omega) or omega <= 0.0 or sigma2_tr <= 0.0:
        warnings.warn(
            "FMA percentile-t bootstrap: the estimated treated-unit residual "
            f"variance is {sigma2_tr:.3e} and Omega_hat is {omega:.3e}, so the "
            "studentized statistic is undefined. The counterfactual "
            "reproduces the treated pre-period exactly, so there is no "
            "residual variation left to calibrate the error draws against. "
            "Returning NaN bounds.",
            UserWarning,
            stacklevel=2,
        )
        return dict(empty)

    F_pre = np.asarray(factors_with_const, dtype=float)[:T0]
    F_post = np.asarray(factors_with_const, dtype=float)[T0:]
    eta = F_post.mean(axis=0)

    XtX = F_pre.T @ F_pre
    try:
        P = np.linalg.solve(XtX, F_pre.T)            # (r+1, T0)
    except np.linalg.LinAlgError:                     # pragma: no cover
        # Unreachable on the estimator's own designs: FMA's factors are
        # orthogonal principal components, so XtX is non-singular whenever
        # T0 exceeds the column count, which the sigma2_tr guard above
        # already requires. Kept because robust_omega takes the same
        # fallback and the two must agree on degenerate input.
        P = np.linalg.pinv(XtX) @ F_pre.T

    # Step 2, vectorised over draws.
    rng = np.random.default_rng(seed)
    M = int(n_replicates)
    u_star = rng.standard_normal((M, T)) * np.sqrt(sigma2_tr)
    u_pre, u_post = u_star[:, :T0], u_star[:, T0:]

    # lambda*_0 - lambda_hat_0 = (F'F)^-1 F' u*_pre
    shift = u_pre @ P.T                              # (M, r+1)
    # ATT* - ATT = -eta'(lambda*_0 - lambda_hat_0) + mean(u*_post)
    att_star_gap = -(shift @ eta) + u_post.mean(axis=1)
    # u*_0t = u*_0t - f_t'(lambda*_0 - lambda_hat_0) for t <= T0
    resid_star = u_pre - shift @ F_pre.T             # (M, T0)

    omega_star, _, _ = robust_omega(
        factors_with_const=factors_with_const, T0=T0, T2=T2,
        residuals_pre=resid_star,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = np.sqrt(T2) * att_star_gap / np.sqrt(omega_star)
    finite = np.isfinite(statistics)
    statistics = statistics[finite]
    n_finite = int(statistics.size)
    if n_finite < 2:                                  # pragma: no cover
        # Requires Omega* to vanish on all but at most one draw. Past the
        # degrees-of-freedom guard the residual maker has rank >= 1, so
        # resid_star is a non-degenerate Gaussian and Omega* > 0 with
        # probability one. Kept so a future change to that guard degrades to
        # NaN bounds and a warning instead of returning inf.
        warnings.warn(
            "FMA percentile-t bootstrap: fewer than two draws produced a "
            "finite studentized statistic. Returning NaN bounds.",
            UserWarning,
            stacklevel=2,
        )
        return dict(empty)

    # Step 3: invert the order statistics. The paper indexes S*_(k) from 1,
    # so k = ceil(alpha M / 2) and ceil((1 - alpha/2) M) become these
    # zero-based positions.
    S_sorted = np.sort(statistics)
    k_lo = int(np.clip(np.ceil(alpha / 2.0 * n_finite), 1, n_finite)) - 1
    k_hi = int(np.clip(np.ceil((1.0 - alpha / 2.0) * n_finite), 1,
                       n_finite)) - 1
    root = float(np.sqrt(omega / T2))
    lower = att - float(S_sorted[k_hi]) * root
    upper = att - float(S_sorted[k_lo]) * root

    # Two-sided bootstrap p-value for H0: ATT = 0, by the same inversion.
    # The (1 + #)/(M + 1) convention keeps it strictly positive, so a
    # p-value is never reported as more extreme than M draws can resolve.
    S_null = float(np.sqrt(T2) * att / np.sqrt(omega))
    below = float(np.count_nonzero(S_sorted <= S_null))
    above = float(np.count_nonzero(S_sorted >= S_null))
    p_value = min(
        1.0,
        2.0 * min((1.0 + below) / (n_finite + 1.0),
                  (1.0 + above) / (n_finite + 1.0)),
    )

    return {
        "se_att": root,
        "lower": float(lower),
        "upper": float(upper),
        "p_value": float(p_value),
        "omega": float(omega),
        "omega1": float(omega1),
        "omega2": float(omega2),
        "statistics": statistics,
        "n_replicates": n_finite,
    }


# ---------------------------------------------------------------------------
# Placebo (Web Appendix G)
# ---------------------------------------------------------------------------

def placebo_inference(
    control_outcomes: np.ndarray,
    treated_outcome: np.ndarray,
    T0: int,
    n_factors: Optional[int],
    stationarity: str,
    preprocessing: str,
    alpha: float = 0.05,
    max_factors: int = 10,
) -> dict:
    """Web Appendix G: pseudo-ATT curves with each control as the treated unit.

    For each control ``k``, swap it into the treated slot, refit the
    factor model on the remaining ``N_co - 1`` controls, project the
    pseudo-treated pre-period onto those factors, compute the pseudo-
    ATT curve. The output band is the pointwise alpha/2 / (1 - alpha/2)
    quantile across the placebo curves at each period.

    Parameters
    ----------
    control_outcomes : np.ndarray
        ``(T, N_co)`` control panel.
    treated_outcome : np.ndarray
        Real treated outcome series, shape ``(T,)``. Used for the
        leading row of the curves matrix (so the caller can compare).
    T0 : int
        Pre-treatment periods.
    n_factors : int or None
        Number of factors to fit on each placebo iteration. ``None``
        means re-select per iteration via the criterion in
        ``stationarity``.
    stationarity : {"stationary", "nonstationary"}
    preprocessing : {"demean", "standardize"}
    alpha : float
        Significance level for the quantile band.
    max_factors : int
        Upper bound on the factor-selection routine.

    Returns
    -------
    dict
        Keys ``curves`` (``(N_co + 1, T)``; first row is the real
        treated unit), ``q_lower``, ``q_upper`` (``(T,)`` bands).
    """
    T, N_co = control_outcomes.shape

    # Real treated unit
    n_real, _, F_real, _ = extract_factors(
        control_outcomes,
        stationarity=stationarity,
        preprocessing=preprocessing,
        n_factors=n_factors,
        max_factors=max_factors,
    )
    _, cf_real, _, _ = estimate_loading_and_counterfactual(
        treated_outcome, F_real, T0
    )
    real_gap = treated_outcome - cf_real

    curves = np.empty((N_co + 1, T), dtype=float)
    curves[0] = real_gap

    for k in range(N_co):
        mask = np.ones(N_co, dtype=bool)
        mask[k] = False
        loo_controls = control_outcomes[:, mask]
        if loo_controls.shape[1] < 1:
            curves[k + 1] = np.nan
            continue
        try:
            _, _, F_k, _ = extract_factors(
                loo_controls,
                stationarity=stationarity,
                preprocessing=preprocessing,
                n_factors=n_factors,
                max_factors=min(max_factors, loo_controls.shape[1]),
            )
            _, cf_k, _, _ = estimate_loading_and_counterfactual(
                control_outcomes[:, k], F_k, T0
            )
        except Exception:
            curves[k + 1] = np.nan
            continue
        curves[k + 1] = control_outcomes[:, k] - cf_k

    placebo_band = curves[1:]
    # Skip rows with any NaNs from failed placebos.
    valid_mask = np.all(np.isfinite(placebo_band), axis=1)
    if valid_mask.any():
        q_lower = np.quantile(placebo_band[valid_mask], alpha / 2.0, axis=0)
        q_upper = np.quantile(placebo_band[valid_mask], 1.0 - alpha / 2.0, axis=0)
    else:
        q_lower = np.full(T, np.nan)
        q_upper = np.full(T, np.nan)

    return {
        "curves": curves,
        "q_lower": q_lower,
        "q_upper": q_upper,
    }
