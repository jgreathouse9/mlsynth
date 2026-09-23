"""The GPITS engine: fit the Gaussian process, read off effects, run placebos.

Follows Cho (2026), Section 3.2. The GP is conditioned on the pre-treatment
periods only, and the posterior evaluated at the post-treatment inputs is the
counterfactual:

    mu_post  = K_*  (K + s2 I)^-1 y            (Eq. 12)
    Sig_post = K_** + s2 I - K_* (K + s2 I)^-1 K_*'   (Eq. 13)

Outcome and continuous covariates are standardised on the pre-period, so the
kernel reads every coordinate on a common scale and the prior's unit variance
means what it says. One-hot columns are scaled by sqrt(0.5) and never centred,
matching the reference implementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from .kernels import KERNELS, PERIODIC_KERNELS, getb_maxvar
from .structures import GPITSDesign, GPITSInputs, GPITSPlacebo

SQRT_HALF = float(np.sqrt(0.5))
# The reference fits the noise variance on [0.05, 1] of the standardised
# outcome, with optimize()'s bracket tolerance at 0.1.
S2_INTERVAL = (0.05, 1.0)
S2_TOL = 0.1


class _GP:
    """A Gaussian process conditioned on one training block."""

    def __init__(self, X_raw: np.ndarray, y: np.ndarray, kernel: str,
                 period: Optional[float], n_categorical: int,
                 length_scale: Optional[float], noise_variance: Optional[float]):
        self.kernel = kernel
        self.kern = KERNELS[kernel]
        self.n_cat = n_categorical

        self.y_mean = float(np.mean(y))
        self.y_sd = float(np.std(y, ddof=1))
        y_s = (y - self.y_mean) / self.y_sd

        X = np.array(X_raw, dtype=float, copy=True)
        cont = slice(n_categorical, X.shape[1])
        self.x_mean = np.zeros(X.shape[1])
        self.x_sd = np.ones(X.shape[1])
        self.x_mean[cont] = X[:, cont].mean(axis=0)
        sd = X[:, cont].std(axis=0, ddof=1)
        if np.any(sd <= 0):
            raise MlsynthEstimationError(
                "A continuous design column is constant over the training "
                "window, so it cannot be standardised."
            )
        self.x_sd[cont] = sd
        X[:, cont] = (X[:, cont] - self.x_mean[cont]) / self.x_sd[cont]
        X[:, :n_categorical] *= SQRT_HALF
        self.X = X

        # The period lives on the same axis as the time column, so it is
        # rescaled by that column's standard deviation. Time leads the
        # continuous block by construction (see setup).
        self.period_scaled = (None if kernel not in PERIODIC_KERNELS
                              else float(period) / self.x_sd[n_categorical])

        self.length_scale_selected = length_scale is None
        self.b = (getb_maxvar(X, kernel, self.period_scaled)
                  if length_scale is None else float(length_scale))
        self.K = self.kern(X, X, self.b, self.period_scaled)

        self.noise_variance_selected = noise_variance is None
        self.s2 = (self._fit_noise(y_s) if noise_variance is None
                   else float(noise_variance))

        n = X.shape[0]
        try:
            self.L = cholesky(self.K + self.s2 * np.eye(n), lower=False)
        except np.linalg.LinAlgError as exc:
            raise MlsynthEstimationError(
                f"The kernel matrix is not positive definite at b={self.b:.4g}, "
                f"s2={self.s2:.4g}; the Cholesky factorisation failed: {exc}"
            ) from exc
        self.alpha = cho_solve((self.L, False), y_s)

    def _log_marginal_likelihood(self, y_s: np.ndarray, s2: float) -> float:
        n = len(y_s)
        L = cholesky(self.K + s2 * np.eye(n), lower=True)
        a = cho_solve((L, True), y_s)
        return float(-0.5 * y_s @ a - np.sum(np.log(np.diag(L)))
                     - (n / 2.0) * np.log(2 * np.pi))

    def _fit_noise(self, y_s: np.ndarray) -> float:
        res = minimize_scalar(
            lambda s2: -self._log_marginal_likelihood(y_s, s2),
            bounds=S2_INTERVAL, method="bounded", options={"xatol": S2_TOL})
        return float(res.x)

    def predict(self, X_raw: np.ndarray, interval_type: str
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance on the original outcome scale."""
        X = np.array(X_raw, dtype=float, copy=True)
        cont = slice(self.n_cat, X.shape[1])
        X[:, cont] = (X[:, cont] - self.x_mean[cont]) / self.x_sd[cont]
        X[:, :self.n_cat] *= SQRT_HALF

        Ks = self.kern(X, self.X, self.b, self.period_scaled)
        Kss = self.kern(X, X, self.b, self.period_scaled)
        mean = Ks @ self.alpha * self.y_sd + self.y_mean
        v = solve_triangular(self.L, Ks.T, trans="T", lower=False)
        f_cov = Kss - v.T @ v
        if interval_type == "prediction":
            f_cov = f_cov + self.s2 * np.eye(f_cov.shape[0])
        return mean, self.y_sd ** 2 * f_cov


def fit_gpits(inputs: GPITSInputs, *, kernel: str, period: Optional[float],
              length_scale: Optional[float], noise_variance: Optional[float],
              interval_type: str, alpha: float) -> GPITSDesign:
    """Condition the GP on the pre-period and predict every period."""
    T0 = inputs.T0
    gp = _GP(inputs.design[:T0], inputs.y[:T0], kernel, period,
             inputs.n_categorical, length_scale, noise_variance)

    mean_all, cov_all = gp.predict(inputs.design, interval_type)
    se_all = np.sqrt(np.maximum(np.diag(cov_all), 0.0))
    post_cov = cov_all[T0:, T0:]

    return GPITSDesign(
        kernel=kernel,
        length_scale=gp.b,
        noise_variance=gp.s2,
        period_scaled=gp.period_scaled,
        length_scale_selected=gp.length_scale_selected,
        noise_variance_selected=gp.noise_variance_selected,
        counterfactual=mean_all,
        counterfactual_se=se_all,
        post_covariance=post_cov,
    )


def summarize_effects(inputs: GPITSInputs, design: GPITSDesign, alpha: float
                      ) -> Tuple[float, Tuple[float, float], np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray,
                                 List[Tuple[float, float]], Dict[str, float]]:
    """Effects, bands, cumulative totals and fit diagnostics.

    The cumulative interval uses the full post-period covariance block, not
    the diagonal: successive counterfactual errors covary, so summing their
    variances alone would understate the uncertainty in a running total.
    """
    T0 = inputs.T0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    obs = inputs.y
    cf = design.counterfactual
    gap = obs - cf
    lower = cf - z * design.counterfactual_se
    upper = cf + z * design.counterfactual_se

    post_gap = gap[T0:]
    n_post = post_gap.size
    cum = np.cumsum(post_gap)
    cum_ci: List[Tuple[float, float]] = []
    for i in range(n_post):
        var = float(design.post_covariance[:i + 1, :i + 1].sum())
        se = float(np.sqrt(max(var, 0.0)))
        cum_ci.append((float(cum[i] - z * se), float(cum[i] + z * se)))

    att = float(np.mean(post_gap))
    att_se = float(np.sqrt(max(float(design.post_covariance.sum()), 0.0)) / n_post)
    att_ci = (att - z * att_se, att + z * att_se)

    resid_pre = gap[:T0]
    ss_tot = float(np.sum((obs[:T0] - np.mean(obs[:T0])) ** 2))
    diagnostics = {
        "rmse_pre": float(np.sqrt(np.mean(resid_pre ** 2))),
        "rmse_post": float(np.sqrt(np.mean(post_gap ** 2))),
        "r_squared_pre": (float(1.0 - np.sum(resid_pre ** 2) / ss_tot)
                          if ss_tot > 0 else None),
        "att_std_err": att_se,
    }
    return att, att_ci, obs, cf, gap, lower, upper, cum, cum_ci, diagnostics


def run_placebo(inputs: GPITSInputs, *, placebo_periods: int, kernel: str,
                period: Optional[float], length_scale: Optional[float],
                noise_variance: Optional[float], interval_type: str,
                alpha: float) -> GPITSPlacebo:
    """Expanding-window one-step-ahead placebo checks (Section 3.3).

    For each of the last ``placebo_periods`` pre-treatment periods, refit on
    everything strictly earlier and predict that period. The window grows by
    one at each step, so every check is the same one-step-ahead extrapolation
    the estimator performs at the first post-treatment period.
    """
    T0 = inputs.T0
    if placebo_periods >= T0:
        raise MlsynthDataError(
            f"placebo_periods={placebo_periods} leaves no training data before "
            f"the first placebo target (T0={T0})."
        )
    first = T0 - placebo_periods
    if first < 3:
        raise MlsynthDataError(
            f"placebo_periods={placebo_periods} leaves only {first} training "
            f"periods before the first placebo target; at least 3 are needed."
        )

    z = float(norm.ppf(1.0 - alpha / 2.0))
    labels, taus, ses = [], [], []
    for i in range(placebo_periods):
        test = first + i
        train = slice(0, test)
        gp = _GP(inputs.design[train], inputs.y[train], kernel, period,
                 inputs.n_categorical, length_scale, noise_variance)
        mean, cov = gp.predict(inputs.design[test:test + 1], interval_type)
        labels.append(inputs.time_index.labels[test])
        taus.append(float(inputs.y[test] - mean[0]))
        ses.append(float(np.sqrt(max(float(cov[0, 0]), 0.0))))

    tau = np.asarray(taus, dtype=float)
    se = np.asarray(ses, dtype=float)
    return GPITSPlacebo(
        time_labels=np.asarray(labels),
        tau=tau, se=se,
        ci_lower=tau - z * se, ci_upper=tau + z * se,
        cover=np.abs(tau) <= z * se,
    )
