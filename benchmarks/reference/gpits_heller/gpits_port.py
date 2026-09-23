"""NumPy port of gpss::gp_its (Cho 2026, "Let Time Tell").

Traced against doeun-kim/gpss:
  R/gp_its.R          gp_its()            -- pre/post split, effects, SEs
  R/gp_functions.R    gp_train()          -- scaling, b, s2, posterior
                      gp_predict()        -- Algorithm 2.1 (Rasmussen-Williams)
                      gp_optimize()       -- s2 by MLE on [0.05, 1], tol 0.1
  R/helper_functions.R getb_maxvar()      -- b by max variance of lower-tri K
                      mixed_data_processing(), one_hot()
  src/kernels.cpp     kernel_gaussian_periodic_linear() and its symmetric twin

Conventions preserved from the reference, including the ones that look odd:
  * the periodic and linear components run over ALL columns of the design,
    one-hot month dummies included, not the time column alone;
  * one-hot columns are multiplied by sqrt(0.5) and are never centred/scaled;
  * the period is rescaled by the SD of the first continuous column (time);
  * column order after processing is [one-hot dummies, continuous];
  * `optimize()` in R is Brent on a bounded interval with tol=0.1 by default.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize_scalar

__all__ = ["gp_its", "gp_its_placebo", "GPITSFit"]

SQRT_HALF = np.sqrt(0.5)


# --------------------------------------------------------------------------
# kernels (src/kernels.cpp)
# --------------------------------------------------------------------------
def _sqdist(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def kernel_gpl(X1, X2, b, period):
    """gaussian + periodic + linear, exactly as kernel_gaussian_periodic_linear."""
    gauss = np.exp(-_sqdist(X1, X2) / b)
    diff = X1[:, None, :] - X2[None, :, :]
    per_sum = (2.0 * np.sin(np.pi * np.abs(diff) / period) ** 2).sum(-1)
    periodic = np.exp(-per_sum / (b / 2.0))
    linear = X1 @ X2.T
    return gauss + periodic + linear


def kernel_gaussian(X1, X2, b, period=None):
    return np.exp(-_sqdist(X1, X2) / b)


_KERNELS = {"gaussian_periodic_linear": kernel_gpl, "gaussian": kernel_gaussian}


# --------------------------------------------------------------------------
# hyperparameters
# --------------------------------------------------------------------------
R_OPTIMIZE_TOL = np.finfo(float).eps ** 0.25   # R optimize() default tol


def getb_maxvar(X, kernel_type, period, maxsearch_b=2000.0,
                tol=R_OPTIMIZE_TOL):
    """b maximising var of the strict lower triangle of K (helper_functions.R).

    R's var() is the n-1 denominator; matched here with ddof=1.
    """
    kern = _KERNELS[kernel_type]
    tril = np.tril_indices(X.shape[0], -1)

    def neg_var(b):
        try:
            K = kern(X, X, b, period)
            return -np.var(K[tril], ddof=1)
        except Exception:
            return 0.0

    res = minimize_scalar(neg_var, bounds=(0.01, maxsearch_b), method="bounded",
                          options={"xatol": tol})
    return float(res.x)


def log_marginal_likelihood(K, y, s2):
    """log_marginal_likelihood_cpp: -0.5 y'a - sum(log diag L) - n/2 log(2pi).

    The reference uses sum(log(diag(L))) with L the LOWER Cholesky, which is
    half the log-determinant; reproduced verbatim so the s2 optimum matches.
    """
    n = len(y)
    L = cholesky(K + s2 * np.eye(n), lower=True)
    a = cho_solve((L, True), y)
    return float(-0.5 * y @ a - np.sum(np.log(np.diag(L))) - (n / 2.0) * np.log(2 * np.pi))


def gp_optimize(K, y, tol=0.1, interval=(0.05, 1.0)):
    """s2 by MLE. R's optimize() is Brent with tol on the bracket width."""
    res = minimize_scalar(lambda s2: -log_marginal_likelihood(K, y, s2),
                          bounds=interval, method="bounded",
                          options={"xatol": tol})
    return float(res.x)


# --------------------------------------------------------------------------
# design matrix (mixed_data_processing + gp_train scaling block)
# --------------------------------------------------------------------------
def _build_design(time_idx, month, levels):
    """[one-hot month (sqrt(.5)-scaled later), time] -- cat first, then cont."""
    onehot = np.zeros((len(month), len(levels)))
    for j, lv in enumerate(levels):
        onehot[:, j] = (month == lv).astype(float)
    return np.column_stack([onehot, time_idx.astype(float)])


class GPITSFit:
    """Trained model; mirrors the gp_train return object."""

    def __init__(self, X_raw, y, kernel_type, period, b=None,
                 optimize=True, n_cat=0, b_tol=R_OPTIMIZE_TOL):
        self.kernel_type = kernel_type
        self.kern = _KERNELS[kernel_type]
        self.n_cat = n_cat

        self.y_mean = float(np.mean(y))
        self.y_sd = float(np.std(y, ddof=1))
        y_s = (y - self.y_mean) / self.y_sd

        X = X_raw.copy()
        cont = slice(n_cat, X.shape[1])
        self.x_mean = np.zeros(X.shape[1])
        self.x_sd = np.ones(X.shape[1])
        self.x_mean[cont] = X[:, cont].mean(axis=0)
        self.x_sd[cont] = X[:, cont].std(axis=0, ddof=1)
        X[:, cont] = (X[:, cont] - self.x_mean[cont]) / self.x_sd[cont]
        # period rescaled by the SD of the FIRST continuous column (time)
        self.period_scaled = None if period is None else period / self.x_sd[n_cat]
        X[:, :n_cat] *= SQRT_HALF
        self.X = X

        self.b = (getb_maxvar(X, kernel_type, self.period_scaled, tol=b_tol)
                  if b is None else b)
        self.K = self.kern(X, X, self.b, self.period_scaled)
        self.s2 = gp_optimize(self.K, y_s) if optimize else 0.3

        n = X.shape[0]
        self.L = cholesky(self.K + self.s2 * np.eye(n), lower=False)  # R chol(): upper
        self.alpha = cho_solve((self.L, False), y_s)
        self.y_s = y_s

    def transform(self, X_raw):
        X = X_raw.copy()
        cont = slice(self.n_cat, X.shape[1])
        X[:, cont] = (X[:, cont] - self.x_mean[cont]) / self.x_sd[cont]
        X[:, :self.n_cat] *= SQRT_HALF
        return X

    def predict(self, X_raw):
        """Returns (mean, Ys_cov, f_cov) on the original outcome scale."""
        Xs = self.transform(X_raw)
        Ks = self.kern(Xs, self.X, self.b, self.period_scaled)
        Kss = self.kern(Xs, Xs, self.b, self.period_scaled)
        mean = Ks @ self.alpha * self.y_sd + self.y_mean
        v = solve_triangular(self.L, Ks.T, trans="T", lower=False)
        f_cov = Kss - v.T @ v
        Ys_cov = f_cov + self.s2 * np.eye(f_cov.shape[0])
        return mean, self.y_sd**2 * Ys_cov, self.y_sd**2 * f_cov


def gp_its(y, month, n_pre, kernel_type="gaussian_periodic_linear", period=12,
           interval_type="prediction", alpha=0.05, optimize=True, b=None,
           b_tol=R_OPTIMIZE_TOL):
    """gp_its() for a single unit. `month` is the categorical covariate array.

    Returns a dict with the counterfactual, per-period and cumulative effects,
    and their standard errors, matching the reference's naming.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    n = len(y)
    time_idx = np.arange(1, n + 1)
    levels = np.unique(month[:n_pre])         # training levels only (CASE 2)
    X_all = _build_design(time_idx, np.asarray(month), levels)

    fit = GPITSFit(X_all[:n_pre], y[:n_pre], kernel_type, period, b=b,
                   optimize=optimize, n_cat=len(levels), b_tol=b_tol)
    mean_post, ys_cov, f_cov = fit.predict(X_all[n_pre:])
    post_cov = ys_cov if interval_type == "prediction" else f_cov

    z = norm.ppf(1 - alpha / 2)
    y0_se = np.sqrt(np.maximum(np.diag(post_cov), 0))
    n_post = n - n_pre
    tau_t = y[n_pre:] - mean_post
    tau_cum = np.cumsum(tau_t)
    tau_cum_se = np.sqrt(np.maximum(
        [post_cov[:i + 1, :i + 1].sum() for i in range(n_post)], 0))
    return dict(
        b=fit.b, s2=fit.s2, n_pre=n_pre,
        counterfactual=mean_post, y0_se=y0_se,
        tau_t=tau_t, tau_t_se=y0_se,
        tau_cum=tau_cum, tau_cum_se=tau_cum_se,
        tau_cum_lwr=tau_cum - z * tau_cum_se, tau_cum_upr=tau_cum + z * tau_cum_se,
        tau_avg=tau_cum / np.arange(1, n_post + 1),
        tau_avg_se=tau_cum_se / np.arange(1, n_post + 1),
    )


def gp_its_placebo(y, month, n_pre, placebo_periods=4, **kw):
    """Expanding-window one-step-ahead placebo (.run_placebo_checks).

    Refits on {1, ..., k-1} and predicts period k, for the last
    `placebo_periods` pre-treatment periods, growing the window each step.

    The reference passes the test row to gp_train as `Xtest`, so the one-hot
    level set is taken from train and test together; that only matters when a
    month is absent from the truncated training window.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    month = np.asarray(month)
    kernel_type = kw.get("kernel_type", "gaussian_periodic_linear")
    period = kw.get("period", 12)
    interval_type = kw.get("interval_type", "prediction")
    z = norm.ppf(1 - kw.get("alpha", 0.05) / 2)

    fake_treat = n_pre - placebo_periods          # 0-based index of first test
    time_idx = np.arange(1, len(y) + 1)
    out = []
    for i in range(placebo_periods):
        test = fake_treat + i
        train = np.arange(0, test)
        levels = np.unique(np.concatenate([month[train], month[[test]]]))
        X = _build_design(time_idx, month, levels)
        fit = GPITSFit(X[train], y[train], kernel_type, period,
                       optimize=kw.get("optimize", True), n_cat=len(levels))
        mean, ys_cov, f_cov = fit.predict(X[[test]])
        se = float(np.sqrt((ys_cov if interval_type == "prediction" else f_cov)[0, 0]))
        tau = float(y[test] - mean[0])
        out.append(dict(time_id=test + 1, relative_time=i + 1, tau=tau, se=se,
                        ci_lwr=tau - z * se, ci_upr=tau + z * se,
                        z_score=tau / se, cover=abs(tau / se) <= z,
                        y_actual=float(y[test]), y_predicted=float(mean[0])))
    return out
