"""Masini & Medeiros (2021), ported: the weighted LASSO and the partial-resampling test.

The two pieces of *"Counterfactual Analysis With Artificial Controls: Inference,
High Dimensions, and Nonstationarity"*, JASA 116(536), 1773-1788. They live here
because two cases need them and neither owns them: ``arco_lasa`` runs the paper's
empirical application and ``arco_resampling_mc`` runs its size and power designs.

The method is a counterfactual in two steps. Fit the treated series on the
controls over the pre-intervention window with a LASSO whose penalty weights are
chosen per regressor from that series' trend type (their Equation 10, weights in
their Table 1), then test the post-intervention gaps by resampling blocks of the
pre-intervention residuals (their Theorem 2). What separates the test from the
alternatives is where the asymptotics run: on the pre-period only, so the number
of post-intervention periods stays fixed and a single post-period is admissible.

Ported from the authors' MATLAB in ``codes/`` of their replication package:
``computeBetaLasso.m`` for step one, ``ressampling.m`` for step two, and
``arco.m`` for the wiring between them. The MATLAB calls ``lasso`` from the
Statistics Toolbox; this uses :func:`sklearn.linear_model.lasso_path` over the
same standardized design, which reproduces their published table to five digits
(see :mod:`benchmarks.cases.arco_lasa`).

Two transcription notes.

The penalty weights enter as a diagonal rescaling of the design (their
Equation 12, ``W_t = L^-1 X_t``), and standardizing the columns is itself a
diagonal rescaling, so standardization absorbs any weight vector: with
``standardize=True`` every scheme in Table 1 returns the same fit. MATLAB's
``lasso`` standardizes by default and the authors did not turn it off, so their
published numbers are the ``w = 1`` column of that table. ``arco_lasa`` measures
this. Keeping the weights as an argument is what lets a case measure it, and the
reading that makes both facts consistent is that a column's pre-sample standard
deviation already carries the Table 1 order -- ``O(sqrt(T0))`` for a driftless
``I(1)``, ``O(T0)`` for a linear trend -- so standardization is a data-driven
stand-in for the table.

``arco.m`` line 23 passes ``mean(abs(u.^2))`` where the absolute-deviation
statistic is ``mean(abs(u))``; their ``farmtreat.m`` passes the latter. This port
follows ``farmtreat.m`` and the paper's Section 3.2. On the application's data
both give a p-value of zero, so the published table does not turn on it.
"""
from __future__ import annotations

import os
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

WeightMode = Literal["unit", "sqrt", "level", "auto"]

_LASA = os.path.join(os.path.dirname(__file__), "..", "basedata",
                     "masini_lasa_sales.parquet")

#: Treatment date of the retail price experiment; the 121st of 134 daily periods.
LASA_TREAT_DATE = pd.Timestamp("2016-10-18")


# --------------------------------------------------------------------------- #
# Step one: the weighted LASSO (their Equation 10, computeBetaLasso.m)
# --------------------------------------------------------------------------- #
def trend_weights(X: np.ndarray, mode: WeightMode = "unit") -> np.ndarray:
    """Per-regressor penalty weights from the trend type of each column.

    Their Table 1. ``X`` is the pre-intervention design, so its last row is
    ``X_{i,T0}``.

    Parameters
    ----------
    X : np.ndarray
        Pre-intervention design, ``(T0, p)``.
    mode : {"unit", "sqrt", "level", "auto"}
        ``"unit"`` gives every column weight 1, the I(0) row of the table and
        the setting behind the authors' published numbers. ``"sqrt"`` gives
        ``sqrt(T0)``, the driftless ``I(1)`` row. ``"level"`` gives
        ``|X_{i,T0}|``, the row shared by polynomial trends and ``I(1)`` with
        drift. ``"auto"`` picks per column by the pretest of their Section 4:
        an augmented Dickey-Fuller rejection gives 1, otherwise a zero-mean
        first difference gives ``sqrt(T0)`` and a non-zero one gives
        ``|X_{i,T0}|``.

    Returns
    -------
    np.ndarray
        Strictly positive weights, ``(p,)``. A column whose weight would be
        zero (a constant column, or one ending at zero) takes 1, since dividing
        the design by it is undefined.
    """
    T0 = X.shape[0]
    if mode == "unit":
        return np.ones(X.shape[1])
    if mode == "sqrt":
        return np.full(X.shape[1], float(np.sqrt(T0)))
    if mode == "level":
        return _positive(np.abs(X[-1]))
    if mode != "auto":
        raise ValueError(f"unknown weight mode {mode!r}")

    import warnings

    from statsmodels.tsa.stattools import adfuller
    w = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.ptp(col) == 0:                      # constant: nothing to test
            continue
        try:
            # A periodic indicator column (the application's day-of-week
            # dummies) makes the pretest's lag augmentation rank-deficient.
            # The pretest still returns, and the branch below sends such a
            # column to weight 1 either way.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_adf = adfuller(col, regression="ct", autolag="AIC")[1]
        except Exception:                          # degenerate column
            continue
        if p_adf < 0.05:                           # stationary around a trend
            w[j] = 1.0
            continue
        d = np.diff(col)
        sd = d.std(ddof=1)
        drift = sd > 0 and abs(d.mean()) / (sd / np.sqrt(d.size)) > 1.96
        w[j] = abs(col[-1]) if drift else np.sqrt(T0)
    return _positive(w)


def _positive(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    return np.where(w > 0, w, 1.0)


def wlasso(
    y: np.ndarray,
    X: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    standardize: bool = True,
    n_lambda: int = 100,
    eps: float = 1e-4,
) -> Tuple[float, np.ndarray, int, float]:
    """Fit their Equation (10) and select the penalty by BIC.

    ``computeBetaLasso.m``: divide the design by the weights, run LASSO over a
    descending penalty path capped at ``round(T0 ** 0.9)`` active coefficients,
    score each with ``log(MSE) + log(T0) * k / T0`` where ``k`` counts the
    intercept, and take the minimum.

    Parameters
    ----------
    y, X : np.ndarray
        Pre-intervention outcome ``(T0,)`` and design ``(T0, p)``.
    weights : np.ndarray, optional
        Penalty weights from :func:`trend_weights`. ``None`` is ``w = 1``.
    standardize : bool, default True
        Scale columns to unit variance before fitting, as MATLAB's ``lasso``
        does by default. See this module's docstring for what it does to
        ``weights``.
    n_lambda, eps : int, float
        Penalty path length and its ratio of smallest to largest penalty.

    Returns
    -------
    (intercept, beta, n_selected, lam)
        ``beta`` is on the scale of ``X``, so the fitted counterfactual on any
        design row is ``intercept + row @ beta``.
    """
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    T0 = y.size
    w = np.ones(X.shape[1]) if weights is None else _positive(weights)
    Z = X / w

    if standardize:
        mu, sd = Z.mean(0), Z.std(0)
        sd = np.where(sd > 0, sd, 1.0)
    else:
        mu, sd = np.zeros(Z.shape[1]), np.ones(Z.shape[1])
    Zs = (Z - mu) / sd

    dfmax = int(round(T0 ** 0.9))
    yc = y - y.mean()
    lams, coefs, _ = lasso_path(Zs, yc, alphas=n_lambda, eps=eps)

    best_bic, best = np.inf, None
    for j, lam in enumerate(lams):
        bz = coefs[:, j]
        k = int(np.count_nonzero(bz))
        if k > dfmax:
            continue
        mse = float(np.mean((yc - Zs @ bz) ** 2))
        if mse <= 0:                               # pragma: no cover - exact fit
            continue
        bic = np.log(mse) + np.log(T0) * (k + 1) / T0
        if bic < best_bic:
            best_bic, best = bic, (bz / sd / w, k, float(lam))
    if best is None:                               # pragma: no cover - dfmax=0
        raise RuntimeError("no penalty on the path respected the dfmax cap")

    beta, k, lam = best
    return float(y.mean() - X.mean(0) @ beta), beta, k, lam


# --------------------------------------------------------------------------- #
# Step two: the partial-resampling test (their Theorem 2, ressampling.m)
# --------------------------------------------------------------------------- #
def partial_resampling(
    phi: Callable[[np.ndarray], float],
    residuals: np.ndarray,
    gaps: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float, np.ndarray]:
    """p-value and quantiles from sliding a post-length window over the residuals.

    Their Section 3.2. The fit is made once on the whole pre-period; its
    residuals are then read in overlapping blocks of length ``T1``, giving
    ``T0 - T1 + 1`` draws of the statistic under the null. The p-value of their
    two-tailed form is ``1 - Qhat(|phi|) + Qhat(-|phi|)``.

    ``phi`` takes a length-``T1`` array and returns a scalar; the paper's
    choices are ``lambda x: np.mean(x ** 2)``, ``lambda x: np.mean(np.abs(x))``
    and the Euclidean norm.

    Parameters
    ----------
    phi : callable
        The test statistic.
    residuals : np.ndarray
        Pre-intervention residuals ``(T0,)``.
    gaps : np.ndarray
        Post-intervention gaps ``(T1,)``, whose ``phi`` is the observed statistic.
    alpha : float, default 0.05
        Two-tailed level for the returned quantiles.

    Returns
    -------
    (p_value, lower, upper, draws)
        ``lower`` / ``upper`` are the ``alpha/2`` and ``1 - alpha/2`` quantiles
        of the null draws, and ``draws`` is the whole null distribution.
    """
    residuals = np.asarray(residuals, float).ravel()
    gaps = np.asarray(gaps, float).ravel()
    T0, T1 = residuals.size, gaps.size
    if T1 < 1 or T0 < T1:
        raise ValueError(f"need T0 >= T1 >= 1, got T0={T0}, T1={T1}")

    n_blocks = T0 - T1 + 1
    draws = np.array([float(phi(residuals[j:j + T1])) for j in range(n_blocks)])
    stat = float(phi(gaps))

    q_upper = float(np.mean(draws <= abs(stat)))
    q_lower = float(np.mean(draws <= -abs(stat)))
    p = 1.0 - q_upper + q_lower
    # ``ressampling.m`` reads the quantiles off MATLAB's ``quantile``, which is
    # Hyndman-Fan type 5 -- Hazen plotting positions (k - 0.5) / n. NumPy's
    # default is type 7, and on this application's null draws the two differ by
    # up to 1811 units on the squared statistic, which is 1.2% of the upper
    # band. Cross-checked against Octave running their file verbatim, where
    # ``method="hazen"`` agrees to machine precision and the default does not.
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], method="hazen")
    return p, float(lo), float(hi), draws


# --------------------------------------------------------------------------- #
# The application's panel
# --------------------------------------------------------------------------- #
def load_lasa() -> pd.DataFrame:
    """The retail panel of their Section 6, long, ready for ``dataprep``.

    233 Brazilian municipalities over 134 daily periods (2016-06-20 to
    2016-10-31); in 107 of them the price of one product rose on 2016-10-18 and
    stayed up for the remaining 14 days.

    Their ``arco.m`` treats the sum over the 107 as the series to explain, so
    the frame returned here carries that aggregate as a unit named ``0``
    alongside the 126 untouched municipalities. ``dataprep`` then reads the
    aggregate as the treated unit and the 126 as the donor pool.
    """
    raw = pd.read_parquet(os.path.abspath(_LASA))
    controls = raw[raw.treated_group == 0].copy()
    treated = (raw[raw.treated_group == 1]
               .groupby("date", as_index=False)["quantity"].sum())
    treated["municipality"] = 0
    treated["treated_group"] = 1
    treated["shops"] = int(raw.loc[raw.treated_group == 1, ["municipality", "shops"]]
                           .drop_duplicates()["shops"].sum())
    treated["treat"] = (treated.date >= LASA_TREAT_DATE).astype("int8")
    return (pd.concat([treated, controls], ignore_index=True)
            .sort_values(["municipality", "date"], ignore_index=True))


def weekday_dummies(dates: pd.DatetimeIndex) -> np.ndarray:
    """Seven day-of-week indicators, in MATLAB's ``weekday`` order (1 = Sunday).

    ``arco.m`` appends ``dummyvar(weekday(data))`` to the control columns, which
    is seven columns and not six; with the intercept the design is collinear,
    and the LASSO handles it. Reproducing their regressor count of 133 needs all
    seven.
    """
    matlab_weekday = (pd.DatetimeIndex(dates).dayofweek + 1) % 7 + 1
    return np.eye(7)[matlab_weekday - 1]


# --------------------------------------------------------------------------- #
# The simulation design of their Section 5
# --------------------------------------------------------------------------- #
def simulate_masini_panel(
    rng: np.random.Generator,
    *,
    design: Literal["deterministic", "stochastic"] = "deterministic",
    T: int = 100,
    n: int = 200,
    s0: int = 5,
) -> np.ndarray:
    """One draw of their Equations (15)-(17). Unit 0 is the treated one.

    ``Z_it = c_i + mu_i F_t + U_it`` with ``mu_i = 1`` for the first ``s0 + 1``
    units and 0 after, standard normal innovations, and ``c_i = 0`` -- the paper
    leaves ``c_i`` unspecified and the regression carries an unpenalized
    intercept either way. ``design="deterministic"`` is their trend-stationary
    ``F_t = t + U^F_t`` (Equation 17 at the baseline ``f^F_t = t``);
    ``"stochastic"`` is their driftless unit root ``F_t = F_{t-1} + U^F_t``
    (Equation 16 at ``f^F_t = 0``).

    Cross-validated against an independent transcription of the same equations
    in MATLAB syntax, run under Octave, which returns the same rejection rates
    to Monte Carlo error.

    Returns
    -------
    np.ndarray
        ``(T, n)``, the treated unit in column 0.
    """
    uF = rng.standard_normal(T)
    F = np.cumsum(uF) if design == "stochastic" else np.arange(1, T + 1) + uF
    mu = np.zeros(n)
    mu[:s0 + 1] = 1.0
    return mu[None, :] * F[:, None] + rng.standard_normal((T, n))


def sigma_at_T0(design: str, T0: int) -> float:
    """``sigma`` of their Table 4: the standard deviation of unit 1 at ``t = T0``.

    ``sqrt(2)`` for the trend-stationary design, where the factor shock and the
    idiosyncratic shock each contribute a unit variance; ``sqrt(T0 + 1)`` for
    the driftless unit root, whose factor has accumulated ``T0`` innovations.
    """
    return float(np.sqrt(2.0) if design == "deterministic" else np.sqrt(T0 + 1))
