"""The Time-Based Regression posterior, from Kerman, Wang and Vaver (2017).

Section 3.2 fits the pretest relation between the two group aggregates,

    y_t = alpha + beta x_t + eps_t,    eps_t ~ N(0, sigma^2),   (eqn 1)

and section 9.1 derives the posterior of the cumulative causal effect in closed
form under a prior uniform on ``(alpha, beta, log sigma)``. The median is

    Delta(T) = T (ybar_T - alpha - xbar_T beta)                  (eqn 4)

and the scale of its t-distribution, on ``n - 2`` degrees of freedom for ``n``
pretest points, is

    T s (v_a + 2 xbar_T v_ab + v_b xbar_T^2 + 1/T)^(1/2)         (eqn 6)

where ``v_a``, ``v_b`` and ``v_ab`` are entries of the unscaled
``V = (X'X)^-1`` and ``s`` is the classical residual standard deviation. The
first three terms carry the uncertainty in the fitted parameters and the ``1/T``
the unobserved errors of the test period itself, so the parameter contribution
grows in ``T^2`` and the observation contribution only in ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from scipy import stats

from ..groupfit import (
    GroupSums, fit_on_sums, group_sums, is_identified, prediction_variance)
from ..groupfit import unscaled_cov as _identified_unscaled_cov

#: The sums this module used to define. They are the shared regression's
#: sufficient statistics, so the definition lives with the regression and this
#: name is kept as the one TBR's own docstrings refer to.
PretestSums = GroupSums


@dataclass(frozen=True)
class PretestFit:
    """The fitted eqn 1, with everything the posterior needs from it."""

    alpha: float
    beta: float
    sigma_sq: float
    df: int
    n_pretest: int
    unscaled_cov: np.ndarray
    rank_deficient: bool
    sums: PretestSums
    resid: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.alpha + self.beta * np.asarray(x, dtype=float)


def fit_pretest(y_pre: np.ndarray, x_pre: np.ndarray) -> PretestFit:
    """Least squares on eqn 1, with the unscaled covariance section 9.1 uses.

    The design is two columns wide, so the covariance section 9.1 needs is
    available in closed form and ``_unscaled_cov`` computes it there.

    When the regressor is constant through the pretest ``X'X`` is singular,
    which is not a pathology but section 3.4's zero-cost case, where the
    counterfactual is zero with certainty. The pseudoinverse returns the zero
    fit the paper describes; the inverse raises. ``rank_deficient`` marks that
    case, and marks it exactly: a constant regressor is ``ptp(x) == 0``, where
    the singular values of the design place the boundary a tolerance away.
    """
    y_pre = np.asarray(y_pre, dtype=float)
    x_pre = np.asarray(x_pre, dtype=float)
    n = int(y_pre.size)
    sums = group_sums(y_pre, x_pre)
    if is_identified(sums):
        fit = fit_on_sums(sums, y_pre, x_pre)
        alpha, beta, resid = fit.alpha, fit.beta, fit.resid
    else:
        design = np.column_stack([np.ones(n), x_pre])
        coef, *_ = np.linalg.lstsq(design, y_pre, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])
        resid = y_pre - (alpha + beta * x_pre)
    df = n - 2
    return PretestFit(
        alpha=alpha,
        beta=beta,
        sigma_sq=float(resid @ resid) / df,
        df=df,
        n_pretest=n,
        unscaled_cov=_unscaled_cov(sums, x_pre),
        rank_deficient=bool(np.ptp(x_pre) == 0.0),
        sums=sums,
        resid=resid,
    )


def _unscaled_cov(sums: PretestSums, x_pre: np.ndarray) -> np.ndarray:
    """``(X'X)^+`` for ``X = [1, x]``: eqn 6's ``v_a``, ``v_ab`` and ``v_b``.

    The identified case is the shared closed form,
    :func:`mlsynth.utils.groupfit.unscaled_cov`, which carries the measurements
    for why it is written about ``S_xx``.

    What is TBR's own is the other branch. When the regressor is constant through
    the pretest ``X'X`` is singular, which is not a pathology but section 3.4's
    zero-cost case, where the counterfactual is zero with certainty. The
    pseudoinverse returns the zero fit the paper describes; an inverse raises.
    """
    if not is_identified(sums):
        sx = sums.sum_x
        return np.linalg.pinv(
            np.array([[float(sums.n), sx], [sx, float(x_pre @ x_pre)]]))
    return _identified_unscaled_cov(sums)


def cumulative_posterior(fit: PretestFit, y_test: np.ndarray,
                         x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eqns 4 and 6: the location and scale of ``Delta(T)`` at every ``T``."""
    y_test = np.asarray(y_test, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    horizon = np.arange(1, y_test.size + 1, dtype=float)
    ybar = np.cumsum(y_test) / horizon
    xbar = np.cumsum(x_test) / horizon
    loc = horizon * (ybar - fit.alpha - xbar * fit.beta)
    # ``v_a + 2 xbar v_ab + v_b xbar^2`` is the fitted value's variance, which is
    # the shared prediction variance; the remaining ``1 / T`` is eqn 6's own term
    # for the test period's unobserved errors.
    fitted_var = (prediction_variance(fit.sums, xbar) if fit.sums.s_xx > 0.0
                  else _quadratic_form(fit.unscaled_cov, xbar))
    scale = horizon * np.sqrt(fit.sigma_sq) * np.sqrt(fitted_var + 1.0 / horizon)
    return loc, scale


def _quadratic_form(cov: np.ndarray, xbar: np.ndarray) -> np.ndarray:
    """``[1, xbar]' V [1, xbar]`` from an explicit ``V``.

    Section 3.4's constant regressor leaves ``V`` a pseudoinverse that no closed
    form about ``S_xx`` describes, so the contraction is done directly there.
    """
    return (cov[0, 0] + 2.0 * xbar * cov[0, 1] + cov[1, 1] * xbar ** 2)


def interval(loc: np.ndarray, scale: np.ndarray, df: int,
             level: float) -> Tuple[np.ndarray, np.ndarray]:
    """The two-sided middle ``level`` interval of the t-distribution.

    Section 9.1: the half-width is the scale times the
    ``0.5 (1 + level)`` quantile on ``df`` degrees of freedom.
    """
    half = stats.t.ppf(0.5 * (1.0 + level), df) * np.asarray(scale, dtype=float)
    return np.asarray(loc, dtype=float) - half, np.asarray(loc, float) + half


def iroas_fixed_cost(loc: np.ndarray, scale: np.ndarray, df: int, level: float,
                     total_cost: float):
    """Section 3.4 with the denominator a known constant.

    "the posterior distribution of iROAS(t) is again a shifted and scaled
    t-distribution, making it unnecessary to resort to simulations". Dividing a
    t by a constant is a t, so the response posterior is rescaled, not
    sampled.
    """
    lo, hi = interval(loc / total_cost, np.asarray(scale, float) / total_cost,
                      df, level)
    return loc / total_cost, lo, hi


def iroas_simulated(resp_loc: np.ndarray, resp_scale: np.ndarray,
                    cost_loc: np.ndarray, cost_scale: np.ndarray, df: int,
                    level: float, n_draws: int = 10000, seed: int = 0):
    """Section 3.4's general route: draw from both posteriors and divide.

    The ratio of two t-distributed quantities has no closed form, so the paper
    simulates it. Only the final horizon is needed for the reported estimand,
    and the median is the point estimate because the ratio's mean need not
    exist.
    """
    rng = np.random.default_rng(seed)
    num = stats.t.rvs(df, loc=resp_loc[-1], scale=resp_scale[-1],
                      size=n_draws, random_state=rng)
    den = stats.t.rvs(df, loc=cost_loc[-1], scale=cost_scale[-1],
                      size=n_draws, random_state=rng)
    draws = num / den
    tail = 0.5 * (1.0 - level)
    return (float(np.median(draws)),
            float(np.quantile(draws, tail)),
            float(np.quantile(draws, 1.0 - tail)))

def _newey_west(resid: np.ndarray, design: np.ndarray, lag: int
                ) -> Tuple[float, np.ndarray]:
    """Bartlett-kernel long-run variance of ``resid``, and the meat for the fit.

    Li and Van den Bulte's proof D.2 writes both of Proposition 3.4's variance
    terms as sums truncated at ``|s - t| <= l``, consistent by the argument of
    Newey and West (1987). The Bartlett taper ``1 - j / (l + 1)`` keeps the
    estimate positive semi-definite, which an untapered truncation does not.

    At ``lag = 0`` only the diagonal survives, which is the independent case the
    same proof reduces to -- so this is a generalisation of equation 6 and not a
    different estimator.

    Parameters
    ----------
    resid : np.ndarray
        Pretest residuals, shape ``(T0,)``.
    design : np.ndarray
        The pretest design ``[1, xbar]``, shape ``(T0, 2)``.
    lag : int
        Truncation lag.

    Returns
    -------
    tuple
        The residuals' long-run variance, and the ``(2, 2)`` meat matrix.
    """
    n = resid.size
    # The same n / (n - 2) correction eqn 6's sigma^2 carries, so a lag of zero
    # lands on HC1 and the two scales are on one footing.
    dfc = n / float(n - design.shape[1])
    lrv = float(resid @ resid) / n
    scaled = design * resid[:, None]
    meat = scaled.T @ scaled
    for j in range(1, lag + 1):
        weight = 1.0 - j / (lag + 1.0)
        lrv += 2.0 * weight * float(resid[j:] @ resid[:-j]) / n
        cross = scaled[j:].T @ scaled[:-j]
        meat += weight * (cross + cross.T)
    return lrv * dfc, meat * dfc


def _hac_scale(y_pre: np.ndarray, x_pre: np.ndarray, x_post: np.ndarray,
               n_post: int, lag: int) -> float:
    """Proposition 3.4's scale for the cumulative effect.

    The same two pieces equation 6 has -- what is unknown about the coefficients,
    and the test window's own errors -- with each sum truncated instead of taken
    on the diagonal.
    """
    design = np.column_stack([np.ones_like(x_pre), x_pre])
    coef, *_ = np.linalg.lstsq(design, y_pre, rcond=None)
    lrv, meat = _newey_west(y_pre - design @ coef, design, lag)
    cov = np.linalg.pinv(design.T @ design)
    contrast = np.array([float(n_post), float(x_post.sum())])
    coefficient_term = float(contrast @ (cov @ meat @ cov) @ contrast)
    return float(np.sqrt(max(coefficient_term + n_post * lrv, 0.0)))


def _bandwidth(n_pre: int, requested: Optional[int]) -> int:
    """``ceil(T0 ** 0.25)`` unless the caller named one; refuse one too long.

    A truncation at or beyond the pretest length has no lags left to average, so
    it is a configuration error and not a degenerate-but-usable choice.
    """
    lag = int(np.ceil(n_pre ** 0.25)) if requested is None else int(requested)
    if lag >= n_pre:
        raise MlsynthEstimationError(
            f"the Newey-West bandwidth must be shorter than the {n_pre} pretest "
            f"periods it truncates over; got {lag}.")
    return lag


def cumulative_posterior_hac(fit, y_pre, x_pre, y_test, x_test, *,
                             bandwidth=None):
    """Eqn 4's location with Proposition 3.4's scale, at every horizon.

    :func:`cumulative_posterior` prices the interval on an iid pretest
    residual. Where that residual is autocorrelated the iid scale understates
    the spread, and Li and Van den Bulte's Proposition 3.4 replaces each sum
    with a Newey-West truncation. The location is untouched: this is a
    variance correction and the point estimate is eqn 4 either way.

    The scale is computed per horizon, so a caller reading the path at T sees
    the correction applied over the first T test periods and not over the whole
    window scaled down.
    """
    y_pre = np.asarray(y_pre, dtype=float)
    x_pre = np.asarray(x_pre, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    loc, _ = cumulative_posterior(fit, y_test, x_test)
    lag = _bandwidth(int(y_pre.size), bandwidth)
    scale = np.array([
        _hac_scale(y_pre, x_pre, x_test[:T], T, lag)
        for T in range(1, y_test.size + 1)
    ], dtype=float)
    return loc, scale

