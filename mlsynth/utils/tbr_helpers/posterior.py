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
from typing import Tuple

import numpy as np
from scipy import stats


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

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.alpha + self.beta * np.asarray(x, dtype=float)


def fit_pretest(y_pre: np.ndarray, x_pre: np.ndarray) -> PretestFit:
    """Least squares on eqn 1, with the unscaled covariance section 9.1 uses.

    The pseudoinverse, not the inverse: when the regressor is constant through
    the pretest ``X'X`` is singular, which is not a pathology but section 3.4's
    zero-cost case, where the counterfactual is zero with certainty. Taking the
    pseudoinverse there returns the zero fit the paper describes; taking the
    inverse raises. ``rank_deficient`` records which happened so a caller is
    never left to infer it.
    """
    y_pre = np.asarray(y_pre, dtype=float)
    x_pre = np.asarray(x_pre, dtype=float)
    n = int(y_pre.size)
    design = np.column_stack([np.ones(n), x_pre])
    coef, *_ = np.linalg.lstsq(design, y_pre, rcond=None)
    resid = y_pre - design @ coef
    df = n - 2
    return PretestFit(
        alpha=float(coef[0]),
        beta=float(coef[1]),
        sigma_sq=float(resid @ resid) / df,
        df=df,
        n_pretest=n,
        unscaled_cov=np.linalg.pinv(design.T @ design),
        rank_deficient=bool(np.linalg.matrix_rank(design) < 2),
    )


def cumulative_posterior(fit: PretestFit, y_test: np.ndarray,
                         x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eqns 4 and 6: the location and scale of ``Delta(T)`` at every ``T``."""
    y_test = np.asarray(y_test, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    horizon = np.arange(1, y_test.size + 1, dtype=float)
    ybar = np.cumsum(y_test) / horizon
    xbar = np.cumsum(x_test) / horizon
    loc = horizon * (ybar - fit.alpha - xbar * fit.beta)
    v_a, v_b = fit.unscaled_cov[0, 0], fit.unscaled_cov[1, 1]
    v_ab = fit.unscaled_cov[0, 1]
    scale = horizon * np.sqrt(fit.sigma_sq) * np.sqrt(
        v_a + 2.0 * xbar * v_ab + v_b * xbar ** 2 + 1.0 / horizon)
    return loc, scale


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
