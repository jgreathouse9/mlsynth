"""TBR, ported from Kerman, Wang & Vaver (2017) sections 3.2 and 9.1.

The paper, not the reference implementation, is the source. Equation numbers
below are the paper's.

Pretest model (eqn 1):      y_t = alpha + beta x_t + eps_t,  eps ~ N(0, sigma^2)
Cumulative effect (eqn 4):  E[Delta(T)] = T (ybar_T - alpha - xbar_T beta)
Scale (eqn 6):              T s (v_a + 2 xbar_T v_ab + v_b xbar_T^2 + 1/T)^(1/2)
Posterior:                  shifted, scaled t with n - 2 degrees of freedom,
                            n the number of pretest time points.

``v_a``, ``v_b``, ``v_ab`` are entries of the unscaled covariance matrix
V = (X'X)^-1 and ``s`` is the classical residual standard deviation, which the
noninformative prior of section 9.1 makes the posterior's scale.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class TBR:
    """Time-Based Regression on two aggregated group series."""

    def fit(self, y_pre: np.ndarray, x_pre: np.ndarray) -> "TBR":
        y_pre = np.asarray(y_pre, float)
        x_pre = np.asarray(x_pre, float)
        X = np.column_stack([np.ones(x_pre.size), x_pre])
        self.coef_, *_ = np.linalg.lstsq(X, y_pre, rcond=None)
        resid = y_pre - X @ self.coef_
        self.n_ = y_pre.size
        self.df_ = self.n_ - 2
        self.s2_ = float(resid @ resid) / self.df_
        # pinv, not inv: when the regressor is constant through the pretest
        # -- the zero-cost case of section 3.4 -- X'X is singular and the
        # counterfactual is zero with certainty. statsmodels takes the same
        # route, which is why the reference survives that panel.
        self.V_ = np.linalg.pinv(X.T @ X)         # unscaled, section 9.1
        self.degenerate_ = np.linalg.matrix_rank(X) < 2
        return self

    def counterfactual(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return self.coef_[0] + self.coef_[1] * x

    def cumulative(self, y_test: np.ndarray, x_test: np.ndarray):
        """Posterior of Delta(T) at every T in the test period."""
        y_test = np.asarray(y_test, float)
        x_test = np.asarray(x_test, float)
        T = np.arange(1, y_test.size + 1, dtype=float)
        ybar = np.cumsum(y_test) / T
        xbar = np.cumsum(x_test) / T
        loc = T * (ybar - self.coef_[0] - xbar * self.coef_[1])      # eqn 4
        va, vb, vab = self.V_[0, 0], self.V_[1, 1], self.V_[0, 1]
        scale = T * np.sqrt(self.s2_) * np.sqrt(                      # eqn 6
            va + 2.0 * xbar * vab + vb * xbar ** 2 + 1.0 / T)
        return stats.t(self.df_, loc=loc, scale=scale)


def is_fixed_cost(cost_pre_all: np.ndarray, cost_test_control: np.ndarray,
                  tol: float = 1e-10) -> bool:
    """Section 3.4's special case: no cost anywhere but the treated test cells.

    The reference declares it when the pretest cost over every geo plus the
    control group's test-period cost sums to approximately zero.
    """
    return abs(float(np.sum(cost_pre_all)) + float(np.sum(cost_test_control))) < tol


def iroas_fixed_cost(resp: "stats.rv_continuous", total_cost: float):
    """Section 3.4: with the denominator a constant, iROAS is again a t.

    "the posterior distribution of iROAS(t) is again a shifted and scaled
    t-distribution, making it unnecessary to resort to simulations".
    """
    loc = np.atleast_1d(resp.kwds["loc"]) / total_cost
    scale = np.atleast_1d(resp.kwds["scale"]) / total_cost
    return stats.t(resp.kwds.get("df", resp.args[0]), loc=loc, scale=scale)


def iroas_simulated(resp, cost, n_draws: int = 10000, seed: int = 0):
    """Section 3.4's general route: draw from both posteriors and divide."""
    rng = np.random.default_rng(seed)
    k = len(np.atleast_1d(resp.kwds["loc"]))
    return (resp.rvs(size=(n_draws, k), random_state=rng) /
            cost.rvs(size=(n_draws, k), random_state=rng))
