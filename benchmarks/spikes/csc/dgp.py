"""Port of the CSC paper's Monte Carlo DGP.

Source: ``Simulation/1_Simulate_Data_Int_Fixed_Effects_Model.R`` in
``tzvetanmoev/Correlated-Synthetic-Controls`` (the script that produced the
paper's Tables 1-2), read against Appendix A.f of arXiv:2507.08918.

The outcome follows an interactive fixed effects model

    y_it = x_i' beta + D_it tau + lambda_t' mu_i + eps_it,
    mu_i = Gamma' x_i + v_i,

with a single categorical covariate ``x_i ~ U{1..K}`` entered as K dummies
(the reference keeps every category -- no reference level is dropped, which
is what makes CSC's adding-up constraint feasible). Treatment happens in the
last period only, and assignment is correlated with the factor loadings
``mu_i`` so that DiD's parallel-trends assumption fails.

Two places where the R script departs from the paper's Appendix A.f, kept as
the script has them because the script is what generated the tables:

* the continuous covariate x^(2) is commented out in the script, so only the
  categorical covariate survives;
* assignment probabilities are ``pi_i = 1 / (exp(a_i) (exp(a_i) + 1))``,
  not the logistic ``exp(a_i)/(1 + exp(a_i))`` of paper eq. (17); they are
  then shifted to have mean ``prob_tr``, which can send some below zero. The
  R draw ``runif(n) < pi`` treats those as never-treated, which is what the
  clip below reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimPanel:
    """One simulated panel."""

    Y: np.ndarray  # (N, T) observed outcomes (treated units bumped in the last period)
    Y0: np.ndarray  # (N, T) untreated potential outcomes (never observed in practice)
    D: np.ndarray  # (N,) ever-treated indicator
    X_dummies: np.ndarray  # (N, K) unit-level category dummies
    category: np.ndarray  # (N,) raw category label
    mu: np.ndarray  # (N, F) factor loadings
    lam: np.ndarray  # (T, F) common factors
    mu_mean: float
    lambda_mean: float
    tau: float

    @property
    def treated_idx(self) -> np.ndarray:
        return np.flatnonzero(self.D > 0.5)

    @property
    def donor_idx(self) -> np.ndarray:
        return np.flatnonzero(self.D < 0.5)


def simulate(
    *,
    T: int = 5,
    N: int = 100,
    num_factors: int = 2,
    num_categories: int = 5,
    prob_tr: float = 0.15,
    lambda_mean: float = 2.0,
    mu_mean: float = 1.0,
    tau: float = 1.0,
    sd_tau: float = 0.3,
    treat_not_at_random: bool = True,
    corr_scale: float = 1.0,
    fixed_lam: np.ndarray | None = None,
    fixed_Gamma: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> SimPanel:
    """Simulate one panel. Defaults are the driver's baseline row.

    ``tunning_params[1, ] <- c(100, 5, 5, 2, 0.15, 1)`` in
    ``3_Simulation_CSC_IDiD_FDiD.R`` -- N=100, T=5, 5 categories, 2 factors,
    E[pi]=0.15, treatment not at random.
    """
    rng = np.random.default_rng() if rng is None else rng

    # eps_it ~ N(0, 1)
    eps = rng.normal(size=(N, T))

    # lambda_t: one N(exp_lambda + lambda_mean, 1) draw per period per factor.
    # The R loop re-seeds identically each iteration, so exp_lambda is the same
    # integer for every factor; kept.
    exp_lambda = rng.integers(-3, 4)
    lam = rng.normal(loc=exp_lambda + lambda_mean, scale=1.0, size=(T, num_factors))
    if fixed_lam is not None:
        # Appendix A.f holds lambda at one draw across every replication; the
        # shipped driver redraws it each time. Passing it in reproduces the
        # paper's protocol.
        lam = np.asarray(fixed_lam, dtype=float)[:T, :num_factors]

    # x_i ~ U{1..K}, entered as K dummies with no reference level dropped.
    category = rng.integers(1, num_categories + 1, size=N)
    X_dummies = np.zeros((N, num_categories))
    X_dummies[np.arange(N), category - 1] = 1.0

    # mu_i = Gamma' x_i + v_i, with a per-factor mean shift.
    Gamma = rng.normal(size=(num_categories, num_factors))
    if fixed_Gamma is not None:
        Gamma = np.asarray(fixed_Gamma, dtype=float)[:num_categories, :num_factors]
    mu = np.empty((N, num_factors))
    for f in range(num_factors):
        expect_mu = mu_mean + rng.integers(-3, 4)
        v = rng.normal(loc=expect_mu, scale=1.0, size=N)
        mu[:, f] = X_dummies @ Gamma[:, f] + v

    # beta ~ U[0, 1] per category (the driver draws K+1 and drops the first,
    # which was the coefficient of the retired continuous covariate).
    beta = rng.uniform(size=num_categories)

    Y0 = X_dummies @ beta[:, None] + mu @ lam.T + eps

    if treat_not_at_random:
        # ``corr_scale`` dials how strongly assignment depends on the
        # unobserved loadings: 1.0 is the reference, 0.0 makes assignment
        # ignorable while keeping the same marginal treatment rate. The
        # paper's "Less Correlation" row has no counterpart in the shipped
        # driver, so this is how that row is approximated.
        a = corr_scale * (
            mu.mean(axis=1)
            + X_dummies @ (Gamma.mean(axis=1) + rng.normal(scale=0.1, size=num_categories))
        ) + rng.normal(size=N)
        with np.errstate(over="ignore"):
            ea = np.exp(np.clip(a, -50, 50))
        pi = 1.0 / (ea * (ea + 1.0))
        pi = pi + (prob_tr - pi.mean())
        pi[rng.integers(N)] = 1.0
        pi[pi == pi.max()] = 1.0
        D = (rng.uniform(size=N) < pi).astype(float)
    else:
        D = (rng.uniform(size=N) < prob_tr).astype(float)

    # Treatment lands in the last period only.
    Y = Y0.copy()
    n1 = int(D.sum())
    Y[D > 0.5, -1] += tau + rng.normal(scale=sd_tau, size=n1)

    return SimPanel(
        Y=Y,
        Y0=Y0,
        D=D,
        X_dummies=X_dummies,
        category=category,
        mu=mu,
        lam=lam,
        mu_mean=mu_mean,
        lambda_mean=lambda_mean,
        tau=tau,
    )


def to_long(panel: SimPanel) -> "pd.DataFrame":  # noqa: F821
    """The panel as a long frame, the shape mlsynth ingests."""
    import pandas as pd

    N, T = panel.Y.shape
    unit = np.repeat(np.arange(1, N + 1), T)
    time = np.tile(np.arange(1, T + 1), N)
    treat = np.zeros(N * T)
    treat[(np.repeat(panel.D, T) > 0.5) & (time == T)] = 1.0
    return pd.DataFrame(
        {
            "unit": unit,
            "time": time,
            "y": panel.Y.reshape(-1),
            "treat": treat,
            "category": np.repeat(panel.category, T),
        }
    )
