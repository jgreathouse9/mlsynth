"""Data-generating processes from Zhang (2026), Section 8.

Three designs, all with treatment selection on BOTH the time-invariant latent
factor and the lagged outcome:

    D_i = 1(alpha_i / 2 + Y_{i,T0-1} / 2 + eps_{D,i} >= 0),  eps_D ~ Logistic(0,1)

8.1 additive fixed effects   Y_it = rho Y_it-1 + alpha_i + gamma_t + eps
8.2 interactive fixed effects Y_it = rho Y_it-1 + alpha_i * gamma_t + eps
8.3 nonlinear (binary)        Y_it = 1(-2 + rho Y_it-1 + 4 alpha_i + eps >= 0)

The paper does not state the initial condition for the autoregression. Every
design here is burned in for ``burn_in`` periods from the conditional stationary
mean, so the reported t = 0, ..., T0-1 are draws from the stationary law. This is
an assumption of the port, not of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Panel:
    """One simulated panel.

    Attributes
    ----------
    Y : (n, T0 + n_post) float
        Observed outcomes; columns 0..T0-1 are pre-treatment, T0.. are post.
    D : (n,) int
        Treatment indicator (block assignment at T0).
    alpha : (n,) float
        The latent factor, for the oracle ("infeasible") estimators.
    T0 : int
        Number of pre-treatment periods, so treatment happens in column T0.
    att : (n_post,) float
        True dynamic ATT at each post period.
    """

    Y: np.ndarray
    D: np.ndarray
    alpha: np.ndarray
    T0: int
    att: np.ndarray


def _select(alpha: np.ndarray, y_last: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paper eq. (8.2)/(8.4)/(8.6): logistic selection on alpha and the last lag."""
    index = alpha / 2.0 + y_last / 2.0
    return (index + rng.logistic(size=alpha.size) >= 0.0).astype(np.int64)


def additive_fe(
    n: int,
    T0: int,
    *,
    n_post: int = 5,
    rho: float = 0.8,
    effect: float = 0.5,
    burn_in: int = 60,
    rng: np.random.Generator,
) -> Panel:
    """Section 8.1 -- dynamic panel with additive fixed effects."""
    alpha = rng.uniform(-0.25, 0.25, size=n)
    n_tot = burn_in + T0 + n_post
    gamma = rng.uniform(-0.25, 0.25, size=n_tot)
    eps = rng.uniform(-0.5, 0.5, size=(n, n_tot))

    Y = np.empty((n, n_tot))
    y = alpha / (1.0 - rho)                      # conditional stationary mean
    for t in range(burn_in + T0):                # pre-treatment: no effect yet
        y = rho * y + alpha + gamma[t] + eps[:, t]
        Y[:, t] = y

    D = _select(alpha, Y[:, burn_in + T0 - 1], rng)

    for t in range(burn_in + T0, n_tot):
        y = rho * y + alpha + gamma[t] + effect * D + eps[:, t]
        Y[:, t] = y

    # dynamic ATT: effect * (1 + rho + ... + rho^h) at horizon h = t - T0
    att = effect * np.cumsum(rho ** np.arange(n_post))
    return Panel(Y=Y[:, burn_in:], D=D, alpha=alpha, T0=T0, att=att)


def interactive_fe(
    n: int,
    T0: int,
    *,
    n_post: int = 5,
    rho: float = 0.8,
    effect: float = 0.5,
    burn_in: int = 60,
    rng: np.random.Generator,
) -> Panel:
    """Section 8.2 -- dynamic panel with interactive fixed effects alpha_i * gamma_t."""
    alpha = rng.uniform(-1.0, 1.0, size=n)
    n_tot = burn_in + T0 + n_post
    gamma = rng.uniform(-2.0, 2.0, size=n_tot)
    eps = rng.uniform(-0.5, 0.5, size=(n, n_tot))

    Y = np.empty((n, n_tot))
    y = np.zeros(n)
    for t in range(burn_in + T0):
        y = rho * y + alpha * gamma[t] + eps[:, t]
        Y[:, t] = y

    D = _select(alpha, Y[:, burn_in + T0 - 1], rng)

    for t in range(burn_in + T0, n_tot):
        y = rho * y + alpha * gamma[t] + effect * D + eps[:, t]
        Y[:, t] = y

    att = effect * np.cumsum(rho ** np.arange(n_post))
    return Panel(Y=Y[:, burn_in:], D=D, alpha=alpha, T0=T0, att=att)


def nonlinear(
    n: int,
    T0: int,
    *,
    n_post: int = 5,
    rho: float = 0.5,
    effect: float = 1.0,
    burn_in: int = 60,
    rng: np.random.Generator,
) -> Panel:
    """Section 8.3 -- dynamic binary panel. The ATT has no closed form."""
    alpha = rng.uniform(-1.0, 1.0, size=n)
    n_tot = burn_in + T0 + n_post
    eps = rng.logistic(size=(n, n_tot))

    Y = np.empty((n, n_tot))
    y = np.zeros(n)
    for t in range(burn_in + T0):
        y = (-2.0 + rho * y + 4.0 * alpha + eps[:, t] >= 0.0).astype(float)
        Y[:, t] = y

    D = _select(alpha, Y[:, burn_in + T0 - 1], rng)

    for t in range(burn_in + T0, n_tot):
        y = (-2.0 + rho * y + 4.0 * alpha + effect * D + eps[:, t] >= 0.0).astype(float)
        Y[:, t] = y

    return Panel(Y=Y[:, burn_in:], D=D, alpha=alpha, T0=T0, att=np.full(n_post, np.nan))


DESIGNS = {"additive_fe": additive_fe, "interactive_fe": interactive_fe, "nonlinear": nonlinear}
