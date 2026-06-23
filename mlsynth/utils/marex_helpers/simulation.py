"""Linear-factor DGP for the MAREX simulation study (Abadie & Zhao 2026, Sec. 5).

Reimplements the paper's baseline data-generating process (Assumption 1,
equations 12a/12b) so the simulation study can be replicated without the
authors' code. Potential outcomes are

    Y^N_jt = delta_t + theta_t' Z_j + lambda_t' mu_j + eps_jt
    Y^I_jt = upsilon_t + gamma_t' Z_j + eta_t' mu_j + xi_jt

with Z_j (R observed) and mu_j (F unobserved) covariates, sorted time effects,
and i.i.d. Normal(0, sigma^2) noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MAREXSample:
    """One simulated sample.

    Attributes
    ----------
    Y_N : np.ndarray
        Potential outcomes under no treatment, shape ``(J, T)``.
    Y_I : np.ndarray
        Potential outcomes under treatment, shape ``(J, T)``.
    tau : np.ndarray
        True average treatment effect ``tau_t`` per period, shape ``(T,)``
        (zero in the pre-period).
    T0 : int
        Number of pre-treatment periods.
    Z : np.ndarray
        Observed (time-invariant) covariates that generate the outcome,
        shape ``(J, R)``.
    """

    Y_N: np.ndarray
    Y_I: np.ndarray
    tau: np.ndarray
    T0: int
    Z: np.ndarray


def generate_marex_sample(
    J: int = 15, R: int = 7, F: int = 11, T: int = 30, T0: int = 25,
    sigma: float = 1.0, rng: Optional[np.random.Generator] = None,
) -> MAREXSample:
    """Draw one sample from the paper's baseline linear-factor DGP (Sec. 5).

    Returns
    -------
    MAREXSample
    """
    rng = rng or np.random.default_rng()

    # sorted (small-to-large) time effects
    delta = np.sort(rng.uniform(0, 20, T))
    upsilon = np.zeros(T)
    upsilon[T0:] = np.sort(rng.uniform(0, 20, T - T0))

    Z = rng.uniform(0, 1, (J, R))     # observed covariates
    mu = rng.uniform(0, 1, (J, F))    # unobserved covariates
    theta = rng.uniform(0, 10, (T, R))
    gamma = rng.uniform(0, 10, (T, R))
    lam = rng.uniform(0, 10, (T, F))
    eta = rng.uniform(0, 10, (T, F))

    eps = rng.normal(0, sigma, (J, T))
    xi = rng.normal(0, sigma, (J, T))

    Y_N = delta[None, :] + Z @ theta.T + mu @ lam.T + eps          # (J, T)
    Y_I = upsilon[None, :] + Z @ gamma.T + mu @ eta.T + xi         # (J, T)

    tau = np.zeros(T)
    tau[T0:] = (Y_I[:, T0:] - Y_N[:, T0:]).mean(axis=0)            # f_j = 1/J
    return MAREXSample(Y_N=Y_N, Y_I=Y_I, tau=tau, T0=T0, Z=Z)
