"""Metropolis-Hastings update for the soft-constraint variance \\tau.

Implements the log-random-walk proposal of Xu & Zhou (2025), Eq. (S1):

    log \\tau^*  =  log \\tau + N(0, 1),

with a reflective barrier at ``log tau_min`` so the chain never drops
below a numerical floor. The acceptance probability comes from

    \\rho = min{1,
              p(y | \\mu, \\tau^*, \\phi) p(\\tau^*) /
              [p(y | \\mu, \\tau, \\phi) p(\\tau)] * (\\tau^* / \\tau)},

where the Jacobian factor ``\\tau^* / \\tau`` arises from the log-RW
proposal. The (\\tau prior) is Gamma(``a``, ``b``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import gamma as gamma_dist

from .posterior import loglike


def MH_tau(
    tau: float,
    gamma_vec: np.ndarray,
    mu: np.ndarray,
    phi: float,
    Y: np.ndarray,
    X: np.ndarray,
    Gram: np.ndarray,
    a: float = 0.01,
    b: float = 0.1,
    tau_min: float = 1e-6,
    nrep: int = 11,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Run ``nrep`` MH steps for ``\\tau`` and return the final draw.

    Parameters
    ----------
    tau : float
        Current value of \\tau.
    gamma_vec : np.ndarray
        Current inclusion indicator.
    mu : np.ndarray
        Current weight vector.
    phi : float
        Current observation-noise precision.
    Y, X, Gram : np.ndarray
        Response, predictor matrix, and Gram matrix.
    a, b : float
        Gamma-prior shape and rate parameters for \\tau.
    tau_min : float
        Numerical floor; proposals below are reflected.
    nrep : int
        Number of MH steps per call (paper's ``n_\\tau``).
    rng : numpy.random.Generator, optional
        Used for both proposal noise and accept/reject draws. Falls
        back to :mod:`numpy.random` when ``None``.

    Returns
    -------
    float
        \\tau value after the ``nrep``th step.
    """

    tau_M = np.empty(nrep)
    tau_M[0] = tau

    for i in range(1, nrep):
        x = tau_M[i - 1]
        if rng is None:
            noise = np.random.randn()
        else:
            noise = rng.standard_normal()

        logy = np.log(x) + noise
        if logy < np.log(tau_min):
            logy = 2 * np.log(tau_min) - logy
        y = np.exp(logy)

        r = (
            gamma_dist.logpdf(y, a, scale=1 / b)
            - gamma_dist.logpdf(x, a, scale=1 / b)
            + loglike(gamma_vec, y, mu, phi, Y, X, Gram)
            - loglike(gamma_vec, x, mu, phi, Y, X, Gram)
            + np.log(y)
            - np.log(x)
        )

        if rng is None:
            log_u = np.log(np.random.rand())
        else:
            log_u = np.log(rng.random())

        if log_u < r:
            tau_M[i] = y
        else:
            tau_M[i] = x

    return float(tau_M[-1])
