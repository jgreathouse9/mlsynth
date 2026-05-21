"""Outer Metropolis-within-Gibbs loop for BVS-SS.

Implements Algorithm 1 of Xu & Zhou (2025), arXiv:2503.06454. The sweep
order per outer iteration:

    1. For every donor pair ``(i, j)`` with ``i < j``: jointly update
       ``(\\gamma_i, \\gamma_j, \\mu_i, \\mu_j)`` via the closed-form
       four-case Gibbs draw (Lemma S2 + Lemma S1).
    2. Sample ``\\phi`` from its full conditional ``Gamma((M+\\kappa_1)/2,
       (\\kappa_2 + RSS) / 2)`` (Eq. (6)).
    3. Run ``n_\\tau`` MH steps for ``\\tau`` against its target
       conditional (Eq. (S1)).

The ``w`` Gibbs step (Eq. (7)) is *not* drawn in this implementation —
counterfactuals are computed directly from ``\\mu`` samples, matching
the reference script's behavior and reducing posterior variance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import gamma as gamma_dist

from .gibbs_pair import _compute_candidate_posteriors, _sample_pair
from .mh import MH_tau
from .posterior import RSS


def _update_phi_tau(
    mutemp: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    tau_prev: float,
    M: int,
    kappa1: float,
    kappa2: float,
    Gram: np.ndarray,
    tau_min: float = 1e-6,
    a: float = 0.01,
    b: float = 0.1,
    n_tau: int = 11,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Joint update of ``\\phi`` (Gibbs) and ``\\tau`` (MH)."""

    zmu = Y - X @ mutemp
    gtemp = (mutemp != 0).astype(int)
    rss_val = RSS(gtemp, tau_prev, zmu, X, Gram)

    shape = (M + kappa1) / 2
    scale = 2 / (kappa2 + rss_val)
    if rng is None:
        phi_new = gamma_dist.rvs(shape, scale=scale)
    else:
        phi_new = gamma_dist.rvs(shape, scale=scale, random_state=rng)

    tau_new = MH_tau(
        tau_prev, gtemp, mutemp, phi_new, Y, X, Gram,
        a=a, b=b, tau_min=tau_min, nrep=n_tau, rng=rng,
    )
    return float(phi_new), tau_new


def gibbs_BVS(
    Y: np.ndarray,
    X: np.ndarray,
    Gram: np.ndarray,
    M: int,
    N: int,
    size: int,
    kappa1: float = 1.0,
    kappa2: float = 1.0,
    theta: float = 0.25,
    tau_min: float = 1e-6,
    a: float = 0.01,
    b: float = 0.1,
    n_tau: int = 11,
    init_mu: Optional[np.ndarray] = None,
    init_phi: float = 0.8,
    init_tau: float = 1.0,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Run ``size`` MCMC iterations targeting the BVS-SS posterior.

    Parameters
    ----------
    Y : np.ndarray
        Demeaned outcome vector of length ``M``.
    X : np.ndarray
        Demeaned predictor matrix of shape ``(M, N)``.
    Gram : np.ndarray
        ``X.T @ X``.
    M : int
        Number of observations (pre-treatment time periods).
    N : int
        Number of donor predictors.
    size : int
        Total number of MCMC iterations (including burn-in).
    kappa1, kappa2 : float
        Gamma hyperparameters for the prior on ``\\phi``.
    theta : float
        Prior inclusion probability for each predictor.
    tau_min : float
        Numerical floor for ``\\tau``.
    a, b : float
        Gamma-prior shape / rate for ``\\tau``.
    n_tau : int
        Number of MH steps for ``\\tau`` per outer iteration.
    init_mu : np.ndarray, optional
        Initial weight vector. Defaults to the uniform-simplex
        ``\\mu = 1 / N``.
    init_phi, init_tau : float
        Initial values for the precision parameters.
    verbose : bool
        Show a tqdm progress bar.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. Falls back to
        :mod:`numpy.random` and the global :mod:`scipy.stats` state when
        ``None``.

    Returns
    -------
    dict
        Mapping with keys:

        * ``musample`` : ``(N, size)`` posterior samples of ``\\mu``.
        * ``phisample`` : length-``size`` posterior samples of ``\\phi``.
        * ``tausample`` : length-``size`` posterior samples of ``\\tau``.
        * ``gammasample`` : ``(N, size)`` 0/1 inclusion indicators
          implied by ``\\mu``.
    """

    musample = np.zeros((N, size))
    phisample = np.zeros(size)
    tausample = np.zeros(size)

    if init_mu is None:
        musample[:, 0] = 1 / N
    else:
        if init_mu.shape != (N,):
            raise ValueError(
                f"init_mu must have shape ({N},); got {init_mu.shape}."
            )
        musample[:, 0] = init_mu
    phisample[0] = init_phi
    tausample[0] = init_tau

    combinations = [(i, j) for i in range(N) for j in range(i + 1, N)]

    if verbose:
        from tqdm import tqdm
        outer = tqdm(range(1, size), desc="Gibbs Sampling")
    else:
        outer = range(1, size)

    for h in outer:
        mutemp = musample[:, h - 1].copy()
        for i, j in combinations:
            s, _, L, O, ptotal = _compute_candidate_posteriors(
                mutemp, i, j, X, Y, tausample[h - 1], phisample[h - 1],
                Gram, theta,
            )
            if abs(s) > 1e-12:
                _sample_pair(
                    mutemp, i, j, s, L, O, phisample[h - 1], ptotal, rng=rng
                )
        musample[:, h] = mutemp
        phisample[h], tausample[h] = _update_phi_tau(
            mutemp, X, Y, tausample[h - 1], M, kappa1, kappa2, Gram,
            tau_min=tau_min, a=a, b=b, n_tau=n_tau, rng=rng,
        )

    gammasample = (musample != 0).astype(int)

    return {
        "musample": musample,
        "phisample": phisample,
        "tausample": tausample,
        "gammasample": gammasample,
    }
