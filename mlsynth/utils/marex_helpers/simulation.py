"""Linear-factor DGP for the MAREX simulation study (Abadie & Zhao 2026, Sec. 5).

Reimplements the paper's baseline data-generating process (Assumption 1,
equations 12a/12b) so the simulation study can be replicated without the
authors' code. Potential outcomes are

    Y^N_jt = delta_t + theta_t' Z_j + lambda_t' mu_j + eps_jt
    Y^I_jt = upsilon_t + gamma_t' Z_j + eta_t' mu_j + xi_jt

with Z_j (R observed) and mu_j (F unobserved) covariates, sorted time effects,
and i.i.d. Normal(0, sigma^2) noise.

:func:`generate_marex_sample` draws from a numpy generator.
:func:`generate_section5_sample` reproduces the authors' own simulations draw
for draw: simulation ``i`` of their 1000 is ``set.seed(i)`` followed by their
``Generate_Model_Primitives`` (SCDesign, "2. Many Simulations",
``Main_LazyRun.R``), and the function returns the same panel from R's stream,
ported here.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import ndtri

from ...exceptions import MlsynthConfigError


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
    mu : np.ndarray, optional
        Unobserved (time-invariant) covariates, shape ``(J, F)``. A design
        never sees them; a simulation that knows them can compute the bias a
        design carries.
    """

    Y_N: np.ndarray
    Y_I: np.ndarray
    tau: np.ndarray
    T0: int
    Z: np.ndarray
    mu: Optional[np.ndarray] = None


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
    return MAREXSample(Y_N=Y_N, Y_I=Y_I, tau=tau, T0=T0, Z=Z, mu=mu)


#: ``1 / 2^32``, R's scale from a 32-bit Mersenne-Twister output to [0, 1).
_I2_32 = 2.3283064365386963e-10
#: ``1 / (2^32 - 1)``, the constant R's ``fixup`` moves a draw of 0 by.
_I2_32M1 = 2.328306437080797e-10
#: ``2^27``: R's inversion builds one normal from two uniforms at this scale.
_BIG = 134217728.0


def _fixup(u: np.ndarray) -> np.ndarray:
    """R's ``fixup``: keep a uniform draw strictly inside (0, 1)."""
    u = np.where(u <= 0.0, 0.5 * _I2_32M1, u)
    # A 32-bit output scaled by 1/2^32 is at most 1 - 2^-32, so R's other
    # branch, for a draw at 1, cannot fire.
    return u


def _check_seed(seed) -> int:
    if (isinstance(seed, bool) or not isinstance(seed, numbers.Integral)
            or not 0 <= int(seed) <= 2 ** 31 - 1):
        raise MlsynthConfigError(
            f"seed must be an integer from 0 to 2^31 - 1, as R's set.seed "
            f"takes; got {seed!r}.")
    return int(seed)


class _RStream:
    """R's default generator after ``set.seed(seed)``.

    R's kinds by default are "Mersenne-Twister" and "Inversion". ``set.seed``
    scrambles the seed with 50 steps of the congruential map
    ``s -> 69069 s + 1`` (mod 2^32), fills the generator's 625-word state with
    the next 625 steps, and marks the 624-word key exhausted, so the first
    draw regenerates it. ``runif`` scales each 32-bit output by ``1 / 2^32``;
    ``rnorm`` spends two uniforms per draw,
    ``qnorm((floor(2^27 u1) + u2) / 2^27)``.
    """

    def __init__(self, seed: int):
        s = _check_seed(seed)
        state = []
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        for _ in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            state.append(s)
        self._bits = np.random.MT19937()
        # The first word is R's position counter; ``FixupSeeds`` sets it to 624.
        self._bits.state = {"bit_generator": "MT19937",
                            "state": {"key": np.array(state[1:], dtype=np.uint32),
                                      "pos": 624}}

    def runif(self, n: int) -> np.ndarray:
        """``runif(n)``."""
        return _fixup(self._bits.random_raw(n).astype(np.float64) * _I2_32)

    def rnorm(self, n: int) -> np.ndarray:
        """``rnorm(n)``."""
        u = self.runif(2 * n)
        return ndtri((np.floor(_BIG * u[0::2]) + u[1::2]) / _BIG)


def generate_section5_sample(seed: int) -> MAREXSample:
    """Simulation ``seed`` of Abadie and Zhao's Section 5, as their R code draws it.

    The panel is what ``set.seed(seed)`` followed by the authors'
    ``Generate_Model_Primitives`` produces: ``J = 15`` units, ``R = 7``
    observed and ``F = 11`` unobserved covariates on Uniform(0, 1),
    ``T = 30`` periods with ``T0 = 25``, coefficients on Uniform(0, 10),
    intercepts on Uniform(0, 20) sorted small to large (``delta`` over all 30
    periods together, ``upsilon`` over the five experimental ones), and N(0, 1)
    noise. Every coefficient vector is drawn afresh each period. The treated
    outcomes exist only on the experimental periods, which is where the
    authors define them; ``Y_I`` is NaN before ``T0``.

    Their driver runs simulation ``i`` of 1000 as ``set.seed(i)``, so seeds 1
    to 1000 give the panels behind their Table 2.

    Parameters
    ----------
    seed : int
        From 0 to 2^31 - 1.

    Returns
    -------
    MAREXSample
        With ``mu``, the unobserved covariates, filled.

    Raises
    ------
    MlsynthConfigError
        If ``seed`` is not an integer in R's range.
    """
    J, R, F, T, T0 = 15, 7, 11, 30, 25
    r = _RStream(seed)
    # The order of the draws is the authors': each block below is one loop of
    # Generate_Model_Primitives, in the order it runs.
    delta = np.sort(np.r_[20.0 * r.runif(T0), 20.0 * r.runif(T - T0)])
    upsilon = np.sort(20.0 * r.runif(T - T0))
    Z = r.runif(J * R).reshape(J, R)                    # unit by unit
    mu = r.runif(J * F).reshape(J, F)
    theta = 10.0 * r.runif(T * R).reshape(T, R)         # period by period
    gamma = 10.0 * r.runif((T - T0) * R).reshape(T - T0, R)
    lam = 10.0 * r.runif(T * F).reshape(T, F)
    eta = 10.0 * r.runif((T - T0) * F).reshape(T - T0, F)
    eps = r.rnorm(T * J).reshape(T, J).T
    xi = r.rnorm((T - T0) * J).reshape(T - T0, J).T

    Y_N = delta[None, :] + Z @ theta.T + mu @ lam.T + eps
    Y_I = np.full((J, T), np.nan)
    Y_I[:, T0:] = upsilon[None, :] + Z @ gamma.T + mu @ eta.T + xi
    tau = np.zeros(T)
    tau[T0:] = (Y_I[:, T0:] - Y_N[:, T0:]).mean(axis=0)
    return MAREXSample(Y_N=Y_N, Y_I=Y_I, tau=tau, T0=T0, Z=Z, mu=mu)
