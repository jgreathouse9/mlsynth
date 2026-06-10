r"""Reference data-generating processes for the proximal estimators.

Reusable simulators behind the durable proximal Path-B benchmarks, each a
faithful port of an authors' own reference data-generating code:

* :func:`simulate_spsc_ifem` -- the *"Toy Example from Interactive Fixed Effect
  Models"* shipped in the README of the SPSC R package
  (``github.com/qkrcks0218/SPSC``), accompanying Park & Tchetgen Tchetgen
  (2025), *"Single Proxy Synthetic Control,"* JCI 13(1), 20230079. Drives the
  ``spsc_ifem_mc`` benchmark.
* :func:`simulate_dr_proximal_normal` -- the ``DR_Proximal_SC/simulation/normal``
  design of Qiu, Shi, Miao, Dobriban & Tchetgen Tchetgen (2024), *"Doubly robust
  proximal synthetic controls,"* Biometrics 80(2), ujae055. Drives the
  ``dr_proximal_mc`` benchmark.
* :func:`simulate_proximal_surrogates` -- the ``freshtaste/proximal`` ``dgp.py``
  (Section 4.1 robustness) of Liu, Tchetgen Tchetgen & Varjao (2024), *"Proximal
  Causal Inference for Synthetic Control with Surrogates,"* AISTATS. Drives the
  ``proximal_surrogates_mc`` benchmark.

The SPSC interactive-fixed-effects DGP follows.

The DGP
-------
One treated unit and ``n_valid + n_invalid`` donors are observed over
``T = T0 + T1`` periods. Two latent factors drive a *valid* donor block ``W`` and
an *invalid* donor block ``V``; only ``W`` shares the treated unit's factors, so a
**single proxy** (the donor outcomes) suffices to identify the bridge while the
invalid donors act as noise the estimator must down-weight.

.. math::

   \lambda_t = b_t + \xi_t, \qquad
   \xi_t = \rho\,\xi_{t-1} + \tfrac{\rho}{2}\,\xi_{t-2} + \nu_t,

with a deterministic baseline trend ``b_t = 1 + t / T0`` (so the untreated
trajectories drift -- the regime where detrending, SPSC-DT, matters). The valid
donors load on ``lambda`` via ``W_coef``; the treated unit loads on the column
means of ``W_coef`` (so a convex-ish combination of valid donors reproduces the
treated factor); the invalid donors load on an independent factor ``zeta``. Idiosyncratic
errors are correlated across the treated unit and the valid donors through a
common shock (``corr``), making the SC problem non-trivial:

.. math::

   Y^{(0)}_t = \lambda_t' \mu_{Y} + e_{Yt}, \quad
   W_t = \lambda_t' \mathrm{Wcoef} + e_{Wt}, \quad
   V_t = \zeta_t' \mathrm{Vcoef} + e_{Vt}.

The post-treatment effect is a constant ``true_att`` perturbed by a small
mean-zero noise (``beta_eps``), matching the reference's *error-prone* treatment
effect. The donor matrix handed to the estimator stacks ``[W, V]`` -- the single
proxy group -- and the treated unit's own (optionally detrended) pre-period path
is the instrument.

Determinism
-----------
Every random draw comes from one :class:`numpy.random.Generator`, so a fixed
``seed`` reproduces a draw bit-for-bit. Factor innovations and idiosyncratic
errors are redrawn on each call, so repeated calls are independent draws from the
sampling distribution -- the correct design for a bias / coverage study.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "SPSCSimSample",
    "simulate_spsc_ifem",
    "DRSimSample",
    "simulate_dr_proximal_normal",
    "ProxSurrogateSimSample",
    "simulate_proximal_surrogates",
]

_RHO = 0.5            # latent-factor serial dependence
_CORR = 0.5          # corr(treated error, valid-donor error)
_SD = 0.1            # idiosyncratic error scale
_TRUE_ATT = 3.0


@dataclass(frozen=True)
class SPSCSimSample:
    """One draw from the SPSC interactive-fixed-effects DGP.

    Attributes
    ----------
    y : np.ndarray
        Observed treated path ``(T,)`` -- pre-period treatment-free, post-period
        with the (noisy) effect added.
    donors : np.ndarray
        Single-proxy donor matrix ``(T, n_valid + n_invalid)`` -- the valid block
        ``W`` followed by the invalid block ``V``.
    T0 : int
        Number of pre-treatment periods.
    true_att : float
        Average post-treatment effect actually realised on this draw (the mean of
        the per-period error-prone effects); the estimand a consistent SPSC ATT
        should recover.
    true_effect : np.ndarray
        Per-post-period error-prone effects ``(T1,)`` (``mean == true_att``).
    n_valid, n_invalid : int
        Counts of valid (factor-sharing) and invalid donors.
    """

    y: np.ndarray
    donors: np.ndarray
    T0: int
    true_att: float
    true_effect: np.ndarray
    n_valid: int
    n_invalid: int


def _ifem_factor(Tt: int, rng: np.random.Generator, rho: float, sd: float,
                 baseline: np.ndarray) -> np.ndarray:
    """Two-dimensional latent factor with a baseline trend, shape ``(Tt + 2, 2)``.

    ``xi_t = rho * xi_{t-1} + (rho / 2) * xi_{t-2} + nu_t`` (innovation scale
    ``sd``), seeded by two ``N(0, sd^2)`` draws, plus the deterministic baseline.
    """
    eps = np.zeros((Tt + 2, 2))
    eps[:2, :] = rng.standard_normal((2, 2)) * sd
    for t in range(Tt):
        eps[t + 2, :] = rho * eps[t + 1, :] + (rho / 2.0) * eps[t, :] + rng.standard_normal(2) * sd
    return eps + baseline[:, None]


def simulate_spsc_ifem(
    T0: int = 50,
    T1: int = 50,
    n_valid: int = 8,
    n_invalid: int = 8,
    rho: float = _RHO,
    corr: float = _CORR,
    sd: float = _SD,
    true_att: float = _TRUE_ATT,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> SPSCSimSample:
    """Draw one sample from the SPSC interactive-fixed-effects DGP.

    Parameters
    ----------
    T0, T1 : int, default 50
        Pre- and post-treatment period counts.
    n_valid, n_invalid : int, default 8
        Number of valid (factor-sharing) and invalid donors. ``n_valid`` may not
        exceed 8 (the reference fixes eight valid-donor loadings).
    rho : float, default 0.5
        Latent-factor serial-dependence coefficient.
    corr : float, default 0.5
        Correlation between the treated unit's and the valid donors'
        idiosyncratic errors (a shared common shock).
    sd : float, default 0.1
        Idiosyncratic / innovation error scale.
    true_att : float, default 3.0
        Mean post-treatment effect (the reference's ``True.ATT``).
    rng : numpy.random.Generator, optional
        Generator; takes precedence over ``seed``.
    seed : int, optional
        Convenience seed used to build a generator when ``rng`` is None.

    Returns
    -------
    SPSCSimSample
        The observed treated path, the stacked ``[W, V]`` donor matrix, ``T0``,
        and the realised effect.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if not 1 <= n_valid <= 8:
        raise ValueError("n_valid must be in 1..8 (reference fixes 8 loadings).")
    if n_invalid < 0:
        raise ValueError("n_invalid must be non-negative.")
    if T0 < 1 or T1 < 1:
        raise ValueError("T0 and T1 must be positive.")

    Tt = T0 + T1
    # Reference factor loadings (2 x 8 each).
    W_coef = np.vstack([np.arange(8, 0, -1) / 4.0,
                        np.repeat([0.8, 0.6, 0.4, 0.2], 2)])
    V_coef = np.vstack([np.ones(8), np.full(8, 0.5)])
    Y0_coef = W_coef[:, :n_valid] @ np.full(n_valid, 1.0 / n_valid)   # treated loadings
    # Baseline trend b_t = 1 + t / T0 (first two warm-up rows held flat at 1).
    baseline = np.concatenate([[0.0, 0.0], np.arange(1, Tt + 1) / T0]) + 1.0

    lam = _ifem_factor(Tt, rng, rho, sd, baseline)       # valid-donor / treated factor
    zeta = _ifem_factor(Tt, rng, rho, sd, baseline)      # invalid-donor factor

    common = rng.standard_normal(Tt + 2)
    y0_idio = rng.standard_normal(Tt + 2)
    w_idio = rng.standard_normal((Tt + 2, n_valid))
    v_idio = rng.standard_normal((Tt + 2, n_invalid))
    mix = np.sqrt(1.0 - corr ** 2)
    y0_eps = sd * (corr * common + mix * y0_idio)
    w_eps = sd * (corr * common[:, None] + mix * w_idio)
    v_eps = sd * v_idio

    Y0 = lam @ Y0_coef + y0_eps
    W = lam @ W_coef[:, :n_valid] + w_eps
    V = zeta @ V_coef[:, :n_invalid] + v_eps

    beta_eps = np.concatenate([np.zeros(T0 + 2), rng.standard_normal(T1)])
    beta = np.concatenate([np.zeros(T0 + 2), np.full(T1, true_att)])
    beta_noisy = beta + beta_eps * sd
    Y1 = Y0 + beta_noisy

    Yobs = np.empty(Tt + 2)
    Yobs[: 2 + T0] = Y0[: 2 + T0]
    Yobs[2 + T0:] = Y1[2 + T0:]

    # Drop the two warm-up rows; donors are the single proxy group [W, V].
    donors = np.hstack([W, V])[2:]
    y = Yobs[2:]
    true_effect = beta_noisy[2 + T0:]
    return SPSCSimSample(
        y=y,
        donors=donors,
        T0=T0,
        true_att=float(np.mean(true_effect)),
        true_effect=true_effect,
        n_valid=n_valid,
        n_invalid=n_invalid,
    )


# --------------------------------------------------------------------------- #
# Qiu, Shi, Miao, Dobriban & Tchetgen Tchetgen (2024) doubly-robust proximal
# Monte Carlo -- the authors' ``DR_Proximal_SC/simulation/normal`` design.
# --------------------------------------------------------------------------- #
_DR_TRUE_ATE = 2.0


@dataclass(frozen=True)
class DRSimSample:
    """One draw from the doubly-robust proximal ``normal`` DGP.

    Attributes
    ----------
    y : np.ndarray
        Treated outcome ``Y``, shape ``(T,)`` (post-period carries ``true_att``).
    donor_outcomes : np.ndarray
        Outcome proxies ``W`` (negative-control outcomes), shape ``(T, nU)``.
    donor_proxies : np.ndarray
        Treatment proxies ``Z`` (negative-control exposures), shape ``(T, nU)``.
    T0 : int
        Number of pre-treatment periods (``T // 2``).
    true_att : float
        The data-generating ATE (constant post-period shift).
    n_confounders : int
        Number of latent confounders ``nU``.
    misspecified : bool
        Whether the outcome bridge is misspecified (a nonlinear ``U`` signal was
        injected into ``Y``, breaking a linear outcome model while the treatment
        bridge stays correct -- the double-robustness stress test).
    """

    y: np.ndarray
    donor_outcomes: np.ndarray
    donor_proxies: np.ndarray
    T0: int
    true_att: float
    n_confounders: int
    misspecified: bool


def simulate_dr_proximal_normal(
    T: int = 1000,
    n_confounders: int = 2,
    true_att: float = _DR_TRUE_ATE,
    misspecify: bool = False,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> DRSimSample:
    r"""Draw one sample from the DR-proximal ``normal`` DGP (Qiu et al. 2024).

    A faithful port of the reference ``DR_Proximal_SC/simulation/normal``
    data-generating code:

    .. math::

        U_t = 0.1\,U_{t-1} + 0.9\,\nu_t, \quad
        Y_t = \tau\,\mathbb{1}\{t > T_0\} + 2\,s(U_t) + e_t, \\
        W_{tj} = 2 U_{tj} + \varepsilon^W_{tj}, \qquad
        Z_{tj} = 2 U_{tj} + \varepsilon^Z_{tj},

    with ``nU`` latent AR(1) confounders, ``T0 = T // 2``, and unit-variance
    Gaussian noise throughout. The donor outcomes ``W`` and the donor proxies
    ``Z`` are independent error-prone shadows of the confounder. By default the
    confounding signal is linear, ``s(U) = sum_j U_j``; with ``misspecify=True``
    it is ``s(U) = sum_j U_j + 0.7 (sum_j U_j)^2`` -- a nonlinearity an
    intercept-free linear outcome bridge cannot absorb, so the outcome-only PI
    estimator is biased while the doubly-robust estimator (with a correct
    treatment bridge) stays consistent.

    Parameters
    ----------
    T : int, default 1000
        Total number of periods (pre-period is ``T // 2``).
    n_confounders : int, default 2
        Number of latent confounders ``nU`` (and the donor / proxy count).
    true_att : float, default 2.0
        Constant post-period treatment effect (the reference's ``true.ATE``).
    misspecify : bool, default False
        Inject the nonlinear confounding signal that breaks a linear outcome
        bridge (the double-robustness stress test).
    rng : numpy.random.Generator, optional
        Generator; takes precedence over ``seed``.
    seed : int, optional
        Convenience seed used to build a generator when ``rng`` is None.

    Returns
    -------
    DRSimSample
        The treated outcome, the ``W`` / ``Z`` proxy blocks, ``T0``, and the
        truth.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if T < 2:
        raise ValueError("T must be at least 2.")
    if n_confounders < 1:
        raise ValueError("n_confounders must be positive.")

    nU = n_confounders
    T0 = T // 2
    U = np.empty((T, nU))
    U[0] = rng.standard_normal(nU)
    for t in range(1, T):
        U[t] = 0.1 * U[t - 1] + 0.9 * rng.standard_normal(nU)
    sig = U.sum(1)
    signal = sig + 0.7 * sig ** 2 if misspecify else sig
    post = (np.arange(1, T + 1) > T0).astype(float)
    y = true_att * post + 2.0 * signal + rng.standard_normal(T)
    W = 2.0 * U + rng.standard_normal((T, nU))
    Z = 2.0 * U + rng.standard_normal((T, nU))
    return DRSimSample(
        y=y,
        donor_outcomes=W,
        donor_proxies=Z,
        T0=T0,
        true_att=float(true_att),
        n_confounders=nU,
        misspecified=bool(misspecify),
    )


# --------------------------------------------------------------------------- #
# Liu, Tchetgen Tchetgen & Varjao (2024) proximal-with-surrogates Monte Carlo
# -- the authors' ``freshtaste/proximal`` ``dgp.py`` (Section 4.1 robustness).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProxSurrogateSimSample:
    """One draw from the proximal-with-surrogates DGP (Liu et al. 2024).

    Attributes
    ----------
    y : np.ndarray
        Treated outcome ``Y``, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes ``W`` (negative-control outcomes), shape ``(T, F)``.
    donor_proxies : np.ndarray
        Donor proxies ``Z0`` -- a second error-prone copy of the latent factor
        that instruments ``W``, shape ``(T, F)``.
    surrogate_outcomes : np.ndarray
        Post-period surrogates ``X`` (correlates of the effect), shape ``(T, K)``.
    surrogate_proxies : np.ndarray
        Surrogate proxies ``Z1`` instrumenting ``X``, shape ``(T, K)``.
    T0 : int
        Number of pre-treatment periods.
    true_att : float
        Realised average post-treatment effect (mean of the per-period effects).
    n_donor_factors, n_surrogate_factors : int
        Factor counts ``F`` and ``K``.
    """

    y: np.ndarray
    donor_outcomes: np.ndarray
    donor_proxies: np.ndarray
    surrogate_outcomes: np.ndarray
    surrogate_proxies: np.ndarray
    T0: int
    true_att: float
    n_donor_factors: int
    n_surrogate_factors: int


def _ar1_noise(T: int, cols: Optional[int], rng: np.random.Generator,
               phi: float, ar: bool) -> np.ndarray:
    """AR(1) (``ar=True``) or i.i.d. Gaussian noise, shape ``(T,)`` or ``(T, cols)``."""
    shape = (T,) if cols is None else (T, cols)
    nu = rng.standard_normal(shape)
    if not ar:
        return nu
    e = np.zeros_like(nu)
    for t in range(1, T):
        e[t] = phi * e[t - 1] + nu[t]
    return e


def simulate_proximal_surrogates(
    T: int = 200,
    T0: Optional[int] = None,
    n_donor_factors: int = 1,
    n_surrogate_factors: int = 1,
    ar_errors: bool = True,
    ar_phi: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> ProxSurrogateSimSample:
    r"""Draw one sample from the proximal-with-surrogates DGP (Liu et al. 2024).

    A faithful port of the authors' reference ``freshtaste/proximal`` ``dgp.py``
    used in their Section 4.1 robustness study. A **trending** latent factor
    drives the treated outcome and the donor block; a second factor drives the
    post-period surrogates:

    .. math::

        \lambda_t = \log t + \nu_t, \quad
        Y_t = \lambda_t' \boldsymbol{1} + e_{Yt}
              + \mathbb{1}\{t > T_0\}(\rho_t' \boldsymbol{1} + \delta_t), \\
        W_t = \lambda_t + e^W_t, \quad Z0_t = \lambda_t + e^{Z0}_t, \quad
        X_t = \rho_t + e^X_t, \quad Z1_t = \rho_t + e^{Z1}_t,

    with AR(1) (or i.i.d.) idiosyncratic errors. ``W`` and ``Z0`` are two
    independent error-prone copies of the donor factor (so ``Z0`` instruments
    ``W``); ``X`` and ``Z1`` are two copies of the effect factor ``rho``. The
    trending ``log t`` factor is the regime where classical SC, fitting an
    error-laden donor regression, extrapolates with growing bias while the
    proximal estimators stay consistent.

    Parameters
    ----------
    T : int, default 200
        Total periods.
    T0 : int, optional
        Pre-period length; defaults to ``T // 2``.
    n_donor_factors, n_surrogate_factors : int, default 1
        Factor counts ``F`` and ``K`` (and the donor / surrogate block widths).
    ar_errors : bool, default True
        Use AR(1) idiosyncratic errors (the reference's robustness regime).
    ar_phi : float, default 0.1
        AR(1) coefficient.
    rng : numpy.random.Generator, optional
        Generator; takes precedence over ``seed``.
    seed : int, optional
        Convenience seed used to build a generator when ``rng`` is None.

    Returns
    -------
    ProxSurrogateSimSample
        The treated outcome, the donor / surrogate outcome and proxy blocks,
        ``T0``, and the realised effect.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if T < 2:
        raise ValueError("T must be at least 2.")
    if n_donor_factors < 1 or n_surrogate_factors < 1:
        raise ValueError("factor counts must be positive.")
    T0 = T // 2 if T0 is None else int(T0)
    if not 1 <= T0 < T:
        raise ValueError("T0 must satisfy 1 <= T0 < T.")
    F, K = n_donor_factors, n_surrogate_factors

    epsY = _ar1_noise(T, None, rng, ar_phi, ar_errors)
    epsW = _ar1_noise(T, F * 2, rng, ar_phi, ar_errors)
    epsX = _ar1_noise(T, K * 2, rng, ar_phi, ar_errors)
    delta = _ar1_noise(T, None, rng, ar_phi, ar_errors)

    lam = np.log(np.linspace(1, T, T))[:, None] + rng.standard_normal((T, F))  # trending
    rho = rng.standard_normal((T, K))
    rho[:, 0] += 1.0

    y = lam @ np.ones(F) + epsY
    effect = rho[T0:] @ np.ones(K)
    y[T0:] += effect + delta[T0:]
    W = lam + epsW[:, :F]
    Z0 = lam + epsW[:, F:]
    X = rho + epsX[:, :K]
    Z1 = rho + epsX[:, K:]
    return ProxSurrogateSimSample(
        y=y,
        donor_outcomes=W,
        donor_proxies=Z0,
        surrogate_outcomes=X,
        surrogate_proxies=Z1,
        T0=T0,
        true_att=float(np.mean(effect)),
        n_donor_factors=F,
        n_surrogate_factors=K,
    )
