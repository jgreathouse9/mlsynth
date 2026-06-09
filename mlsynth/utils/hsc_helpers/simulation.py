"""Regime-adaptation DGP for the HSC simulation study (Liu & Xu, Appendix C.1).

Re-implements the paper's Monte Carlo so the regime-adaptation finding can be
benchmarked: a treated unit (row 0) and ``N0`` donors load on three latent
trend factors (a random walk, an ARIMA(1,1,0), and an extra integrated trend)
plus an integrated-AR(1) idiosyncratic term whose *commonality* is controlled by
``rho_u`` -- ``rho_u=1`` gives a **shared** stochastic trend (SC-on-levels is
best), ``rho_u=0`` an **idiosyncratic** one (SC-on-differences is best). The
treated unit gets no effect, so post-period error is pure counterfactual error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _integ_ar1(T: int, phi: float, innov: np.ndarray) -> np.ndarray:
    d = np.zeros(T)
    for t in range(1, T):
        d[t] = phi * d[t - 1] + innov[t]
    return np.cumsum(d)


@dataclass(frozen=True)
class HSCLoadings:
    """Fixed (across-rep) loadings for the HSC regime DGP."""

    Lam: np.ndarray          # (N0, 3) donor factor loadings
    lam0: np.ndarray         # (3,) treated loading (inside the donor hull)
    alpha_d: np.ndarray      # (N0,) donor fixed effects


def make_hsc_loadings(N0: int = 20, seed: int = 0) -> HSCLoadings:
    """Draw the fixed donor/treated loadings once (paper's setup block)."""
    m = np.random.default_rng(seed)
    Lam = np.clip(m.normal(0, 0.5, (N0, 3)), -2, 2)
    s8 = m.choice(N0, 8, replace=False)
    lam0 = m.dirichlet(np.ones(8) * 0.5) @ Lam[s8]     # treated loading, in hull
    alpha_d = m.uniform(5, 15, N0)
    return HSCLoadings(Lam=Lam, lam0=lam0, alpha_d=alpha_d)


def simulate_hsc_regime(
    rng: np.random.Generator, loadings: HSCLoadings, *,
    N0: int = 20, T0: int = 100, Tpost: int = 10,
    kappa: float = 2.0, rho_u: float = 0.0, phe: float = 0.25,
) -> np.ndarray:
    """Draw one ``(N0+1, T)`` untreated panel; row 0 = treated, ``T = T0+Tpost``.

    ``rho_u=1`` -> shared stochastic trend; ``rho_u=0`` -> idiosyncratic.
    """
    Lam, lam0, alpha_d = loadings.Lam, loadings.lam0, loadings.alpha_d
    T = T0 + Tpost
    F = np.column_stack([
        np.cumsum(rng.normal(0, 2, T)),                  # random walk
        _integ_ar1(T, 0.5, rng.normal(0, 2, T)),         # ARIMA(1,1,0)
        np.r_[0.0, np.cumsum(rng.normal(0, 1, T - 1))],  # extra trend factor
    ])
    L = np.vstack([lam0, Lam]) @ F.T
    uc = rng.normal(0, np.sqrt(1 - phe ** 2), T)
    E = np.zeros((N0 + 1, T))
    for j in range(N0 + 1):
        ui = rng.normal(0, np.sqrt(1 - phe ** 2), T)
        E[j] = _integ_ar1(T, phe, np.sqrt(rho_u) * uc + np.sqrt(1 - rho_u) * ui)
    alpha = np.concatenate([[0.0], alpha_d])
    return L + kappa * E + rng.normal(0, 1, (N0 + 1, T)) + alpha[:, None] + rng.normal(0, 1, T)
