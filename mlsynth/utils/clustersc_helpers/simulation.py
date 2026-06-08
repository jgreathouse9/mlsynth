"""Subgroup sine-mixture DGP for ClusterSC validation (Rho et al. 2025).

Re-implements the synthetic data-generating process of Rho, Tang, Bergam,
Cummings & Misra (2025), *ClusterSC: Advancing Synthetic Control with Donor
Selection* (Section 6.1), generalised from the paper's two-subgroup example
(``generate_sine_dataset_A`` / ``_B`` in the authors' ``syclib.gendata``) to an
arbitrary number of subgroups ``K``.

Each subgroup is a rank-``rank`` additive sine mixture occupying a distinct
frequency band; donor rows are random convex-ish mixtures of that band's basis
plus i.i.d. Gaussian observation noise. With ``K`` well-separated subgroups the
pooled donor matrix has true rank ``K * rank`` -- so once ``K * rank`` exceeds
the pre-period length ``T0`` the whole-pool RSC fit must under-denoise, which is
exactly the curse-of-dimensionality regime where the paper argues donor
clustering pays off.

The generator deliberately mirrors the authors' signal model: ``generate_sine_wave``
fast-forwards time by ``10π`` and drops a 20% burn-in (:func:`_sine_wave`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


def _sine_wave(alpha: float, omega: float, phi: float, num_time: int,
               rng: np.random.Generator) -> np.ndarray:
    """One basis signal, matching ``syclib.gendata.generate_sine_wave`` (noise-free)."""
    time = np.arange(num_time) * 10 * np.pi
    return alpha * np.sin(2 * np.pi * omega * time / 360 + phi)


def _subgroup(n: int, T: int, noise: float, rank: int,
              omega_low: float, omega_high: float,
              rng: np.random.Generator) -> np.ndarray:
    """Generate a ``(T, n)`` subgroup block (mirrors ``generate_sine_dataset_A``)."""
    burned = int(T * 1.2)
    basis = np.zeros((rank, T))
    for i in range(rank):
        alpha = rng.beta(2, 2)
        omega = rng.uniform(omega_low, omega_high)
        phi = rng.normal(0, 1)
        y = _sine_wave(alpha, omega, phi, burned, rng)
        basis[i] = y[burned - T:]            # drop the 20% burn-in
    weights = rng.uniform(0, 1, (n, rank))
    data = weights @ basis + rng.normal(0, noise, (n, T))
    return data.T                            # (T, n)


@dataclass
class SubgroupPanel:
    """A generated subgroup panel.

    Attributes
    ----------
    wide : pd.DataFrame
        ``(T, n)`` outcome matrix; columns are integer unit ids ``0..n-1``.
    labels : np.ndarray
        Subgroup id per unit, shape ``(n,)``.
    target_ids : list[int]
        Units drawn from subgroup 0, the pool the placebo targets come from.
    T0 : int
        Pre-period length.
    rank : int
        Per-subgroup rank.
    K : int
        Number of subgroups.
    """

    wide: pd.DataFrame
    labels: np.ndarray
    target_ids: List[int]
    T0: int
    rank: int
    K: int


def simulate_subgroup_panel(
    *,
    K: int = 5,
    rank: int = 3,
    n_per_group: int = 200,
    T: int = 16,
    T0: int = 8,
    noise: float = 0.30,
    seed: int = 0,
) -> SubgroupPanel:
    """Generate a ``K``-subgroup sine-mixture panel (true pooled rank ``K*rank``).

    Targets are drawn from subgroup 0; with ``K*rank > T0`` the pooled donor
    matrix is higher-rank than the pre-period can support, the regime where
    donor clustering improves the synthetic-control fit.
    """
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for g in range(K):
        blocks.append(_subgroup(n_per_group, T, noise, rank,
                                omega_low=1 + 2 * g, omega_high=3 + 2 * g, rng=rng))
        labels.extend([g] * n_per_group)
    wide = pd.DataFrame(np.concatenate(blocks, axis=1))
    wide.columns = range(wide.shape[1])
    labels = np.asarray(labels)
    target_ids = list(np.where(labels == 0)[0])
    return SubgroupPanel(wide=wide, labels=labels, target_ids=target_ids,
                         T0=T0, rank=rank, K=K)


@dataclass
class RSCPanel:
    """A latent-variable panel with the noise-free mean matrix retained.

    Attributes
    ----------
    means : np.ndarray
        ``(N, T)`` true (noise-free) mean matrix ``M``.
    observed : np.ndarray
        ``(N, T)`` observed matrix ``X = M + N(0, noise^2)``.
    T0 : int
        Pre-intervention length.
    noise : float
        Observation-noise standard deviation.
    """

    means: np.ndarray
    observed: np.ndarray
    T0: int
    noise: float


def simulate_rsc_panel(
    *,
    N: int = 100,
    T: int = 2000,
    T0: int = 1600,
    noise: float = 1.0,
    seed: int = 0,
) -> RSCPanel:
    """Latent-variable DGP of Amjad, Shah & Shen (2018), RSC Section 5.3.

    Each unit ``i`` has a latent feature ``θ_i ~ U[0, 1]``; time is the latent
    variable ``ρ_t = t``. The mean is

    .. math::

       m_{it} = θ_i + (0.3\\,θ_i\\,ρ_t/T)\\,e^{ρ_t/T}
                + \\cos(f_1 π/180) + 0.5\\sin(f_2 π/180)
                + 1.5\\cos(f_3 π/180) - 0.5\\sin(f_4 π/180),

    with periodicities ``f_1 = ρ_t mod 360``, ``f_2 = ρ_t mod 180``,
    ``f_3 = 2ρ_t mod 360``, ``f_4 = 2ρ_t mod 180`` (shared across units). The
    observed matrix adds i.i.d. ``N(0, noise^2)`` noise. The signal is
    approximately rank 3 (unit intercept + shared seasonal pattern +
    ``θ``-scaled trend), the low-rank regime RSC targets.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 1.0, N)
    t = np.arange(1, T + 1)
    f1 = t % 360
    f2 = t % 180
    f3 = (2 * t) % 360
    f4 = (2 * t) % 180
    seasonal = (np.cos(f1 * np.pi / 180) + 0.5 * np.sin(f2 * np.pi / 180)
                + 1.5 * np.cos(f3 * np.pi / 180) - 0.5 * np.sin(f4 * np.pi / 180))
    trend = 0.3 * (t / T) * np.exp(t / T)
    means = theta[:, None] + theta[:, None] * trend[None, :] + seasonal[None, :]
    observed = means + rng.normal(0.0, noise, (N, T))
    return RSCPanel(means=means, observed=observed, T0=T0, noise=noise)
