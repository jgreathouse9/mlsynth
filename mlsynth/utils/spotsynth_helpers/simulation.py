"""The paper's local-linear-trend DGP for SPOTSYNTH examples/replications.

Implements the data-generating process of O'Riordan & Gilligan-Lee (2025),
Appendix B: a target series driven by a sum of latent local-linear-trend
processes, a large pool of donors that are noisy proxies of those latents, and
a random subset of donors hit by a constant spillover effect from the
intervention. Returns a tidy long panel (one treated unit + donor pool) ready
for :class:`mlsynth.SPOTSYNTH`, together with the ground-truth validity mask.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def simulate_spillover_panel(
    n_donors: int = 120,
    n_latent: int = 10,
    T0: int = 80,
    n_post: int = 20,
    sigma_x: float = 0.5,
    frac_invalid: float = 0.8,
    tau: float = 2.0,
    spillover: float = -2.0,
    spillover_ramp: int = 1,
    sigma_u: float = 1.0,
    sigma_delta: float = 0.1,
    sigma_y: float = 0.1,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Simulate one panel from the Appendix B local-linear-trend DGP.

    .. math::

        u_j^{t+1} &\\sim \\mathcal N(u_j^t + \\delta_j^t,\\ \\sigma_u), \\\\
        \\delta_j^{t+1} &\\sim \\mathcal N(S_j + \\rho_j(\\delta_j^t - S_j),\\ \\sigma_\\delta), \\\\
        y^t &\\sim \\mathcal N\\Bigl(\\sum_j u_j^t + \\tau I^t,\\ \\sigma_y\\Bigr), \\\\
        x_i^t &\\sim \\mathcal N\\Bigl(\\sum_j u_j^t + \\tau_{x_i} I^t,\\ \\sigma_x\\Bigr),

    with :math:`S_j \\sim \\mathcal N(0.1, 0.1)`, :math:`\\rho_j \\sim U(0, 1)`,
    intervention indicator :math:`I^t = \\mathbb 1\\{t \\ge T_0\\}`, target effect
    :math:`\\tau`, and donor spillover :math:`\\tau_{x_i} =` ``spillover`` for the
    invalid donors and 0 for the valid ones.

    Parameters
    ----------
    n_donors : int
        Size of the donor pool.
    n_latent : int
        Number of latent local-linear-trend processes.
    T0, n_post : int
        Pre- and post-intervention period counts.
    sigma_x : float
        Donor-noise standard deviation (low/medium/high ~ 0.1/0.5/1.0 in the
        paper). When ``sigma_x`` approaches ``|spillover|`` the screen's
        false-negative rate rises (Figure 2).
    frac_invalid : float
        Fraction of donors hit by the spillover effect (0.8 in the paper).
    tau, spillover : float
        Treatment effect on the target and spillover effect on invalid donors.
    spillover_ramp : int
        Onset speed of the donor spillover: ``1`` is a sharp/immediate level
        shift (the paper's DGP); larger values ramp the spillover linearly to its
        final level over that many post-periods (a gradual onset), holding the
        final level fixed. Used to study detection power vs onset speed.
    sigma_u, sigma_delta, sigma_y : float
        Latent-level, slope, and target-noise standard deviations.
    seed : int
        RNG seed.

    Returns
    -------
    (df, valid_mask) : tuple
        ``df`` is a long panel with columns ``unit``, ``time``, ``Y``,
        ``treated``; the treated unit is named ``"target"`` and donors
        ``"d{j}"``. ``valid_mask`` is a boolean array (length ``n_donors``,
        aligned with sorted donor names) flagging the *valid* donors.
    """
    rng = np.random.default_rng(seed)
    T = T0 + n_post

    S = rng.normal(0.1, 0.1, n_latent)
    rho = rng.uniform(0.0, 1.0, n_latent)
    u = np.zeros((T, n_latent))
    delta = np.zeros((T, n_latent))
    u[0] = rng.normal(0.0, 1.0, n_latent)
    delta[0] = S.copy()
    for t in range(1, T):
        delta[t] = rng.normal(S + rho * (delta[t - 1] - S), sigma_delta)
        u[t] = rng.normal(u[t - 1] + delta[t - 1], sigma_u)

    I = np.zeros(T)
    I[T0:] = 1.0
    signal = u.sum(axis=1)
    y = rng.normal(signal + tau * I, sigma_y)

    # Spillover onset profile: ``spillover_ramp`` controls how fast the donor
    # spillover reaches its final level (1 = sharp/immediate, larger = gradual
    # ramp over that many post-periods). Holds the final level fixed.
    ramp = np.zeros(T)
    n_post_eff = T - T0
    if spillover_ramp <= 1:
        ramp[T0:] = 1.0
    else:
        prof = np.minimum(1.0, (np.arange(n_post_eff) + 1) / float(spillover_ramp))
        ramp[T0:] = prof

    invalid = np.zeros(n_donors, dtype=bool)
    invalid[: int(round(frac_invalid * n_donors))] = True
    rng.shuffle(invalid)
    X = np.column_stack([
        rng.normal(signal + (spillover * ramp if invalid[i] else 0.0), sigma_x)
        for i in range(n_donors)
    ])

    rows = []
    for t in range(T):
        rows.append({"unit": "target", "time": t, "Y": float(y[t]),
                     "treated": int(I[t])})
        for i in range(n_donors):
            rows.append({"unit": f"d{i}", "time": t, "Y": float(X[t, i]),
                         "treated": 0})
    df = pd.DataFrame(rows)

    # valid_mask aligned with the donor order mlsynth uses (sorted names).
    donor_names = sorted(f"d{i}" for i in range(n_donors))
    valid_mask = np.array([not invalid[int(name[1:])] for name in donor_names])
    return df, valid_mask
