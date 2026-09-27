"""The v2 simulation design of Shen (2606.18512v2), content/simulations.tex.

Everything below is stated in v2 except the numeric entries of A_0 and A_1,
which the paper gives only structurally: both are 4 x 8 and fixed across
replications, and "the lowest-frequency harmonic is absent under control and
present under treatment". The reconstruction therefore zeroes A_0's first two
columns (the omega_0 sine/cosine pair) and draws the rest once from a fixed
seed. That is the one remaining free choice, and it is the thing the gate is
testing: with the scaling and sigma now specified, does a non-degenerate pair
respecting the stated structure give nominal coverage?

v1 left the harmonics, the scaling and sigma unreported as well, and the
reconstruction then produced a Page spectrum spanning 500,000x with the
smallest signal direction at ~1e-4 against a noise floor of 3.7.
"""
from __future__ import annotations

import numpy as np

N_MAX = 150
H_MAX = 10
SIGMA = 0.10
R_Y, R_Z = 4, 8


def dims(n: int):
    """N_1 = T_0 = L = n, T_1(n) = 8(n + h_max), T(n) = T_0 + T_1(n)."""
    T1 = 8 * (n + H_MAX)
    return dict(N1=n, T0=n, L=n, T1=T1, T=n + T1)


def basis(t: np.ndarray, T_star: int) -> np.ndarray:
    """b(t), the 8 harmonic components of simulations.tex eq. (b(t))."""
    w0, w1, w2, w3 = 2 * np.pi / (8 * T_star), 2 * np.pi / 12, 2 * np.pi / 37, 2 * np.pi / 91
    return np.stack([
        np.sin(w0 * t - np.pi / 3), np.cos(w0 * t - np.pi / 3),
        np.sin(w1 * t), np.cos(w1 * t),
        np.sin(w2 * t), np.cos(w2 * t),
        np.sin(w3 * t), np.cos(w3 * t),
    ])                                              # 8 x len(t)


def loadings(seed: int = 0):
    """A_0, A_1 in R^{4x8}. The only unreported piece; see the module docstring."""
    rng = np.random.default_rng(seed)
    A1 = rng.standard_normal((4, 8))
    A0 = A1.copy()
    A0[:, :2] = 0.0                                 # omega_0 absent under control
    return A0, A1


def latent(seed: int, A0: np.ndarray, A1: np.ndarray):
    """One latent replication at n_max: donor factors, target factor, time factors."""
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((N_MAX, 3))
    xi = (xi - xi.mean(0)) / xi.std(0)              # standardised across the 150 donors
    U = np.column_stack([np.ones(N_MAX), xi])       # N_MAX x 4

    A_set = rng.choice(25, size=8, replace=False)   # 8 donors from the first 25
    lam = rng.dirichlet(np.ones(8))
    u_N = lam @ U[A_set]                            # target factor, in the donor span

    d = dims(N_MAX)
    T_star = d["T"] + H_MAX
    t = np.arange(1, T_star + 1)
    b = basis(t, T_star)                            # 8 x T_star
    V0, V1 = A0 @ b, A1 @ b                         # 4 x T_star each

    # common scale factor so max_{i,t,d} |<u_i, v_t(d)>| <= 0.8
    allU = np.vstack([U, u_N])
    peak = max(np.abs(allU @ V0).max(), np.abs(allU @ V1).max())
    c = 0.8 / peak
    return U * 1.0, u_N, V0 * c, V1 * c, T_star


def panel(n: int, U, u_N, V0, V1, T_star, noise_seed: int):
    """Assemble one panel at dimension n from a latent replication.

    Nested design: keep the first n donors and the trailing window of length
    T(n) + h_max, so every n forecasts the same future dates.
    """
    d = dims(n)
    T0, T1, T, L, N1 = d["T0"], d["T1"], d["T"], d["L"], d["N1"]
    win = T + H_MAX
    sl = slice(T_star - win, T_star)                # trailing window
    V0w, V1w = V0[:, sl], V1[:, sl]
    Ud, uN = U[:N1], u_N

    rng = np.random.default_rng(noise_seed)
    # signals: control everywhere pre; donors treated post, target control post
    sig_d_pre = Ud @ V0w[:, :T0]
    sig_d_post = Ud @ V1w[:, T0:T]                  # donors treated
    sig_t_pre = uN @ V0w[:, :T0]

    noise = rng.normal(0, SIGMA, (N1 + 1, win))
    Y_pre_donors = sig_d_pre + noise[:N1, :T0]
    Y_post_donors = sig_d_post + noise[:N1, T0:T]
    y_pre_target = sig_t_pre + noise[N1, :T0]

    # estimand: the target's noiseless TREATED outcome after T
    theta_path = uN @ V1w[:, T:T + H_MAX]
    return y_pre_target, Y_pre_donors, Y_post_donors, theta_path, d


def noiseless(n: int, U, u_N, V0, V1, T_star):
    """The same panel with sigma = 0, for the algebra check."""
    d = dims(n)
    T0, T, N1 = d["T0"], d["T"], d["N1"]
    win = T + H_MAX
    sl = slice(T_star - win, T_star)
    V0w, V1w = V0[:, sl], V1[:, sl]
    Ud, uN = U[:N1], u_N
    return (uN @ V0w[:, :T0], Ud @ V0w[:, :T0], Ud @ V1w[:, T0:T],
            uN @ V1w[:, T:T + H_MAX], d)
