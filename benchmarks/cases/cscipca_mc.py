"""CSC-IPCA Path-B: the authors' instrumented-PCA Monte Carlo.

Validates mlsynth's ``CSCIPCA`` against the linear factor simulation Wang
(2024) uses to establish the estimator's finite-sample properties (the
``CongWang141/JMP`` replication, ``test1_data_generating`` DGP, paper eq. 13):

    Wang, C. (2024). "Counterfactual and Synthetic Control Method: Causal
    Inference with Instrumented Principal Component Analysis." Job Market Paper.

The DGP is a ``K``-factor interactive-fixed-effects model with covariate-driven
loadings,

    Y_it = (X_it Gamma) F_t + X_it beta + alpha_i + xi_t + D_it delta_t + eps_it,

where the covariates ``X_it`` follow unit-specific VAR(1) processes and the
treated unit carries a covariate drift, pushing it outside the donor convex
hull (so a plain synthetic control must extrapolate and is biased). The paper's
Table 1 headline: the CSC-IPCA bias shrinks as the share ``alpha`` of
*observed* covariates rises -- because the instrumented loadings extract more
signal from more covariates -- and CSC-IPCA is far less biased than a
simplex synthetic control on the same panel.

This case reproduces that geometry with a single treated unit (mlsynth's
``BaseEstimatorResults`` contract, and the paper's own Brexit application),
driving the public ``CSCIPCA.fit()`` at ``L = 9`` covariates, ``K = 3``
factors, ``T0 = 40`` pre-periods over ``M`` draws per ``alpha``:

  =========================  ==========  ==================
  Quantity                   mlsynth     reference
  =========================  ==========  ==================
  bias at alpha = 1/3        ~ +2.0      large, positive
  bias at alpha = 2/3        ~ +1.1      shrinking
  bias at alpha = 1          ~ +0.1      ~ 0 (Table 1)
  naive simplex-SC bias      ~ +7        biased (extrapolation)
  monotone bias in alpha     yes         yes
  CSC-IPCA < SC at every a   yes         yes
  =========================  ==========  ==================

Path B (the authors' own DGP): the case asserts the qualitative shape --
monotone bias reduction in the observed-covariate share, near-unbiasedness once
all covariates are observed, and a strict improvement over the naive SC it is
built to beat -- not exact Monte Carlo cells.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_TRUE_K = 3
_L = 9
_T0 = 40
_T1 = 8
_N_CO = 45
M = 150            # draws per alpha grid point (single-treated is noisy; average down)
_ALPHAS = (1 / 3, 2 / 3, 1.0)


def _stationary_ar1(n: int, rng) -> np.ndarray:
    while True:
        A = rng.random((n, n)) - 0.5
        if np.all(np.abs(np.linalg.eigvals(A)) < 1):
            return A


def _draw(seed: int, alpha_obs: float):
    """One single-treated draw of the paper's eq-13 DGP (vectorized)."""
    rng = np.random.default_rng(seed)
    K, L, T0, T1, N_co = _TRUE_K, _L, _T0, _T1, _N_CO
    N, T = N_co + 1, T0 + T1

    A1 = _stationary_ar1(K, rng)
    F = np.zeros((K, T)); F[:, 0] = rng.uniform(-1, 1, K)
    for t in range(1, T):                                  # factor VAR(1) (recursive)
        F[:, t] = A1 @ F[:, t - 1] + rng.normal(0, 1, K)

    A2 = np.stack([_stationary_ar1(L, rng) for _ in range(N)])  # (N, L, L)
    drift = np.zeros((N, L)); drift[N_co] = 2.0                 # treated covariate drift
    X = np.zeros((N, T, L)); X[:, 0, :] = rng.uniform(-1, 1, (N, L))
    for t in range(1, T):                                  # covariate VAR(1) (recursive in t)
        X[:, t, :] = (np.einsum("nij,nj->ni", A2, X[:, t - 1, :])
                      + drift + rng.normal(0, 1, (N, L)))

    Gamma = rng.uniform(-0.1, 0.1, (L, K))
    beta = rng.uniform(0, 1, L)
    a_fe = rng.uniform(0, 1, N)
    xi = rng.uniform(0, 1, T)
    delta = np.concatenate([np.zeros(T0), np.arange(1, T1 + 1) + rng.normal(0, 1, T1)])

    Y = (np.einsum("itl,lk,kt->it", X, Gamma, F)               # (X Gamma) F
         + X @ beta                                            # linear covariate part
         + a_fe[:, None] + xi[None, :]                         # unit + time FE
         + rng.normal(0, 1, (N, T)))                           # noise
    Y[N_co, T0:] += delta[T0:]                                 # treatment on the treated unit

    n_obs = max(1, int(round(alpha_obs * L)))
    covs = [f"x{l}" for l in range(n_obs)]
    rows = []
    for i in range(N):
        treated = int(i == N_co)
        for t in range(T):
            row = {"unit": i, "time": t, "y": float(Y[i, t]),
                   "D": int(treated and t >= T0)}
            for l in range(n_obs):
                row[f"x{l}"] = float(X[i, t, l])
            rows.append(row)
    return pd.DataFrame(rows), covs, float(delta[T0:].mean()), T0


def _naive_sc_bias(df: pd.DataFrame, T0: int, true_att: float) -> float:
    from mlsynth.utils.bilevel.active_set import solve_simplex_qp
    piv = df.pivot(index="unit", columns="time", values="y")
    treated_id = df[df["D"] == 1]["unit"].iloc[0]
    y_tr = piv.loc[treated_id].to_numpy()
    donors = piv.drop(index=treated_id).to_numpy().T          # (T, N_co)
    w = solve_simplex_qp(donors[:T0], y_tr[:T0])
    return float((y_tr - donors @ w)[T0:].mean() - true_att)


def _mc(alpha_obs: float, base_seed: int):
    from mlsynth import CSCIPCA

    ipca_bias = np.empty(M)
    sc_bias = np.empty(M)
    for i in range(M):
        df, covs, true_att, T0 = _draw(base_seed + i, alpha_obs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            att = CSCIPCA({
                "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                "time": "time", "covariates": covs, "n_factors": _TRUE_K,
                "inference": False,
            }).fit().effects.att
        ipca_bias[i] = att - true_att
        sc_bias[i] = _naive_sc_bias(df, T0, true_att)
    return float(ipca_bias.mean()), float(sc_bias.mean())


def run() -> dict:
    seeds = {1 / 3: 4000, 2 / 3: 5000, 1.0: 6000}
    ipca, sc_by_alpha = {}, {}
    for a in _ALPHAS:
        ipca[a], sc_by_alpha[a] = _mc(a, seeds[a])
    sc = sc_by_alpha[1.0]             # SC bias is alpha-invariant (outcome-only)
    biases = [abs(ipca[a]) for a in _ALPHAS]
    monotone = all(biases[i] >= biases[i + 1] for i in range(len(biases) - 1))
    beats_sc = all(abs(ipca[a]) < abs(sc) for a in _ALPHAS)
    return {
        "bias_alpha_1_3": ipca[1 / 3],
        "bias_alpha_2_3": ipca[2 / 3],
        "bias_alpha_1": ipca[1.0],
        "naive_sc_bias": sc,
        "bias_monotone_in_alpha": float(monotone),
        "cscipca_beats_sc": float(beats_sc),
    }


# Deterministic (seeds 4000 / 5000 / 6000, M=150, single treated, L=9, K=3,
# T0=40; ~75s). The reproduced Table-1 facts: CSC-IPCA bias shrinks
# monotonically as the observed-covariate share rises (1.27 -> 0.81 -> 0.00) and
# is near-zero once all covariates are observed, and it is far less biased than a
# simplex SC (~7.7) that must extrapolate for a treated unit outside the donor
# hull. Single-treated is noisier than the paper's 5-treated grid, so M=150
# averages the per-draw variance down enough for the monotone ordering to hold;
# centres are the measured means, tolerances absorb platform/BLAS drift. The two
# flag checks carry the qualitative claim at zero tolerance.
EXPECTED = {
    "bias_alpha_1_3": (1.27, 0.5),              # large positive bias, few covariates
    "bias_alpha_2_3": (0.81, 0.5),              # shrinking
    "bias_alpha_1": (0.0, 0.4),                 # near-unbiased, all covariates observed
    "naive_sc_bias": (7.7, 2.0),                # SC biased by extrapolation
    "bias_monotone_in_alpha": (1.0, 0.0),
    "cscipca_beats_sc": (1.0, 0.0),
}
