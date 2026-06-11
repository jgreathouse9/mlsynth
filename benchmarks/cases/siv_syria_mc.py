"""SIV Path-B: Gulek & Vives (2024) Syrian-calibrated Monte Carlo (Table 1).

Validates mlsynth's ``SIV`` (Synthetic IV) against the paper's Section-6 Table-1
simulation. The DGP
(:func:`mlsynth.utils.siv_helpers.simulation.simulate_siv_sample`) couples the
treatment to the outcome through both a common factor *and* an idiosyncratic
correlation ``r`` swept across the bivariate-normal shock pairs, so 2SLS-TWFE is
biased even with valid post-period instrument assignment. SIV's interactive-factor
debiasing removes that bias.

At the true coefficient ``theta = -0.16``, over Monte Carlo draws at each
``r in {0.5, 0.7, 0.9}`` SIV has **substantially lower absolute bias** than
2SLS-TWFE (the paper's headline ranking):

  =====  ===============  ===============
  r      |bias| SIV       |bias| 2SLS
  =====  ===============  ===============
  0.5    ~0.02            ~0.11
  0.7    ~0.09            ~0.22
  0.9    ~0.24            ~0.38
  =====  ===============  ===============

Path B (the paper's simulation): the case asserts SIV beats 2SLS-TWFE at every
correlation level and that the 2SLS bias matches the paper's Table-1 cells, not
exact SIV cells (fewer reps than the paper's 1,000). Deterministic (seeded).
"""
from __future__ import annotations

import warnings

import numpy as np

M = 80
_THETA = -0.16
_RS = (0.5, 0.7, 0.9)


def _tsls_twfe(Y, R, Z, T0):
    """Two-way-FE 2SLS slope -- the biased baseline the paper benchmarks against."""
    T = Y.shape[1]

    def demean(X):
        X = X - X.mean(axis=1, keepdims=True)
        return X - X.mean(axis=0, keepdims=True)

    Yd, Rd, Zd = demean(Y), demean(R), demean(Z)
    mask = np.arange(T) >= T0
    y, r, z = Yd[:, mask].flatten(), Rd[:, mask].flatten(), Zd[:, mask].flatten()
    z_c = np.column_stack([np.ones_like(z), z])
    b_fs, *_ = np.linalg.lstsq(z_c, r, rcond=None)
    rhat = z_c @ b_fs
    r_c = np.column_stack([np.ones_like(y), rhat])
    b_ss, *_ = np.linalg.lstsq(r_c, y, rcond=None)
    return float(b_ss[1])


def _bias_pair(r: float):
    from mlsynth import SIV
    from mlsynth.utils.siv_helpers.simulation import simulate_siv_sample

    siv, tsls = np.empty(M), np.empty(M)
    for s in range(M):
        sample = simulate_siv_sample(r=r, rng=np.random.default_rng(s))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = SIV({"df": sample.df, "outcome": "y", "treat": "r",
                       "instrument": "z", "unitid": "unit", "time": "time",
                       "T0": sample.T0, "mode": "siv",
                       "display_graphs": False}).fit()
        siv[s] = float(est.theta_hat)
        tsls[s] = _tsls_twfe(sample.Y, sample.R, sample.Z, sample.T0)
    return abs(float(siv.mean()) - _THETA), abs(float(tsls.mean()) - _THETA)


def run() -> dict:
    out = {}
    siv_wins = 0
    for r in _RS:
        sb, tb = _bias_pair(r)
        out[f"siv_bias_r{int(r*10)}"] = sb
        out[f"tsls_bias_r{int(r*10)}"] = tb
        siv_wins += int(sb < tb)
    out["siv_beats_tsls_all"] = float(siv_wins == len(_RS))
    return out


# Deterministic (seeded). Tolerances absorb the Monte Carlo noise at M=80 (the
# paper uses 1,000). Reproduces Gulek & Vives Table 1: SIV's debiasing yields
# substantially lower bias than 2SLS-TWFE at every correlation level; the 2SLS
# baseline matches the paper's cells (0.111 / 0.218 / 0.360).
EXPECTED = {
    "siv_bias_r5": (0.02, 0.08),
    "tsls_bias_r5": (0.111, 0.05),
    "siv_bias_r7": (0.09, 0.10),
    "tsls_bias_r7": (0.218, 0.06),
    "siv_bias_r9": (0.24, 0.14),
    "tsls_bias_r9": (0.360, 0.08),
    "siv_beats_tsls_all": (1.0, 0.0),
}
