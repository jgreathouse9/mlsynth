"""Benchmark: RMSI vs Agarwal, Choi & Yuan (2026) -- "Robust Matrix Estimation
with Side Information" (arXiv:2603.24833).

RMSI decomposes a matrix into four components -- a (possibly nonlinear) row x
column interaction, a row-characteristic-driven part, a column-driven part, and a
residual low-rank part -- estimated by sieve projection + nuclear-norm
penalization, with MAR/MNAR extensions for panel causal inference. The paper ships
no reference code, so this is a Path-B reproduction of its own simulations plus a
Path-A check on the tobacco application.

Path B -- Section 5.1 (robustness across component weights)
----------------------------------------------------------
The paper writes ``M = a1 M1 + a2 M2 + a3 M3 + a4 M4`` with each ``||M_r||_F =
2 sqrt(NT)`` and ``sum a_r = 1``, and shows the estimator's advantage over a
no-side-information baseline varies with -- but is robust across -- the component
weights. We reproduce that: over several seeds and three weight regimes
(interaction-dominant / balanced / residual-dominant), RMSI's relative recovery
error is **below** the no-side-info nuclear-norm baseline in every case, and near
the noise floor.

Path A -- Section 5.2 (tobacco)
------------------------------
The Prop 99 application: RMSI recovers California's tobacco ATT in the
Abadie-Diamond-Hainmueller range (``basedata/P99data.csv``).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_N, _T, _NOISE, _J = 60, 50, 0.3, 3
_SEEDS = range(5)
_REGIMES = {
    "interaction": [0.55, 0.15, 0.15, 0.15],
    "balanced": [0.25, 0.25, 0.25, 0.25],
    "residual": [0.15, 0.15, 0.15, 0.55],
}


def _components(rng, N, T):
    """Four normalized components (||M_r||_F = 2 sqrt(NT)), paper Section 5.1."""
    X = rng.uniform(-1, 1, (N, 4))
    Z = rng.uniform(-1, 1, (T, 4))
    gX = np.column_stack([np.sin(X[:, 0]), X[:, 1] ** 2, X[:, 2]])
    qZ = np.column_stack([Z[:, 0], np.cos(Z[:, 1]), Z[:, 2]])
    M1 = gX @ qZ.T                                           # interaction
    M2 = gX @ rng.normal(size=(T, 3)).T                      # row-driven
    M3 = rng.normal(size=(N, 3)) @ qZ.T                      # column-driven
    M4 = rng.normal(size=(N, 3)) @ rng.normal(size=(T, 3)).T  # residual low-rank
    scale = lambda A: A / np.linalg.norm(A) * 2.0 * np.sqrt(N * T)
    return [scale(M1), scale(M2), scale(M3), scale(M4)], X, Z


def run() -> dict:
    from mlsynth.utils.rmsi_helpers import algorithm1
    from mlsynth.utils.rmsi_helpers.replication import replicate_prop99

    side, base, beats = [], [], 0
    n_cells = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in _SEEDS:
            rng = np.random.default_rng(seed)
            comps, X, Z = _components(rng, _N, _T)
            for alpha in _REGIMES.values():
                M = sum(a * Mi for a, Mi in zip(alpha, comps))
                Y = M + rng.normal(scale=_NOISE, size=M.shape)
                Mhat, _ = algorithm1(Y, X, Z, J=_J)
                Mbase, _ = algorithm1(Y, np.zeros((_N, 0)), np.zeros((_T, 0)), J=_J)
                rel = float(np.linalg.norm(Mhat - M) / np.linalg.norm(M))
                rel_base = float(np.linalg.norm(Mbase - M) / np.linalg.norm(M))
                side.append(rel); base.append(rel_base)
                beats += rel < rel_base
                n_cells += 1

        prop99_att = float(replicate_prop99(str(_BASE / "P99data.csv"),
                                            rank=3, verbose=False).att)

    return {
        "rmsi_beats_baseline_frac": beats / n_cells,
        "rmsi_mean_rel_sideinfo": float(np.mean(side)),
        "rmsi_mean_rel_baseline": float(np.mean(base)),
        "rmsi_sideinfo_advantage": float(np.mean(base) - np.mean(side)),
        "prop99_att": prop99_att,
    }


# Deterministic (fixed seeds). Side information helps in EVERY (seed, regime) cell
# (Section 5.1 robustness), the recovery error sits near the noise floor, and the
# Prop 99 ATT lands in the ADH range (Section 5.2).
EXPECTED = {
    "rmsi_beats_baseline_frac": (1.0, 0.0),       # side info wins in every cell
    "rmsi_mean_rel_sideinfo": (0.30, 0.08),       # near noise floor
    "rmsi_sideinfo_advantage": (0.045, 0.05),     # positive advantage over baseline
    "prop99_att": (-21.0, 7.0),                   # ADH range (~ -19 to -20)
}
