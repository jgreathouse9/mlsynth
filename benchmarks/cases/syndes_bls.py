"""Path-B benchmark: SYNDES vs the paper's Monte Carlo (Doudchenko et al. 2021).

Reproduces the simulation study (Section 5, Table 1) of

    Doudchenko, Khosravi, Pouget-Abadie, Lahaie, Lubin, Mirrokni, Spiess &
    Imbens (2021). "Synthetic Design: An Optimization Approach to Experimental
    Design with Synthetic Controls." (arXiv:2112.00278).

The paper has no public reference *code* (only the data link in footnote 4), so
this is a Path-B replication of the paper's own Monte Carlo rather than a
code cross-validation: we check that mlsynth's ``SYNDES`` design-and-analysis
modes attain the paper's reported root-mean-square errors and beat the
randomized baseline.

DGP (Section 5)
---------------
* Data: ``basedata/urate_cps.csv`` -- US BLS state unemployment rates, 40 months
  x 50 states, the exact file the paper uses (footnote 4:
  ``synth-inference/synthdid .../bdm/data/urate_cps.csv``).
* Each simulation samples a 10x10 panel (10 random states, 10 consecutive
  months from a random start), uses 7 pre- and 3 post-periods, treats ``K`` units
  chosen by the design on the pre-periods, and adds the homogeneous additive
  effect ``0.05`` to the treated units' post-periods.
* RMSE is over the post periods, averaged across simulations, reported x1000.

Table 1 (homogeneous, ATET RMSE x1000): per-unit 8.5 / 8.3, two-way 8.4 / 8.4,
one-way 8.5 / 8.5 (K = 3 / 7); randomized diff-in-means 12.1 / 11.5. The design
modes beat the randomized baseline -- the paper's headline. Compute is bounded by
sampling a 10x10 panel per simulation (the paper's own subsetting); ``N_SIMS``
can be raised toward the paper's 500 to tighten the Monte-Carlo error.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "urate_cps.csv"

N_SIMS = 100
SEED = 0
EFFECT = 0.05
_MODES = {"per_unit": "per_unit", "two_way": "global_2way", "one_way": "global_equal_weights"}


def _rmse_x1000(errors) -> float:
    e = np.asarray(errors, dtype=float)
    return float(1000.0 * np.sqrt(np.mean(e ** 2)))


def _run_cell(Y, K, n_sims, seed):
    """One Table-1 cell: ATET RMSE for each design mode + randomized diff-in-means."""
    from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design

    rng = np.random.default_rng(seed)
    T, N = Y.shape
    errs = {m: [] for m in _MODES}
    errs["dim_random"] = []
    for _ in range(n_sims):
        cols = rng.choice(N, size=10, replace=False)
        start = int(rng.integers(0, T - 10 + 1))
        panel = Y[start:start + 10][:, cols]            # 10 periods x 10 units
        pre, post = panel[:7], panel[7:]

        # design-and-analysis modes: select K treated on the pre-period, then
        # apply the design's contrast weights to the effect-laden post-period.
        for name, hmode in _MODES.items():
            design = solve_synthetic_design(pre, K=K, mode=hmode)
            sel = design.selected_unit_indices
            cw = design.contrast_weights
            if cw is None:
                continue
            post_inj = post.copy()
            post_inj[:, sel] += EFFECT
            errs[name].extend((post_inj @ cw - EFFECT).tolist())

        # randomized baseline (v): random treatment, difference in means
        treated = rng.choice(10, size=K, replace=False)
        control = np.setdiff1d(np.arange(10), treated)
        post_inj = post.copy()
        post_inj[:, treated] += EFFECT
        tau = post_inj[:, treated].mean(axis=1) - post_inj[:, control].mean(axis=1)
        errs["dim_random"].extend((tau - EFFECT).tolist())

    return {k: _rmse_x1000(v) for k, v in errs.items()}


def run() -> dict:
    Y = np.loadtxt(_DATA, delimiter=",")                # (40 months, 50 states)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for K in (3, 7):
            cell = _run_cell(Y, K, N_SIMS, SEED + K)
            for mode in _MODES:
                out[f"syndes_{mode}_atet_rmse_k{K}"] = cell[mode]
            out[f"dim_random_atet_rmse_k{K}"] = cell["dim_random"]
            design_min = min(cell[m] for m in _MODES)
            out[f"design_beats_dim_k{K}"] = float(design_min < cell["dim_random"])
    return out


# Targets are Table 1's homogeneous ATET RMSE (x1000). Tolerances bracket the
# Monte-Carlo error at N_SIMS=100 (vs the paper's 500) plus solver differences.
EXPECTED = {
    "syndes_per_unit_atet_rmse_k3": (8.5, 2.0),
    "syndes_two_way_atet_rmse_k3": (8.4, 2.0),
    "syndes_one_way_atet_rmse_k3": (8.5, 2.0),
    "dim_random_atet_rmse_k3": (12.1, 2.5),
    "design_beats_dim_k3": (1.0, 0.0),
    "syndes_per_unit_atet_rmse_k7": (8.3, 2.0),
    "syndes_two_way_atet_rmse_k7": (8.4, 2.0),
    "syndes_one_way_atet_rmse_k7": (8.5, 2.0),
    "dim_random_atet_rmse_k7": (11.5, 2.5),
    "design_beats_dim_k7": (1.0, 0.0),
}
