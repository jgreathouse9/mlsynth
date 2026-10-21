"""Gate 0: does the DGP port reproduce Table 1's DID row?

Target (Zhang 2026, Table 1, N=1000, T0=20; bias and SD in units of 0.01,
coverage in percent):  bias 2.16   SD 1.94   coverage 79.8
"""

from __future__ import annotations

import sys

import numpy as np

from dgp import additive_fe
from did import Z95, did


def run(n: int, T0: int, reps: int, seed: int = 20260927) -> dict:
    rng = np.random.default_rng(seed)
    estimates, covered, shares = [], [], []
    for _ in range(reps):
        panel = additive_fe(n, T0, n_post=1, rng=rng)
        est, se = did(panel.Y, panel.D, panel.T0)
        truth = panel.att[0]
        estimates.append(est)
        covered.append(abs(est - truth) <= Z95 * se)
        shares.append(panel.D.mean())
    estimates = np.asarray(estimates)
    truth = 0.5
    return {
        "bias": (estimates.mean() - truth) * 100,
        "sd": estimates.std(ddof=1) * 100,
        "coverage": np.mean(covered) * 100,
        "mc_se_bias": estimates.std(ddof=1) / np.sqrt(reps) * 100,
        "treated_share": float(np.mean(shares)),
    }


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    out = run(n=1000, T0=20, reps=reps)
    print(f"DID, N=1000, T0=20, {reps} reps")
    print(f"  bias      {out['bias']:+7.2f}   (paper 2.16, sign unstated)  +/- {out['mc_se_bias']:.2f} MC")
    print(f"  SD        {out['sd']:7.2f}   (paper 1.94)")
    print(f"  coverage  {out['coverage']:7.1f}   (paper 79.8)")
    print(f"  treated share {out['treated_share']:.3f}")
