"""Path B benchmark: Li (2024) Forward DiD, Table 5 PMSE grid.

Pure Python (no R). Reproduces a subset of the paper's Web-Appendix-E Monte
Carlo and checks the headline cells against the published values. Uses a
modest M for speed; tolerances absorb the resulting MC noise.
"""
from __future__ import annotations

import numpy as np

from mlsynth import FDID
from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_sample

M = 400          # paper uses 10,000; benchmark trades precision for runtime


def _pmse(dgp, N, T1, T2, M=M, seed=0):
    fdid_sq, did_sq = [], []
    for j in range(M):
        rng = np.random.default_rng(seed + j)
        s = simulate_fdid_sample(dgp=dgp, N=N, T1=T1, T2=T2, rng=rng)
        res = FDID({"df": s.df, "outcome": "y", "treat": "treat",
                    "unitid": "unit", "time": "time",
                    "display_graphs": False, "verbose": False}).fit()
        fdid_sq.append(res.fdid.att ** 2)
        did_sq.append(res.did.att ** 2)
    return float(np.mean(fdid_sq)), float(np.mean(did_sq))


def run() -> dict:
    out = {}
    # the two diagnostic cells: DGP1 (DiD valid) and DGP2 (half contaminated)
    f1, d1 = _pmse(1, 60, 48, 24)
    f2, d2 = _pmse(2, 60, 48, 24)
    out["fdid_dgp1_48_24"] = f1
    out["fdid_dgp2_48_24"] = f2
    out["did_dgp2_48_24"] = d2
    return out


# Li (2024) Table 5 values at (48,24); generous tol for M=400 vs 10,000.
EXPECTED = {
    "fdid_dgp1_48_24": (0.071, 0.030),   # FDID small when DiD valid
    "fdid_dgp2_48_24": (0.082, 0.040),   # FDID collapses despite contamination
    "did_dgp2_48_24": (0.473, 0.150),    # DiD stays large under contamination
}
