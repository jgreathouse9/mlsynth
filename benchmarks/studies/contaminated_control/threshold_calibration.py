"""Does a pre-period statistic predict the post-window reconstruction error?

Calibration is a property of reconstructing one unit from the others, so it
needs no design solve: every unit stands in as ``k*`` in turn. Two estimators
are compared against the realised RMS post-window error.

    python threshold_calibration.py 25 results/threshold_calibration.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.solvers.simplex import simplex_lstsq

from dgps import DGPS
from repair import block_rms

MAX_BLANK_BLOCKS = 2


def rows_for(name: str, seeds) -> list[dict]:
    out = []
    for seed in seeds:
        YN, _, T0 = DGPS[name](seed)
        T, J = YN.shape
        Tp = T - T0
        if T0 // Tp < 2:
            continue
        nbk = MAX_BLANK_BLOCKS
        while nbk >= 1 and T0 - nbk * Tp < Tp:
            nbk -= 1
        if nbk < 1:
            continue
        fit_end = T0 - nbk * Tp
        for k in range(J):
            donors = np.array([j for j in range(J) if j != k])
            full = simplex_lstsq(YN[:T0, donors], YN[:T0, k])
            in_sample = block_rms(YN[:T0, k] - YN[:T0, donors] @ full, Tp)
            held = simplex_lstsq(YN[:fit_end, donors], YN[:fit_end, k])
            oos = block_rms(YN[fit_end:T0, k] - YN[fit_end:T0, donors] @ held, Tp)
            truth = float(np.mean(YN[T0:, k] - YN[T0:, donors] @ full))
            out.append(dict(dgp=name, seed=seed, k=k, n_blank_blocks=nbk,
                            in_sample=in_sample, oos=oos, truth=truth))
    return out


def main(reps: int, out: str) -> None:
    warnings.filterwarnings("ignore")
    rows: list[dict] = []
    for name in DGPS:
        rows += rows_for(name, range(reps))
        print(f"done {name}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)

    rms = lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))
    summary = []
    for name, g in frame.groupby("dgp"):
        truth = rms(g.truth)
        summary.append(dict(dgp=name, n=len(g),
                            blank_blocks=int(g.n_blank_blocks.iloc[0]),
                            truth=truth,
                            in_sample=rms(g.in_sample),
                            in_ratio=rms(g.in_sample) / truth,
                            oos=rms(g.oos), oos_ratio=rms(g.oos) / truth))
    print(pd.DataFrame(summary).sort_values("oos_ratio").round(3).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
