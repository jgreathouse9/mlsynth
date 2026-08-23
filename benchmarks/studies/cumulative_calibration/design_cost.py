"""What a placebo replicate costs, against what the design costs.

Conformal calibration is downstream of the design. The mixed-integer program
chooses which units to treat under the cardinality and disjointness constraints;
once a treated set and its weights are fixed, the contrast is a linear functional
and a replicate that fixes a pseudo-treated set has no integer decisions left in
it. Its remaining weights are a least-squares fit on the probability simplex.

This times both on one panel, and reports the rank arithmetic that decides whether
an in-space placebo can produce a finite band at all: split conformal needs
``ceil((n + 1) (1 - alpha)) <= n``, which is 19 usable donors at the five percent
level.

    MLSYNTH_CAL_PANEL=<panel.csv> python design_cost.py [seed]
"""
from __future__ import annotations

import sys
from pathlib import Path

# run from anywhere: the repo root supplies mlsynth, this directory supplies panel
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
import time
import warnings

import numpy as np

from mlsynth import MAREX
from mlsynth.utils.bilevel.simplex import simplex_lstsq
from panel import ALPHA, H, T0, TB, TE, draw, frame

J = 20


def main(seed=0):
    YN = draw(seed, J=J)
    fit = slice(0, TE)

    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MAREX({"df": frame(YN), "outcome": "y", "unitid": "unit",
                     "time": "time", "T0": T0, "blank_periods": TB, "T_post": H,
                     "m_eq": 3, "inference": True, "alpha": ALPHA}).fit()
    t_design = time.perf_counter() - t

    w = np.asarray(res.globres.treated_weights_agg, dtype=float)
    treated = set(np.flatnonzero(w > 1e-8).tolist())
    m = len(treated)
    print(f"design MIQP        {t_design:7.2f}s   |S| = {m}")

    rng = np.random.default_rng(seed)
    pool = np.array([j for j in range(J) if j not in treated])
    gaps, times = [], []
    for _ in range(pool.size):
        S = pool[rng.choice(pool.size, size=m, replace=False)]
        rest = np.array([j for j in range(J) if j not in set(S.tolist())])
        target = YN[S].mean(axis=0)
        t = time.perf_counter()
        v = simplex_lstsq(YN[rest][:, fit].T, target[fit])
        times.append(time.perf_counter() - t)
        gaps.append(float((target - v @ YN[rest])[T0:T0 + H].sum()))

    times, gaps = np.asarray(times), np.asarray(gaps)
    print(f"placebo QP         {times.mean() * 1e3:7.2f}ms mean, "
          f"{times.max() * 1e3:.2f}ms max, over {times.size} replicates")
    print(f"all {times.size} replicates  {times.sum():7.3f}s  "
          f"= {times.sum() / t_design * 100:.2f}% of one design solve")

    k = int(np.ceil((gaps.size + 1) * (1 - ALPHA)))
    print(f"\nplacebo cumulative gaps: sd {gaps.std(ddof=1):.4f}, "
          f"rank k = {k} of n = {gaps.size}"
          + ("" if k <= gaps.size else "   -> the band is infinite"))
    print(f"a finite band at alpha = {ALPHA} needs "
          f"{int(np.ceil(1 / ALPHA)) - 1} usable donors, so J >= "
          f"{int(np.ceil(1 / ALPHA)) - 1 + m} at |S| = {m}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
