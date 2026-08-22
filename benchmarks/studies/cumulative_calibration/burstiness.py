"""Do the pointwise band's violations cluster?

The standard charge against split conformal on dependent data is not that
marginal coverage is wrong but that the misses arrive in bursts: long quiet
stretches punctuated by runs of consecutive failures, which a marginal average
cannot see. This records MAREX's per-period violation indicator over the post
periods of each draw, then asks the pooled indicators two questions -- whether a
violation raises the odds of the next period violating, and whether there are
fewer runs than independence implies (Wald-Wolfowitz).

The cumulative band is scored on the same draws. A total over H periods has no
within-window pattern to cluster, so if clustering is the disease the cumulative
band is the wrong patient.

    MLSYNTH_CAL_PANEL=<panel.csv> python burstiness.py <n_draws> <first_seed> <out.jsonl>
"""
from __future__ import annotations

import sys
from pathlib import Path

# run from anywhere: the repo root supplies mlsynth, this directory supplies panel
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import sys
import warnings

import numpy as np

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.inference import compute_inference
from panel import ALPHA, DELTA, H, T0, TB, draw, frame


def one(seed, J=20, n_sim=2000):
    YN = draw(seed, J=J)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MAREX({"df": frame(YN), "outcome": "y", "unitid": "unit",
                     "time": "time", "T0": T0, "blank_periods": TB, "T_post": H,
                     "m_eq": 3, "inference": True, "alpha": ALPHA}).fit()
    w = np.asarray(res.globres.treated_weights_agg, dtype=float)
    v = np.asarray(res.globres.control_weights_agg, dtype=float)
    Y = YN.copy()
    Y[w > 1e-8, T0:] += DELTA
    inf = compute_inference(w @ Y, v @ Y, T0, T0 - TB, TB, alpha=ALPHA,
                            random_state=seed, cumulative_band=True,
                            cumulative_block=0, cumulative_n_sim=n_sim,
                            cumulative_seed=seed)
    err = np.asarray(inf.treated_effects, dtype=float) - DELTA
    q = float(np.quantile(np.abs(np.asarray(inf.placebo_effects, dtype=float)),
                          1 - ALPHA))
    return {"seed": seed,
            "miss": (np.abs(err) > q).astype(int).tolist(),
            "cum_miss": int(abs(float(err.sum())) > inf.cumulative_band.half_width)}


def summarise(rows):
    M = np.array([r["miss"] for r in rows])
    n = len(rows)
    print(f"draws {n}, H {M.shape[1]}")
    print(f"pointwise marginal coverage  {1 - M.mean():.4f}")
    print(f"cumulative coverage          "
          f"{1 - np.mean([r['cum_miss'] for r in rows]):.4f}")
    a, b = M[:, :-1].ravel(), M[:, 1:].ravel()
    print(f"\nP(miss)               {a.mean():.4f}")
    print(f"P(miss | prev miss)   {b[a == 1].mean():.4f}   (n={int(a.sum())})")
    print(f"P(miss | prev ok)     {b[a == 0].mean():.4f}")
    print(f"lag-1 corr of misses  {np.corrcoef(a, b)[0, 1]:+.4f}")

    runs = exp = var = 0.0
    used = 0
    for m in M:
        n1, n0 = int(m.sum()), int((1 - m).sum())
        if n1 == 0 or n0 == 0:
            continue
        N = n1 + n0
        runs += 1 + int((m[1:] != m[:-1]).sum())
        exp += 2 * n1 * n0 / N + 1
        var += 2 * n1 * n0 * (2 * n1 * n0 - N) / (N * N * (N - 1))
        used += 1
    if used:
        print(f"\nruns  observed {runs:.0f}  expected {exp:.1f}  "
              f"z = {(runs - exp) / np.sqrt(var):+.2f}   ({used} usable draws)")
        print("  z < 0 means fewer runs than independence implies, i.e. clustering")

    longest = [max((len(s) for s in "".join(map(str, m)).split("0")), default=0)
               for m in M]
    longest = np.array(longest)
    print(f"\nlongest miss-run per draw: mean {longest.mean():.2f}, "
          f"max {longest.max()}, share with a run >= 2: {(longest >= 2).mean():.3f}")


if __name__ == "__main__":
    n, start, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with open(path, "a") as fh:
        for s in range(start, start + n):
            fh.write(json.dumps(one(s)) + "\n")
            fh.flush()
            print(f"seed {s}", flush=True)
