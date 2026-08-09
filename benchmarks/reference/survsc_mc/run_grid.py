"""Run the Section 4 grid and compare against the paper's Table 1.

Common random numbers: the latent design for replication ``r`` is seeded by
``r`` alone, so the same 20 designs are reused at every ``K`` and the
K-comparison isolates the patient-sampling effect Figure 4 is about.
"""
from __future__ import annotations

import sys

import numpy as np

from survsc_oracle import replication

KS = (100, 300, 700)
DGPS = ("cox", "aalen")
N_REPS = 20
SEED0 = 20_251_114

# Table 1 of arXiv:2511.14133v1: (mean, sd) of the sup-norm error.
PAPER = {
    ("cox", 100): {"ssc": (0.1177, 0.0835), "oracle": (0.0247, 0.0121)},
    ("cox", 300): {"ssc": (0.0652, 0.0497), "oracle": (0.0204, 0.0096)},
    ("cox", 700): {"ssc": (0.0542, 0.0413), "oracle": (0.0194, 0.0104)},
    ("aalen", 100): {"ssc": (0.0621, 0.0307), "oracle": (0.0185, 0.0052)},
    ("aalen", 300): {"ssc": (0.0507, 0.0388), "oracle": (0.0136, 0.0067)},
    ("aalen", 700): {"ssc": (0.0245, 0.0102), "oracle": (0.0099, 0.0044)},
}
DGP_OFFSET = {"cox": 0, "aalen": 500_000}


def cell(dgp, K, reps=N_REPS, oracle=True, **kw):
    rows = [
        replication(
            dgp, K,
            seed=SEED0 + DGP_OFFSET[dgp] + 1000 * K + r,
            design_seed=SEED0 + DGP_OFFSET[dgp] + r,     # common across K
            run_oracle=oracle, **kw
        )
        for r in range(reps)
    ]
    out = {}
    for key in ("ssc", "oracle", "rank", "monotone", "in_unit_range", "horizon"):
        if key in rows[0]:
            vals = np.array([r[key] for r in rows], dtype=float)
            out[key] = (float(vals.mean()), float(vals.std(ddof=1)),
                        float(np.median(vals)))
    return out


def main(argv: list[str]) -> None:
    rule = argv[0] if argv else "usvt"
    horizon = argv[1] if len(argv) > 1 else "pooled"
    kw = {"rank_rule": rule} if not rule.isdigit() else {"fixed_rank": int(rule)}
    kw["horizon_mode"] = horizon
    if len(argv) > 2:
        kw["weight_mode"] = argv[2]

    print(f"rank rule: {rule}   horizon: {horizon}   "
          f"weights: {kw.get('weight_mode', 'pcr')}   reps: {N_REPS}   "
          f"common random numbers across K\n")
    print(f"{'DGP':6} {'K':>4} | {'SSC (mine)':>19} {'SSC (paper)':>19} | "
          f"{'oracle (mine)':>19} {'oracle (paper)':>19} | {'mono':>5} {'in01':>5}")
    summary = {}
    for dgp in DGPS:
        col = []
        for K in KS:
            c = cell(dgp, K, **kw)
            p = PAPER[(dgp, K)]
            col.append(c["ssc"][0])
            print(f"{dgp:6} {K:>4} | {c['ssc'][0]:>8.4f} +/- {c['ssc'][1]:<8.4f} "
                  f"{p['ssc'][0]:>8.4f} +/- {p['ssc'][1]:<8.4f} | "
                  f"{c['oracle'][0]:>8.4f} +/- {c['oracle'][1]:<8.4f} "
                  f"{p['oracle'][0]:>8.4f} +/- {p['oracle'][1]:<8.4f} | "
                  f"{c['monotone'][0]:>5.2f} {c['in_unit_range'][0]:>5.2f}")
        summary[dgp] = col
    print()
    for dgp in DGPS:
        col = summary[dgp]
        falls = all(b < a for a, b in zip(col, col[1:]))
        print(f"{dgp:6} error falls monotonically in K: {falls}   "
              f"({' -> '.join(f'{v:.4f}' for v in col)})")


if __name__ == "__main__":
    main(sys.argv[1:])
