"""The end-to-end arm: design, contaminate, repair, score both thresholds.

Each replication is swept over two ``pi`` grids off one design solve, because
no single grid answers both questions.

``thr_oos``
    ``pi = ratio * thr_oos``. A calibrated out-of-sample threshold puts the
    naive/iterative crossover at ratio 1.0, and the DGPs become comparable
    despite their different scales. This grid cannot score the decision rules:
    ``pi`` is then a fixed multiple of ``thr_oos``, so the out-of-sample rule
    fires exactly when ``ratio > 1`` and carries no per-replication
    information, while the in-sample rule keeps its cross-sectional variation.
    Comparing the two on this grid measures the grid, not the rules.

``panel``
    ``pi = ratio * scale``, where ``scale`` is the median across units of the
    standard deviation of their pre-period series. This depends on neither
    threshold, so both rules vary replication to replication and the decision
    comparison is a fair one.

    python run.py 40 results/end_to_end.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.config import MAREXConfig

from dgps import DGPS
from repair import arms, reconstruct

RATIOS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def design(Y: np.ndarray, T0: int, m_eq: int):
    """MAREX's design on the pre-period. Returns treated and control weights."""
    T, J = Y.shape
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), T),
                       "time": np.tile(np.arange(T), J),
                       "y": Y.T.reshape(-1)})
    res = MAREX(MAREXConfig(df=df, outcome="y", unitid="unit", time="time",
                            T0=T0, program_type="MIQP", display_graph=False,
                            inference=False, m_eq=m_eq)).fit()
    w = np.zeros(J)
    v = np.zeros(J)
    for unit, x in res.design_weights.donor_weights.items():
        w[int(unit)] = x
    for unit, x in res.design_weights.summary_stats["control_weights_agg"].items():
        v[int(unit)] = x
    return w, v


def replication(name: str, seed: int) -> list[dict]:
    YN, YI, T0 = DGPS[name](seed)
    T, J = YN.shape
    w, v = design(YN, T0, max(2, J // 6))
    treated = np.flatnonzero(w > 1e-8)
    rec = reconstruct(YN, v, T0)

    post = slice(T0, T)
    scale = float(np.median(YN[:T0].std(axis=0)))
    Yt = YN.copy()
    if YI is not None:
        Yt[post, treated] = YI[post, treated]
    else:
        Yt[post, treated] += scale
    e_post = float(np.mean(Yt[post, rec.kstar]
                           - Yt[post][:, rec.clean] @ rec.weights))

    rows = []
    for grid, unit in (("thr_oos", rec.thr_oos), ("panel", scale)):
        for ratio in RATIOS:
            pi = ratio * unit
            est = arms(Yt, w, v, T0, rec, pi)
            rows.append(dict(dgp=name, seed=seed, grid=grid, ratio=ratio, pi=pi,
                             J=J, vk=rec.vk, scale=scale,
                             n_blank_blocks=rec.n_blank_blocks,
                             thr_in=rec.thr_in, thr_oos=rec.thr_oos,
                             e_post=e_post,
                             rule_in=abs(pi) > rec.thr_in,
                             rule_oos=abs(pi) > rec.thr_oos,
                             truth=abs(pi) > abs(e_post), **est))
    return rows


def main(reps: int, out: str) -> None:
    warnings.filterwarnings("ignore")
    rows: list[dict] = []
    failures: dict[str, int] = {}
    for name in DGPS:
        for seed in range(reps):
            try:
                rows += replication(name, seed)
            except Exception as exc:                       # noqa: BLE001
                failures[name] = failures.get(name, 0) + 1
                if failures[name] <= 2:
                    print(f"{name} seed={seed}: {type(exc).__name__}: {exc}",
                          flush=True)
        print(f"done {name} ({failures.get(name, 0)} failures)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}; failures={failures}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
