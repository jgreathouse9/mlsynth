"""Gate 1b: how much of the variance is the untrimmed propensity tail?

Algorithm 1 puts (1 - pi-hat) in a denominator and says nothing about bounding
pi-hat. Nadaraya-Watson on a handful of effective neighbours returns pi-hat = 1
whenever a unit's kernel neighbourhood happens to be all-treated, and then
pi/(1-pi) explodes. This sweeps the clip and reports what it costs.

For reference, the TRUE propensity in design 8.1 is Lambda(alpha/2 + Y_{T0-1}/2)
with an index of standard deviation ~0.5, so true pi almost never leaves
[0.12, 0.88]: a pi-hat near 1 is estimation error, not a real overlap failure.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from collections import defaultdict

import numpy as np

from dgp import additive_fe
from distance import pseudo_distance
from estimator import (KERNELS, Z95, _fold_plan, _select_bandwidths, _weights,
                       epanechnikov)

TRIMS = (0.01, 0.03, 0.05, 0.10)
VARIANTS = [("DR2*", 2, True), ("DR2", 2, False), ("DR3*", 3, True), ("DR3", 3, False)]


def nuisances(panel, dist_in, folds, rep, oracle, kappa=0.2):
    Y, D, T0 = panel.Y, panel.D, panel.T0
    n = Y.shape[0]
    dist = np.abs(panel.alpha[:, None] - panel.alpha[None, :]) if oracle else dist_in.copy()
    X = Y[:, T0 - 1]
    xdiff = np.abs(X[:, None] - X[None, :])
    off = ~np.eye(n, dtype=bool)
    dist = dist / (dist[off].std() or 1.0)
    xdiff = xdiff / (xdiff[off].std() or 1.0)
    grid = np.geomspace(0.02, 3.0, 14)
    rng = np.random.default_rng([7, rep, folds, int(oracle)])
    fold, sm, sp = _fold_plan(n, folds, rng)
    h_pi, h_m, _ = _select_bandwidths(dist, xdiff, D, fold, grid, grid,
                                      epanechnikov, kappa, 0.8, n)
    in_m = np.asarray(sm)[fold][:, None] == fold[None, :]
    in_pi = np.asarray(sp)[fold][:, None] == fold[None, :]
    y = Y[:, T0]
    Wm = _weights(dist, xdiff, h_m, epanechnikov) * in_m * (1 - D)[None, :]
    dm = Wm.sum(1)
    m_hat = np.divide(Wm @ y, dm, out=np.zeros(n), where=dm > 0)
    empty = dm <= 0
    m_hat[empty] = y[D == 0].mean()
    Wp = _weights(dist, xdiff, h_pi, epanechnikov) * in_pi
    dp = Wp.sum(1)
    pi_raw = np.divide(Wp @ D.astype(float), dp, out=np.full(n, D.mean()), where=dp > 0)
    return y, D, m_hat, pi_raw, empty


def one_rep(rep):
    rng = np.random.default_rng([20260927, rep])
    panel = additive_fe(1000, 20, n_post=1, rng=rng)
    dist = pseudo_distance(panel.Y[:, :20], "zhang_range")
    row = {}
    for tag, folds, oracle in VARIANTS:
        y, D, m_hat, pi_raw, empty = nuisances(panel, dist, folds, rep, oracle)
        n1 = int(D.sum())
        resid = y - m_hat
        for tr in TRIMS:
            pi = np.clip(pi_raw, tr, 1 - tr)
            psi = np.where(D == 1, resid, -pi / (1 - pi) * resid)
            att = psi.sum() / n1
            se = np.sqrt(((psi - D * att) ** 2).sum()) / n1
            row[(tag, tr)] = (att, se, abs(att - 0.5) <= Z95 * se)
        row[(tag, "info")] = (pi_raw.max(), float((pi_raw > 0.9).mean()),
                              float(empty[D == 1].mean()), float(empty[D == 0].mean()))
    return row


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    with mp.Pool(4) as pool:
        rows = pool.map(one_rep, range(reps), chunksize=4)

    paper = {"DR2*": (0.54, 2.07, 94.0), "DR2": (0.84, 2.15, 92.1),
             "DR3*": (0.21, 2.41, 93.7), "DR3": (0.54, 2.44, 93.7)}
    acc = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            acc[k].append(v)

    print(f"N=1000 T0=20 reps={reps} design=additive_fe scale=std kappa=0.2")
    print(f"{'variant':7} {'trim':>5} {'bias':>7} {'SD':>7} {'cov':>6}   "
          f"paper: {'bias':>5} {'SD':>5} {'cov':>5}")
    for tag, _, _ in VARIANTS:
        pb, ps, pc = paper[tag]
        for tr in TRIMS:
            a = np.array([x[0] for x in acc[(tag, tr)]])
            c = np.mean([x[2] for x in acc[(tag, tr)]]) * 100
            print(f"{tag:7} {tr:5.2f} {(a.mean()-0.5)*100:+7.2f} {a.std(ddof=1)*100:7.2f} "
                  f"{c:6.1f}          {pb:5.2f} {ps:5.2f} {pc:5.1f}")
        info = np.array(acc[(tag, 'info')])
        print(f"        -> max pi-hat over reps {info[:,0].max():.4f}; "
              f"reps with any pi-hat>0.9: {np.mean(info[:,0]>0.9)*100:.1f}%; "
              f"empty m-hat: treated {info[:,2].mean()*100:.2f}% control {info[:,3].mean()*100:.2f}%")
    print()
    print("MC se on bias (units of 0.01):",
          ", ".join(f"{t} {np.array([x[0] for x in acc[(t,0.05)]]).std(ddof=1)*100/np.sqrt(reps):.2f}"
                    for t, _, _ in VARIANTS))


if __name__ == "__main__":
    main()
