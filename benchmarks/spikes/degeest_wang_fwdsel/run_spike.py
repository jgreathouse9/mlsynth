"""Spike: De Geest and Wang (2025) greedy forward selection, on geo data.

Runs the four experiments behind ``REPORT.md`` on
``basedata/geolift_test_data.csv`` (40 US markets, 105 daily periods, T0 = 91,
a 14-day post window).

    python run_spike.py                 # stages 1-3 (about 40s)
    python run_spike.py --stage all     # adds the K=4 enumeration (about 3min)
    python run_spike.py --stage 4       # pooling vs separate only

Stage 1  greedy vs exhaustive enumeration over every C(40, K) treated set,
         both scored by the same inner FDID: what greediness costs.
Stage 2  greedy vs random search at a matched evaluation budget and the same
         selection criterion: whether the greedy path beats blind sampling.
Stage 3  the greedy path itself -- the objective against the minimum detectable
         effect, and the tolerance the paper leaves unspecified.
Stage 4  Section 4.1: pooling against the separate approach, treated set held
         fixed, over rolling placebo-in-time windows.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from mlsynth.utils.design_compare import simulated_mde
from mlsynth.utils.fdid_helpers.estimation import forward_did_select
from mlsynth.utils.syndes_helpers.setup import prepare_syndes_inputs

from fwdsel import _score_pooled, greedy_forward_design

T0, HORIZON = 91, 14
GRID = np.arange(0.25, 30.25, 0.25)
DATA = Path(__file__).resolve().parents[3] / "basedata" / "geolift_test_data.csv"


def load():
    df = pd.read_csv(DATA)
    inp = prepare_syndes_inputs(df, outcome="Y", unitid="location", time="date",
                                T0=T0)
    Y = np.vstack([inp.Y_pre, inp.Y_post])
    return Y, [str(u) for u in inp.unit_index.labels]


def evaluate(Y, labels, treated):
    """The paper's objective plus the two quantities a designer decides on.

    ``dm_rmse`` is the demeaned pre-period gap. The raw contrast RMSE would
    charge an FDID design for a constant level gap that FDID's intercept
    removes, so it is not comparable across the two estimator families; the
    demeaned gap is what both analyses see, and it is also what the MDE
    simulation uses.
    """
    idx = [labels.index(u) for u in treated]
    s = _score_pooled(Y, T0, idx, labels)
    wt, wc = 1.0 / len(idx), 1.0 / len(s["donors"])
    c = np.zeros(len(labels))
    for u in treated:
        c[labels.index(u)] += wt
    for u in s["donors"]:
        c[labels.index(u)] -= wc
    g = Y[:T0] @ c
    base = float(Y[:T0][:, idx].mean())
    mde = 100 * simulated_mde(g, horizon=HORIZON,
                              effects_abs=GRID / 100 * abs(base),
                              alpha=0.10, power_target=0.80) / abs(base)
    return dict(r2=s["r2"], dm_rmse=float(np.std(g)), mde=mde)


def stage1(Y, labels, kmax):
    print("\n=== 1. greedy vs exhaustive enumeration (same inner FDID) ===")
    print(f'{"K":>2} {"exact R2":>9} {"greedy R2":>10} {"gap":>9} {"shared":>7} '
          f'{"exact MDE":>10} {"greedy MDE":>11} {"enumerated":>11}')
    N = len(labels)
    for K in range(1, kmax + 1):
        best_r2, best = -np.inf, None
        n = 0
        for combo in itertools.combinations(range(N), K):
            r2 = _score_pooled(Y, T0, combo, labels)["r2"]
            n += 1
            if r2 > best_r2:
                best_r2, best = r2, combo
        exact = [labels[i] for i in best]
        g = greedy_forward_design(Y, T0, labels, tol=-np.inf, max_treated=K)
        gset = g.history[K - 1].treated
        shared = len(set(gset) & set(exact))
        me, mg = evaluate(Y, labels, exact), evaluate(Y, labels, gset)
        print(f"{K:>2} {best_r2:>9.5f} {mg['r2']:>10.5f} "
              f"{best_r2 - mg['r2']:>9.1e} {f'{shared}/{K}':>7} "
              f"{me['mde']:>10.2f} {mg['mde']:>11.2f} {n:>11,}")
        if shared < K:
            print(f"     exact : {sorted(exact)}")
            print(f"     greedy: {sorted(gset)}")


def stage2(Y, labels, runs=25):
    print("\n=== 2. greedy vs random search, matched budget, same criterion ===")
    N = len(labels)
    print(f'{"K":>2} {"procedure":>24} {"evals":>6} {"R2":>9} {"dmRMSE":>8} {"MDE%":>6}')
    for K in (3, 4, 5):
        budget = K * N                      # greedy's own evaluation count
        g = greedy_forward_design(Y, T0, labels, tol=-np.inf, max_treated=K)
        m = evaluate(Y, labels, g.history[K - 1].treated)
        print(f"{K:>2} {'greedy (paper)':>24} {budget:>6} {m['r2']:>9.5f} "
              f"{m['dm_rmse']:>8.1f} {m['mde']:>6.2f}")
        out = []
        for seed in range(runs):
            rng = np.random.default_rng(1000 + seed)
            bt, br2 = None, -np.inf
            for _ in range(budget):
                t = [labels[i] for i in rng.choice(N, K, replace=False)]
                r = _score_pooled(Y, T0, [labels.index(u) for u in t], labels)["r2"]
                if r > br2:
                    br2, bt = r, t
            e = evaluate(Y, labels, bt)
            out.append((br2, e["dm_rmse"], e["mde"]))
        a = np.array(out)
        print(f"{K:>2} {'random search (median)':>24} {budget:>6} "
              f"{np.median(a[:,0]):>9.5f} {np.median(a[:,1]):>8.1f} "
              f"{np.median(a[:,2]):>6.2f}")
        print(f"{K:>2} {'':>24} {'':>6} random beats greedy on R2 in "
              f"{100*(a[:,0] > m['r2']).mean():.0f}% of {runs} runs")


def stage3(Y, labels):
    print("\n=== 3. the greedy path: objective vs decision quantity ===")
    g = greedy_forward_design(Y, T0, labels, tol=-np.inf, max_treated=12)
    print(f'{"K":>3} {"R2":>9} {"dmRMSE":>8} {"MDE%":>6}')
    rows = []
    for h in g.history:
        m = evaluate(Y, labels, h.treated)
        rows.append(dict(K=len(h.treated), **m))
        print(f"{len(h.treated):>3} {m['r2']:>9.5f} {m['dm_rmse']:>8.1f} {m['mde']:>6.2f}")
    d = pd.DataFrame(rows)
    print(f"  the paper's rule (argmax R2) selects K={int(d.loc[d.r2.idxmax(),'K'])}; "
          f"the MDE-minimising size on the same path is K={int(d.loc[d.mde.idxmin(),'K'])}")

    print("\n  design size as a function of the unstated tolerance:")
    for tol in (0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2):
        dd = greedy_forward_design(Y, T0, labels, tol=tol)
        print(f"    tol={tol:<7} K*={len(dd.treated):<3} R2={dd.r2:.5f}")

    print("\n  greedy's percentile among 400 random designs, and how well the "
          "objective\n  orders them (Spearman against each reported metric):")
    rng = np.random.default_rng(11)
    N = len(labels)
    for K in (3, 5):
        rr = [evaluate(Y, labels, [labels[i] for i in rng.choice(N, K, replace=False)])
              for _ in range(400)]
        r2 = np.array([x["r2"] for x in rr])
        dm = np.array([x["dm_rmse"] for x in rr])
        md = np.array([x["mde"] for x in rr])
        ok = np.isfinite(md)
        m = evaluate(Y, labels, greedy_forward_design(
            Y, T0, labels, tol=-np.inf, max_treated=K).history[K - 1].treated)
        print(f"    K={K}: R2 {100*(r2 < m['r2']).mean():5.1f}%  "
              f"dmRMSE {100*(dm > m['dm_rmse']).mean():5.1f}%  "
              f"MDE {100*(md[ok] > m['mde']).mean():5.1f}%  |  "
              f"rho(R2,dmRMSE)={pd.Series(r2).corr(pd.Series(dm), method='spearman'):+.3f}  "
              f"rho(R2,MDE)={pd.Series(r2[ok]).corr(pd.Series(md[ok]), method='spearman'):+.3f}")


def stage4(Y, labels):
    print("\n=== 4. Section 4.1: pooling vs separate, treated set held fixed ===")
    N = len(labels)
    T = Y.shape[0]

    def estimate(Yw, t0, treated_idx, pooling):
        donor_idx = [j for j in range(N) if j not in set(treated_idx)]
        names = [labels[j] for j in donor_idx]
        if pooling:
            r = forward_did_select(Yw[:, treated_idx].mean(axis=1),
                                   Yw[:, donor_idx], t0, names)["FDID"]
            return float(r["Effects"]["ATT"])
        return float(np.mean([
            forward_did_select(Yw[:, i], Yw[:, donor_idx], t0, names)["FDID"]["Effects"]["ATT"]
            for i in treated_idx]))

    rows = []
    for t0 in range(60, T - HORIZON + 1, 2):        # rolling pseudo-treatment date
        Yw = Y[: t0 + HORIZON]
        d = greedy_forward_design(Yw, t0, labels, tol=1e-3)   # design on pre only
        tre = [labels.index(u) for u in d.treated]
        for lift, tag in ((0.0, "A/A"), (0.10, "+10%")):
            Yl = Yw.copy()
            if lift:
                Yl[t0:, tre] *= 1 + lift
            b = float(Yl[t0:][:, tre].mean())
            for pooling in (True, False):
                rows.append(dict(t0=t0, K=len(tre), scenario=tag,
                                 arm="pooling" if pooling else "separate",
                                 att_pct=100 * estimate(Yl, t0, tre, pooling) / b,
                                 target=100 * lift))
    res = pd.DataFrame(rows)
    print(f"  {res.t0.nunique()} placebo windows, treated-set sizes "
          f"{sorted(res.K.unique())}")
    for scen, g in res.groupby("scenario", sort=False):
        tgt = g.target.iloc[0]
        print(f"  --- {scen} (truth {tgt:.0f}%) ---")
        for arm, gg in g.groupby("arm"):
            err = gg.att_pct - tgt
            print(f"    {arm:>8}: mean {gg.att_pct.mean():+6.2f}%  bias {err.mean():+6.2f}pp  "
                  f"sd {gg.att_pct.std():5.2f}  RMSE {np.sqrt((err**2).mean()):5.2f}")
        p = g[g.arm == "pooling"].set_index("t0").att_pct
        s = g[g.arm == "separate"].set_index("t0").att_pct
        print(f"    pooling closer to truth in "
              f"{100*(abs(p-tgt) < abs(s-tgt)).mean():.0f}% of windows")

    print("\n  design size by arm at tol=1e-3 (the paper's p.4 claim):")
    for pooling in (True, False):
        d = greedy_forward_design(Y, T0, labels, tol=1e-3, pooling=pooling)
        print(f"    {'pooling' if pooling else 'separate':>8}: K={len(d.treated)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="fast",
                    choices=["fast", "all", "1", "2", "3", "4"])
    args = ap.parse_args()
    Y, labels = load()
    if args.stage in ("fast", "all", "1"):
        stage1(Y, labels, kmax=4 if args.stage == "all" else 3)
    if args.stage in ("fast", "all", "2"):
        stage2(Y, labels)
    if args.stage in ("fast", "all", "3"):
        stage3(Y, labels)
    if args.stage in ("all", "4"):
        stage4(Y, labels)


if __name__ == "__main__":
    main()
