"""Gate 1c: the consolidated Table 1 comparison, with every open porting choice named.

Three configurations, all at trim = 0.10 (justified: the TRUE propensity in design
8.1 never exceeds 0.83, so clipping pi-hat at 0.90 cannot bias anything):

    std/nn    standardize both kernel arguments; nearest-control fallback
    raw/nn    the kernel formula exactly as printed (one h, two scales)
    std/mean  pooled-control-mean fallback, to price the fallback itself

Also reports interval width, to test whether near-nominal coverage is real or an
artifact of the standard error exploding in the same reps the point estimate does.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from collections import defaultdict

import numpy as np

from dgp import additive_fe
from did import did
from distance import pseudo_distance
from estimator import Z95, dr_att

PAPER = {"DID": (2.16, 1.94, 79.8), "DR2*": (0.54, 2.07, 94.0), "DR2": (0.84, 2.15, 92.1),
         "DR3*": (0.21, 2.41, 93.7), "DR3": (0.54, 2.44, 93.7)}
VARIANTS = [("DR2*", 2, True), ("DR2", 2, False), ("DR3*", 3, True), ("DR3", 3, False)]
CONFIGS = [("std/nn", "std", "nn"), ("raw/nn", "raw", "nn"), ("std/mean", "std", "mean")]


def one_rep(rep: int):
    rng = np.random.default_rng([20260927, rep])
    panel = additive_fe(1000, 20, n_post=1, rng=rng)
    dist = pseudo_distance(panel.Y[:, :20], "zhang_range")
    out = {}
    est, se = did(panel.Y, panel.D, panel.T0)
    out[("DID", "-")] = (est, se, 2 * Z95 * se, 0.0)
    for cname, scale, fb in CONFIGS:
        for tag, folds, oracle in VARIANTS:
            a, s, d = dr_att(
                panel, n_folds=folds, oracle=oracle, scale=scale, fallback=fb,
                kappa=0.2, trim=0.10,
                rng=np.random.default_rng([7, rep, folds, int(oracle)]),
                dist_cache=None if oracle else dist)
            out[(tag, cname)] = (a, s, d.extras["ci_width"], d.extras["max_abs_psi"])
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    with mp.Pool(4) as pool:
        rows = pool.map(one_rep, range(reps), chunksize=3)
    acc = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            acc[k].append(v)

    print(f"design=additive_fe  N=1000  T0=20  reps={reps}  kappa=0.2  trim=0.10  truth=0.50")
    print("bias and SD in units of 0.01; coverage in percent; width = mean 95% CI width x100")
    print()
    hdr = f"{'variant':8}{'config':10}{'bias':>7}{'SD':>7}{'cov':>7}{'width':>8}{'wid med':>8}{'wid p95':>8}{'max|psi|':>9}"
    print(hdr); print("-" * len(hdr))
    for tag in ("DID", "DR2*", "DR2", "DR3*", "DR3"):
        pb, ps, pc = PAPER[tag]
        for cname in (["-"] if tag == "DID" else [c[0] for c in CONFIGS]):
            v = acc[(tag, cname)]
            a = np.array([x[0] for x in v]); s = np.array([x[1] for x in v])
            w = np.array([x[2] for x in v]); mp_ = np.array([x[3] for x in v])
            cov = np.mean(np.abs(a - 0.5) <= Z95 * s) * 100
            print(f"{tag:8}{cname:10}{(a.mean()-0.5)*100:+7.2f}{a.std(ddof=1)*100:7.2f}"
                  f"{cov:7.1f}{w.mean()*100:8.2f}{np.median(w)*100:8.2f}"
                  f"{np.percentile(w,95)*100:8.2f}{np.median(mp_):9.2f}")
        print(f"{'':8}{'PAPER':10}{pb:+7.2f}{ps:7.2f}{pc:7.1f}")
        print()
    print("MC se on bias (units of 0.01):",
          ", ".join(f"{t} {np.array([x[0] for x in acc[(t,'std/nn')]]).std(ddof=1)*100/np.sqrt(reps):.2f}"
                    for t, _, _ in VARIANTS))


if __name__ == "__main__":
    main()
