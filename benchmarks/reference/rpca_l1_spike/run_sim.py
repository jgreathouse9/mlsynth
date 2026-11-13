"""Placebo simulation: RPCA-SC's denoisers against the rank-k l1 fit.

Synthetic-control-shaped panels: 20 donors and one treated unit, T = 40,
T0 = 30, no treatment effect. Untreated outcomes follow a rank-3 factor model
(a level and two random-walk factors), so the signal matrix has rank exactly 3.
The treated unit's loadings are a convex combination of four donors', so it is
spanned. Donors are observed with Gaussian noise sigma and one of four outlier
patterns; the treated unit gets the same noise and no outliers.

  none    no outliers
  sparse5 each donor entry is an outlier with probability 0.05, size +-6
  sparse15 the same at 0.15
  spells  each donor, with probability 0.3, has one spell of 5 consecutive
          periods shifted by +6 (a unit-specific shock)

Every method denoises the full donor matrix, fits non-negative weights on the
pre-period (RPCA-SC's default), and projects. Clustering is not run: all 20
donors are used, so the comparison is of the denoisers alone.

Score: post-period RMSE of the counterfactual against the treated unit's
untreated signal (the truth, without noise). `oracle` fits the weights on the
true donor signals, which is the floor.

Usage: python run_sim.py REPS OUT.csv [WORKERS]
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

N, T, T0, R = 20, 40, 30, 3
NOISE = [0.0, 0.2, 0.5]
OUTLIERS = ["none", "sparse5", "sparse15", "spells"]


def panel(seed, sigma, pattern):
    rng = np.random.default_rng(seed)
    F = np.c_[np.ones(T), np.cumsum(rng.normal(0, 0.3, (T, R - 1)), axis=0)]
    lam = np.c_[rng.uniform(8, 12, N), rng.uniform(0.5, 1.5, (N, R - 1))]
    mix = np.zeros(N)
    pick = rng.choice(N, 4, replace=False)
    mix[pick] = rng.dirichlet(np.ones(4))
    lam_t = mix @ lam
    S_d = lam @ F.T                      # donors x T, rank 3
    s_t = F @ lam_t
    Y_d = S_d + sigma * rng.standard_normal(S_d.shape)
    if pattern.startswith("sparse"):
        p = 0.05 if pattern == "sparse5" else 0.15
        mask = rng.random(S_d.shape) < p
        Y_d = Y_d + mask * rng.choice([-6.0, 6.0], S_d.shape)
    elif pattern == "spells":
        for i in range(N):
            if rng.random() < 0.3:
                start = rng.integers(0, T - 5)
                Y_d[i, start:start + 5] += 6.0
    y_t = s_t + sigma * rng.standard_normal(T)
    return Y_d, y_t, s_t, S_d


def _fit(L, y_t):
    from mlsynth.utils.clustersc_helpers.rpca.weights import solve_nnls
    w = solve_nnls(L[:, :T0].T, y_t[:T0])
    return L.T @ w


def one(args):
    seed, sigma, pattern = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.simplefilter("ignore")
    from l1rpca import certify, l1_lowrank
    from mlsynth.utils.clustersc_helpers.rpca.hqf import hqf_decompose
    from mlsynth.utils.clustersc_helpers.rpca.pcp import pcp_decompose
    from mlsynth.utils.pcr.core import hsvt
    from mlsynth.utils.pcr.rank import select_rank

    Y_d, y_t, s_t, S_d = panel(seed, sigma, pattern)
    r_usvt = int(select_rank(Y_d, method="usvt"))
    fits = {"oracle": S_d}
    t0 = time.perf_counter()
    fits["pcp_1000"] = pcp_decompose(Y_d).low_rank
    fits["pcp_conv"] = pcp_decompose(Y_d, max_iter=20000).low_rank
    fits["hqf"] = hqf_decompose(Y_d, random_state=seed).low_rank
    fits["hsvt_usvt"] = hsvt(Y_d, max(r_usvt, 1))[0]
    fits["hsvt_true"] = hsvt(Y_d, R)[0]
    l1_true = l1_lowrank(Y_d, R, seed=seed)
    fits["l1_true"] = l1_true.low_rank
    l1_usvt = l1_lowrank(Y_d, max(r_usvt, 1), seed=seed) if r_usvt != R else l1_true
    fits["l1_usvt"] = l1_usvt.low_rank
    rows = []
    for name, L in fits.items():
        cf = _fit(L, y_t)
        rows.append(dict(
            seed=seed, sigma=sigma, pattern=pattern, method=name,
            post_rmse=float(np.sqrt(np.mean((cf[T0:] - s_t[T0:]) ** 2))),
            pre_rmse=float(np.sqrt(np.mean((cf[:T0] - y_t[:T0]) ** 2))),
            L_err=float(np.linalg.norm(L - S_d) / np.linalg.norm(S_d)),
        ))
    cert = certify(Y_d, l1_true.low_rank, R)
    rows.append(dict(seed=seed, sigma=sigma, pattern=pattern, method="l1_true_cert",
                     post_rmse=cert.t_star, pre_rmse=float(cert.critical),
                     L_err=cert.support_share))
    rows.append(dict(seed=seed, sigma=sigma, pattern=pattern, method="usvt_rank",
                     post_rmse=float(r_usvt), pre_rmse=float("nan"),
                     L_err=time.perf_counter() - t0))
    return rows


if __name__ == "__main__":
    reps, out = int(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    jobs = [(1000 * i + s, sig, pat) for i, (sig, pat) in
            enumerate([(sg, pt) for sg in NOISE for pt in OUTLIERS]) for s in range(reps)]
    rows, t0 = [], time.perf_counter()
    if out.exists():                       # resume: keep finished panels
        prev = pd.read_csv(out)
        done = set(zip(prev.seed, prev.sigma, prev.pattern))
        rows = prev.to_dict("records")
        jobs = [j for j in jobs if j not in done]
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs), 1):
            rows += r
            if k % 12 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
            if k % 12 == 0 or k == len(jobs):
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
