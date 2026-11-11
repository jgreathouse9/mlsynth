"""Size and power of the EFSC no-leakage placebo test (Algorithm 2).

The paper shows placebo histograms (Figure 15) but never a rejection rate
under the null. This script measures it.

Panel (Appendix C.6): Poisson, N=16, T=32, rank 2, m=100, the last 4 units
treated, the last 6 periods post. Fits use 500 epochs, S=2, lr 0.1 and
N(mu, 3^2) priors with the unit-effect prior mean at the baseline (C.3).

The four fits per assignment follow C.7.5 restricted to a common adoption time:

  post-target   all cells except the target post block and, for placebo
                assignments, the actually treated post block (no leakage)
  post-observed the target post block only
  pre-target    every unit's pretreatment cells -- identical for every
                assignment, so it is fit once per panel
  pre-observed  the target units' pretreatment cells only

  Delta = mean KL(post-obs || post-tgt) over the target post block
        - mean KL(pre-obs  || pre-tgt)  over the target pre block

p = (1 + #{placebo Delta >= observed Delta}) / (B + 1), placebo target sets
drawn from the control units without replacement.

Usage: python run_placebo.py SCENARIO REPS B OUT.csv [WORKERS]
  SCENARIO: null | power05 | power10 | confounded
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N, T, R_RANK, M_CELL, K, POST = 16, 32, 2, 100, 4, 6
BASELINE = 1.0

SCENARIOS = {
    "null": dict(tau=0.0, scales=(0.05, 0.05, 0.05), select="last"),
    "power05": dict(tau=0.05, scales=(0.05, 0.05, 0.05), select="last"),
    "power10": dict(tau=0.10, scales=(0.05, 0.05, 0.05), select="last"),
    # Treated units chosen on their first loading, with factor scales large
    # enough for the choice to matter.
    "confounded": dict(tau=0.0, scales=(0.05, 0.05, 0.3), select="loading"),
}


def _setup():
    import torch

    torch.set_num_threads(1)
    import fastvi

    fastvi.install()


def _fit(panel, mask, seed):
    import efsc.fitting
    from efsc import InferenceConfig

    cfg = InferenceConfig(epochs=500, learning_rate=0.1, prior_scale=3.0, elbo_samples=2)
    d = N + T + N * R_RANK + T * R_RANK
    prior_mean = np.zeros(d, dtype=np.float32)
    prior_mean[:N] = BASELINE
    fit = efsc.fitting.fit_factorization(panel, mask, rank=R_RANK, config=cfg,
                                         seed=seed, prior_mean=prior_mean)
    return fit.natural_parameters.numpy().astype(float)


def _delta(panel, target, actual, pre_tgt_eta, seed):
    import methods as M

    pre = np.zeros((N, T), bool)
    pre[:, : T - POST] = True
    post = ~pre
    tgt_rows = np.zeros((N, T), bool)
    tgt_rows[target] = True
    act_rows = np.zeros((N, T), bool)
    act_rows[actual] = True
    target_post = tgt_rows & post
    target_pre = tgt_rows & pre
    train = ~(target_post | (act_rows & post))
    e_post_tgt = _fit(panel, train, seed)
    e_post_obs = _fit(panel, target_post, seed + 1)
    e_pre_obs = _fit(panel, target_pre, seed + 2)
    ecd_post = M.poisson_kl(e_post_obs, e_post_tgt)[target_post].mean()
    ecd_pre = M.poisson_kl(e_pre_obs, pre_tgt_eta)[target_pre].mean()
    return float(ecd_post - ecd_pre)


def one_panel(args):
    scenario, rep, B = args
    _setup()
    import dgp

    spec = SCENARIOS[scenario]
    panel, truth = dgp.poisson_panel(n=N, T=T, r=R_RANK, m_fixed=M_CELL, treated=K,
                                     post=POST, tau=spec["tau"], baseline=BASELINE,
                                     scales=spec["scales"], select=spec["select"],
                                     seed=5000 + rep)
    actual = truth["treated_idx"]
    controls = np.setdiff1d(np.arange(N), actual)
    pre = np.zeros((N, T), bool)
    pre[:, : T - POST] = True
    pre_tgt_eta = _fit(panel, pre, 10 * rep)
    obs = _delta(panel, actual, actual, pre_tgt_eta, 100 * rep)
    rng = np.random.default_rng(rep)
    placebos = []
    for b in range(B):
        target = np.sort(rng.choice(controls, K, replace=False))
        placebos.append(_delta(panel, target, actual, pre_tgt_eta, 100 * rep + 10 * (b + 1)))
    placebos = np.array(placebos)
    p = (1 + np.sum(placebos >= obs)) / (B + 1)
    return dict(scenario=scenario, rep=rep, observed=obs, p=p,
                placebo_mean=placebos.mean(), placebo_sd=placebos.std(),
                placebo_min=placebos.min(), placebo_max=placebos.max())


if __name__ == "__main__":
    scenario, reps, B, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    rows = []
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        for row in pool.imap_unordered(one_panel, [(scenario, r, B) for r in range(reps)]):
            rows.append(row)
            pd.DataFrame(rows).sort_values("rep").to_csv(out, index=False)
            print(f"{scenario} rep={row['rep']} p={row['p']:.3f} "
                  f"({len(rows)}/{reps}, {time.perf_counter() - t0:.0f}s)", flush=True)
