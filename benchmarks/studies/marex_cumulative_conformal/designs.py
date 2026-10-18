"""Candidate 2 through MAREX's own designs on a factor-model panel.

The panel follows Assumption 1 of Abadie and Zhao with the dimensions of their
Section 5: ``J = 15`` units, ``R = 7`` observed and ``F = 11`` unobserved
covariates on Uniform(0, 1), period intercepts on Uniform(0, 20), coefficient
vectors on Uniform(0, 10), errors N(0, 1),

    y_jt = delta_t + theta_t' Z_j + lambda_t' mu_j + eps_jt.

How the period effects move over time is what Theorem 2 rests on, so it is the
treatment here:

* ``iid`` -- every period effect drawn independently: the periods are
  exchangeable, Theorem 2's assumption.
* ``ar`` -- ``theta_t`` and ``lambda_t`` an AR(1) with coefficient 0.8 about
  their means, same marginal: stationary and serially dependent.
* ``sorted`` -- ``delta_t`` and each coordinate of ``lambda_t`` sorted small to
  large, the rearrangement Section 5 uses: trending.

Each draw solves one MAREX design (``m_eq = 3``, exact backend) on the fitting
periods, forms the contrast ``g_t = w'Y_t - v'Y_t`` on the blank and post
periods, adds an effect path to the treated units, and asks every arm whether it
covers the total.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.designs [draws] [config] [dgp] [seed0]
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from benchmarks.studies.marex_cumulative_conformal.candidate2 import shapes
from mlsynth.exceptions import MlsynthDataError
from benchmarks.studies.marex_cumulative_conformal.inversion import (
    blocks, pvalue, reference)

J, R_OB, F_UNOB = 15, 7, 11
#: (fitting periods, blank periods, post periods)
CONFIGS = {"az": (20, 5, 5), "mid": (40, 26, 13)}
ALPHAS = (0.05, 0.2)
SCALE = 2.0


def _factor(T, k, hi, dgp, rng):
    x = rng.uniform(0.0, hi, (T, k))
    if dgp == "ar":
        # an AR(1) with the uniform's mean and spread, built from normals
        z = np.empty((T + 200, k))
        z[0] = rng.standard_normal(k)
        for t in range(1, T + 200):
            z[t] = 0.8 * z[t - 1] + np.sqrt(1 - 0.64) * rng.standard_normal(k)
        x = hi / 2.0 + z[200:] * hi / np.sqrt(12.0)
    if dgp == "sorted":
        x = np.sort(x, axis=0)
    return x


def panel(T, dgp, rng):
    Z = rng.uniform(0, 1, (R_OB, J))
    mu = rng.uniform(0, 1, (F_UNOB, J))
    delta = _factor(T, 1, 20.0, dgp, rng)
    theta = _factor(T, R_OB, 10.0, "ar" if dgp == "ar" else "iid", rng)
    lam = _factor(T, F_UNOB, 10.0, dgp, rng)
    return delta + theta @ Z + lam @ mu + rng.standard_normal((T, J))


def design(Y, TE, Tb, H):
    from mlsynth import MAREX
    T0 = TE + Tb
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), Y.shape[0]),
                       "time": np.tile(np.arange(Y.shape[0]), J),
                       "y": Y.T.reshape(-1)})
    res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                 "T0": T0, "blank_periods": Tb, "T_post": H, "m_eq": 3,
                 "backend": "exact", "inference": False,
                 "display_graph": False}).fit()
    g = res.globres
    return (np.asarray(g.treated_weights_agg, float),
            np.asarray(g.control_weights_agg, float))


def shipped_half(blank, H, alpha, seed):
    from mlsynth.utils.conformal import block_error_paths
    m = blank.size // H
    paths = block_error_paths(blank, horizon=H, block=0, n_sim=2000,
                              rng=np.random.default_rng(seed),
                              n_windows=m if m >= 2 else 0)
    return float(np.quantile(np.abs(paths.sum(axis=1)), 1.0 - alpha))


def run(config, dgp, draws, seed0):
    TE, Tb, H = CONFIGS[config]
    T = TE + Tb + H
    paths = {"constant": np.zeros(H), **{k: SCALE * v for k, v in shapes(H).items()}}
    hit = {}
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        Y = panel(T, dgp, rng)
        w, v = design(Y, TE, Tb, H)
        treated = w > 1e-8
        g0 = Y @ w - Y @ v                       # contrast with no effect
        blank = g0[TE:TE + Tb]
        u_post = g0[TE + Tb:]
        for a in ALPHAS:
            try:
                half = shipped_half(blank, H, a, seed0 + d)
            except MlsynthDataError:
                # a horizon longer than half the blank window is refused
                hit[("shipped", "refused", a)] = hit.get(("shipped", "refused", a), 0) + 1
                continue
            hit[("shipped", "any", a)] = hit.get(("shipped", "any", a), 0) + \
                (abs(u_post.sum()) <= half)
        for scheme in ("subsets", "cyclic"):
            idx = blocks(Tb, H, scheme, max_subsets=999, rng=rng)
            for name, dev in paths.items():
                tau = 1.0 + dev                 # every unit's effect in period t
                # the treated aggregate carries tau_t, the control none
                gaps = u_post + tau * w[treated].sum()
                ref = reference(blank, gaps, idx)
                p_sum = pvalue(ref, float(tau.sum()))[0]
                v_null = np.concatenate([blank, gaps - tau.mean()])
                S = np.abs(v_null[idx]).mean(axis=1)
                p_abs = np.mean(S >= S[0] - 1e-9)
                for a in ALPHAS:
                    for stat, p in (("sum", p_sum), ("abs", p_abs)):
                        key = (f"{scheme}:{stat}", name, a)
                        hit[key] = hit.get(key, 0) + (p > a)
    for a in ALPHAS:
        cells = " ".join(f"{k[0]}:{k[1]}={c / draws:.3f}"
                         for k, c in sorted(hit.items()) if k[2] == a)
        print(f"{config} (TE={TE} Tb={Tb} H={H}) {dgp} alpha={a} draws={draws}: {cells}",
              flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    configs = (sys.argv[2],) if len(sys.argv) > 2 else tuple(CONFIGS)
    dgps = (sys.argv[3],) if len(sys.argv) > 3 else ("iid", "ar", "sorted")
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    for c in configs:
        for g in dgps:
            run(c, g, draws, seed0)
