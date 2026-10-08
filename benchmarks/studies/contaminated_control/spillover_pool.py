"""Where `iterative` and `iscm` stop being the same estimator.

The study's headline identity holds because the treated markets are kept out of
the contaminated market's donor pool, which makes the cross-weight ``l1`` zero
and the inclusive system triangular. That is the exogenous case: nothing the
treatment did caused the contamination, so nothing is lost by excluding the
treated markets.

Under genuine spillover the exclusion may not be available. Di Stefano and
Mellace's own example is this: Austria carries 42 percent of synthetic West
Germany, and dropping West Germany from Austria's pool gives implausible
spillover estimates. This arm admits the treated markets to ``k*``'s pool and
measures what happens.

With the treated markets at total weight ``l1``, the rebuild of ``k*`` picks up
``l1 * tau`` over the post-period, and the two arms part company:

    iterative error = v_k * e - v_k * l1 * tau
    iscm error      = v_k * (e + delta * l1) / (1 - v_k * l1)

with ``e`` the rebuild error and ``delta`` the design's own fit error. The leak
in ``iterative`` scales with the treatment effect; the inclusive system removes
that term. Two predictions follow, and both are checked here: the gap between
the arms grows linearly in ``l1 * tau``, and at ``tau = 0`` the arms agree even
when ``l1`` is far from zero.

A homogeneous treatment effect is imposed on every DGP in this arm, including
the one that ships its own treated potential outcomes, so that ``l1 * tau`` is
exact and the mechanism stays legible.

    python spillover_pool.py 40 results/spillover_pool.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.solvers.simplex import simplex_lstsq

from dgps import DGPS
from repair import block_rms
from run import design

RATIOS = (0.0, 1.0, 2.0)
TAU_MULTS = (0.0, 1.0, 3.0)


def replication(name: str, seed: int) -> list[dict]:
    YN, _, T0 = DGPS[name](seed)
    T, J = YN.shape
    Tp = T - T0
    w, v = design(YN, T0, max(2, J // 6))
    treated = np.flatnonzero(w > 1e-8)
    kstar = int(np.argmax(v))
    vk = float(v[kstar])
    clean = np.array([j for j in range(J) if v[j] > 0 and j != kstar])
    if clean.size < 2:
        raise ValueError("too few clean control markets")

    scale = float(np.median(YN[:T0].std(axis=0)))
    post = slice(T0, T)

    pools = {
        "clean": clean,
        "inclusive": np.concatenate([clean, treated]),
    }

    rows = []
    for pool_name, pool in pools.items():
        fit = simplex_lstsq(YN[:T0, pool], YN[:T0, kstar])
        # total weight the treated markets carry in k*'s own rebuild
        l1 = float(sum(fit[i] for i, j in enumerate(pool) if j in set(treated.tolist())))
        thr = block_rms(YN[:T0, kstar] - YN[:T0, pool] @ fit, Tp)

        for tau_mult in TAU_MULTS:
            tau = tau_mult * scale
            Yt = YN.copy()
            Yt[post, treated] += tau
            e_post = float(np.mean(Yt[post, kstar] - Yt[post][:, pool] @ fit))

            for ratio in RATIOS:
                pi = ratio * thr
                Yc = Yt.copy()
                Yc[post, kstar] += pi
                rebuilt = Yc[post][:, pool] @ fit

                oracle = float(np.mean(Yt[post] @ w - Yt[post] @ v))
                naive = float(np.mean(Yc[post] @ w - Yc[post] @ v))

                Yi = Yc.copy()
                Yi[post, kstar] = rebuilt
                iterative = float(np.mean(Yi[post] @ w - Yi[post] @ v))

                gamma = float(np.mean(Yc[post, kstar] - rebuilt))
                omega = np.array([[1.0, -vk], [-l1, 1.0]])
                det = float(np.linalg.det(omega))
                iscm = (float(np.linalg.solve(omega, np.array([naive, gamma]))[0])
                        if abs(det) > 1e-10 else float("nan"))

                rows.append(dict(dgp=name, seed=seed, pool=pool_name,
                                 tau_mult=tau_mult, tau=tau, ratio=ratio, pi=pi,
                                 vk=vk, l1=l1, det=det, thr=thr, e_post=e_post,
                                 oracle=oracle, naive=naive,
                                 iterative=iterative, iscm=iscm,
                                 predicted_leak=vk * l1 * tau))
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
