"""Search for an effect path that makes the signed-sum inversion over-reject.

The observed statistic depends on the path only through the total, so a path
reaches the test only through the reference blocks that straddle the post
window, each shifted by a partial sum of the path's deviations. A shift moves a
symmetric unimodal sum away from zero in distribution (Anderson's inequality),
which raises the p-value; whether that holds for the joint rank, and for laws
that are skewed or bimodal, is what this measures.

Small cases, so every subset or shift is enumerated and the rejection rate at
the null path is the exact level of the test. Random paths with zero mean are
scored on common random numbers and the worst is re-scored on fresh draws.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.path_search
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.inversion import blocks


def draws(law, reps, N, rng):
    if law == "gauss":
        return rng.standard_normal((reps, N))
    if law == "skew":
        return rng.exponential(1.0, (reps, N)) - 1.0
    if law == "negskew":
        return 1.0 - rng.exponential(1.0, (reps, N))
    if law == "bimodal":
        return rng.choice([-2.0, 2.0], (reps, N)) + 0.3 * rng.standard_normal((reps, N))
    if law == "spike_exch":
        U = rng.standard_normal((reps, N))
        U[np.arange(reps), rng.integers(N, size=reps)] += 200.0
        return U
    if law == "spike_sym":
        U = rng.standard_normal((reps, N))
        U[np.arange(reps), rng.integers(N, size=reps)] += rng.choice([-200.0, 200.0], reps)
        return U
    if law == "cauchy":
        return rng.standard_cauchy((reps, N))
    raise ValueError(law)


def reject_rate(law, n, H, d, alpha, scheme, reps, seed):
    rng = np.random.default_rng(seed)
    idx = blocks(n, H, scheme, max_subsets=10 ** 6)
    V = draws(law, reps, n + H, rng)
    V[:, n:] += d[None, :]
    S = np.abs(V[:, idx].sum(axis=2))           # row 0 of idx is the post block
    p = np.mean(S >= S[:, :1] - 1e-12, axis=1)
    return float(np.mean(p <= alpha))


def main(laws=("gauss", "skew", "negskew", "bimodal", "spike_exch")):
    for law in [x for x in laws if x != "negskew"]:
        rates = [(dl, reject_rate(law, 1, 2, np.array([dl, -dl]), 0.34, "subsets",
                                  200000, 1))
                 for dl in (0, .25, .5, 1, 1.5, 2, 3, 5)]
        print("N=3 H=2", law, " ".join(f"{a:g}:{r:.4f}" for a, r in rates),
              "(level 1/3)", flush=True)
    for (n, H, scheme, alpha) in ((5, 3, "subsets", 0.1), (9, 3, "cyclic", 0.17),
                                  (12, 4, "cyclic", 0.13)):
        for law in laws:
            base = reject_rate(law, n, H, np.zeros(H), alpha, scheme, 40000, 2)
            rng = np.random.default_rng(3)
            best = (base, None)
            for _ in range(150):
                d = rng.standard_normal(H) * rng.choice([0.3, 1.0, 3.0, 10.0])
                d -= d.mean()
                r = reject_rate(law, n, H, d, alpha, scheme, 20000, 4)
                if r > best[0]:
                    best = (r, d)
            conf = (reject_rate(law, n, H, best[1], alpha, scheme, 200000, 99)
                    if best[1] is not None else base)
            shown = np.round(best[1], 2) if best[1] is not None else "none above the null"
            print(f"n={n} H={H} {scheme} alpha={alpha} {law}: null {base:.4f}, "
                  f"worst path {shown} -> {conf:.4f}", flush=True)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("gauss", "skew", "negskew", "bimodal", "spike_exch"))
