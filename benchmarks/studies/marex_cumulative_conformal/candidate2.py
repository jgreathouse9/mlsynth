"""Candidate 2: the signed-sum inversion, by scheme, law, path and size.

Coverage of the total ``C`` is the event ``p(C) > alpha`` (the accepted set,
holes and all); width is the hull of the accepted set, in units of the true
standard deviation of the post sum, so a Gaussian oracle sits at 3.92 at 95%.

Every path has total ``H`` and deviations about its mean of standard deviation
``s``: ``tau_t = 1 + s d_t`` with ``d`` a fixed shape scaled to unit spread.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.candidate2 [draws] [n] [H] [law]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.inversion import (
    band, blocks, pvalue, reference)

ALPHAS = (0.05, 0.2)
SCALES = (0.5, 2.0, 8.0)
LAWS = ("gauss", "t2", "skew", "bimodal", "ar5", "ar8", "spike_exch")
SIZES = ((12, 4), (26, 13), (47, 13), (104, 13))


def _ar(phi, N, rng):
    e = rng.standard_normal(N + 300)
    x = np.empty_like(e)
    x[0] = e[0] / np.sqrt(1 - phi ** 2)
    for t in range(1, e.size):
        x[t] = phi * x[t - 1] + e[t]
    return x[300:]


def errors(law, N, rng):
    if law == "gauss":
        return rng.standard_normal(N)
    if law == "t2":
        return rng.standard_t(2, N)
    if law == "skew":
        return rng.exponential(1.0, N) - 1.0
    if law == "bimodal":
        return rng.choice([-2.0, 2.0], N) + 0.3 * rng.standard_normal(N)
    if law == "ar5":
        return _ar(0.5, N, rng)
    if law == "ar8":
        return _ar(0.8, N, rng)
    if law == "spike_exch":
        e = rng.standard_normal(N)
        e[rng.integers(N)] += 200.0
        return e
    raise ValueError(law)


def shapes(H):
    t = np.arange(H, dtype=float)
    raw = {"ramp": t,
           "decay": np.exp(-t / 3.0),
           "one_week": (t == 0).astype(float),
           "alternate": np.cos(np.pi * t)}
    out = {}
    for k, v in raw.items():
        d = v - v.mean()
        out[k] = d / d.std() if d.std() > 0 else d
    return out


def sum_sd(law, n, H, rng, reps=20000):
    """Standard deviation of the post sum, by simulation (t2 by its IQR / 1.349)."""
    s = np.array([errors(law, n + H, rng)[n:].sum() for _ in range(reps)])
    if law == "t2":
        q1, q3 = np.quantile(s, [0.25, 0.75])
        return (q3 - q1) / 1.349
    return s.std()


def run(n, H, law, draws, seed):
    rng = np.random.default_rng(seed)
    sd = sum_sd(law, n, H, np.random.default_rng(seed + 1))
    S = shapes(H)
    cover = {}
    width = {}
    unbounded = {}
    for _ in range(draws):
        u = errors(law, n + H, rng)
        blank, post = u[:n], u[n:]
        for scheme in ("cyclic", "subsets"):
            idx = blocks(n, H, scheme, max_subsets=999, rng=rng)
            # constant path: coverage and width
            ref = reference(blank, post + 1.0, idx)
            p = pvalue(ref, float(H))[0]
            for a in ALPHAS:
                key = (scheme, "constant", 0.0, a)
                cover[key] = cover.get(key, 0) + (p > a)
                b = band(ref, a)
                w = b.upper - b.lower
                if np.isfinite(w):
                    width.setdefault((scheme, a), []).append(w / sd)
                else:
                    unbounded[(scheme, a)] = unbounded.get((scheme, a), 0) + 1
            for name, d in S.items():
                for s in SCALES:
                    ref = reference(blank, post + 1.0 + s * d, idx)
                    p = pvalue(ref, float(H))[0]
                    for a in ALPHAS:
                        key = (scheme, name, s, a)
                        cover[key] = cover.get(key, 0) + (p > a)
    for a in ALPHAS:
        for scheme in ("cyclic", "subsets"):
            w = width.get((scheme, a), [])
            head = (f"n={n} H={H} {law} alpha={a} {scheme}: "
                    f"width/sd {np.median(w) if w else float('nan'):.2f} "
                    f"unbounded {unbounded.get((scheme, a), 0) / draws:.2f} | ")
            cells = [f"{k[1]}{'' if k[1] == 'constant' else '@' + format(k[2], 'g')}"
                     f"={v / draws:.3f}"
                     for k, v in cover.items() if k[0] == scheme and k[3] == a]
            print(head + " ".join(cells), flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    sizes = ((int(sys.argv[2]), int(sys.argv[3])),) if len(sys.argv) > 3 else SIZES
    laws = (sys.argv[4],) if len(sys.argv) > 4 else LAWS
    for n, H in sizes:
        for law in laws:
            run(n, H, law, draws, seed=1000 * n + 10 * H + LAWS.index(law))
