"""Block permutations and recency weights for the signed-sum inversion.

The all-subsets reference set fails under serial dependence because the post
block is ``H`` consecutive periods and a reference subset is ``H`` scattered
ones: under positive autocorrelation the consecutive sum has the larger
variance, so the observed statistic sits too far out and the band is too
narrow. Cyclic shifts keep the reference blocks consecutive but give only ``N``
of them. Permuting contiguous blocks of length ``L`` sits between the two:
dependence shorter than ``L`` stays inside a block, and there are
``C(N / L, H / L)`` references.

Recency weights (Barber, Candes, Ramdas and Tibshirani 2023) are the remedy
their Theorem 2 offers for drift: a reference block far from the post window
gets less weight, so its distance in distribution costs less coverage.

The horizon is 12, so ``L`` can be 1, 2, 3, 4, 6 or 12. ``drift`` adds
``0.01 t`` to iid normal errors, a contrast bias that grows by a hundredth of
the noise each period.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.blocks_weights [draws] [n] [law]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.candidate2 import _ar, shapes
from benchmarks.studies.marex_cumulative_conformal.inversion import (
    band, blocks, pvalue, recency_weights, reference)

H = 12
ALPHAS = (0.05, 0.2)
LAWS = ("gauss", "ar5", "ar8", "drift")
ARMS = (("subsets", "blocks", 1, None), ("L3", "blocks", 3, None),
        ("L4", "blocks", 4, None), ("L6", "blocks", 6, None),
        ("windows", "blocks", 12, None), ("cyclic", "cyclic", 1, None),
        ("cyclic_w", "cyclic", 1, 0.97), ("L4_w", "blocks", 4, 0.97))


def errors(law, N, rng):
    if law == "gauss":
        return rng.standard_normal(N)
    if law == "ar5":
        return _ar(0.5, N, rng)
    if law == "ar8":
        return _ar(0.8, N, rng)
    if law == "drift":
        return rng.standard_normal(N) + 0.01 * np.arange(N)
    raise ValueError(law)


def noise_sd(law, n, rng, reps=20000):
    base = "gauss" if law == "drift" else law
    return np.std([errors(base, n + H, rng)[n:].sum() for _ in range(reps)])


def run(n, law, draws, seed):
    rng = np.random.default_rng(seed)
    sd = noise_sd(law, n, np.random.default_rng(seed + 1))
    ramp = 2.0 * shapes(H)["ramp"]
    cover, width, unb = {}, {}, {}
    for _ in range(draws):
        u = errors(law, n + H, rng)
        blank, post = u[:n], u[n:]
        for name, scheme, L, rho in ARMS:
            idx = blocks(n, H, scheme, L=L, max_subsets=999, rng=rng)
            w = None if rho is None else recency_weights(idx, n, rho)
            ref = reference(blank, post + 1.0, idx, w)
            p_const = pvalue(ref, float(H))[0]
            p_ramp = pvalue(reference(blank, post + 1.0 + ramp, idx, w), float(H))[0]
            for a in ALPHAS:
                for path, p in (("constant", p_const), ("ramp", p_ramp)):
                    cover[(name, path, a)] = cover.get((name, path, a), 0) + (p > a)
            b = band(ref, 0.05)
            if np.isfinite(b.upper - b.lower):
                width.setdefault(name, []).append((b.upper - b.lower) / sd)
            else:
                unb[name] = unb.get(name, 0) + 1
    for a in ALPHAS:
        cells = []
        for name, *_ in ARMS:
            c, r = cover[(name, "constant", a)] / draws, cover[(name, "ramp", a)] / draws
            extra = ""
            if a == 0.05:
                w = width.get(name, [])
                extra = (f" w{np.median(w):.2f}" if w else " w-") + \
                        (f" inf{unb.get(name, 0) / draws:.2f}" if unb.get(name) else "")
            cells.append(f"{name}={c:.3f}/{r:.3f}{extra}")
        print(f"n={n} H={H} {law} alpha={a}: " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    sizes = (int(sys.argv[2]),) if len(sys.argv) > 2 else (24, 48, 96)
    laws = (sys.argv[3],) if len(sys.argv) > 3 else LAWS
    for n in sizes:
        for law in laws:
            run(n, law, draws, seed=100 * n + LAWS.index(law))
