"""Pricing time per candidate: the shipped solve against bordering, m = 2 to 6.

Issue #769's measurement, on GeoLift (``n = 40``) and on the synthetic panel
(``n = 80``). Each case prices every candidate in lexicographic order, in the
search's batches of 20,000, three ways:

* ``shipped``: ``exact.treated_fit_bound``, one ``m``-by-``m`` solve each;
* ``bordered``: :func:`bordered.updated_bound`, the same values from the
  prefix's inverse;
* ``bordered_safe``: bordering, then :func:`bordered.safe_bound` at the affine
  minimiser, the form whose validity does not depend on the solve.

It records the largest relative difference between the shipped and bordered
values, how many candidates each sets to 0, and the largest ratio of the safe
form's a priori rounding bound to the search's margin. ``n = 80, m = 6`` has
300,500,200 candidates; there the prefixes of 4 million are drawn with
probability proportional to their number of extensions, which keeps the cost per
candidate unbiased, and all their extensions are priced.

Single-threaded. From the repository root::

    python benchmarks/studies/marex_bordered_pricing/pricing.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
import json
import platform
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_bordered_pricing import bordered, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402

FULL_LIMIT = 25_000_000
SAMPLE = 4_000_000
SIZES = range(2, 7)


def all_chunks(n, m):
    walk = itertools.combinations(range(n), m)
    while True:
        flat = np.fromiter(itertools.chain.from_iterable(
            itertools.islice(walk, bordered.CHUNK)), dtype=np.int64)
        if not flat.size:
            return
        yield flat.reshape(-1, m)


def sampled_chunks(n, m, seed=11):
    rng = np.random.default_rng(seed)
    seen, parts, total = set(), [], 0
    while total < SAMPLE:
        S = np.sort(rng.choice(n, m, replace=False))
        P = tuple(int(i) for i in S[:-1])
        if P in seen:
            continue
        seen.add(P)
        ext = np.arange(P[-1] + 1, n)
        parts.append(np.column_stack([np.tile(P, (len(ext), 1)), ext]))
        total += len(ext)
    leaves = np.concatenate(parts)
    leaves = leaves[np.lexsort(leaves.T[::-1])]
    for lo in range(0, len(leaves), bordered.CHUNK):
        yield leaves[lo:lo + bordered.CHUNK]


def price(B, A, m):
    n = B.shape[1]
    total = comb(n, m)
    full = total <= FULL_LIMIT
    blocks = list(all_chunks(n, m) if full else sampled_chunks(n, m))
    count = sum(len(b) for b in blocks)
    Bc, G = bordered.gram(B, A)
    Habs = np.abs(Bc).T @ np.abs(Bc)
    secs = {"shipped": 0.0, "bordered": 0.0, "bordered_safe": 0.0}
    rel = err = 0.0
    zeros_shipped = zeros_bordered = 0
    for blk in blocks:
        t0 = time.perf_counter()
        d = exact.treated_fit_bound(B, A, blk)
        secs["shipped"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        b = bordered.updated_bound(B, A, blk)
        secs["bordered"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        s, t, u, x, y, _ = bordered.bordered(G, blk)
        v = bordered.affine_minimiser(s, t, x, y)
        safe = bordered.safe_bound(G, blk, v)
        secs["bordered_safe"] += time.perf_counter() - t0
        e = bordered.safe_error_bound(Habs, blk, v, B.shape[0])
        ok = np.isfinite(safe) & (safe > 0)
        if ok.any():
            err = max(err, float((e[ok] / (exact.PRUNE_MARGIN
                                           * (1.0 + np.abs(safe[ok])))).max()))
        both = (d > 0) & (b > 0)
        if both.any():
            rel = max(rel, float((np.abs(d - b)[both] / d[both]).max()))
        zeros_shipped += int((d == 0).sum())
        zeros_bordered += int((b == 0).sum())
    return {"candidates": total, "priced": count, "all_priced": full,
            "us_per_candidate": {k: round(v / count * 1e6, 3) for k, v in secs.items()},
            "speedup_bordered": round(secs["shipped"] / secs["bordered"], 2),
            "speedup_bordered_safe": round(secs["shipped"] / secs["bordered_safe"], 2),
            "max_relative_difference": rel, "zero_bounds_shipped": zeros_shipped,
            "zero_bounds_bordered": zeros_bordered,
            "max_rounding_bound_over_margin": err}


def main():
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "T": panel.T0, "cases": {}}
    for name, (B, A) in (("geolift_n40", panel.geolift()),
                         ("synthetic_n80", panel.synthetic())):
        for m in SIZES:
            r = price(B, A, m)
            out["cases"][f"{name}_m{m}"] = r
            print(f"{name} m={m}: {r['us_per_candidate']} speedup "
                  f"{r['speedup_bordered']} (safe {r['speedup_bordered_safe']}), "
                  f"max rel diff {r['max_relative_difference']:.2g}, rounding/margin "
                  f"{r['max_rounding_bound_over_margin']:.2g}", flush=True)
    panel.RESULTS.mkdir(exist_ok=True)
    (panel.RESULTS / "pricing.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
