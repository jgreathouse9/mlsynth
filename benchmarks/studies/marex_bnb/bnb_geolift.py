"""The 12-market branch and bound at GeoLift's scale: 40 markets, 4 treated.

The tree and the node bounds are those of ``bnb12.py``. One step changes at the
bottom of the tree: a node that fixes three markets has complete treated sets as
children, and those are priced the way the exact backend prices every set, by
the library's closed-form treated bound. Only children whose closed form is
under the incumbent are fitted exactly, in closed-form order. So the count to
watch is how many of the 91,390 sets get priced at all: the exact backend
prices every one.

Every bounded node is checked against the minimum of its subtree, read from the
enumeration cached by ``panels.leaves``, and every run must return the
enumeration's optimum. Each bound is also run bounding only the first one or
two levels: a node bound costs about 0.6 ms against about a microsecond to
price a set, so a node pays only when its subtree holds hundreds of sets.

From the repository root::

    python -m benchmarks.studies.marex_bnb.bnb_geolift
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from math import comb

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from benchmarks.studies.marex_bnb.bnb12 import Bounds, fit, objective
from benchmarks.studies.marex_bnb.panels import RESULTS, geolift, leaves

NAME, M = "geolift_m4", 4


def subtree_minima(sets, f, order, n):
    """``{depth: {code: min f}}``, a node's code being its positions in base n."""
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    keyed = np.sort(rank[sets], axis=1)
    out = {}
    for d in range(1, M):
        code = np.zeros(len(sets), dtype=np.int64)
        for c in range(d):
            code = code * n + keyed[:, c]
        uniq, inv = np.unique(code, return_inverse=True)
        mins = np.full(len(uniq), np.inf)
        np.minimum.at(mins, inv, f)
        out[d] = dict(zip(uniq.tolist(), mins.tolist()))
    return out


def run(args):
    bname, oname, order, incumbent, depth_cap = args
    from mlsynth.utils.marex_helpers.exact import treated_fit_bound
    B, A = geolift()
    n = B.shape[1]
    sets, g, h = leaves(NAME)
    f = g + h
    mins = subtree_minima(sets, f, order, n)
    bd = Bounds(B, A, m=M)
    bound = getattr(bd, bname)
    best = [incumbent, None]
    st = {"bounded": [0] * M, "pruned": [0] * M, "sets_cut": 0,
          "priced": 0, "fitted": 0, "bound_s": 0.0, "leaf_s": 0.0,
          "ratio_d1": []}

    def visit(pos):
        d = len(pos)
        last = pos[-1] if pos else -1
        for q in range(last + 1, n - (M - d) + 1):
            child = pos + (q,)
            k = len(child)
            I = [int(order[p]) for p in child]
            F = [int(order[p]) for p in range(q + 1, n)]
            O = [int(order[p]) for p in range(q + 1) if p not in child]
            if k < M and k > depth_cap:
                leaves_of(child) if k == M - 1 else visit(child)
                continue
            if k < M:
                t0 = time.perf_counter()
                lb = bound(I, F, O, M - k)
                st["bound_s"] += time.perf_counter() - t0
                st["bounded"][k] += 1
                code = 0
                for p in child:
                    code = code * n + p
                sub = mins[k][code]
                assert lb <= sub * (1 + 1e-9) + 1e-9, (bname, I, lb, sub)
                if k == 1:
                    st["ratio_d1"].append(lb / sub)
                if lb >= best[0]:
                    st["pruned"][k] += 1
                    st["sets_cut"] += comb(n - 1 - q, M - k)
                    continue
                if k < M - 1:
                    visit(child)
                else:
                    leaves_of(child)

    def leaves_of(pos):
        t0 = time.perf_counter()
        kids = [pos + (q,) for q in range(pos[-1] + 1, n)]
        S = np.array([sorted(int(order[p]) for p in c) for c in kids])
        lbs = treated_fit_bound(B, A, S)
        st["priced"] += len(S)
        for i in np.argsort(lbs, kind="stable"):
            if lbs[i] >= best[0]:
                break
            st["fitted"] += 1
            value = objective(B, A, S[i])
            if value < best[0]:
                best[0], best[1] = value, tuple(int(x) for x in S[i])
        st["leaf_s"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    visit(())
    st["total_s"] = time.perf_counter() - t0
    fstar = float(f.min())
    S_star = tuple(int(x) for x in sets[int(np.argmin(f))])
    if np.isinf(incumbent):
        assert best[1] == S_star and abs(best[0] - fstar) <= 1e-9 * fstar, best
    r = np.array(st.pop("ratio_d1"))
    half = r[: len(r) // 2]
    st["ratio_d1_first_half_median"] = float(np.median(half))
    st["ratio_d1_median"] = float(np.median(r))
    start = "cold" if np.isinf(incumbent) else "at f*"
    return f"{bname}<={depth_cap}", oname, start, st


def existing_search():
    from mlsynth.utils.marex_helpers import exact as ex
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    B, A = geolift()
    calls = [0]
    real = ex.side_fit

    def spy(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    ex.side_fit = spy
    t0 = time.perf_counter()
    d = ex.search_cluster(B, A, [M], filt=SubsetFilter())[0]
    elapsed = time.perf_counter() - t0
    ex.side_fit = real
    return d, calls[0] // 2, elapsed


def main():
    B, A = geolift()
    n = B.shape[1]
    sets, g, h = leaves(NAME)
    f = g + h
    fstar = float(f.min())
    single = np.array([fit(B, A, [j]) for j in range(n)])
    orders = {"index": np.arange(n),
              "best_first": np.argsort(single, kind="stable"),
              "worst_first": np.argsort(-single, kind="stable"),
              "random": np.random.default_rng(0).permutation(n)}
    jobs = [(b, o, order, inc, cap) for b in ("monotone", "sparse")
            for o, order in orders.items()
            for inc in (np.inf, fstar * (1 + 1e-12))
            for cap in (M - 1, 2, 1)]
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(run, jobs))
    d, fits, secs = existing_search()
    print(f"GeoLift, {n} markets, {M} treated, {len(sets):,} sets, f* = "
          f"{fstar:.10g}")
    print(f"existing search: prices {len(sets):,} sets, fits {fits}, "
          f"{secs:.2f} s, optimum {d.treated}")
    print(f"{'bound<=depth':12s} {'order':12s} {'start':6s} {'bounded d1/d2/d3':>18s} "
          f"{'pruned d1/d2/d3':>16s} {'priced':>7s} {'fitted':>6s} "
          f"{'ratio d1':>15s} {'seconds':>8s}")
    rows = []
    for bname, oname, start, st in results:
        b, p = st["bounded"][1:], st["pruned"][1:]
        print(f"{bname:12s} {oname:12s} {start:6s} "
              f"{'/'.join(map(str, b)):>18s} {'/'.join(map(str, p)):>16s} "
              f"{st['priced']:7d} {st['fitted']:6d} "
              f"{st['ratio_d1_first_half_median']:6.3f} "
              f"{st['ratio_d1_median']:6.3f}  {st['total_s']:8.1f}")
        rows.append({"bound": bname, "order": oname, "start": start, **st})
    (RESULTS / "bnb_geolift.json").write_text(json.dumps(
        {"sets": len(sets), "f_star": fstar,
         "existing": {"priced": len(sets), "fitted": fits, "seconds": secs},
         "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
