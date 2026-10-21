"""The exact backend's design step by design and cluster count, on GeoLift.

``timing.py`` times the standard design. This times every design the search
solves without a coupling term, ``standard``, ``penalized`` and
``unit_penalized``, on GeoLift's 40 markets with the 90 pre-periods as the fit
window, as one cluster and as 2 and 4 clusters of equal size. The clusters
split the markets in order along the first principal component of their series,
each divided by its mean, so every cluster has room for its treated markets.

Each run counts the candidates the closed-form bound prices and the simplex fits
the search makes, by wrapping ``exact.treated_fit_bound`` and ``exact.side_fit``
from the outside. On one cluster with four treated, it also takes the arguments
the search receives and counts, with the cutoff at the optimum, the candidates
three bounds leave to fit:

* the shipped closed-form bound ``lb``, the affine relaxation of the treated fit;
* ``max(lb, 1/s + min over S of a)``: the treated fit is at least the affine
  relaxation of its quadratic term plus the smallest treated penalty, since a
  minimum of a sum is at least the sum of the minimums and the weights are on
  the simplex;
* that, plus the penalised control fit over every market in the cluster, which
  bounds every candidate's control fit since a smaller donor pool only fits
  worse.

Single-threaded, the median of three runs where one takes under 3 s. The
penalised designs at five treated on one cluster are not run: they take tens of
minutes. From the repository root::

    python benchmarks/studies/marex_exact_backend/designs.py \
        benchmarks/studies/marex_exact_backend/results/designs.json
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

import numpy as np
import pandas as pd

from mlsynth.utils.marex_helpers import exact

DESIGNS = {
    "standard": {},
    "penalized_0.1": dict(design="penalized", lambda1=0.1, lambda2=0.1),
    "penalized_1": dict(design="penalized", lambda1=1.0, lambda2=1.0),
    "unit_penalized_light": dict(design="unit_penalized", xi=0.1, lambda1_unit=0.01),
    "unit_penalized_heavy": dict(design="unit_penalized", xi=1.0, lambda1_unit=0.1),
}
#: (clusters, treated per cluster, designs to run or None for all)
CASES = ((1, 4, None), (1, 5, ("standard", "unit_penalized_light")),
         (2, 2, None), (2, 3, None), (4, 1, None), (4, 2, None))

_count = {"priced": 0, "fits": 0}
_seen = []
_bound, _fit, _search = exact.treated_fit_bound, exact.side_fit, exact.search_cluster


def _counted_bound(*a, **k):
    _count["priced"] += len(a[2])
    return _bound(*a, **k)


def _counted_fit(*a, **k):
    _count["fits"] += 1
    return _fit(*a, **k)


def _recorded_search(B, A, sizes, **kw):
    out = _search(B, A, sizes, **kw)
    _seen.append((B, A, kw, out[0].value))
    return out


def panel():
    d = pd.read_csv("basedata/geolift_market_data.csv")
    return d.pivot(index="location", columns="date", values="Y").sort_index(axis=1)


def clusters(Y):
    shape = Y.to_numpy(float)
    shape = shape / shape.mean(axis=1, keepdims=True)
    shape -= shape.mean(axis=0)
    pc1 = np.linalg.svd(shape, full_matrices=False)[0][:, 0]
    rank = np.argsort(np.argsort(pc1))
    n = len(Y)
    return {1: np.zeros(n, dtype=int), 2: rank // (n // 2), 4: rank // (n // 4)}


def left_to_fit(B, A, m, kw, fstar):
    """Candidates each bound leaves under the optimum plus the search's margin."""
    a, b = kw.get("linear_treated"), kw.get("linear_control")
    cut = fstar + exact.PRUNE_MARGIN * (1.0 + abs(fstar))
    sets = np.array(list(itertools.combinations(range(B.shape[1]), m)))
    lb = _bound(B, A, sets, a)
    treated = np.maximum(lb, _bound(B, A, sets, None)
                         + (a[sets].min(axis=1) if a is not None else 0.0))
    control = _fit(B, A, list(range(B.shape[1])), b)[0]
    return {"optimum": fstar, "control_fit_over_the_cluster": control,
            "shipped": int((lb < cut).sum()),
            "with_treated_bound": int((treated < cut).sum()),
            "with_both_bounds": int((treated + control < cut).sum())}


def main(out_path):
    Y = panel()
    T0 = Y.shape[1]
    groups = clusters(Y)
    exact.treated_fit_bound, exact.side_fit = _counted_bound, _counted_fit
    exact.search_cluster = _recorded_search
    out = {"python": platform.python_version(), "numpy": np.__version__, "T0": T0,
           "cluster_sizes": {str(k): np.bincount(g).tolist() for k, g in groups.items()},
           "runs": [], "left_to_fit_one_cluster_m4": {}}
    try:
        exact.solve_design_exact(Y, T0, groups[1], m_eq=2)
        for K, m, only in CASES:
            for name, kw in DESIGNS.items():
                if only and name not in only:
                    continue
                times = []
                for _ in range(3):
                    _count["priced"] = _count["fits"] = 0
                    _seen.clear()
                    t0 = time.perf_counter()
                    res = exact.solve_design_exact(Y, T0, groups[K], m_eq=m, **kw)
                    times.append(time.perf_counter() - t0)
                    if times[-1] > 3.0:
                        break
                row = {"clusters": K, "treated_per_cluster": m, "design": name,
                       "candidates": int(sum(comb(int(s), m) for s in np.bincount(groups[K]))),
                       "priced": _count["priced"], "simplex_fits": _count["fits"],
                       "seconds": round(float(np.median(times)), 3), "runs": len(times),
                       "objective": float(res["objective"])}
                out["runs"].append(row)
                print(f"{K} cluster(s), m={m}, {name}: {row['seconds']} s, "
                      f"{row['candidates']:,} candidates, {row['simplex_fits']:,} simplex fits",
                      flush=True)
                if K == 1 and m == 4 and name != "standard":
                    B, A, kw_s, fstar = _seen[-1]
                    out["left_to_fit_one_cluster_m4"][name] = left_to_fit(B, A, m, kw_s, fstar)
                    print("   left to fit:", out["left_to_fit_one_cluster_m4"][name], flush=True)
    finally:
        exact.treated_fit_bound, exact.side_fit = _bound, _fit
        exact.search_cluster = _search
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
