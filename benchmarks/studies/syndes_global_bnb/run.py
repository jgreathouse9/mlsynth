"""The search's table: one panel, one size, the search and optionally every set.

From the repository root::

    python -m benchmarks.studies.syndes_global_bnb.run geolift 40 10
    python -m benchmarks.studies.syndes_global_bnb.run factor 60 6 nolift
    python -m benchmarks.studies.syndes_global_bnb.run geolift 40 10 enumerate
    python -m benchmarks.studies.syndes_global_bnb.run factor 40 6 library
    python -m benchmarks.studies.syndes_global_bnb.run geolift 40 10 dual

``nolift`` turns the lift off; ``dual`` lifts by the regularised dual of
``dual.py`` in place of the interior-point solve; ``enumerate`` prices every
set through the study's own path (no bound); ``library`` runs
``solve_synthetic_design_exact`` with its candidate limit raised. Panels: ``factor`` and ``noise`` (seed 0) from
``quality.py``, and ``geolift`` (its first ``N`` markets, 90 days).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np

from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact

from .quality import common_shock, factor, geolift
from .search import search

PANELS = {"factor": lambda N: factor(N, seed=0),
          "noise": lambda N: common_shock(N, seed=0),
          "geolift": geolift}


def main(argv):
    name, N, K = argv[0], int(argv[1]), int(argv[2])
    Y = PANELS[name](N)
    if "library" in argv:
        t = time.perf_counter()
        d = solve_synthetic_design_exact(Y, K, candidate_limit=10 ** 12)
        print(f"{name} N={N} K={K} library {time.perf_counter() - t:.1f}s "
              f"value={d.objective_value:.10g} "
              f"treated={sorted(int(i) for i in d.selected_unit_indices)}")
        return
    if "enumerate" in argv:
        r = search(Y, K, price_cost=0.0, bound_cost=1.0, lift_cost=np.inf)
        label = "every set"
    else:
        lift = "dual" if "dual" in argv else "ipm"
        r = search(Y, K, lift_cost=np.inf if "nolift" in argv else None, lift=lift)
        label = "search" + (" (no lift)" if "nolift" in argv else "") + (" (dual)" if lift == "dual" else "")
    print(f"{name} N={N} K={K} {label} {r.seconds:.1f}s bounded={r.nodes_bounded} "
          f"lifted={r.nodes_lifted} pruned={r.nodes_pruned} priced={r.sets_priced:,} "
          f"value={r.value:.10g} treated={sorted(int(i) for i in r.treated)}")


if __name__ == "__main__":
    main(sys.argv[1:])
