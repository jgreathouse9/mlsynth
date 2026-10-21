"""The library's exact search, its candidate limit raised, as an independent check.

``search_cluster`` prices every treated set by its closed-form bound and fits the
few the bound cannot rule out. It shares no code with the certified search
beyond ``side_fit``, so agreement on the optimum is a cross-check of both.
Its answer is not certified: it rests on floating-point bounds.

From the repository root::

    python -m benchmarks.studies.marex_bnb.exact_reference geolift 10
    python -m benchmarks.studies.marex_bnb.exact_reference walmart 6 8
"""
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

from benchmarks.studies.marex_bnb.panels import geolift, walmart
from mlsynth.utils.marex_helpers.exact import count_candidates, search_cluster
from mlsynth.utils.syndes_helpers.exact import SubsetFilter


def main(name, ms):
    B, A = walmart() if name == "walmart" else geolift()
    n = B.shape[1]
    for m in ms:
        total = count_candidates(n, [m], SubsetFilter())
        t0 = time.perf_counter()
        best = search_cluster(B, A, [m], filt=SubsetFilter(), candidate_limit=total)[0]
        print(f"{name} m={m}: {total:,} sets; exact backend {best.value:.10g} at "
              f"{best.treated} in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(a) for a in sys.argv[2:]])
