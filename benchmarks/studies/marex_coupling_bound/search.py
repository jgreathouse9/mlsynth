"""The exact search as shipped, and with the hull-distance bound in front of each fit.

Both runs call :func:`~mlsynth.utils.marex_helpers.exact.search_cluster` on the
same panel. The second replaces ``exact.coupled_fit`` for its duration with a
guard that sees what the fit sees: the treated fit's weights ``w0``, so
``g = ||A - B_S w0||^2``, and the cutoff, the incumbent plus the search's
margin. The guard discards the candidate when ``g + beta * dist_lb^2`` clears the
cutoff, which is the fit's own discard (it returns ``None``), and otherwise
fits. The incumbent, the order of candidates and everything else are the
search's own, so the fit counts are what the bound would do inside it.

Single-threaded, as ``marex_exact_backend/timing.py``, after one untimed run
of each search at the smaller ``beta``. From the repository root::

    python benchmarks/studies/marex_coupling_bound/search.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from benchmarks.studies.marex_coupling_bound import bound, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402
from mlsynth.utils.syndes_helpers.exact import SubsetFilter  # noqa: E402


def run(B, A, beta, guard):
    shipped = exact.coupled_fit
    pair_d2 = bound.squared_pair_distances(B)
    n = {"bounded": 0, "discarded_by_bound": 0, "fits": 0, "fits_finished": 0}

    def fit(Bm, Am, treated, control, beta_, *, w0=None, cutoff=None, **kw):
        if guard and cutoff is not None and w0 is not None:
            r = Am - Bm[:, list(treated)] @ w0
            g = float(r @ r)
            need = (cutoff - g) / beta_
            if need > 0:
                n["bounded"] += 1
                d, _, _ = bound.hull_distance_bound(Bm, treated, control, need,
                                                    pair_d2=pair_d2)
                if d > 0 and g + beta_ * d * d > cutoff:
                    n["discarded_by_bound"] += 1
                    return None
        n["fits"] += 1
        out = shipped(Bm, Am, treated, control, beta_, w0=w0, cutoff=cutoff, **kw)
        n["fits_finished"] += out is not None
        return out

    exact.coupled_fit = fit
    try:
        t0 = time.perf_counter()
        best = exact.search_cluster(B, A, [panel.M], filt=SubsetFilter(),
                                    beta=beta)[0]
        seconds = time.perf_counter() - t0
    finally:
        exact.coupled_fit = shipped
    return dict(n, value=best.value, treated=[int(i) for i in best.treated],
                seconds=round(seconds, 2))


def main():
    B, A = panel.geolift()
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "T0": panel.T0, "m": panel.M, "markets": B.shape[1], "runs": {}}
    run(B, A, panel.BETAS[0], guard=False)    # warm-up, untimed: first calls
    run(B, A, panel.BETAS[0], guard=True)     # pay one-off costs
    for beta in panel.BETAS:
        shipped = run(B, A, beta, guard=False)
        guarded = run(B, A, beta, guard=True)
        same = (shipped["treated"] == guarded["treated"]
                and shipped["value"] == guarded["value"])
        out["runs"][str(beta)] = {"shipped": shipped, "with_bound": guarded,
                                  "same_design": same}
        print(f"beta={beta:g}: shipped {shipped['fits']} fits in "
              f"{shipped['seconds']} s; with the bound {guarded['fits']} fits in "
              f"{guarded['seconds']} s ({guarded['discarded_by_bound']} of "
              f"{guarded['bounded']} discarded by it); same design {same}",
              flush=True)
    panel.RESULTS.mkdir(exist_ok=True)
    (panel.RESULTS / "search.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
