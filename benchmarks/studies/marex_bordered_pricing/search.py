"""The GeoLift searches end to end, with the shipped pricing and with bordering.

The standard design at four, five and six treated markets, where pricing is
most of the search, and the weakly-targeted design at ``beta = 1`` with four,
where the fits are. The second run of each swaps ``exact.treated_fit_bound``
for :func:`bordered.updated_bound` for its duration; everything else is the
search's own. After one untimed warm-up search.

Single-threaded. From the repository root::

    python benchmarks/studies/marex_bordered_pricing/search.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.studies.marex_bordered_pricing import bordered, panel  # noqa: E402
from mlsynth.utils.marex_helpers import exact  # noqa: E402
from mlsynth.utils.syndes_helpers.exact import SubsetFilter  # noqa: E402

CASES = (("standard", 4, None), ("standard", 5, None), ("standard", 6, None),
         ("weakly_targeted", 4, 1.0))


def search(B, A, m, beta, pricing):
    shipped = exact.treated_fit_bound
    exact.treated_fit_bound = pricing
    try:
        t0 = time.perf_counter()
        best = exact.search_cluster(B, A, [m], filt=SubsetFilter(), beta=beta)[0]
        seconds = time.perf_counter() - t0
    finally:
        exact.treated_fit_bound = shipped
    return {"seconds": round(seconds, 3), "value": best.value,
            "treated": [int(i) for i in best.treated]}


def main():
    B, A = panel.geolift()
    exact.search_cluster(B, A, [3], filt=SubsetFilter())
    out = {}
    for design, m, beta in CASES:
        shipped = search(B, A, m, beta, exact.treated_fit_bound)
        updated = search(B, A, m, beta, bordered.updated_bound)
        same = (shipped["treated"] == updated["treated"]
                and shipped["value"] == updated["value"])
        key = f"{design}_m{m}" + ("" if beta is None else f"_beta{beta:g}")
        out[key] = {"shipped": shipped, "bordered": updated, "same_design": same}
        print(f"{key}: {shipped['seconds']} s -> {updated['seconds']} s, "
              f"same design {same}", flush=True)
    panel.RESULTS.mkdir(exist_ok=True)
    (panel.RESULTS / "search.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
