"""The swap-search fallback on every instance of a group, without SCIP.

    python -m benchmarks.studies.syndes_one_way_exact.fallback GROUP

Writes ``results/<group>_fallback.csv``, joined to the group's main CSV by
``name``: where the instance was enumerated, it says whether the fallback found
the optimum.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.studies.syndes_one_way_exact.search import fallback  # noqa: E402
from benchmarks.studies.syndes_per_unit_exact.run import instances  # noqa: E402
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIELDS = ["name", "N", "K", "fb_design", "fb_value", "n_candidates", "rayleigh",
          "rayleigh_gap", "fb_t"]


def main(group):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{group}_fallback.csv", "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=FIELDS)
        out.writeheader()
        for name, Y, K in instances(group):
            Y = np.ascontiguousarray(Y, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = fallback(Y, K, float(estimate_lambda(Y)))
            out.writerow({"name": name, "N": Y.shape[1], "K": K,
                          "fb_design": f["design"], "fb_value": f["value"],
                          "n_candidates": f["n_candidates"], "rayleigh": f["rayleigh"],
                          "rayleigh_gap": f["rayleigh_gap"], "fb_t": f["t_total"]})
            fh.flush()
            print(f"{name}: candidates={f['n_candidates']} rayleigh gap="
                  f"{f['rayleigh_gap']:.3f} t={f['t_total']:.2f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
