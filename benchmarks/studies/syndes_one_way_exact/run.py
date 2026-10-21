"""Run the one-way study's grid and write one CSV row per instance.

    python -m benchmarks.studies.syndes_one_way_exact.run GROUP [scip=default|proof|none]

The groups and panels are the per-unit study's
(``benchmarks/studies/syndes_per_unit_exact/run.py``): ``bls``, ``small``,
``factor``, ``noise`` and ``real``. Instances with more than ``LIMIT`` treated
sets are recorded as beyond enumeration and still given to SCIP.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.studies.syndes_one_way_exact.search import scip, search  # noqa: E402
from benchmarks.studies.syndes_per_unit_exact.run import instances  # noqa: E402
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
LIMIT = 20_000_000
SCIP_SETTINGS = {"default": (60.0, 0.05), "proof": (300.0, 0.0)}
FIELDS = ["name", "N", "T", "K", "lam", "count", "skipped", "value", "design",
          "gap_two_way", "gap_ridge", "pruned_two_way", "pruned_ridge",
          "ridge_ge_two_way", "priced", "max_rel_gap", "t_bounds", "t_total",
          "scip_status", "scip_wall", "scip_time", "scip_value", "scip_design",
          "scip_excess"]


def main(argv):
    group = argv[0]
    mode = dict(a.split("=", 1) for a in argv[1:]).get("scip", "default")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{group}_scip-{mode}.csv", "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=FIELDS)
        out.writeheader()
        for name, Y, K in instances(group):
            Y = np.ascontiguousarray(Y, dtype=float)
            T, N = Y.shape
            lam = float(estimate_lambda(Y))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = search(Y, K, lam, limit=LIMIT)
                m = ({} if mode == "none" else
                     scip(Y, K, lam, time_limit=SCIP_SETTINGS[mode][0],
                          gap_limit=SCIP_SETTINGS[mode][1]))
            row = {"name": name, "N": N, "T": T, "K": K, "lam": lam, **s}
            if m:
                excess = (math.nan if s["skipped"] or not m["design"]
                          else (m["value"] - s["value"]) / s["value"])
                row.update(scip_status=m["status"], scip_wall=m["wall"],
                           scip_time=m["scip_time"], scip_value=m["value"],
                           scip_design=m["design"], scip_excess=excess)
            out.writerow({k: row.get(k) for k in FIELDS})
            fh.flush()
            if s["skipped"]:
                line = f"{name}: beyond enumeration ({s['count']:,} sets)"
            else:
                line = (f"{name}: sets={s['count']} gap two-way={s['gap_two_way']:.4f} "
                        f"ridge={s['gap_ridge']:.4f} pruned={s['pruned_ridge']:.4f} "
                        f"priced={s['priced']} t={s['t_total']:.2f}s")
            if m:
                line += (f" | scip {m['status']} {m['wall']:.1f}s"
                         + (f" excess={row['scip_excess']:.2e}"
                            if not s["skipped"] else ""))
            print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
