"""The interior case: the study's searches against the authors' synthx, for speed.

    python -m benchmarks.studies.as_exact.run_interior    # -> results/interior.json

The designs are those of results/reference_interior.json (capture_interior.py): the four
Proposition 99 donors that lie inside the other donors' hull, each taken as a placebo's
treated unit, and 24 seeded designs with the treated unit pulled inward. Each is solved
by ``synthx.estimate`` under diagonal and positive definite V, as the authors' package
routes an interior unit, and by ``interior.solve_interior_diag`` and
``solve_interior_pd``. Single-threaded wall clock, the median of three calls after a
warm-up, except the authors' placebo runs, which take seconds to minutes and are timed
once.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse          # noqa: E402
import json              # noqa: E402
import pathlib           # noqa: E402
import platform          # noqa: E402
import time              # noqa: E402
import warnings          # noqa: E402

import numpy as np       # noqa: E402
import scipy             # noqa: E402

from . import interior, reference   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def _timed(fn, calls):
    """(result, median seconds) over ``calls`` calls, after one warm-up when calls > 1."""
    if calls > 1:
        fn()
    ts, out = [], None
    for _ in range(calls):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "interior.json"))
    args = ap.parse_args(argv)
    paths = reference.fetch()
    estimate = reference.load(paths)["synthx"].estimate
    cases = json.loads((HERE / "results" / "reference_interior.json").read_text())["cases"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case in cases:
            X1, X0, Z1, Z0 = (np.array(case[k], float) for k in ("X1", "X0", "Z1", "Z0"))
            row = dict(name=case["name"], kind=case["kind"], K=X0.shape[0], J=X0.shape[1])
            for cls, solve in (("diagonal", interior.solve_interior_diag), ("pd", interior.solve_interior_pd)):
                calls = 1 if case["kind"] == "placebo" else 3
                ref, t_ref = _timed(lambda: estimate(X1, X0, Z1, Z0, V_class=cls), calls)
                st = {}
                (loss, w), t_study = _timed(lambda: solve(X1, X0, Z1, Z0, stats=st), 3)
                row[cls] = dict(authors=t_ref, study=t_study, ratio=t_ref / t_study,
                                rel_loss_diff=abs(loss - ref["loss"]) / max(1.0, abs(ref["loss"])),
                                max_abs_w_diff=float(np.abs(w - ref["weights"]).max()),
                                **{k: v for k, v in st.items() if k != "cell"})
            rows.append(row)
            print(json.dumps(row), flush=True)
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               machine=platform.machine(), threads=os.environ.get("OMP_NUM_THREADS"),
               commits={k: v[1] for k, v in paths.items()})
    pathlib.Path(args.out).write_text(json.dumps(dict(environment=env, rows=rows), indent=1))
    for kind in ("placebo", "random"):
        sub = [r for r in rows if r["kind"] == kind]
        for cls in ("diagonal", "pd"):
            ratio = [r[cls]["ratio"] for r in sub]
            print(f"{kind} {cls}: authors {sum(r[cls]['authors'] for r in sub):.2f} s, study "
                  f"{sum(r[cls]['study'] for r in sub):.2f} s in all; per design {min(ratio):.1f}x-"
                  f"{max(ratio):.1f}x (median {np.median(ratio):.1f}x); largest weight difference "
                  f"{max(r[cls]['max_abs_w_diff'] for r in sub):.1e}")


if __name__ == "__main__":
    main()
