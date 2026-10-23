"""Seeded random designs of Proposition 99's size and larger: the diagonal search in both codes.

    python -m benchmarks.studies.as_exact.scaling     # -> results/scaling.json

Designs come from ``capture.random_instance`` with J = 38 donors, the Proposition 99
donor count, and K predictors. At K = 7 each design is timed in both codes (the
replication's median of 3 calls after a warm-up, the study's median of 5); at K = 8
each code runs once per design; at K = 9 only the study runs. Every row carries the
study's search counters, among them ``fallbacks``: faces whose normal cone has too
many walls for the batched ray enumeration, listed instead by one LP per sign pattern.
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

from . import exact, reference              # noqa: E402
from .capture import random_instance        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
J = 38
PLAN = [(7, 8, 3, 5), (8, 4, 1, 1), (9, 4, 0, 1)]   # (K, designs, replication calls, study calls)


def _median_time(fn, calls):
    ts, out = [], None
    for _ in range(calls):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))


def design_row(seed, K, inst, rep_calls, study_calls, vfs):
    X1, X0, Z1, Z0 = inst
    st = {}
    t = time.perf_counter()
    loss, w = exact.solve_diag(X1, X0, Z1, Z0, stats=st)
    t_study = time.perf_counter() - t
    if study_calls > 1:                          # the counted run above was the warm-up
        (loss, w), t_study = _median_time(lambda: exact.solve_diag(X1, X0, Z1, Z0), study_calls)
    row = dict(seed=seed, K=K, J=J, study=t_study, study_calls=study_calls, **st)
    if rep_calls:
        if rep_calls > 1:
            vfs.solve_exterior_hullfree(X0, X1, Z1, Z0)
        (l_r, w_r, _), t_rep = _median_time(lambda: vfs.solve_exterior_hullfree(X0, X1, Z1, Z0), rep_calls)
        row.update(replication=t_rep, replication_calls=rep_calls, ratio=t_rep / t_study,
                   rel_loss_diff=abs(loss - l_r) / max(1.0, abs(l_r)),
                   max_abs_w_diff=float(np.abs(w - w_r).max()))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "scaling.json"))
    args = ap.parse_args(argv)
    paths = reference.fetch()
    vfs = reference.load(paths)["visible_facet_solver"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for K, n, rep_calls, study_calls in PLAN:
            seed, got = 1000 * K, 0
            while got < n:
                inst = random_instance(seed, K, J)
                seed += 1
                if inst is None:
                    continue
                rows.append(design_row(seed - 1, K, inst, rep_calls, study_calls, vfs))
                print(json.dumps(rows[-1]), flush=True)
                got += 1
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               machine=platform.machine(), threads=os.environ.get("OMP_NUM_THREADS"),
               commits={k: v[1] for k, v in paths.items()}, plan=PLAN)
    pathlib.Path(args.out).write_text(json.dumps(dict(environment=env, rows=rows), indent=1))
    for K, *_ in PLAN:
        sub = [r for r in rows if r["K"] == K]
        line = (f"K={K}: study {min(r['study'] for r in sub):.2f}-{max(r['study'] for r in sub):.2f} s, "
                f"fallback faces {[r['fallbacks'] for r in sub]}")
        if "ratio" in sub[0]:
            ratio = [r["ratio"] for r in sub]
            line += (f"; replication {min(r['replication'] for r in sub):.1f}-"
                     f"{max(r['replication'] for r in sub):.1f} s; speed-up {min(ratio):.1f}x-{max(ratio):.1f}x "
                     f"(median {np.median(ratio):.1f}x)")
        print(line)


if __name__ == "__main__":
    main()
