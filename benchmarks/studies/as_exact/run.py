"""Proposition 99: the cvxpy-free solver against the authors' current code, for speed.

    python -m benchmarks.studies.as_exact.run            # pinned commits (reference.py)
    python -m benchmarks.studies.as_exact.run --latest   # their default-branch heads

Inputs are built by the authors' own ``build_california`` from their repository's
``data/smoking_data.csv``, so every implementation solves the same arrays. Two panels:
the headline specification (validation 1970-1988, Tables 1, 2, 4, D.1) and the
leakage-free variant of Section 6 (validation on the 16 pre-treatment years that are
not lagged predictors). Each task is timed ``--reps`` times after one warm-up call,
single-threaded, in one process; results go to ``results/raw.json``.
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

from . import exact, rationalize, reference   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def timed(fn, reps):
    out = fn()                                    # warm-up: imports, caches, first solver call
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, ts


def summary(ts):
    a = np.array(ts)
    return dict(times=[round(float(t), 6) for t in a], median=float(np.median(a)), min=float(a.min()),
                q25=float(np.quantile(a, 0.25)), q75=float(np.quantile(a, 0.75)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "results" / "raw.json"))
    args = ap.parse_args(argv)

    paths = reference.fetch(latest=args.latest)
    ref = reference.load(paths)
    dp, vfs, pw, sx = ref["data_preparation"], ref["visible_facet_solver"], ref["predictor_weights"], ref["synthx"]
    data = dp.build_california(str(paths["synth-AS-replication"][0] / "data" / "smoking_data.csv"))
    X1, X0, Z1, Z0, donors = data["X1"], data["X0"], data["Z1"], data["Z0"], data["donors"]
    names = list(dp.CA_PREDICTORS)
    hold = list(dp.CA_HOLDOUT_IDX)
    panels = {"prop99": (Z1, Z0), "prop99_leakage_free": (Z1[hold], Z0[hold])}

    import cvxpy
    import clarabel
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               cvxpy=cvxpy.__version__, clarabel=clarabel.__version__, machine=platform.machine(),
               cpus_visible=os.cpu_count(), threads=os.environ.get("OMP_NUM_THREADS"),
               reps=args.reps, commits={k: v[1] for k, v in paths.items()})
    rows = []

    def record(panel, task, impl, ts, **extra):
        rows.append(dict(panel=panel, task=task, impl=impl, **summary(ts), **extra))
        print(f"{panel:22s} {task:26s} {impl:24s} median {np.median(ts):9.4f}s", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for panel, (z1, z0) in panels.items():
            # donor weights, diagonal class
            (l_r, w_r, _), t = timed(lambda: vfs.solve_exterior_hullfree(X0, X1, z1, z0), args.reps)
            record(panel, "weights, diagonal", "replication", t, loss=l_r)
            out, t = timed(lambda: sx.estimate(X1, X0, z1, z0, V_class="diagonal"), args.reps)
            record(panel, "weights, diagonal", "synthx.estimate", t, loss=out["loss"],
                   max_abs_w_diff=float(np.abs(out["weights"] - w_r).max()))
            (l_o, w_o), t = timed(lambda: exact.solve_diag(X1, X0, z1, z0), args.reps)
            record(panel, "weights, diagonal", "study (no cvxpy)", t, loss=l_o,
                   max_abs_w_diff=float(np.abs(w_o - w_r).max()))
            # donor weights, positive definite class
            (lp_r, wp_r, _), t = timed(lambda: vfs.solve_exterior_pd(X0, X1, z1, z0), args.reps)
            record(panel, "weights, positive definite", "replication", t, loss=lp_r)
            out, t = timed(lambda: sx.estimate(X1, X0, z1, z0, V_class="pd"), args.reps)
            record(panel, "weights, positive definite", "synthx.estimate", t, loss=out["loss"],
                   max_abs_w_diff=float(np.abs(out["weights"] - wp_r).max()))
            (lp_o, wp_o), t = timed(lambda: exact.solve_pd(X1, X0, z1, z0), args.reps)
            record(panel, "weights, positive definite", "study (no cvxpy)", t, loss=lp_o,
                   max_abs_w_diff=float(np.abs(wp_o - wp_r).max()))
            if panel != "prop99":
                continue
            # predictor weights for the diagonal solution (Table 2) and the PD class (Table 4)
            ref_d, t = timed(lambda: pw.diag_centers(X0, X1, w_r, names, decimals=None), args.reps)
            record(panel, "V, diagonal (Table 2)", "replication", t)
            our_d, t = timed(lambda: rationalize.diag_rationalizers(X1, X0, w_r), args.reps)
            diff_d = max(np.abs(our_d["logdet"] - np.array([ref_d["logdet"][n] for n in names])).max(),
                         np.abs(our_d["center"] - np.array([ref_d["analytic_center"][n] for n in names])).max())
            record(panel, "V, diagonal (Table 2)", "study (no cvxpy)", t, max_abs_diff=float(diff_d))

            def ref_pd():
                _, a = pw.compute_pd_logdet_and_ranges(X1, X0, w_r, names, decimals=None, ranges=True)
                _, b = pw.analytic_center(X1, X0, w_r, names, decimals=None)
                return a, b
            (ref_a, ref_b), t = timed(ref_pd, args.reps)
            record(panel, "V, PD + ranges (Table 4)", "replication", t)
            our_p, t = timed(lambda: rationalize.pd_rationalizers(X1, X0, w_r, ranges=True), args.reps)
            diff_p = max(np.abs(our_p["logdet_diag"] - np.array([ref_a["diag"][n] for n in names])).max(),
                         np.abs(our_p["center_diag"] - np.array([ref_b["diag"][n] for n in names])).max(),
                         np.abs(our_p["lo"] - np.array([ref_a["ranges"][n][0] for n in names])).max(),
                         np.abs(our_p["hi"] - np.array([ref_a["ranges"][n][1] for n in names])).max())
            record(panel, "V, PD + ranges (Table 4)", "study (no cvxpy)", t, max_abs_diff=float(diff_p))

    out = dict(environment=env, donors=donors, rows=rows)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print("\n| panel | task | replication | synthx.estimate | study (no cvxpy) | speed-up vs replication |")
    print("|---|---|---:|---:|---:|---:|")
    for panel in panels:
        for task in dict.fromkeys(r["task"] for r in rows if r["panel"] == panel):
            cell = {r["impl"]: r for r in rows if r["panel"] == panel and r["task"] == task}
            rep, sxr, ours = (cell.get(k) for k in ("replication", "synthx.estimate", "study (no cvxpy)"))
            fmt = lambda r: f"{r['median']:.4f} s" if r else "--"  # noqa: E731
            ratio = rep["median"] / ours["median"]
            speed = f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"
            print(f"| {panel} | {task} | {fmt(rep)} | {fmt(sxr)} | {fmt(ours)} | {speed} |")


if __name__ == "__main__":
    main()
