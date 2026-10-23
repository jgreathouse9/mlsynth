"""Where the time goes on Proposition 99, in both codes.

    python -m benchmarks.studies.as_exact.breakdown   # -> results/breakdown.json

Each implementation's solver calls are wrapped with a timer and a counter, and each
task runs once as a warm-up and then ``reps`` times. Labels can nest (the replication's
seeding LP runs inside its facet enumeration), and a wrapper adds a few microseconds
a call, so the shares are approximate. The cvxpy wrapper is added only for the V
tasks, where it is the only view into the semidefinite programs.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import collections       # noqa: E402
import json              # noqa: E402
import pathlib           # noqa: E402
import time              # noqa: E402
import warnings          # noqa: E402

from . import exact, hull, patterns, rationalize, reference   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TIME, CALLS = collections.defaultdict(float), collections.Counter()


def _wrap(owner, name, label):
    f = getattr(owner, name)

    def timed(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            TIME[label] += time.perf_counter() - t
            CALLS[label] += 1
    setattr(owner, name, timed)


def _profile(fn, reps):
    fn()
    TIME.clear()
    CALLS.clear()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    total = (time.perf_counter() - t) / reps
    return dict(seconds=total, reps=reps,
                parts={k: dict(seconds=TIME[k] / reps, share=TIME[k] / reps / total, calls=CALLS[k] // reps)
                       for k in TIME})


def main():
    paths = reference.fetch()
    ref = reference.load(paths)
    dp, vfs, pw = ref["data_preparation"], ref["visible_facet_solver"], ref["predictor_weights"]
    d = dp.build_california(str(paths["synth-AS-replication"][0] / "data" / "smoking_data.csv"))
    X1, X0, Z1, Z0 = d["X1"], d["X0"], d["Z1"], d["Z0"]
    names = list(dp.CA_PREDICTORS)
    out, stats = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, w, _ = vfs.solve_exterior_hullfree(X0, X1, Z1, Z0)
        _wrap(vfs, "qp_simplex", "cvxpy quadratic programs")
        _wrap(vfs, "linprog", "SciPy LPs")
        _wrap(vfs, "visible_facets", "visible facets")
        _wrap(hull.Hull, "visible_facets", "visible facets (gift wrapping)")
        _wrap(exact, "simplex_ls", "least squares, face fits")
        _wrap(exact, "sign_piece_ls", "least squares, piece fits")
        _wrap(patterns.lsq, "sign_piece_ls", "least squares, piece fits")   # the branch and bound's own
        _wrap(patterns, "linprog", "SciPy LPs")
        out["weights, diagonal: replication"] = _profile(lambda: vfs.solve_exterior_hullfree(X0, X1, Z1, Z0), 3)
        out["weights, diagonal: study"] = _profile(lambda: exact.solve_diag(X1, X0, Z1, Z0, stats=stats), 20)
        out["weights, diagonal: study"]["search"] = dict(stats)

        import cvxpy
        _wrap(cvxpy.Problem, "solve", "cvxpy programs")
        _wrap(pw, "linprog", "SciPy LPs")
        _wrap(rationalize, "linprog", "SciPy LPs")
        _wrap(rationalize, "_newton_eq", "Newton runs")
        out["V, diagonal: replication"] = _profile(lambda: pw.diag_centers(X0, X1, w, names, decimals=None), 10)
        out["V, diagonal: study"] = _profile(lambda: rationalize.diag_rationalizers(X1, X0, w), 10)
        out["V, PD + ranges: replication"] = _profile(
            lambda: (pw.compute_pd_logdet_and_ranges(X1, X0, w, names, decimals=None, ranges=True),
                     pw.analytic_center(X1, X0, w, names, decimals=None)), 5)
        out["V, PD + ranges: study"] = _profile(lambda: rationalize.pd_rationalizers(X1, X0, w, ranges=True), 5)
    out = dict(commits={k: v[1] for k, v in paths.items()}, tasks=out)
    (HERE / "results" / "breakdown.json").write_text(json.dumps(out, indent=1))
    for task, r in out["tasks"].items():
        print(f"{task}: {1e3 * r['seconds']:.1f} ms" + (f"  {r['search']}" if "search" in r else ""))
        for label, p in r["parts"].items():
            print(f"    {label:32s} {1e3 * p['seconds']:8.1f} ms {100 * p['share']:4.0f}%  "
                  f"{p['calls']:4d} calls, {1e3 * p['seconds'] / max(p['calls'], 1):.2f} ms each")


if __name__ == "__main__":
    main()
