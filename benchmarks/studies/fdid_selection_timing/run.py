"""What Li's forward selection costs, in four implementations, on two panels.

Forward DiD's cost is its selection: the search adds one donor at a time and
scores every remaining candidate at every step. Li's released R and MATLAB both
rebuild each candidate's donor average from scratch, which is ``O(N^3 T0)`` over
the whole path. mlsynth keeps a running centred sum and caches each donor's
squared norm and its cross-product with the treated unit, so a step is one matvec
and the path is ``O(N^2 T0)``.

That is a claim about an asymptotic, so it is measured on two panels an order
apart in donor count, and correctness is established before any timing is
reported: all four implementations must select the same donors in the same order.
A faster answer to a different question is not faster.

Timings are machine-dependent, so nothing here is a pinned benchmark value. The
study records the ratio and the machine.

    python -m benchmarks.studies.fdid_selection_timing.run
    python -m benchmarks.studies.fdid_selection_timing.run --reps 10
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

PANELS = {
    "hcw_hongkong": dict(
        csv="basedata/HongKong.csv", unit="Country", time="Time",
        outcome="GDP", treat="Integration",
        note="Hsiao, Ching and Wan (2012); Li's own released companion panel"),
    "fspda_china_watches": dict(
        csv="basedata/china_watches_long.csv", unit="unit", time="time",
        outcome="y", treat="treat",
        note="Shi and Huang fsPDA anti-corruption panel, p > n"),
}


def _wide(spec: dict):
    """Treated column first, then the donors, in mlsynth's donor order."""
    df = pd.read_csv(ROOT / spec["csv"])
    from mlsynth.utils.datautils import dataprep
    p = dataprep(df, spec["unit"], spec["time"], spec["outcome"], spec["treat"])
    y = np.asarray(p["y"], dtype=float)
    X = np.asarray(p["donor_matrix"], dtype=float)
    return y, X, int(p["pre_periods"]), list(p["donor_names"])


def _naive_select(y: np.ndarray, X: np.ndarray, T0: int):
    """Li's algorithm as she writes it: rebuild every candidate average.

    This is the reference cost, in Python, so the comparison against mlsynth
    isolates the algorithm from the language.
    """
    y1 = y[:T0]
    denom = float(((y1 - y1.mean()) ** 2).mean())
    N = X.shape[1]

    def r2(cols):
        xb = X[:T0, cols].mean(axis=1)
        resid = y1 - (float((y1 - xb).mean()) + xb)
        return 1.0 - float((resid ** 2).mean()) / denom

    sel, path = [], []
    remaining = list(range(N))
    while remaining:
        scores = [r2(sel + [j]) for j in remaining]
        k = int(np.argmax(scores))
        sel.append(remaining.pop(k))
        path.append(scores[k])
    return sel[:int(np.argmax(path)) + 1], sel


def _mlsynth_select(y: np.ndarray, X: np.ndarray, T0: int, names):
    from mlsynth.utils.fdid_helpers.estimation import forward_did_select
    res = forward_did_select(y, X, T0, names)["FDID"]
    return list(res["selected_controls"])


def _time(fn, reps: int):
    fn()                                    # once, untimed, to warm any import
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return median(out), min(out)


def _external(cmd, cwd=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or ROOT,
                          timeout=3600)
    if proc.returncode != 0:
        return None, proc.stderr.strip()[-300:]
    got = {}
    for line in proc.stdout.splitlines():
        f = line.split("\t")
        if len(f) == 2:
            got[f[0]] = f[1]
    return got, None


def run(reps: int = 5) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        return _run(reps, Path(tmpdir))


def _run(reps: int, tmpdir: Path) -> dict:
    out = {"machine": {"platform": platform.platform(),
                       "processor": platform.processor() or "unknown",
                       "python": platform.python_version()},
           "reps": reps, "panels": {}}

    for name, spec in PANELS.items():
        y, X, T0, names = _wide(spec)
        rec = {"donors": X.shape[1], "pre_periods": T0,
               "total_periods": len(y), "note": spec["note"], "impls": {}}

        ml_sel = _mlsynth_select(y, X, T0, names)
        nv_sel, _ = _naive_select(y, X, T0)
        rec["impls"]["mlsynth"] = dict(zip(
            ("median_seconds", "min_seconds"),
            _time(lambda: _mlsynth_select(y, X, T0, names), reps)))
        rec["impls"]["mlsynth"]["selected"] = [int(i) for i in ml_sel]
        rec["impls"]["python_naive"] = dict(zip(
            ("median_seconds", "min_seconds"),
            _time(lambda: _naive_select(y, X, T0), reps)))
        rec["impls"]["python_naive"]["selected"] = [int(i) for i in nv_sel]

        # The two external implementations read a treated-first CSV. It is derived
        # from basedata on every run, so it goes to a tempdir and not into the
        # repository.
        tmp = tmpdir / f"{name}_wide.csv"
        pd.DataFrame(np.column_stack([y, X]),
                     columns=["treated"] + [str(n) for n in names]
                     ).to_csv(tmp, index=False)

        if shutil.which("Rscript"):
            got, err = _external(["Rscript", str(HERE / "time_selection.R"),
                                  str(tmp), str(T0), str(reps)])
            rec["impls"]["R_li"] = (
                {"error": err} if got is None else
                {"median_seconds": float(got["median_seconds"]),
                 "min_seconds": float(got["min_seconds"]),
                 "selected": [int(v) - 1 for v in got["selected"].split(",")]})
        else:
            rec["impls"]["R_li"] = {"error": "Rscript not on PATH"}

        octave = shutil.which("octave-cli") or shutil.which("octave")
        if octave:
            got, err = _external([octave, "--no-gui",
                                  str(HERE / "time_selection.m"),
                                  "--data", str(tmp), "--t1", str(T0),
                                  "--reps", str(reps)])
            rec["impls"]["matlab_li_octave"] = (
                {"error": err} if got is None else
                {"median_seconds": float(got["median_seconds"]),
                 "min_seconds": float(got["min_seconds"]),
                 "selected": [int(v) - 1 for v in got["selected"].split(",")]})
        else:
            rec["impls"]["matlab_li_octave"] = {"error": "octave not on PATH"}

        # correctness first: every implementation that ran must agree
        sels = {k: v["selected"] for k, v in rec["impls"].items() if "selected" in v}
        ref = sels["mlsynth"]
        rec["all_select_the_same_donors"] = all(v == ref for v in sels.values())
        rec["disagreements"] = {k: v for k, v in sels.items() if v != ref}
        out["panels"][name] = rec

    return out


def scaling(reps: int = 3, sizes=(20, 40, 80, 160, 320),
            solo=(320, 640, 1280, 2560), T0: int = 40) -> dict:
    """Fit the exponents, so the complexity claim is measured and not asserted.

    Two sweeps, because one cannot answer both questions. Over the shared range
    both implementations run, and what is fitted there is the *ratio*: if they are
    ``O(N^2 T0)`` and ``O(N^3 T0)`` the ratio grows like ``N``, a log-log slope of
    1. The naive version is cubic, so it cannot be pushed far enough for
    mlsynth's own exponent to emerge above a fixed per-call overhead of about a
    millisecond -- at 20 and 40 donors mlsynth's time is overhead, and a slope
    fitted there reads near zero whatever the algorithm is. So mlsynth is swept
    alone to 2560 donors, where the overhead is a rounding error, and its own
    slope is fitted on that.
    """
    rng = np.random.default_rng(0)

    def panel(N):
        T = T0 + 10
        f = rng.standard_normal((T, 3))
        lam = rng.standard_normal((N, 3))
        X = 10.0 + f @ lam.T + 0.3 * rng.standard_normal((T, N))
        y = 10.0 + f @ rng.standard_normal(3) + 0.3 * rng.standard_normal(T)
        return y, X, [f"d{j}" for j in range(N)]

    rows = []
    for N in sizes:
        y, X, names = panel(N)
        ml = _time(lambda: _mlsynth_select(y, X, T0, names), reps)[0]
        nv = _time(lambda: _naive_select(y, X, T0), reps)[0]
        rows.append({"donors": N, "mlsynth_seconds": ml,
                     "python_naive_seconds": nv, "ratio": nv / ml})

    solo_rows = []
    for N in solo:
        y, X, names = panel(N)
        solo_rows.append({"donors": N,
                          "mlsynth_seconds": _time(
                              lambda: _mlsynth_select(y, X, T0, names), reps)[0]})

    slope = lambda xs, ys: float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

    def local(rows, key):
        """Slope between each adjacent pair, which is where the crossover shows."""
        out = []
        for a, b in zip(rows, rows[1:]):
            out.append({"from": a["donors"], "to": b["donors"],
                        "slope": float(np.log(b[key] / a[key])
                                       / np.log(b["donors"] / a["donors"]))})
        return out

    return {
        "shared_range": rows,
        "ratio_log_log_slope": slope([r["donors"] for r in rows],
                                     [r["ratio"] for r in rows]),
        "naive_log_log_slope": slope([r["donors"] for r in rows],
                                     [r["python_naive_seconds"] for r in rows]),
        "mlsynth_only": solo_rows,
        "mlsynth_log_log_slope_large_N": slope(
            [r["donors"] for r in solo_rows],
            [r["mlsynth_seconds"] for r in solo_rows]),
        "mlsynth_local_slopes": local(solo_rows, "mlsynth_seconds"),
        "naive_local_slopes": local(rows, "python_naive_seconds"),
        "T0": T0, "reps": reps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--scaling", action="store_true",
                    help="also fit the complexity exponent over donor counts")
    ap.add_argument("--out", default=str(HERE / "results" / "timings.json"))
    a = ap.parse_args()

    res = run(a.reps)
    if a.scaling:
        res["scaling"] = scaling(max(a.reps, 3))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")

    print(f"machine: {res['machine']['platform']}  reps={res['reps']}\n")
    for name, rec in res["panels"].items():
        print(f"{name}: {rec['donors']} donors, T0={rec['pre_periods']}")
        print(f"  all implementations select the same donors: "
              f"{rec['all_select_the_same_donors']}")
        if rec["disagreements"]:
            print(f"  DISAGREE: {list(rec['disagreements'])}")
        base = rec["impls"]["mlsynth"]["median_seconds"]
        print(f"  {'implementation':26} {'median s':>11} {'min s':>11} {'vs mlsynth':>11}")
        for impl, d in rec["impls"].items():
            if "error" in d:
                print(f"  {impl:26} {'skipped':>11}   {d['error'][:40]}")
                continue
            print(f"  {impl:26} {d['median_seconds']:11.4f} "
                  f"{d['min_seconds']:11.4f} {d['median_seconds'] / base:10.1f}x")
        print()
    if "scaling" in res:
        sc = res["scaling"]
        print(f"complexity, synthetic panels at T0={sc['T0']}")
        print(f"  {'donors':>7} {'mlsynth s':>11} {'naive s':>11} {'ratio':>8}")
        for r in sc["shared_range"]:
            print(f"  {r['donors']:7d} {r['mlsynth_seconds']:11.5f} "
                  f"{r['python_naive_seconds']:11.5f} {r['ratio']:7.1f}x")
        print(f"  ratio log-log slope {sc['ratio_log_log_slope']:.2f} "
              f"(O(N) predicts 1.00); naive {sc['naive_log_log_slope']:.2f} "
              f"(O(N^3) predicts 3.00)")
        print(f"  mlsynth alone, where the per-call overhead is negligible:")
        for r in sc["mlsynth_only"]:
            print(f"  {r['donors']:7d} {r['mlsynth_seconds']:11.5f}")
        print(f"  mlsynth log-log slope {sc['mlsynth_log_log_slope_large_N']:.2f} "
              f"(O(N^2) predicts 2.00)")
        print("  local slopes, mlsynth: " + "  ".join(
            f"{d['from']}->{d['to']} {d['slope']:.2f}"
            for d in sc["mlsynth_local_slopes"]))
        print("  local slopes, naive  : " + "  ".join(
            f"{d['from']}->{d['to']} {d['slope']:.2f}"
            for d in sc["naive_local_slopes"]) + "\n")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
