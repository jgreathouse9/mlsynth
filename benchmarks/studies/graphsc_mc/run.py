"""Classic SC against the Graph-SC arms, plain and CWZ-debiased, over four simulation families.

Run::

    python -m benchmarks.studies.graphsc_mc.run                 # every family, 4 workers
    python -m benchmarks.studies.graphsc_mc.run --family hz --workers 2

writes ``results/graphsc_mc.json``. Replications are independent, so they run
in a process pool, one BLAS thread per worker.

For every arm and cell the plain estimate is the mean post-period gap of the
weights fit on the whole pre-period; the debiased estimate and its interval are
``mlsynth.utils.inferutils.debiased_sc_ttest`` with the arm as ``weight_fn``
(Chernozhukov, Wuthrich & Zhu 2025: ``K`` blocks at the end of the pre-period,
each refit without its block, the held-out gap subtracted, a ``t_{K-1}``
interval). The oracle row is the same t-test on the true weights.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import designs as D
from . import methods as M

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "graphsc_mc.json"
SEED = 20261023
ALPHA = 0.1
BOOT = 2000


def evaluate(d: D.Draw, K: int, alpha: float, seed: int, methods=M.METHODS) -> dict:
    """Every arm on one draw: plain and debiased errors, coverage, interval length, weight quality."""
    from scipy.stats import t as student_t

    from benchmarks.cwz_common import oracle_ttest
    from mlsynth.utils.inferutils import debiased_sc_ttest

    out = {"plain": {}, "debiased": {}, "oracle": None, "uncertified": 0, "fallbacks": 0}
    pre, post = np.arange(d.T0), slice(d.T0, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for m in methods:
            fn = M.weight_fn(m, d.y, d.Y0, d.T0, seed=seed)
            w = fn(pre)
            row = {"err": float(np.mean(d.y[post] - d.Y0[post] @ w)) - d.tau}
            if d.target is not None:
                row["sse"] = float(np.sum((d.target[post] - d.Y0[post] @ w) ** 2))
            if d.risk is not None:
                row["risk_ratio"] = d.risk(w) / d.risk_min
            out["plain"][m] = row
            r = debiased_sc_ttest(d.y, d.Y0, d.T0, d.T1, K=K, alpha=alpha, weight_fn=fn)
            out["debiased"][m] = {"err": r["att"] - d.tau, "length": r["ci_upper"] - r["ci_lower"],
                                  "covered": bool(r["ci_lower"] <= d.tau <= r["ci_upper"])}
            out["fallbacks"] += fn.stats["fallback"]
        if d.w_true is not None:
            att, se = oracle_ttest(d.y, d.Y0, d.T0, d.T1, d.w_true, K)
            crit = float(student_t.ppf(1 - alpha / 2, K - 1))
            out["oracle"] = {"err": att - d.tau, "length": 2 * crit * se,
                             "covered": bool(abs(att - d.tau) <= crit * se)}
    out["uncertified"] = count_uncertified(caught)
    return out


def count_uncertified(caught) -> int:
    """Weight vectors returned without a certificate; other warnings are not solver failures."""
    from .solver import CertificateWarning

    return sum(issubclass(c.category, CertificateWarning) for c in caught)


def _ratio_ci(num, den, rng):
    """Paired bootstrap 95 percent interval for ``mean(num) / mean(den)``."""
    idx = rng.integers(0, len(num), size=(BOOT, len(num)))
    r = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    return [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def summarise(reps, methods=M.METHODS) -> dict:
    """Bias, RMSE and the MSE ratio to SC per arm; coverage and length for the debiased rows."""
    rng = np.random.default_rng(SEED)
    out = {"n": len(reps), "uncertified": int(sum(r["uncertified"] for r in reps)),
           "fallbacks": int(sum(r.get("fallbacks", 0) for r in reps))}
    for kind in ("plain", "debiased"):
        out[kind] = {}
        sc_err = np.array([r[kind]["SC"]["err"] for r in reps]) if all("SC" in r[kind] for r in reps) else None
        for m in methods:
            rows = [r[kind][m] for r in reps if m in r[kind]]
            if not rows:
                continue
            e = np.array([x["err"] for x in rows])
            s = {"bias": float(e.mean()), "sd": float(e.std(ddof=1)) if e.size > 1 else float("nan"),
                 "rmse": float(np.sqrt(np.mean(e ** 2)))}
            if sc_err is not None and m != "SC" and np.mean(sc_err ** 2) > 0:
                s["mse_ratio"] = float(np.mean(e ** 2) / np.mean(sc_err ** 2))
                s["mse_ratio_ci"] = _ratio_ci(e ** 2, sc_err ** 2, rng)
            if "covered" in rows[0]:
                c = np.array([x["covered"] for x in rows], dtype=float)
                s["coverage"] = float(c.mean())
                s["coverage_se"] = float(np.sqrt(c.mean() * (1 - c.mean()) / c.size))
                s["length"] = float(np.mean([x["length"] for x in rows]))
            if "sse" in rows[0] and kind == "plain":
                ratios = [r[kind][m]["sse"] / r[kind]["SC"]["sse"] for r in reps
                          if m in r[kind] and r[kind]["SC"]["sse"] > 0]
                s["sse_ratio_median"] = float(np.median(ratios))
            if "risk_ratio" in rows[0]:
                s["risk_ratio_mean"] = float(np.mean([x["risk_ratio"] for x in rows]))
            out[kind][m] = s
    orc = [r["oracle"] for r in reps if r["oracle"] is not None]
    if orc:
        e = np.array([x["err"] for x in orc])
        c = np.array([x["covered"] for x in orc], dtype=float)
        out["oracle"] = {"bias": float(e.mean()), "rmse": float(np.sqrt(np.mean(e ** 2))),
                         "coverage": float(c.mean()), "coverage_se": float(np.sqrt(c.mean() * (1 - c.mean()) / c.size)),
                         "length": float(np.mean([x["length"] for x in orc]))}
    return out


def one_rep(task):
    family, ci, rep = task
    spec = D.FAMILIES[family]
    rng = np.random.default_rng([SEED, list(D.FAMILIES).index(family), ci, rep])
    d = D.draw(family, spec["cells"][ci], rng)
    return family, ci, evaluate(d, K=spec["K"], alpha=ALPHA, seed=int(rng.integers(2**62)))


def _init_worker():
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def main(families=None, workers=4, reps=None, dump=None) -> dict:
    families = families or list(D.FAMILIES)
    tasks = [(f, ci, r) for f in families for ci in range(len(D.FAMILIES[f]["cells"]))
             for r in range(reps or D.FAMILIES[f]["reps"])]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    collected = {(f, ci): [] for f in families for ci in range(len(D.FAMILIES[f]["cells"]))}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for i, (f, ci, out) in enumerate(pool.map(one_rep, tasks, chunksize=4)):
            collected[(f, ci)].append(out)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(tasks)} replications, {time.perf_counter() - t0:.0f} s", flush=True)
    if dump:
        import pickle

        Path(dump).write_bytes(pickle.dumps(collected))
    for f in families:
        res[f] = {"label": D.FAMILIES[f]["label"], "K": D.FAMILIES[f]["K"], "alpha": ALPHA,
                  "cells": [{"cell": D.FAMILIES[f]["cells"][ci], **summarise(collected[(f, ci)])}
                            for ci in range(len(D.FAMILIES[f]["cells"]))]}
    res["_meta"] = {"seed": SEED, "lambdas": M.LAM, "knn": M.KNN, "methods": M.METHODS,
                    "seconds_last_run": round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", choices=sorted(D.FAMILIES))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--reps", type=int)
    ap.add_argument("--dump", help="pickle every replication's record to this path")
    a = ap.parse_args()
    main(a.family, a.workers, a.reps, a.dump)
