"""Run the per-unit study's grid and write one CSV row per instance.

    python -m benchmarks.studies.syndes_per_unit_exact.run GROUP [scip=default|proof|none]

Groups:

- ``bls``: the panels of ``benchmarks/cases/syndes_bls.py``, 10 states by 7
  months, drawn with that case's generator; the first 30 draws at K = 3 and 7.
- ``factor``: rank-4 factors plus noise 0.3, T = 30 (the issue's panel), N in
  {20, 40, 60, 100}, K in {3, 5, 10}, seeds 0..2.
- ``noise``: a common shock plus independent noise, no structure, N in {20, 40,
  60}, K in {3, 5, 10}, seeds 0..2. Every unit's root fit is about as good as
  every other's, which is where a bound-ordered search visits the most sets.
- ``real``: GeoLift (40 markets, 90 days), Walmart (45 stores, first 100
  weeks), and BLS county panels of 40, 70 and 100 counties over 36 months; K in
  {3, 5, 10}.
- ``small``: the 20-unit factor and noise panels, for SCIP's proof runs.

``scip=default`` runs the library's MIP at its defaults (60 s, gap 0.05);
``scip=proof`` with no gap limit and 300 s; ``scip=none`` skips it.
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
import pandas as pd  # noqa: E402

from benchmarks.studies.syndes_per_unit_exact.search import (  # noqa: E402
    branch_and_bound, scip, search)
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent / "results"

MAX_SETS = 2_000_000
TIME_LIMIT = 1200.0
ROOT_ONLY_LIMIT = 120.0
SCIP_SETTINGS = {"default": (60.0, 0.05), "proof": (300.0, 0.0)}


def _factor(N, seed, T=30):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 4)) @ rng.standard_normal((4, N))
            + 0.3 * rng.standard_normal((T, N)))


def _noise(N, seed, T=30):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, N)) + rng.normal(size=(T, 1)) * 2.0


def _bls():
    """The syndes_bls case's draws, in its order: per K, seed 0 + K."""
    Y = np.loadtxt(ROOT / "basedata" / "urate_cps.csv", delimiter=",")
    T, N = Y.shape
    for K in (3, 7):
        rng = np.random.default_rng(0 + K)
        for draw in range(30):
            cols = rng.choice(N, size=10, replace=False)
            start = int(rng.integers(0, T - 10 + 1))
            panel = Y[start:start + 10][:, cols]
            rng.choice(10, size=K, replace=False)      # the case's random baseline
            yield f"bls_k{K}_d{draw}", panel[:7], K


def _real():
    from benchmarks.studies.marex_bnb.panels import county, geolift
    g, _ = geolift()
    d = pd.read_csv(ROOT / "basedata" / "walmart_weekly_sales.csv")
    w = d.pivot(index="week", columns="store", values="sales").sort_index()
    panels = {"geolift_n40": g,
              "walmart_n45": w.to_numpy(float)[:100],
              **{f"county_n{n}": county(n)[0] for n in (40, 70, 100)}}
    for name, Y in panels.items():
        for K in (3, 5, 10):
            yield f"{name}_k{K}", Y, K


def instances(group):
    if group == "bls":
        yield from _bls()
    elif group == "factor":
        for N in (20, 40, 60, 100):
            for K in (3, 5, 10):
                for seed in range(3):
                    yield f"factor_n{N}_k{K}_s{seed}", _factor(N, seed), K
    elif group == "noise":
        for N in (20, 40, 60):
            for K in (3, 5, 10):
                for seed in range(3):
                    yield f"noise_n{N}_k{K}_s{seed}", _noise(N, seed), K
    elif group == "real":
        yield from _real()
    elif group == "small":
        for kind, make in (("factor", _factor), ("noise", _noise)):
            for K in (3, 5, 10):
                for seed in range(3):
                    yield f"{kind}_n20_k{K}_s{seed}", make(20, seed), K
    else:
        raise SystemExit(f"unknown group {group!r}")


def _variants(Y, K, lam):
    """The three searches on one instance: the issue's root-bound order, the
    same order pruned on pair bounds, and the branch and bound on units."""
    root = search(Y, K, lam, pair=False, max_sets=MAX_SETS, time_limit=ROOT_ONLY_LIMIT)
    pair = search(Y, K, lam, pair=True, max_sets=MAX_SETS, time_limit=TIME_LIMIT)
    bnb = branch_and_bound(Y, K, lam, time_limit=TIME_LIMIT)
    return root, pair, bnb


def main(argv):
    group = argv[0]
    mode = dict(a.split("=", 1) for a in argv[1:]).get("scip", "default")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{group}_scip-{mode}.csv"
    fields = ["name", "N", "T", "K", "lam", "value", "design", "agree", "root_bound",
              "root_gap", "max_rel_gap",
              "root_closed", "root_popped", "root_qp", "root_t",
              "pair_closed", "pair_popped", "pair_pruned", "pair_qp", "pair_t",
              "bnb_closed", "bnb_nodes", "bnb_qp", "bnb_t",
              "scip_status", "scip_wall", "scip_time", "scip_value", "scip_design",
              "scip_excess"]
    with open(path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=fields)
        out.writeheader()
        for name, Y, K in instances(group):
            Y = np.ascontiguousarray(Y, dtype=float)
            T, N = Y.shape
            lam = float(estimate_lambda(Y))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                root, pair, bnb = _variants(Y, K, lam)
                m = ({} if mode == "none" else
                     scip(Y, K, lam, time_limit=SCIP_SETTINGS[mode][0],
                          gap_limit=SCIP_SETTINGS[mode][1]))
            closed = [r for r in (root, pair, bnb) if r["closed"]]
            best = min(closed, key=lambda r: r["value"]) if closed else bnb
            agree = all(abs(r["value"] - best["value"]) <= 1e-9 * best["value"]
                        for r in closed)
            row = {"name": name, "N": N, "T": T, "K": K, "lam": lam,
                   "value": best["value"], "design": best["design"], "agree": agree,
                   "root_bound": bnb["root_bound"],
                   "root_gap": (best["value"] - bnb["root_bound"]) / best["value"],
                   "max_rel_gap": max(r["max_rel_gap"] for r in (root, pair, bnb)),
                   "root_closed": root["closed"], "root_popped": root["popped"],
                   "root_qp": root["qp"], "root_t": root["t_total"],
                   "pair_closed": pair["closed"], "pair_popped": pair["popped"],
                   "pair_pruned": pair["pair_pruned"], "pair_qp": pair["qp"],
                   "pair_t": pair["t_total"],
                   "bnb_closed": bnb["closed"], "bnb_nodes": bnb["nodes"],
                   "bnb_qp": bnb["qp"], "bnb_t": bnb["t_total"]}
            if m:
                excess = ((m["value"] - best["value"]) / best["value"]
                          if m["design"] else math.nan)
                row.update(scip_status=m["status"], scip_wall=m["wall"],
                           scip_time=m["scip_time"], scip_value=m["value"],
                           scip_design=m["design"], scip_excess=excess)
            out.writerow(row)
            fh.flush()
            line = (f"{name}: gap={row['root_gap']:.4f} agree={agree} "
                    f"root {root['closed']} {root['t_total']:.2f}s | "
                    f"pair {pair['closed']} {pair['t_total']:.2f}s | "
                    f"bnb {bnb['closed']} {bnb['nodes']} nodes {bnb['t_total']:.2f}s")
            if m:
                line += (f" | scip {m['status']} {m['wall']:.1f}s "
                         f"excess={row['scip_excess']:.2e} same={m['design'] == best['design']}")
            print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
