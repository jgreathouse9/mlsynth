"""Pair exclusion on google/trimmed_match's example geos: nested trimming against the cardinality-optimal pairs.

``python -m benchmarks.studies.supergeo.exclusion_tm <example_data_for_design.csv>`` prints the in-sample table
(every number of pairs dropped, under supergeo_total by the sorted program and under ss_res by the matcher with
dummies) and the rolling held-out table of the README's "Pair exclusion" section. The CSV is the package's
``trimmed_match/example_datasets/example_data_for_design.csv`` (``benchmarks/reference/trimmed_match/install.sh``
clones it).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from mlsynth.utils.pangeo_helpers.matching import min_weight_matching
from . import exclusion

DESIGN_START, DESIGN_END = "2020-01-01", "2020-12-01"
WINDOW_STARTS = ("2020-08-04", "2020-09-01", "2020-09-29", "2020-10-27", "2020-12-02")
WINDOW_DAYS = 28
DROPS = (1, 2, 3, 5, 10, 20, 30, 43, 60, 80)


def panel(df):
    """Geos by dates of the response, geos sorted."""
    return df.pivot(index="geo", columns="date", values="response").sort_index()


def ss_res_costs(Y):
    """PANGEO's ss_res cost of every pair of single markets: the squared distance between the two demeaned
    trajectories (``parallelism.split_cost``), as one matrix."""
    C = Y - Y.mean(1, keepdims=True)
    sq = (C * C).sum(1)
    W = sq[:, None] + sq[None, :] - 2 * C @ C.T
    np.fill_diagonal(W, 0.0)
    return np.maximum((W + W.T) / 2, 0.0)


def in_sample(P, start=DESIGN_START, end=DESIGN_END):
    """For each number of pairs dropped, the nested and re-paired in-sample costs under both objectives, the pairs
    the re-paired design holds outside the full pairing, and the solve times."""
    cols = P.columns
    Y = P.loc[:, (cols >= start) & (cols <= end)].to_numpy(float)
    z, Wss = Y.sum(1), ss_res_costs(Y)
    m = len(z) // 2
    t0 = time.perf_counter()
    opt_tot, pairs_tot = exclusion.sorted_k_pairs(z)
    t_sorted = time.perf_counter() - t0
    nest_tot = exclusion.nested_trim((z[:, None] - z[None, :]) ** 2, pairs_tot[m])[0]
    full_ss = min_weight_matching(Wss)["pairs"]
    nest_ss = exclusion.nested_trim(Wss, full_ss)[0]
    rows, times = [], []
    for drop in range(m):
        k = m - drop
        t0 = time.perf_counter()
        pk, ck = exclusion.matched_k_pairs(Wss, k)
        times.append(time.perf_counter() - t0)
        rows.append(dict(drop=drop, tot_opt=float(opt_tot[k]), tot_nested=float(nest_tot[k]),
                         tot_new=len(set(pairs_tot[k]) - set(pairs_tot[m])), ss_opt=ck,
                         ss_nested=float(nest_ss[k]), ss_new=len(set(pk) - set(full_ss))))
    return dict(rows=rows, t_sorted=t_sorted, t_matcher=(min(times, default=0.0), max(times, default=0.0)))


def window_totals(P, start, days=WINDOW_DAYS):
    """Each geo's total over the design window (DESIGN_START to the day before ``start``) and over the evaluation
    window (``days`` days from ``start``). The two windows share no day."""
    cols = P.columns
    s = pd.Timestamp(start)
    e = s + pd.Timedelta(days=days - 1)
    z = P.loc[:, (cols >= DESIGN_START) & (cols < s)].to_numpy(float).sum(1)
    zt = P.loc[:, (cols >= s) & (cols <= e)].to_numpy(float).sum(1)
    return z, zt


def rolling(P, starts=WINDOW_STARTS, days=WINDOW_DAYS, drops=DROPS):
    """For each window, design on every day from DESIGN_START to the day before it and evaluate the pairs kept on
    the window: the sum over pairs of the squared difference of their window totals (sum D_k^2, the variance of
    the ratio estimator times B^2, Chen et al. 2023 eq. 3), relative to the full pairing's. Returns, per drop, the
    lists over windows for each method, in and out of sample."""
    out = {d: dict(opt=[], nest=[], in_opt=[], in_nest=[]) for d in drops}
    for s in starts:
        z, zt = window_totals(P, s, days)

        def held(ps):
            return float(sum((zt[a] - zt[b]) ** 2 for a, b in ps))

        def ins(ps):
            return float(sum((z[a] - z[b]) ** 2 for a, b in ps))

        full = exclusion.design_excluding(totals=z)["pairs"]
        h0, i0 = held(full), ins(full)
        for d in drops:
            o = exclusion.design_excluding(totals=z, drop_pairs=d)["pairs"]
            t = exclusion.design_excluding(totals=z, drop_pairs=d, method="nested")["pairs"]
            out[d]["opt"].append(held(o) / h0)
            out[d]["nest"].append(held(t) / h0)
            out[d]["in_opt"].append(ins(o) / i0)
            out[d]["in_nest"].append(ins(t) / i0)
    return out


def main(argv):  # pragma: no cover - the script's printing; the two tables are tested through in_sample and rolling
    df = pd.read_csv(argv[0], parse_dates=["date"])
    P = panel(df)
    a = in_sample(P)
    print(f"{len(P)} geos; sorted program {a['t_sorted'] * 1e3:.1f} ms for every k; matcher "
          f"{a['t_matcher'][0] * 1e3:.0f} ms to {a['t_matcher'][1] * 1e3:.0f} ms per k")
    print("drop  total: nested/opt  new pairs   ss_res: nested/opt  new pairs")
    for r in a["rows"]:
        if r["drop"] in (0,) + DROPS[:-1] + (50, 70, 80, 86):
            tr = r["tot_nested"] / r["tot_opt"] if r["tot_opt"] else float("nan")
            sr = r["ss_nested"] / r["ss_opt"] if r["ss_opt"] else float("nan")
            print(f"{r['drop']:4d}  {tr:17.4f}  {r['tot_new']:9d}   {sr:18.4f}  {r['ss_new']:9d}")
    print("drop  in-sample opt / nest   held-out opt [min, max]   held-out nest [min, max]   opt better")
    for d, r in rolling(P).items():
        better = sum(x < y for x, y in zip(r["opt"], r["nest"]))
        print(f"{d:4d}  {np.median(r['in_opt']):.4f} / {np.median(r['in_nest']):.4f}   "
              f"{np.median(r['opt']):.3f} [{min(r['opt']):.2f}, {max(r['opt']):.2f}]   "
              f"{np.median(r['nest']):.3f} [{min(r['nest']):.2f}, {max(r['nest']):.2f}]   "
              f"{better}/{len(r['opt'])}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
