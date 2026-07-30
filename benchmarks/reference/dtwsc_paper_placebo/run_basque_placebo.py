"""Reproduce Cao and Chadefaux's Basque placebo design, run for run.

This is the expensive half of the Path A replication: 1789 placebo runs, each
fitted twice (warped and unwarped) at its own hyperparameters. It writes one row
per run to ``basque_runs.csv``; ``benchmarks/cases/dtwsc_paper_placebo.py`` then
compares the summary against the numbers printed in the paper. Splitting it that
way keeps the benchmark suite cheap -- the fits are hours, the assertion is
milliseconds.

Every design choice below is transcribed from ``Figure_5_1.R`` in the authors'
replication archive rather than inferred from the paper's prose, which describes
the pool construction only loosely. See the README for provenance.

Usage::

    python run_basque_placebo.py                 # all 1789 runs
    python run_basque_placebo.py --limit 20      # smoke test
    python run_basque_placebo.py --patterns symmetricP1,asymmetricP2
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from mlsynth import DTWSC  # noqa: E402

#: Basque Country. The authors' id, from ``Synth::basque``, not our renumbered
#: ``basedata`` copy -- see the README.
TREATED_ID = 17
#: Treatment is marked from 1971 so the warp sees 1955--1970, sixteen periods.
#: The reference's ``t.treat = 1970`` is inclusive (``1:t.treat``) where
#: mlsynth's ``pre_periods`` stops the period before, so the dates differ by one
#: to describe the same window.
TREAT_YEAR = 1971
LAST_PRE = 1970
#: MSE windows, from ``Figure_5_1.R`` lines 185-188 as indices into 1955:1997.
PRE_MSE = (1961, 1970)
POST_MSE = (1971, 1980)

#: The paper's 14 special predictors. The outcome's own 1960--1969 mean is the
#: fourteenth and is passed as ``fit_window``.
COVARIATE_WINDOWS = {
    "invest_ratio": (1964, 1969), "popdens": (1969, 1969),
    "sec.agriculture": (1961, 1969), "sec.energy": (1961, 1969),
    "sec.industry": (1961, 1969), "sec.construction": (1961, 1969),
    "sec.services.venta": (1961, 1969), "sec.services.nonventa": (1961, 1969),
    "school.illit": (1964, 1969), "school.prim": (1964, 1969),
    "school.med": (1964, 1969), "school.high": (1964, 1969),
    "school.post.high": (1964, 1969),
}
FIT_WINDOW = (1960, 1969)

#: Fixed warping arguments, from ``args.TFDTW``. Only ``filter_width``, ``k``
#: and ``step_pattern1`` vary per run.
FIXED = dict(
    buffer=0, match_method="fixed", dist_quant=0.95, n_burn=3, n_iqr=3.0,
    ma=3, default_margin=3, n_q=1, n_r=1, step_pattern2="asymmetricP2",
)


def rescale_panel_like_reference(panel):
    """Map every unit onto the common pre-treatment range, and return the map.

    The reference rescales ONCE on the full 18-unit panel before any pool is
    built, so every placebo run sees the same transform. Doing it per-pool
    instead would give each run a different scale and make the runs
    incommensurable.
    """
    pre = panel[panel.time <= LAST_PRE]
    rng = pre.groupby("unit")["value"].agg(["min", "max"])
    spans = rng["max"] - rng["min"]
    mean_span = float(spans.mean())
    factors = {u: (float(rng["min"][u]), mean_span / float(spans[u]))
               for u in rng.index}
    out = panel.copy()
    out["value"] = [(v - factors[u][0]) * factors[u][1]
                    for v, u in zip(out["value"], out["unit"])]
    return out, factors


def build_datasets(ids):
    """The authors' 119 pools: full panel, 18 single drops, 100 pairs.

    ``combn`` enumerates lexicographically and they keep the first 100 columns,
    so the order is part of the design, not an implementation detail.
    """
    datasets = [set(ids)]
    for i in ids:
        datasets.append(set(ids) - {i, TREATED_ID})
    for pair in list(combinations([i for i in ids if i != TREATED_ID], 2))[:100]:
        datasets.append(set(ids) - set(pair) - {TREATED_ID})
    return datasets


def _window_mse(times, gap, lo, hi):
    m = (times >= lo) & (times <= hi)
    return float(np.nanmean(gap[m] ** 2))


def run_one(panel, dataset_units, target_id, filter_width, k, step_pattern,
            multiplier):
    """Fit one placebo run warped and unwarped; return the two post-period MSEs.

    Gaps are mapped back out of the rescaled units by ``*1000/multiplier``,
    because that is the scale the paper's log MSEs are reported on. Comparing
    against 10.06 without this step compares two different quantities.
    """
    sub = panel[panel.id.isin(dataset_units)].copy()
    sub["treated"] = ((sub.id == target_id) & (sub.time >= TREAT_YEAR)).astype(int)
    present = [c for c in COVARIATE_WINDOWS if c in sub.columns]
    cfg = dict(
        df=sub, outcome="value", treat="treated", unitid="unit", time="time",
        display_graphs=False, sc_backend="mscmt", covariates=present,
        covariate_windows={c: COVARIATE_WINDOWS[c] for c in present},
        fit_window=FIT_WINDOW, filter_width=int(filter_width), k=int(k),
        step_pattern1=step_pattern, **FIXED,
    )
    out = {}
    for tag, warp in (("tfdtw", True), ("raw", False)):
        res = DTWSC({**cfg, "warp": warp}).fit()
        ts = res.time_series
        times = np.asarray(ts.time_periods, dtype=float)
        obs = np.asarray(ts.observed_outcome, dtype=float)
        cf = np.asarray(ts.counterfactual_outcome, dtype=float)
        n = min(times.size, obs.size, cf.size)
        gap = (obs[:n] - cf[:n]) * 1000.0 / multiplier
        out[f"mse_post_{tag}"] = _window_mse(times[:n], gap, *POST_MSE)
        out[f"mse_pre_{tag}"] = _window_mse(times[:n], gap, *PRE_MSE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patterns", default=None,
                    help="comma-separated allow-list, for partial runs")
    ap.add_argument("--out", default=str(_HERE / "basque_runs.csv"))
    args = ap.parse_args()

    panel = pd.read_csv(_HERE / "gold_basque_panel.csv")
    grid = pd.read_csv(_HERE / "gold_gridopt.csv")
    grid = grid[grid.panel == "Figure_5_1"].reset_index(drop=True)
    if args.patterns:
        allow = set(args.patterns.split(","))
        skipped = len(grid) - int(grid["step.pattern"].isin(allow).sum())
        grid = grid[grid["step.pattern"].isin(allow)].reset_index(drop=True)
        print(f"pattern filter: kept {len(grid)}, SKIPPED {skipped} runs "
              f"-- this is a partial design, not the paper's")
    if args.limit:
        grid = grid.head(args.limit)
        print(f"limit: first {len(grid)} runs only -- partial design")

    scaled, factors = rescale_panel_like_reference(panel)
    datasets = build_datasets(sorted(panel.id.unique()))
    mult = {int(i): factors[u][1] for i, u in
            zip(panel.id, panel.unit)}

    rows, t0 = [], time.time()
    for n, r in grid.iterrows():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = run_one(scaled, datasets[int(r["data.id"]) - 1],
                              int(r["id"]), r["filter.width"], r["k"],
                              r["step.pattern"], mult[int(r["id"])])
            rows.append({"unit": r["unit"], "data_id": int(r["data.id"]),
                         "step_pattern": r["step.pattern"],
                         "filter_width": int(r["filter.width"]),
                         "k": int(r["k"]), "status": "ok", **got})
        except Exception as exc:  # a failed run is recorded, never silently lost
            rows.append({"unit": r["unit"], "data_id": int(r["data.id"]),
                         "step_pattern": r["step.pattern"],
                         "filter_width": int(r["filter.width"]),
                         "k": int(r["k"]), "status": f"{type(exc).__name__}: {exc}",
                         "mse_post_tfdtw": np.nan, "mse_post_raw": np.nan,
                         "mse_pre_tfdtw": np.nan, "mse_pre_raw": np.nan})
        if (n + 1) % 10 == 0 or n + 1 == len(grid):
            el = time.time() - t0
            print(f"  {n + 1}/{len(grid)}  {el:.0f}s elapsed, "
                  f"{el / (n + 1):.2f}s/run", flush=True)

    df = pd.DataFrame(rows)
    df["log_ratio"] = np.log(df.mse_post_tfdtw / df.mse_post_raw)
    df.to_csv(args.out, index=False)
    ok = int((df.status == "ok").sum())
    print(f"\nwrote {args.out}: {len(df)} runs, {ok} ok, {len(df) - ok} failed")
    if ok:
        print(f"  mean log(MSE_SC)  = {np.nanmean(np.log(df.mse_post_raw)):.4f}"
              f"   paper 10.06")
        print(f"  mean log(MSE_DSC) = {np.nanmean(np.log(df.mse_post_tfdtw)):.4f}"
              f"   paper  9.88")
        from mlsynth.utils.dtwsc_helpers.inference import icc_corrected_ttest
        good = df[np.isfinite(df.log_ratio)]
        if len(good) > 1:
            st = icc_corrected_ttest(good.log_ratio, good.data_id, good.unit)
            print(f"  t = {st['t']:.4f}  (paper -7.9097),  p = {st['p']:.3g},"
                  f"  df = {st['df']:.1f},  ICC = {st['icc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
