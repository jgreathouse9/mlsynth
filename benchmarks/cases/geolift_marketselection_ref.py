"""GEOLIFT live cross-check vs GeoLiftMarketSelection: the BestMarkets ranking.

Cross-validation against the *running* reference. ``geolift_marketselection`` pins
the public ``GEOLIFT`` selection against GeoLift's *published* BestMarkets table;
this case goes further and checks it against **live GeoLift** -- the real
``GeoLiftMarketSelection`` (``facebookincubator/GeoLift``) on the identical
PreTest panel, the gold-standard cross-validation.

It shells out to ``benchmarks/R/geolift_marketselection.R`` (install the
reference once with ``benchmarks/R/install_geolift.sh``) and compares GeoLift's
ranked BestMarkets table to mlsynth's pooled N=2-5 selection. On the stable
top-five designs mlsynth matches live GeoLift exactly: same candidate sets, same
rank, the CPIC investment to the cent, and the MDE. (This is what licensed the
``include_markets`` generate-then-filter fix: live GeoLift surfaces the same
candidate sets -- including the low-correlation ones like
``{atlanta, chicago, cleveland, las vegas}`` -- that mlsynth now generates.)

The marginal rank-six design (an N=5 set at the es=0.05 / budget boundary) is
*not* pinned: with ``lookback_window=1`` the power is a single-placement binary,
and that design's flush-placement conformal p sits right at ``alpha`` (mlsynth
0.123 vs GeoLift just under), so it tips one way for GeoLift and the other for
mlsynth -- a known small-effect power-methodology difference, not a selection
discrepancy (its true power over placements is ~0.8).

The reference is version-pinned (the install script freezes every source package
to a CRAN tag / commit). Skips itself (``BenchmarkSkipped``) when ``Rscript`` or
the GeoLift package is absent, so it is a no-op in CI and runs only where the
reference is installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings

import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from mlsynth import GEOLIFT

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA = os.path.join(_ROOT, "basedata", "geolift_market_data.csv")  # 90-period PreTest
_RSCRIPT_REF = os.path.join(_ROOT, "benchmarks", "R", "geolift_marketselection.R")


def _geolift_reference() -> pd.DataFrame:
    """Run GeoLiftMarketSelection via Rscript and parse its BestMarkets ROWs."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (run benchmarks/R/install_geolift.sh)")
    probe = subprocess.run([rscript, "-e", "suppressMessages(library(GeoLift))"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package 'GeoLift' not installed")
    out = subprocess.run([rscript, _RSCRIPT_REF, _DATA], capture_output=True, text=True)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"GeoLift reference failed: {out.stderr.strip()[-200:]}")

    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] != "ROW":
            continue
        _, loc, dur, es, power, sl2, inv, az, rank = parts
        rows.append({
            "cand": frozenset(loc.split("|")), "duration": int(dur),
            "EffectSize": float(es), "investment": float(inv), "rank": int(rank),
        })
    if not rows:
        raise BenchmarkSkipped("could not parse GeoLift BestMarkets output")
    return pd.DataFrame(rows)


def _mlsynth_pool() -> pd.DataFrame:
    """mlsynth's pooled N=2-5 selection with GeoLift's composite rank."""
    df = pd.read_csv(_DATA)

    def fit(N):
        cfg = dict(df=df, outcome="Y", unitid="location", time="date",
                   treatment_size=N, to_be_treated=["chicago"],
                   not_to_be_treated=["honolulu"], durations=[10, 15],
                   effect_sizes=[0.0, 0.05, 0.10, 0.15, 0.20], lookback_window=1,
                   how="sum", augment="ridge", fixed_effects=True, alpha=0.1,
                   power_threshold=0.8, cpic=7.5, budget=1e5, ns=1000, seed=0,
                   conformal_type="iid", display_graphs=False, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return GEOLIFT(cfg).fit().power

    pool = pd.concat([fit(N) for N in (2, 3, 4, 5)], ignore_index=True)
    pool["cand"] = pool["candidate"].apply(frozenset)
    pool["rank_mde"] = pool["mde"].abs().rank(method="dense")
    pool["rank_pvalue"] = pool["power"].rank(method="dense")
    pool["rank_abszero"] = pool["abs_lift_in_zero"].rank(method="dense")
    pool["rank"] = pool[["rank_mde", "rank_pvalue", "rank_abszero"]].mean(
        axis=1).rank(method="min")
    return pool


def run() -> dict:
    ref = _geolift_reference()
    pool = _mlsynth_pool()

    top5 = ref[ref["rank"] <= 5]
    inv_diffs, rank_diffs, mde_diffs = [], [], []
    n_present = 0
    for _, g in top5.iterrows():
        m = pool[(pool["cand"] == g["cand"]) & (pool["duration"] == g["duration"])]
        if m.empty:
            continue
        n_present += 1
        m = m.iloc[0]
        inv_diffs.append(abs(m["investment"] - g["investment"]))
        rank_diffs.append(abs(m["rank"] - g["rank"]))
        mde_diffs.append(abs(m["mde"] - g["EffectSize"]))

    return {
        "top5_designs_present": float(n_present == len(top5)),
        "top5_investment_max_abs_diff": float(max(inv_diffs)) if inv_diffs else float("inf"),
        "top5_rank_max_abs_diff": float(max(rank_diffs)) if rank_diffs else float("inf"),
        "top5_mde_max_abs_diff": float(max(mde_diffs)) if mde_diffs else float("inf"),
    }


# mlsynth reproduces live GeoLift's BestMarkets top-five exactly: every design
# present, investment to the cent, identical rank, identical MDE.
EXPECTED = {
    "top5_designs_present": (1.0, 0.5),
    "top5_investment_max_abs_diff": (0.0, 1.0),
    "top5_rank_max_abs_diff": (0.0, 0.5),
    "top5_mde_max_abs_diff": (0.0, 0.001),
}
