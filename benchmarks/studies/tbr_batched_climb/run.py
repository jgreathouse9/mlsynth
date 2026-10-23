"""The batched climb against the exact one: identical designs, and how long each takes.

Run::

    python -m benchmarks.studies.tbr_batched_climb.run

writes ``results/tbr_batched_climb.json``. Inputs come from TBR's own design
pipeline (``build_inputs`` on a ``TBRConfig``), so the scoring window, the geo
order and the eligibility are the estimator's.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.utils.tbr_helpers.design.pipeline import build_inputs
from mlsynth.utils.tbr_helpers.design.search import greedy_search

from .batched import batched, batched_greedy_search

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results" / "tbr_batched_climb.json"
N_TEST = 14


def geolift() -> pd.DataFrame:
    return pd.read_csv(ROOT / "basedata" / "geolift_test_data.csv").rename(columns={"location": "geo"})


def synthetic(n_geos: int, n_days: int = 118, seed: int = 0) -> pd.DataFrame:
    """Geo series with a shared weekly cycle and drift, scales spread over two orders of magnitude."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    season = 1 + 0.2 * np.sin(2 * np.pi * t / 7) + 0.1 * np.sin(2 * np.pi * t / 60)
    scale = np.exp(rng.normal(8, 1, size=n_geos))
    load = rng.normal(1, 0.3, size=n_geos)
    common = np.cumsum(rng.normal(0, 0.01, size=n_days))
    Y = scale * (season[:, None] ** load + common[:, None] + 0.05 * rng.normal(size=(n_days, n_geos)))
    dates = pd.date_range("2021-01-01", periods=n_days).strftime("%Y-%m-%d")
    return pd.DataFrame({"geo": np.repeat([f"g{j:04d}" for j in range(n_geos)], n_days),
                         "date": np.tile(dates, n_geos), "Y": Y.T.ravel()})


def _outcome(o):
    return (o.k, list(o.treatment), list(o.control), list(o.unassigned), o.score.key,
            [tuple(k) for k in o.trace], o.converged)


def _timed(fn, repeats):
    best, out = np.inf, None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def compare(df, k, start, exact=True, repeats=1):
    cfg = TBRConfig(df=df, unitid="geo", time="date", outcome="Y", max_treatment_size=k,
                    n_test=N_TEST, control_start=start, display_graphs=False)
    Y, units, elig = build_inputs(cfg)
    kw = dict(max_treatment_size=k, n_test=N_TEST, objective="reference", control_start=start)
    t_fast, fast = _timed(lambda: batched_greedy_search(Y, elig, **kw), repeats)
    row = {"geos": len(units), "periods": int(Y.shape[0]), "k": k, "control_start": start,
           "batched_s": round(t_fast, 4), "exact_scores_batched": int(sum(o.evaluations for o in fast))}
    if exact:
        t_exact, slow = _timed(lambda: greedy_search(Y, elig, **kw), repeats)
        row.update(exact_s=round(t_exact, 4), speedup=round(t_exact / t_fast, 1),
                   exact_scores=int(sum(o.evaluations for o in slow)),
                   identical=[_outcome(o) for o in fast] == [_outcome(o) for o in slow])
    return row


def end_to_end(df, k):
    """``TBR(...).fit()`` in searched mode, the user-facing call, exact and batched."""
    cfg = dict(df=df, unitid="geo", time="date", outcome="Y", max_treatment_size=k, n_test=N_TEST,
               display_graphs=False)
    t_exact, a = _timed(lambda: TBR(TBRConfig(**cfg)).fit(), 1)
    with batched():
        t_fast, b = _timed(lambda: TBR(TBRConfig(**cfg)).fit(), 1)
    same = ([(d.k, d.treatment_units, d.control_units) for d in a.designs]
            == [(d.k, d.treatment_units, d.control_units) for d in b.designs])
    return {"geos": df["geo"].nunique(), "k": k, "exact_s": round(t_exact, 3), "batched_s": round(t_fast, 3),
            "speedup": round(t_exact / t_fast, 1), "same_designs": same}


def main():
    panels = {"geolift": geolift(), "synthetic100": synthetic(100), "synthetic200": synthetic(200),
              "synthetic500": synthetic(500)}
    res = {"search": [], "end_to_end": []}
    for name, df in panels.items():
        n = df["geo"].nunique()
        starts = ("carried", "pool", "best") if n <= 100 else ("carried",)
        for k in sorted({3, 4, max(4, n // 4)}):
            for start in starts:
                row = compare(df, k, start, exact=(n <= 200 or k <= 4), repeats=3 if n <= 100 else 1)
                row["panel"] = name
                res["search"].append(row)
                print(json.dumps(row), flush=True)
    for name in ("geolift", "synthetic200"):
        df = panels[name]
        row = end_to_end(df, max(4, df["geo"].nunique() // 4))
        row["panel"] = name
        res["end_to_end"].append(row)
        print(json.dumps(row), flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":  # pragma: no cover - command-line entry; main() is tested directly
    main()
