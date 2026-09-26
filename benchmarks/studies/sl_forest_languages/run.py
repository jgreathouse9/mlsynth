"""Where SL's random-forest expert differs between R and scikit-learn.

Reference: Viviano, D. and Bradic, J. (2023), "Synthetic learner: model
evaluation with limited overlap", *Journal of Econometrics* 234(2):691-713, and
their released application package, ``libraries/library.R`` lines 107-116 and
``analysis_main_text/analyze_main_text.R`` line 403.

``docs/replications/sl.rst`` used to say the residual gap between mlsynth's
effect and their Table 4 was "the one member that cannot be matched across
languages". This study measures that claim. It does not pin anything: R's
``randomForest`` and scikit-learn's ``RandomForestRegressor`` are both random, so
every number here is a seed sweep and a spread.

Run::

    python -m benchmarks.studies.sl_forest_languages.run
    python -m benchmarks.studies.sl_forest_languages.run --employment <path>

The second form adds the arm that needs their ``employment_BFRSS.txt``, which is
in their replication package and is not vendored here. Without it the study runs
the language comparison, which needs only ``basedata/``, and reports the
predictor-set arm as skipped.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PANEL = ROOT / "basedata" / "sl_tennessee_medcost.csv"
TRAIN, WIN, T0 = 30, slice(30, 50), 50
POST = slice(51, 88)            # their window, analyze_main_text.R:426
ETA_MLSYNTH = 48.2462327323     # 1/(sqrt(100) var(y)), the paper's formula
ETA_THEIRS = 51.4307            # 1/(sqrt(88) var(y)), the line their script runs
THEIR_EFFECT, THEIR_STATISTIC = 5.2227, 0.6910      # Table 4, m = 0, second half


def _inputs():
    from mlsynth.utils.sl_helpers.setup import prepare_sl_inputs

    df = pd.read_csv(PANEL)
    return prepare_sl_inputs(df, unitid="state", time="quarter",
                             outcome="medcost", treat="expansion",
                             covariates=["employment"])


def _quarterly(monthly: np.ndarray) -> np.ndarray:
    """Their averaging of months into quarters, analyze_main_text.R lines 27-34."""
    return np.stack([monthly[3 * i:3 * i + 3].mean(axis=0) for i in range(100)])


def _sklearn_paths(design, y, *, max_features, min_samples_leaf, seeds):
    from sklearn.ensemble import RandomForestRegressor

    out = np.empty((design.shape[0], seeds))
    for s in range(seeds):
        model = RandomForestRegressor(n_estimators=500, max_leaf_nodes=20,
                                      max_features=max_features,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=s, n_jobs=1)
        model.fit(design[:TRAIN], y[:TRAIN])
        out[:, s] = model.predict(design)
    return out


def _r_paths(tmp: Path, design, y, *, mtry, nodesize, seeds, tag):
    np.savetxt(tmp / "y.csv", y, delimiter=",")
    np.savetxt(tmp / f"d_{tag}.csv", design, delimiter=",")
    proc = subprocess.run(
        ["Rscript", str(HERE / "forests.R"), str(tmp), f"d_{tag}.csv",
         str(mtry), str(nodesize), f"r_{tag}.csv", str(seeds), str(TRAIN)],
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-400:])
    return pd.read_csv(tmp / f"r_{tag}.csv").to_numpy(float)


def _ensemble(P3, y, forest_path, eta):
    """SL's Equation 12 weights and Equation 10 effect, with one column swapped.

    The other three experts are held at mlsynth's, which agree with the R
    reference to 2.3e-12, 7.0e-07 and 4.9e-11, so what moves here is the forest.
    """
    from mlsynth.utils.sl_helpers.weights import exponential_weights

    P = np.column_stack([P3[:, 0], P3[:, 1], forest_path, P3[:, 2]])
    ssr = np.sum((P[WIN] - y[WIN][:, None]) ** 2, axis=0)
    w = exponential_weights(ssr, eta)
    gap = y - P @ w
    g = gap[POST]
    return {"effect": (float(np.mean(g)) - float(np.mean(gap[WIN]))) * 100,
            "statistic": float(np.sum(g ** 2) / np.sqrt(len(g))) * 100,
            "forest_weight": float(w[2]), "forest_ssr": float(ssr[2])}


def _summarise(P3, y, paths, eta):
    rows = [_ensemble(P3, y, paths[:, s], eta) for s in range(paths.shape[1])]
    eff = np.array([r["effect"] for r in rows])
    return {"effect_mean": float(eff.mean()), "effect_sd": float(eff.std()),
            "effect_min": float(eff.min()), "effect_max": float(eff.max()),
            "statistic_mean": float(np.mean([r["statistic"] for r in rows])),
            "forest_weight_mean": float(np.mean([r["forest_weight"] for r in rows])),
            "forest_ssr_mean": float(np.mean([r["forest_ssr"] for r in rows]))}


def _pair_spread(A, B=None):
    """Mean over pairs of the largest per-period difference between two paths."""
    if B is None:
        return float(np.mean([np.max(np.abs(A[:, i] - A[:, j]))
                              for i in range(A.shape[1])
                              for j in range(i + 1, A.shape[1])]))
    return float(np.mean([np.max(np.abs(A[:, i] - B[:, j]))
                          for i in range(A.shape[1]) for j in range(B.shape[1])]))


def run(seeds: int = 30, employment: Path | None = None) -> dict:
    from mlsynth.utils.sl_helpers.experts import build_experts

    inp = _inputs()
    y, X, Z7 = inp.y, inp.Yco, inp.covariates
    P3 = build_experts(X, y, slice(0, TRAIN), ("lasso", "factor", "did")).predictions
    d13 = np.column_stack([X, Z7])

    out = {"machine": {"platform": platform.platform(),
                       "python": platform.python_version()},
           "seeds": seeds, "their_effect": THEIR_EFFECT,
           "their_statistic": THEIR_STATISTIC, "arms": {}, "spreads": {}}

    designs = {"13": d13}
    if employment is not None:
        monthly = pd.read_csv(employment, sep=r"\s+").to_numpy(float)
        designs["57"] = np.column_stack([X, _quarterly(monthly)])
        out["employment_columns"] = int(monthly.shape[1])
    else:
        out["skipped"] = ("the 57-predictor arm needs their employment_BFRSS.txt; "
                          "pass --employment to include it")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, d in designs.items():
            p = d.shape[1]
            arms = {
                # what each side runs on its own defaults
                f"sklearn_{name}_own_defaults": _sklearn_paths(
                    d, y, max_features=1.0, min_samples_leaf=1, seeds=seeds),
                f"R_{name}_own_defaults": _r_paths(
                    tmp, d, y, mtry=-1, nodesize=5, seeds=seeds,
                    tag=f"{name}_own"),
                # and what each runs on the other's settings
                f"sklearn_{name}_R_settings": _sklearn_paths(
                    d, y, max_features=max(p // 3, 1) / p, min_samples_leaf=5,
                    seeds=seeds),
                f"R_{name}_sklearn_settings": _r_paths(
                    tmp, d, y, mtry=p, nodesize=1, seeds=seeds,
                    tag=f"{name}_sk"),
            }
            for tag, paths in arms.items():
                out["arms"][tag] = {
                    "mlsynth_eta": _summarise(P3, y, paths, ETA_MLSYNTH),
                    "their_eta": _summarise(P3, y, paths, ETA_THEIRS)}
            # the question the study exists for: is the language difference
            # bigger than the difference between two seeds of either language?
            for label, a, b in (
                    ("matched_settings", f"R_{name}_sklearn_settings",
                     f"sklearn_{name}_own_defaults"),
                    ("each_on_its_own_defaults", f"R_{name}_own_defaults",
                     f"sklearn_{name}_own_defaults")):
                A, B = arms[a], arms[b]
                out["spreads"][f"{name}_{label}"] = {
                    "within_R": _pair_spread(A), "within_sklearn": _pair_spread(B),
                    "across": _pair_spread(A, B),
                    "mean_paths": float(np.max(np.abs(A.mean(1) - B.mean(1))))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--employment", type=Path, default=None,
                    help="their employment_BFRSS.txt, 300 rows by 51 columns")
    ap.add_argument("--out", default=str(HERE / "results" / "forest_languages.json"))
    a = ap.parse_args()

    res = run(a.seeds, a.employment)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")

    print(f"machine: {res['machine']['platform']}  seeds={res['seeds']}\n")
    print(f"{'arm':34} {'effect':>9} {'sd':>7} {'stat':>7} {'w_forest':>9}")
    for tag, rec in res["arms"].items():
        m = rec["mlsynth_eta"]
        print(f"{tag:34} {m['effect_mean']:9.4f} {m['effect_sd']:7.4f} "
              f"{m['statistic_mean']:7.4f} {m['forest_weight_mean']:9.4f}")
    print("\npath spreads, mean largest per-period difference over seed pairs")
    print(f"{'comparison':34} {'within R':>9} {'within sk':>10} {'across':>9} "
          f"{'mean paths':>11}")
    for tag, s in res["spreads"].items():
        print(f"{tag:34} {s['within_R']:9.5f} {s['within_sklearn']:10.5f} "
              f"{s['across']:9.5f} {s['mean_paths']:11.5f}")
    if "skipped" in res:
        print(f"\nskipped: {res['skipped']}")
    print(f"\ntheirs, Table 4 m=0: effect {THEIR_EFFECT}  statistic {THEIR_STATISTIC}")


if __name__ == "__main__":
    main()
