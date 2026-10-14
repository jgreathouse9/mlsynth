"""FEct and TWFE against the authors' own run logs, on both of their examples.

Reference: Liu, Licheng, Ye Wang and Yiqing Xu (2022), "A Practical Guide to
Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional
Data", *American Journal of Political Science* 68(1):160-176, DOI
10.1111/ajps.12723. Replication package: AJPS Dataverse, DOI
10.7910/DVN/ZVC9W5, which ships ``fect`` 0.4.1 and the stdout of the authors'
own runs under ``log/``.

This is a replication spike, not an estimator and not a pinned case. Every
target below is a number the authors printed, cited by log file and line, so
nothing here traces to a constant somebody typed.

What it checks, and what it cannot:

* TWFE and FEct point estimates -- reproduced here, in NumPy, no R.
* The bootstrap standard errors, the placebo and equivalence tests, IFEct and
  MC -- NOT reproduced. They need the factor models and a 1000-draw block
  bootstrap. Their targets are recorded in the README so the next spike has
  them, and the gaps are listed as ``unchecked`` in the JSON.

Run::

    python -m benchmarks.studies.fect_replicate.run
    python -m benchmarks.studies.fect_replicate.run --panels hh2015
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .fect_fe import fect_fe, twfe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "basedata"

#: Every expected value is a number the authors printed. ``source`` cites the
#: log file and line in the Dataverse package; ``tol`` is absolute and chosen to
#: be tighter than the rounding they published at, except where noted.
PANELS: Dict[str, Dict[str, Any]] = {
    "hh2015": {
        "file": "hh2015_naturalization.parquet",
        "unit": "bfs", "time": "year",
        "outcome": "nat_rate_ord", "treat": "indirect",
        "min_T0": 5,                       # fect 0.4.1 default, R/default.R:61
        "study": "Hainmueller & Hangartner (2019), Swiss naturalisation",
        "expected": {
            "twfe.coef":        (1.339,   5e-4,  "log_hh2015.txt:3"),
            "twfe.se_hc1":      (0.161,   5e-4,  "log_hh2015.txt:3"),
            "twfe.t":           (8.301,   1e-3,  "log_hh2015.txt:3"),
            "twfe.residual_sd": (4.206,   5e-4,  "log_hh2015.txt:11"),
            "twfe.dof":         (21779,   0,     "log_hh2015.txt:15-22"),
            "fect.att_obs_weighted":  (1.767, 2e-3, "log_hh2015.txt:66"),
            "fect.att_unit_weighted": (1.249, 2e-3, "log_hh2015.txt:67"),
        },
        "unchecked": {
            "fect.se_obs_weighted": (0.2021, "log_hh2015.txt:66 -- 1000 block bootstrap"),
            "placebo.att": (0.1455, "log_hh2015.txt:86"),
            "placebo.p_value": (0.422, "log_hh2015.txt:86"),
            "mc.lambda_norm_star": (1.0, "log_hh2015.txt:100 -- max regularisation"),
            "mc.att_obs_weighted": (1.767, "log_hh2015.txt:112 -- MC collapses onto FEct"),
        },
    },
    "fm2015": {
        "file": "fm2015_council_grants.parquet",
        "unit": "councilnumber", "time": "year",
        "outcome": "logSgwaPercap", "treat": "treat",
        "min_T0": 4,                       # passed explicitly, 3_ex_FM2015.R:47
        "study": "Fouirnaies & Mutlu-Eren (2015), English council grants",
        "expected": {
            "twfe.coef": (-0.015, 5e-4, "log_fm2015.txt:5"),
            # 0.5% and 0.2% off; the script also passes pre.periods=c(-6,0) and
            # group="cohort", neither of which this reimplementation reads.
            "fect.att_obs_weighted":  (-0.03396, 3e-4, "log_fm2015.txt:53"),
            "fect.att_unit_weighted": (-0.07488, 3e-4, "log_fm2015.txt:54"),
        },
        "unchecked": {
            "twfe.residual_sd": (0.403, "log_fm2015.txt:11 -- fastplm's own dof convention"),
            "ife.att_obs_weighted": (0.006397, "log_fm2015.txt:84 -- method='ife', r=2"),
            "mc.att_obs_weighted": (-0.01591, "log_fm2015.txt:129 -- lambda.norm*=0.1778"),
        },
    },
}


def _flat(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {f"{group}.{k}": v for group, d in results.items() for k, v in d.items()}


def check(name: str) -> Dict[str, Any]:
    spec = PANELS[name]
    df = pd.read_parquet(BASE / spec["file"])
    kw = dict(unit=spec["unit"], time=spec["time"],
              outcome=spec["outcome"], treat=spec["treat"])
    got = _flat({"twfe": twfe(df, **kw),
                 "fect": fect_fe(df, **kw, min_T0=spec["min_T0"])})

    rows, worst = [], 0.0
    for key, (want, tol, source) in spec["expected"].items():
        have = got[key]
        diff = abs(float(have) - float(want))
        ok = diff <= tol
        rel = diff / abs(want) if want else diff
        worst = max(worst, rel)
        rows.append({"quantity": key, "expected": want, "got": have,
                     "abs_diff": diff, "tol": tol, "pass": bool(ok),
                     "source": source})
    return {"panel": name, "study": spec["study"], "min_T0": spec["min_T0"],
            "sample": {k: got[f"fect.{k}"] for k in
                       ("n_units", "n_units_dropped", "n_treated_units", "n_treated_obs")},
            "checks": rows,
            "worst_relative_error": worst,
            "all_pass": all(r["pass"] for r in rows),
            "unchecked": {k: {"expected": v[0], "source": v[1]}
                          for k, v in spec["unchecked"].items()}}


def mlsynth_coverage(name: str) -> Dict[str, Any]:
    """Which mlsynth estimator, if any, can fit this panel through ``.fit()``.

    This is the measurement that decides whether the study can ever become a
    benchmark case. A case pins what an estimator's ``.fit()`` returns; nothing
    in this directory does that, because as of this commit nothing in mlsynth
    accepts either panel. The refusals are recorded verbatim so a later FEct
    build starts from the measured obstacle, not a guess.

    FEct is run by the paper as ``method="ife", r=0``, which makes GSYNTH with
    ``r=0`` look like a substitute. It is not: GSYNTH fits its factor space on
    never-treated units alone, which the authors' own vignette gives as the
    difference from ``method="ife"``. MCNNM is the paper's MC estimator and is
    the other candidate.

    Both are tried on the raw estimation sample and on the one ``fect``'s own
    ``min.T0`` rule leaves, since that preprocessing is part of the method.
    """
    import warnings

    from mlsynth import GSYNTH, MCNNM
    from mlsynth.config_models import GSYNTHConfig, MCNNMConfig
    from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

    from .fect_fe import dropped_units

    spec = PANELS[name]
    raw = pd.read_parquet(BASE / spec["file"]).dropna(
        subset=[spec["outcome"], spec["treat"]])
    filtered = raw[~raw[spec["unit"]].isin(
        dropped_units(raw, spec["unit"], spec["treat"], spec["min_T0"]))]

    def attempt(cls, cfg, extra, frame):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cls(cfg(df=frame, unitid=spec["unit"], time=spec["time"],
                              outcome=spec["outcome"], treat=spec["treat"],
                              display_graphs=False, **extra)).fit()
            res = res[0] if isinstance(res, list) else res
            return {"ran": True, "att": float(res.effects.att), "refusal": None}
        except (MlsynthDataError, MlsynthConfigError) as exc:
            return {"ran": False, "att": None, "refusal": str(exc).split(";")[0]}

    out: Dict[str, Any] = {}
    for sample_name, frame in (("estimation sample", raw),
                               (f"after min.T0={spec['min_T0']}", filtered)):
        out[sample_name] = {
            "MCNNM": attempt(MCNNM, MCNNMConfig, {}, frame),
            "GSYNTH(r=0)": attempt(GSYNTH, GSYNTHConfig,
                                   dict(r=0, force="two-way", inference=False), frame),
        }
    out["any_fit_succeeded"] = any(
        v["ran"] for s in out.values() if isinstance(s, dict) and "MCNNM" in s
        for v in s.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", nargs="+", default=list(PANELS))
    ap.add_argument("--out", default=str(HERE / "results" / "fect_replicate.json"))
    args = ap.parse_args()

    out: Dict[str, Any] = {
        "reference": "Liu, Wang & Xu (2022), 10.1111/ajps.12723",
        "replication_package": "10.7910/DVN/ZVC9W5 (ships fect 0.4.1)",
        "targets_from": "the authors' own stdout logs, cited per check",
        "python": platform.python_version(), "numpy": np.__version__,
        "panels": [],
    }
    ok = True
    for name in args.panels:
        rec = check(name)
        rec["mlsynth_today"] = mlsynth_coverage(name)
        out["panels"].append(rec)
        ok &= rec["all_pass"]
        print(f"\n{name}  --  {rec['study']}")
        s = rec["sample"]
        print(f"  min.T0 = {rec['min_T0']}: {s['n_units_dropped']} treated units dropped, "
              f"{s['n_units']} kept, {s['n_treated_obs']} treated observations")
        print(f"  {'quantity':<26}{'authors':>12}{'mine':>14}{'abs diff':>12}  source")
        for r in rec["checks"]:
            mark = " " if r["pass"] else "  <-- FAIL"
            print(f"  {r['quantity']:<26}{r['expected']:>12.5g}{r['got']:>14.6g}"
                  f"{r['abs_diff']:>12.2e}  {r['source']}{mark}")
        print(f"  worst relative error {rec['worst_relative_error']:.2e}; "
              f"{len(rec['unchecked'])} further targets recorded but not checked")
        m = rec["mlsynth_today"]
        print("  mlsynth estimators on this panel:")
        for sample, got in m.items():
            if not isinstance(got, dict) or "MCNNM" not in got:
                continue
            for est, r in got.items():
                verdict = (f"ATT {r['att']:+.5f}" if r["ran"]
                           else f"refused -- {r['refusal']}")
                print(f"    {sample:<22}{est:<14}{verdict}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\n{'all checks pass' if ok else 'SOME CHECKS FAILED'}; wrote {path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
