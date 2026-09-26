"""ADID against the authors' own MATLAB, run live under Octave.

Reference: Kathleen T. Li and Christophe Van den Bulte (2022), "Augmented
Difference-in-Differences", *Marketing Science*, DOI 10.1287/mksc.2022.1406.
Equation 2.4 is the estimator, Proposition 3.1 the limit distribution, and
Appendix A.1 the variance estimator.

This is a replication spike, not an estimator and not a pinned case. It answers
one question before any build starts: does a port that ingests through
``dataprep`` reproduce the authors' arithmetic cell for cell?

Run::

    python -m benchmarks.studies.adid_replicate.run
    python -m benchmarks.studies.adid_replicate.run --t1 90
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PANEL = ROOT / "basedata" / "adid_showroom.csv"
VENDOR = ROOT / "benchmarks" / "reference" / "adid_showroom" / "showroom_generated.csv"
OCTAVE = ROOT / "benchmarks" / "octave" / "adid_showroom.m"
T1_DEFAULT = 83          # the .m's own default, city = 1


def _inputs(t1: int):
    """The panel through ``dataprep``, which is the only pandas touchpoint.

    The shipped panel marks the treatment at the script's own default split. A
    different split is expressed by rebuilding the indicator here and handing
    ``dataprep`` the same frame, so the pivot is still its job and not ours.
    """
    from mlsynth.utils.datautils import dataprep

    df = pd.read_csv(PANEL)
    df = df.assign(showroom=((df["unit"] == "Treated") & (df["week"] > t1)).astype(int))
    prepped = dataprep(df, "unit", "week", "sales", "showroom")
    return (np.asarray(prepped["y"], dtype=float).ravel(),
            np.asarray(prepped["donor_matrix"], dtype=float),
            int(prepped["pre_periods"]))


def adid(y: np.ndarray, Yco: np.ndarray, T1: int) -> Dict[str, object]:
    """Equation 2.4 and Appendix A.1, in the serially uncorrelated branch.

    The design is ``x_t = (1, mean of the controls at t)``, fit by least squares
    on the pre-period. DID is the same construction with the slope held at one,
    which is what makes ADID a variant of it and not a separate estimator.
    """
    T = y.size
    T2 = T - T1
    ybar = Yco.mean(axis=1)
    X = np.column_stack([np.ones(T), ybar])

    delta = np.linalg.solve(X[:T1].T @ X[:T1], X[:T1].T @ y[:T1])
    cf = X @ delta
    att = float(np.mean(y[T1:] - cf[T1:]))

    resid = y[:T1] - cf[:T1]
    sigma2 = float(np.mean(resid ** 2))                      # Appendix A.1 Sigma2
    eta = X[T1:].mean(axis=0)                                # B, up to Psi^-1
    psi = X[:T1].T @ X[:T1] / T1
    omega1 = float(sigma2 * eta @ np.linalg.solve(psi, eta))  # B V B' at V = sigma2 Psi
    omega = (T2 / T1) * omega1 + sigma2                      # Sigma = Sigma1 + Sigma2
    se = float(np.sqrt(omega / T2))

    # DID, the slope-one restriction, with Appendix A.1's variance for it
    intercept = float(np.mean(y[:T1] - ybar[:T1]))
    cf_did = intercept + ybar
    att_did = float(np.mean(y[T1:] - cf_did[T1:]))
    resid_did = y[:T1] - cf_did[:T1]
    sigma2_did = float(np.mean(resid_did ** 2))

    return {
        "t": T, "t1": T1, "t2": T2,
        "delta1": float(delta[0]), "delta2": float(delta[1]),
        "adid_att": att,
        "adid_att_pct": 100.0 * att / float(np.mean(cf[T1:])),
        "adid_sigma2": sigma2, "adid_omega1": omega1, "adid_omega": omega,
        "adid_std_stat": float(np.sqrt(T2) * att / np.sqrt(omega)),
        "adid_se": se,
        "did_intercept": intercept, "did_att": att_did,
        "did_att_pct": 100.0 * att_did / float(np.mean(cf_did[T1:])),
        "did_r2_pre": float(1 - np.mean(resid_did ** 2)
                            / np.mean((y[:T1] - y[:T1].mean()) ** 2)),
        "did_omega": sigma2_did * (T2 / T1 + 1.0),
        "adid_cf": cf, "did_cf": cf_did,
    }


def octave(t1: int) -> Dict[str, object]:
    """Their script's own output, from Octave, parsed into the same keys."""
    proc = subprocess.run(["octave-cli", str(OCTAVE), str(VENDOR), str(t1)],
                          capture_output=True, text=True, timeout=600, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-500:])
    scalars, paths = {}, {"adid_cf": {}, "did_cf": {}}
    for line in proc.stdout.splitlines():
        f = line.split("\t")
        if len(f) == 2:
            scalars[f[0]] = float(f[1])
        elif len(f) == 3 and f[0] in paths:
            paths[f[0]][int(f[1])] = float(f[2])
    for k, v in paths.items():
        scalars[k] = np.array([v[i] for i in sorted(v)])
    return scalars


def compare(t1: int) -> dict:
    y, Yco, pre = _inputs(t1)
    if pre != t1:
        raise AssertionError(f"dataprep put the split at {pre}, not {t1}")
    ours, theirs = adid(y, Yco, t1), octave(t1)

    out = {"t1": t1, "scalars": {}, "paths": {}}
    for k, v in ours.items():
        if isinstance(v, np.ndarray):
            d = float(np.max(np.abs(v - theirs[k])))
            out["paths"][k] = {"max_abs_diff": d,
                               "max_rel_diff": float(d / np.max(np.abs(theirs[k])))}
        elif k in theirs:
            out["scalars"][k] = {"ours": v, "theirs": theirs[k],
                                 "abs_diff": abs(v - theirs[k])}
    # the quantities their script does not print, reported without a comparison
    out["ours_only"] = {k: ours[k] for k in ("adid_se", "did_omega")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1", type=int, default=T1_DEFAULT,
                    help="pre-period length; the script's own default is 83")
    ap.add_argument("--both", action="store_true",
                    help="also report the other split their script offers, 90")
    ap.add_argument("--out", default=str(HERE / "results" / "adid_replicate.json"))
    a = ap.parse_args()

    splits = [a.t1] + ([90] if a.both and a.t1 != 90 else [])
    res = {"machine": {"platform": platform.platform(),
                       "python": platform.python_version()},
           "splits": {}}
    for t1 in splits:
        res["splits"][str(t1)] = compare(t1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float) + "\n")

    for t1, rec in res["splits"].items():
        print(f"\n=== pre-period {t1} ===")
        print(f"  {'quantity':16} {'mlsynth':>22} {'their MATLAB':>22} {'abs diff':>11}")
        for k, v in rec["scalars"].items():
            print(f"  {k:16} {v['ours']:22.12f} {v['theirs']:22.12f} "
                  f"{v['abs_diff']:11.3e}")
        for k, v in rec["paths"].items():
            print(f"  {k+' (path)':16} max abs diff {v['max_abs_diff']:.3e}  "
                  f"relative {v['max_rel_diff']:.3e}")
        print(f"  not printed by their script: "
              + ", ".join(f"{k} {v:.6f}" for k, v in rec["ours_only"].items()))


if __name__ == "__main__":
    main()
