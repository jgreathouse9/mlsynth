"""Re-derive every number this bundle reports, from the committed artifacts.

Written after a stale `coverage_results.json` -- the 10-replication smoke run --
was briefly staged in place of the 200-replication run, because the gate that
was supposed to wait for the real run tested only that the file existed and a
file from the earlier run already did. Both runs write the same filename, so the
filesystem could not tell them apart and nothing downstream checked.

The fix is that artifacts now carry their own run identity and this script
refuses any that disagrees with `manifest.json`. Run it before trusting or
committing anything here:

    python benchmarks/reference/gpits_heller/verify.py

Exits non-zero on the first mismatch. `--tamper` flips a value to prove the
check has teeth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpits_port import gp_its, gp_its_placebo  # noqa: E402

RTOL = 1e-8          # port vs gpss agree to ~1e-11; this leaves headroom
PANEL_ATOL = 1e-6    # regenerated panel vs committed panel
failures: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def fingerprint():
    h = hashlib.sha256()
    for n in ("gpits_port.py", "run_coverage.py"):
        h.update((HERE / n).read_bytes())
    return h.hexdigest()[:16]


def load_dc():
    dc = pd.read_csv(HERE / "dc_series.csv")
    dc["date"] = pd.to_datetime(dc["date"])
    n_pre = int((dc["date"] < "2008-07-01").sum())
    return dc, dc["date"].dt.strftime("%m").values, n_pre


def main(tamper=False):
    man = json.load(open(HERE / "manifest.json"))
    print(f"verifying {man['case']}\n")

    print("provenance")
    cov = json.load(open(HERE / "coverage_results.json"))
    meta, rows = cov["meta"], cov["rows"]
    if tamper:
        meta["n_reps"] = 10
    exp = man["simulation"]
    check("coverage n_reps matches manifest", meta["n_reps"] == exp["n_reps"],
          f"artifact={meta['n_reps']} manifest={exp['n_reps']}")
    check("coverage grid matches manifest", meta["n_grid"] == exp["n_grid"],
          f"artifact={meta['n_grid']}")
    check("coverage rows complete",
          len(rows) == len(exp["n_grid"]) * len(meta["scenarios"]),
          f"{len(rows)} rows")
    check("every row carries the declared n_reps",
          all(r["n_reps"] == meta["n_reps"] for r in rows))
    fp_ok = meta["code_sha256_16"] == fingerprint()
    check("artifact was produced by the committed code", fp_ok,
          f"artifact={meta['code_sha256_16']} files={fingerprint()}")

    print("\nPath A -- D.C. Heller, port vs gpss and vs the paper")
    dc, months, n_pre = load_dc()
    ref = json.load(open(HERE / "reference.json"))
    p = gp_its(dc["handgun_rate"].values, months, n_pre)
    for k, a, b in (("b", p["b"], ref["b"]), ("s2", p["s2"], ref["s2"]),
                    ("tau_cum", p["tau_cum"], ref["tau_cum"]),
                    ("tau_cum_se", p["tau_cum_se"], ref["tau_cum_se"])):
        rel = float(np.max(np.abs(np.atleast_1d(a) - np.atleast_1d(b)))
                    / np.max(np.abs(np.atleast_1d(b))))
        check(f"port matches gpss: {k}", rel < RTOL, f"rel={rel:.2e}")

    cum = float(p["tau_cum"][-1])
    se = float(p["tau_cum_se"][-1])
    lo, hi = cum - 1.959963985 * se, cum + 1.959963985 * se
    tgt = man["path_a_target"]
    check("headline matches the paper at reported precision",
          (round(cum, 1) == tgt["cumulative"]
           and round(lo, 1) == tgt["ci"][0] and round(hi, 1) == tgt["ci"][1]),
          f"{cum:.4f} [{lo:.4f}, {hi:.4f}] vs paper "
          f"{tgt['cumulative']} {tgt['ci']}")

    print("\nplacebo diagnostic")
    pref = pd.read_csv(HERE / "placebo_reference.csv")
    pp = pd.DataFrame(gp_its_placebo(dc["handgun_rate"].values, months, n_pre,
                                     placebo_periods=4))
    d = float(np.max(np.abs(pp["tau"].values - pref["tau"].values)))
    check("port matches gpss placebo tau", d / np.max(np.abs(pref["tau"])) < RTOL,
          f"max|diff|={d:.2e}")
    check("all four placebo periods cover zero", bool(pp["cover"].all()))

    print("\nFigure 4A -- panel, regenerated and compared")
    pan = pd.read_json(HERE / "panel_results.json").set_index("state")
    check("panel has 50 jurisdictions", len(pan) == 50, f"{len(pan)} rows")
    check("committed panel agrees with a fresh D.C. fit",
          abs(pan.loc["District of Columbia", "cum"] - cum) < PANEL_ATOL,
          f"diff={abs(pan.loc['District of Columbia', 'cum'] - cum):.2e}")
    pan["std_cum"] = pan["cum"] / pan["pre_sd"]
    others = pan.drop("District of Columbia")
    dc_std = pan.loc["District of Columbia", "std_cum"]
    check("D.C. is rank 1 on the paper's standardised scale",
          dc_std > others["std_cum"].max(),
          f"D.C.={dc_std:.2f} next={others['std_cum'].max():.2f}")

    print("\ncoverage claim")
    gp = [r["GP_coverage"] for r in rows]
    seg = [r["Segmented_coverage"] for r in rows]
    check("GP coverage at or above nominal in every cell", min(gp) >= 0.95,
          f"min={min(gp):.3f}")
    check("segmented under-covers in every cell", max(seg) < 0.95,
          f"max={max(seg):.3f}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamper", action="store_true",
                    help="flip n_reps to 10 to prove the provenance check fails")
    sys.exit(main(**vars(ap.parse_args())))
