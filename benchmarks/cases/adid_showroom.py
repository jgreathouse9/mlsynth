"""Cross-validation: ADID against the authors' own MATLAB, run live under Octave.

Reference: Kathleen T. Li and Christophe Van den Bulte (2022), "Augmented
Difference-in-Differences", *Marketing Science*, DOI 10.1287/mksc.2022.1406.
Equation 2.4 is the estimator, Proposition 3.1 the limit distribution and
Appendix A.1 the variance.

The reference side is their own ``Showroom_generated_data_ADID_DID.m``, vendored
verbatim under ``benchmarks/reference/adid_showroom/`` and adapted for Octave in
``benchmarks/octave/adid_showroom.m`` with the four changes its header lists.
Nothing is transcribed and no number is copied from the paper: both sides run on
the same 110 weeks and the differences below are what the case pins.

Their published Table 7 is not pinned and cannot be. Their script's header says
the treated series is generated from a factor model, because the transactions
belong to an eyewear company that shared them under confidence, and the data
agrees -- 69.7 percent of control cells are exact multiples of 23.75, a quarter
of the $95 price point the paper names, against none of the treated cells. The
same script returns 1067 where the paper reports 946. What is checkable is the
arithmetic, and that is what this case checks.

``benchmarks/studies/adid_replicate`` is the spike this came from, and carries
the reasoning.
"""
from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1].parent
PANEL = _ROOT / "basedata" / "adid_showroom.csv"
DATA = _ROOT / "benchmarks" / "reference" / "adid_showroom" / "showroom_generated.csv"
SCRIPT = _ROOT / "benchmarks" / "octave" / "adid_showroom.m"
T1 = 83                      # their script's own default, city = 1


def _octave(t1: int) -> Dict[str, object]:
    from benchmarks.compare import BenchmarkSkipped

    exe = shutil.which("octave-cli") or shutil.which("octave")
    if exe is None:
        raise BenchmarkSkipped("octave-cli not on PATH")
    proc = subprocess.run([exe, str(SCRIPT), str(DATA), str(t1)],
                          capture_output=True, text=True, timeout=900, cwd=_ROOT)
    if proc.returncode != 0:
        raise BenchmarkSkipped(f"octave failed: {proc.stderr.strip()[-200:]}")

    out, path = {}, {}
    for line in proc.stdout.splitlines():
        f = line.split("\t")
        if len(f) == 2:
            out[f[0]] = float(f[1])
        elif len(f) == 3 and f[0] == "adid_cf":
            path[int(f[1])] = float(f[2])
    out["adid_cf"] = np.array([path[i] for i in sorted(path)])
    return out


def run() -> dict:
    from mlsynth import FDID
    from mlsynth.config_models import FDIDConfig

    theirs = _octave(T1)
    res = FDID(FDIDConfig(df=pd.read_csv(PANEL), outcome="sales",
                          treat="showroom", unitid="unit", time="week",
                          display_graphs=False)).fit()
    a = res.adid
    cf_diff = float(np.max(np.abs(np.asarray(a.counterfactual, dtype=float)
                                  - theirs["adid_cf"])))

    return {
        # Equation 2.4's two coefficients
        "delta1_abs_diff": abs(a.intercept - theirs["delta1"]),
        "delta2_abs_diff": abs(a.slope - theirs["delta2"]),
        # the effect and its standardized form, Proposition 3.1
        "att_abs_diff": abs(a.att - theirs["adid_att"]),
        "att_pct_abs_diff": abs(a.att_percent - theirs["adid_att_pct"]),
        "satt_abs_diff": abs(a.satt - theirs["adid_std_stat"]),
        # the whole fitted path, all 110 weeks
        "counterfactual_max_abs_diff": cf_diff,
        "counterfactual_max_rel_diff": cf_diff / float(
            np.max(np.abs(theirs["adid_cf"]))),
        # that the fit is the one their script runs: every donor, slope fitted
        "n_donors": float(len(a.selected_names)),
        "slope_is_reported": float(a.slope is not None),
        # and that the restriction ADID lifts is binding here, so the case is
        # not comparing two ways of computing the same number
        "slope_distance_from_one": abs(a.slope - 1.0),
        "did_att_ratio_to_adid": float(res.did.att / a.att),
    }


EXPECTED = {
    # Both sides solve the same 2x2 system on the same 83 weeks, so these are
    # floating-point reorderings and nothing else. The tolerances are five
    # orders of magnitude above what was measured, to absorb a different BLAS.
    "delta1_abs_diff": (3.45e-12, 1e-7),
    "delta2_abs_diff": (7.99e-14, 1e-9),
    "att_abs_diff": (3.87e-12, 1e-7),
    "att_pct_abs_diff": (1.42e-14, 1e-9),
    "satt_abs_diff": (1.43e-13, 1e-9),
    "counterfactual_max_abs_diff": (7.28e-12, 1e-7),
    "counterfactual_max_rel_diff": (1.05e-15, 1e-11),
    "n_donors": (10.0, 0.0),
    "slope_is_reported": (1.0, 0.0),
    # delta2 is 3.506, so DID's restriction is wrong by a factor of 3.5 on this
    # panel and the two estimates differ by 3.19. Wide tolerances: these pin
    # that the comparison has power, not a published quantity.
    "slope_distance_from_one": (2.506, 0.5),
    "did_att_ratio_to_adid": (3.188, 0.5),
}
