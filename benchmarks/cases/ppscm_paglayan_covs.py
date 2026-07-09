"""Cross-validation benchmark: PPSCM with auxiliary covariates vs
``augsynth::multisynth`` (Ben-Michael, Feller & Rothstein 2022, Sec 5.2).

Extends the ``ppscm_paglayan`` outcome-only case to covariate balancing:
mlsynth's PPSCM with ``covariates=["perinc_1959", "studteachratio_1959"]`` on
the Paglayan (2018) collective-bargaining panel, cross-checked against a live
``augsynth 0.2.0`` run (R 4.3.3) captured in
``benchmarks/reference/ppscm_paglayan_covs/`` -- the covariate formula
``lnppexpend ~ cbr | perinc_1959 + studteachratio_1959`` from the multisynth
vignette. Pins the partial-pooling ``nu``, the average ATT, the global and
individual L2 imbalance, and the full event-study path.

Covariates enter as augsynth does: each is z-scored against the never-treated
controls and rescaled by ``sd(X[[1]][is.finite(trt)])`` -- the sd over the
treated rows of the first cohort's residual block (multi_synth_qp.R:98) -- then
stacked into the pooled and separate QP terms.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_REF = load_reference("ppscm_paglayan_covs")["values"]
AUG_TAU = np.array([_REF[f"tau_{k:02d}"] for k in range(11)])


def _analysis_df() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "Teachingaugsynth.scv")
    d = d[~d.State.isin(["DC", "WI", "AK", "HI"])].copy()   # contiguous US (cov coverage)
    d = d[(d.year >= 1959) & (d.year <= 1997)].copy()
    d["cbr"] = (d["year"] >= d["YearCBrequired"].fillna(np.inf)).astype(int)
    snap = (d[d.year == 1959][["State", "perinc", "studteachratio"]]
            .rename(columns={"perinc": "perinc_1959",
                             "studteachratio": "studteachratio_1959"}))
    return d.merge(snap, on="State")


def run() -> dict:
    from mlsynth import PPSCM

    d = _analysis_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = PPSCM(dict(df=d, outcome="lnppexpend", treat="cbr", unitid="State",
                         time="year", covariates=["perinc_1959", "studteachratio_1959"],
                         run_inference=False, display_graphs=False)).fit()
    tau = np.asarray(fit.event_study.tau, dtype=float)[:11]
    return {
        "nu": float(fit.design.nu_used),
        "att": float(fit.att),
        "global_l2": float(fit.design.global_l2),
        "ind_l2": float(fit.design.ind_l2),
        "event_study_max_abs_diff": float(np.max(np.abs(tau - AUG_TAU))),
    }


EXPECTED = {
    "nu": (_REF["nu"], 3e-3),
    "att": (_REF["att"], 3e-3),
    "global_l2": (_REF["global_l2"], 5e-4),
    "ind_l2": (_REF["ind_l2"], 3e-3),
    "event_study_max_abs_diff": (0.0, 3e-3),
}


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
