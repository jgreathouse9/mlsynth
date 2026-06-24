"""Cross-validation benchmark: PPSCM vs ``augsynth::multisynth`` (Paglayan 2018).

Path A / cross-validation against the reference implementation. ``PPSCM`` is
mlsynth's port of partially-pooled SCM (Ben-Michael, Feller & Rothstein 2021,
``augsynth::multisynth``); this case reproduces the package's own `multisynth
vignette
<https://github.com/ebenmichael/augsynth/blob/master/vignettes/multisynth-vignette.md>`_
cell-for-cell: the partial-pooling ``nu``, the average ATT, the **event-study
path**, and -- crucially -- the standard errors of **both** of augsynth's
inference procedures.

Provenance
----------
* Data: ``basedata/Teachingaugsynth.scv`` -- the Paglayan (2018) public-sector
  collective-bargaining panel (log per-pupil expenditure ``lnppexpend``,
  treatment ``cbr`` from ``YearCBrequired``), restricted as the vignette does:
  drop DC and WI, years 1959-1997, leaving 32 staggered-treated and 17
  never-treated states.
* Reference values come from a live ``augsynth::multisynth`` run, captured in
  ``benchmarks/reference/ppscm_paglayan/`` with its version pinned (augsynth
  0.2.0) -- not transcribed. Point estimates: basic ``nu = 0.2607``, ATT
  ``-0.011``, Global L2 ``0.003``; ``time_cohort`` ``nu = 0.3939``, ATT
  ``-0.018``. Standard errors come in two flavours, both from that run:

  - ``AUG_JACK_SE`` -- ``inf_type="jackknife"`` (delete-one), which mlsynth's
    jackknife reproduces to ``< 1.5e-3``;
  - ``AUG_BOOT_SE`` -- ``inf_type="bootstrap"`` (the **default**, Mammen wild
    multiplier bootstrap), which is what the vignette prints, reproduced by
    mlsynth's ported bootstrap.

The earlier "~10% SE gap" was an artefact of comparing mlsynth's jackknife to
augsynth's *bootstrap* default; matched method-for-method they agree.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"

# augsynth::multisynth's event-study path and per-period SEs, pinned from a live
# run captured in benchmarks/reference/ppscm_paglayan/ (not transcribed). The
# point estimates and jackknife SE are deterministic; the bootstrap SE is a
# seeded stochastic draw (mlsynth's bootstrap is a separate draw).
_REF = load_reference("ppscm_paglayan")["values"]
_arr = lambda p: np.array([_REF[f"{p}_{k:02d}"] for k in range(11)])
AUG_TAU = _arr("tau")
AUG_TAU_TC = _arr("tau_tc")
AUG_JACK_SE = _arr("jack_se")
AUG_BOOT_SE = _arr("boot_se")


def _analysis_df() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "Teachingaugsynth.scv")
    d = d[~d.State.isin(["DC", "WI"])].copy()
    d = d[(d.year >= 1959) & (d.year <= 1997)].copy()
    d["cbr"] = (d["year"] >= d["YearCBrequired"].fillna(np.inf)).astype(int)
    return d


def run() -> dict:
    from mlsynth import PPSCM

    d = _analysis_df()
    common = dict(df=d, outcome="lnppexpend", treat="cbr", unitid="State",
                  time="year", display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jack = PPSCM({**common, "run_inference": True,
                      "inference_method": "jackknife"}).fit()
        cohort = PPSCM({**common, "run_inference": False,
                        "time_cohort": True}).fit()
        boot = PPSCM({**common, "run_inference": True,
                      "inference_method": "bootstrap", "n_boot": 2000,
                      "seed": 0}).fit()

    jse = np.asarray(jack.event_study.se, dtype=float)
    bse = np.asarray(boot.event_study.se, dtype=float)
    tau = np.asarray(jack.event_study.tau, dtype=float)
    tau_tc = np.asarray(cohort.event_study.tau, dtype=float)

    return {
        "ppscm_nu": float(jack.design.nu_used),
        "ppscm_att": float(jack.att),
        "ppscm_global_l2": float(jack.design.global_l2),
        "ppscm_event_study_max_abs_diff": float(np.max(np.abs(tau - AUG_TAU))),
        "ppscm_time_cohort_nu": float(cohort.design.nu_used),
        "ppscm_time_cohort_att": float(cohort.att),
        "ppscm_tc_event_study_max_abs_diff": float(np.max(np.abs(tau_tc - AUG_TAU_TC))),
        "ppscm_jackknife_se_max_abs_diff": float(np.max(np.abs(jse - AUG_JACK_SE))),
        "ppscm_bootstrap_att_se": float(boot.inference_detail.se),
        "ppscm_bootstrap_se_max_abs_diff": float(np.max(np.abs(bse - AUG_BOOT_SE))),
        "n_treated": int(d[d.cbr == 1].State.nunique()),
        "n_control": int(d.State.nunique() - d[d.cbr == 1].State.nunique()),
    }


def comparison() -> dict:
    """mlsynth PPSCM vs ``augsynth::multisynth``, quantity by quantity.

    Pairs mlsynth's PPSCM fit against augsynth's documented / live-R reference
    numbers (the ``AUG_*`` arrays and the vignette anchors): the partial-pooling
    ``nu``, the average ATT, the global L2 imbalance, the time-cohort ``nu``/ATT,
    and the full event-study and standard-error paths for both inference flavours
    (delete-one jackknife and the Mammen wild bootstrap). Propagates the
    ``BenchmarkSkipped`` raised when Rscript / augsynth is absent so a missing R
    toolchain never turns the suite red.
    """
    from mlsynth import PPSCM

    d = _analysis_df()
    cfg = {"outcome": "lnppexpend", "treat": "cbr", "unitid": "State",
           "time": "year"}
    common = dict(df=d, display_graphs=False, **cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jack = PPSCM({**common, "run_inference": True,
                      "inference_method": "jackknife"}).fit()
        cohort = PPSCM({**common, "run_inference": False,
                        "time_cohort": True}).fit()
        boot = PPSCM({**common, "run_inference": True,
                      "inference_method": "bootstrap", "n_boot": 2000,
                      "seed": 0}).fit()
    tau = np.asarray(jack.event_study.tau, dtype=float)
    tau_tc = np.asarray(cohort.event_study.tau, dtype=float)
    jse = np.asarray(jack.event_study.se, dtype=float)
    bse = np.asarray(boot.event_study.se, dtype=float)

    rows = [
        {"quantity": "nu", "mlsynth": round(float(jack.design.nu_used), 4),
         "reference": round(_REF["nu"], 4)},
        {"quantity": "ATT", "mlsynth": round(float(jack.att), 4),
         "reference": round(_REF["att"], 4)},
        {"quantity": "global_L2", "mlsynth": round(float(jack.design.global_l2), 4),
         "reference": round(_REF["global_l2"], 4)},
        {"quantity": "time_cohort/nu", "mlsynth": round(float(cohort.design.nu_used), 4),
         "reference": round(_REF["tc_nu"], 4)},
        {"quantity": "time_cohort/ATT", "mlsynth": round(float(cohort.att), 4),
         "reference": round(_REF["tc_att"], 4)},
    ]

    for k in range(len(AUG_TAU)):
        rows.append({"quantity": f"event_study/tau[{k}]",
                     "mlsynth": round(float(tau[k]), 6),
                     "reference": round(float(AUG_TAU[k]), 6)})
    for k in range(len(AUG_TAU_TC)):
        rows.append({"quantity": f"time_cohort/tau[{k}]",
                     "mlsynth": round(float(tau_tc[k]), 6),
                     "reference": round(float(AUG_TAU_TC[k]), 6)})
    for k in range(len(AUG_JACK_SE)):
        rows.append({"quantity": f"jackknife_se[{k}]",
                     "mlsynth": round(float(jse[k]), 5),
                     "reference": round(float(AUG_JACK_SE[k]), 5)})
    for k in range(len(AUG_BOOT_SE)):
        rows.append({"quantity": f"bootstrap_se[{k}]",
                     "mlsynth": round(float(bse[k]), 5),
                     "reference": round(float(AUG_BOOT_SE[k]), 5)})

    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PPSCM", "config": cfg},
        "reference": {"impl": "R augsynth::multisynth (live run, captured)",
                      "version": "augsynth 0.2.0"},
    }


# All values cross-validate augsynth method-for-method; tolerances bracket the
# OSQP solver residual (point estimates) and the bootstrap's Monte-Carlo error
# (R RNG vs numpy RNG at n_boot=2000).
# Scalar targets are pinned from the live augsynth run (benchmarks/reference/
# ppscm_paglayan/); the *_max_abs_diff metrics compare mlsynth's paths/SEs to the
# bundle's AUG_* arrays. The bootstrap SE is a seeded stochastic draw, so its
# targets keep a wider band.
_pp = lambda k: reference_value("ppscm_paglayan", k)
EXPECTED = {
    "ppscm_nu": (_pp("nu"), 2e-3),
    "ppscm_att": (_pp("att"), 1.5e-3),
    "ppscm_global_l2": (_pp("global_l2"), 5e-4),
    "ppscm_event_study_max_abs_diff": (0.0, 7e-4),
    "ppscm_time_cohort_nu": (_pp("tc_nu"), 2e-3),
    "ppscm_time_cohort_att": (_pp("tc_att"), 2e-3),
    "ppscm_tc_event_study_max_abs_diff": (0.0, 3e-3),
    "ppscm_jackknife_se_max_abs_diff": (0.0, 1.5e-3),     # vs augsynth jackknife
    "ppscm_bootstrap_att_se": (0.022, 2e-3),              # vs augsynth bootstrap (vignette)
    "ppscm_bootstrap_se_max_abs_diff": (0.0, 4e-3),       # vs augsynth bootstrap path
    "n_treated": (32, 0),
    "n_control": (17, 0),
}
