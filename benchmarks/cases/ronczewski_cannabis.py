"""Path A: Ronczewski (2026), cannabis legalisation and alcohol consumption.

Four mlsynth estimators against the four R packages the paper runs, on the
paper's own data and its own primary sample. The replication package is
`Alexr951/Cannabis_Alcohol_Substitution
<https://github.com/Alexr951/Cannabis_Alcohol_Substitution>`_; every target
below is read from a number the paper publishes in ``Results/csv/``.

Provenance
----------
The outcome data is ``Data/pcyr1970-2023.txt`` -- NIAAA Surveillance Report
#122 (Slater and Alpert, 2025), gallons of ethanol per capita (21+) by state,
beverage and year -- fetched from the replication repository at run time and
rebuilt here by the same steps as the paper's loader. Nothing is vendored, and
the case raises :class:`BenchmarkSkipped` when the host is unreachable. It does
not substitute a generated panel: a synthetic fallback that reports as a
real-data replication is the failure ``ppscm_cs_real_panels`` guards against,
and the shape assertions below would not catch it on their own.

The primary sample is the paper's: log ethanol per capita 21+, total beverage,
2000-2019, the five clean-fit treated states and the thirty never-treated
donors. The fit screen that selects those five (a placebo RMSPE ratio, Table 2)
is the paper's own and is taken as given here, from
``Results/csv/sample_rules.csv``; reproducing the screen is a different case
from validating the estimators on the sample it produces.

What agrees, and how closely
----------------------------
Three levels, and the case reports each as what it is.

Exact, to double precision. PPSCM's ``callaway_santanna`` mode against
``did::att_gt`` + ``aggte``, and GSYNTH at a fixed factor number against
``gsynth``'s interactive fixed effects. Same estimator, same arithmetic.

Solver tolerance. PPSCM's default against ``augsynth::multisynth`` (8.2e-08),
and SDID against ``synthdid`` (6.1e-05 on the aggregate). Both sides solve a
quadratic program, and the residual is where the solvers stop.

Tuning, which is not agreement at all. GSYNTH's factor-number cross-validation
picks 4 where the paper's ``gsynth`` 1.4.0 picks 1, and the estimate differs by
a factor of 2.6 -- while the estimators agree to 4e-17 at a common rank. The
paper's ``Results/package_versions.csv`` records ``gsynth,1.4.0``, the version
that forwards to ``fect``, whose cross-validation masks a random share of
observed cells over k rounds where Xu's Algorithm 1 holds out one pre-treatment
period at a time. Measured against gsynth 1.4.0 on this panel:

===  ======================  ====================
r    mlsynth (Algorithm 1)   gsynth 1.4.0 / fect
===  ======================  ====================
0    6.73e-04                2.024e-03
1    4.14e-04                1.481e-03
2    4.63e-04                3.203e-03
3    4.90e-04                2.281e-03
4    3.89e-04                2.026e-03
5    4.17e-04                1.815e-03
===  ======================  ====================

The two schemes rank the ranks differently, and not by breaking a near-tie: r=1
wins fect's by 18% over the runner-up and is second on Algorithm 1's. So this
case validates GSYNTH at a fixed r and records the selected rank as a separate
metric, which is the honest split (#483).

The two aggregations
--------------------
``did::aggte(type = "simple")`` averages every unit-post-period cell equally;
``type = "dynamic"`` averages the event-study horizons. They coincide only when
every cohort is the same size and reaches every horizon, which this panel is
not: cohorts of 2, 1 and 2 states with post windows of 6, 5 and 2 years. The
paper reports the simple one. Both are read off the same PPSCM fit -- the
dynamic one is ``effects.att``, the simple one is the mean over the finite
entries of the ragged per-unit paths -- and both are checked.

Reaching them at all needs ``n_leads = 6``, which the paper sets explicitly
against a default of 2. Before #482 mlsynth clamped that back to the default
and could reproduce neither headline number.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_RAW = ("https://raw.githubusercontent.com/Alexr951/"
        "Cannabis_Alcohol_Substitution/main/Data/pcyr1970-2023.txt")

_COLS = ["year", "fips", "beverage", "gal_bev", "gal_eth", "pop14", "pc_eth14",
         "decile14", "pop21", "pc_eth21", "decile21", "data_source", "abv",
         "eth_tvabv"]

# Results/csv/sample_rules.csv: included == 1, with each state's adoption year.
# The two dropped states (Alaska 2016, Nevada 2017) fail the paper's placebo
# RMSPE ratio screen at 4.3 and 5.1 against a threshold of 2.
CLEAN = {8: 2014, 53: 2014, 41: 2015, 6: 2018, 25: 2018}

# Data/processed/treatment.csv: ever_rec_2023 == 0 and balanced 2000-2023.
# Every state with recreational legalisation through 2023 is excluded, so a
# donor is never a future adopter.
REC_LEGAL = {2, 4, 6, 8, 9, 11, 17, 23, 24, 25, 26, 29, 30, 32, 34, 35, 36, 41,
             44, 50, 53}

YEARS = range(2000, 2020)
N_LEADS = 6                      # R/helpers.R: N_LEADS_PRIMARY

# Results/csv/, written by the paper's R pipeline.
REF = {
    "cs_simple": 0.0124231500223412,        # cs_twfe_primary.csv, did 2.3.0
    "multisynth": 0.0275659204188017,       # multisynth_overall.csv, augsynth 0.2.0
    "gsynth_ife": 0.0179213802156284,       # gsynth_primary.csv, gsynth 1.4.0, r.cv = 1
    "sdid_aggregate": 0.0181113832098199,   # sdid_primary.csv, synthdid 0.0.9
}
# cs_eventstudy.csv, did::aggte(type = "dynamic"), event times 0..5.
REF_CS_EVENT = np.array([
    0.017471628691788, 0.0204811123073007, 0.00368048581046909,
    0.00957077749057922, 0.0137794789016267, -0.004984890567151])
# sdid_primary.csv, per cohort, with the treated-post cell counts it aggregates by.
REF_SDID = {2014: (0.00867449914569538, 12), 2015: (0.016854104123037, 5),
            2018: (0.0479936342606723, 4)}
# gsynth 1.4.0 r.cv on this panel, run live at fect 2.4.5 (#483).
REF_GSYNTH_R = 1


def _panel() -> pd.DataFrame:
    """The paper's analysis frame, rebuilt from the NIAAA source file."""
    try:
        with urllib.request.urlopen(_RAW, timeout=60) as fh:
            raw_bytes = fh.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkSkipped(
            "could not fetch the NIAAA surveillance file from the Ronczewski "
            f"replication repository ({exc}); this case replicates published "
            "numbers and has no meaning against substituted data") from exc

    raw = pd.read_csv(io.BytesIO(raw_bytes), skiprows=129, sep=r"\s+",
                      header=None, names=_COLS, na_values=["."])
    raw["pc_eth21"] = raw["pc_eth21"] / 10_000
    raw = raw[raw["fips"].between(1, 56) & (raw["fips"] != 11)]        # drop DC
    df = raw[(raw["beverage"] == 4)                                    # total
             & raw["year"].between(2000, 2023)
             & raw["pc_eth21"].notna() & (raw["pc_eth21"] > 0)].copy()

    # The paper keeps only states with a balanced total-ethanol series over the
    # full 2000-2023 span, then estimates on the 2000-2019 primary window.
    counts = df.groupby("fips")["year"].nunique()
    balanced = set(counts[counts == 24].index)
    donors = sorted(balanced - REC_LEGAL)
    keep = sorted(CLEAN) + donors

    d = df[df["fips"].isin(keep) & df["year"].isin(YEARS)].copy()
    d["y"] = np.log(d["pc_eth21"])
    d["g"] = d["fips"].map(lambda f: CLEAN.get(f, 0))
    d["trt"] = ((d["g"] > 0) & (d["year"] >= d["g"])).astype(int)
    return d.sort_values(["fips", "year"]).reset_index(drop=True)


def _base(d: pd.DataFrame) -> Dict[str, Any]:
    return {"df": d[["fips", "year", "y", "trt"]], "outcome": "y",
            "treat": "trt", "unitid": "fips", "time": "year",
            "display_graphs": False}


def _stack(res) -> np.ndarray:
    """Per-unit effect paths, ``NaN`` past each unit's own post window."""
    return np.vstack([np.asarray(v.tau, float) for v in res.per_unit.values()])


def _ppscm(d: pd.DataFrame) -> Dict[str, float]:
    from mlsynth import PPSCM

    cfg = {**_base(d), "run_inference": False, "n_leads": N_LEADS}
    cs = PPSCM({**cfg, "method": "callaway_santanna"}).fit()
    pp = PPSCM(cfg).fit()
    cs_s = _stack(cs)
    return {
        # aggte(type="simple"): every unit-post-period cell weighted equally.
        "cs_simple": float(np.nanmean(cs_s)),
        # aggte(type="dynamic")$att.egt, horizon by horizon.
        "cs_event": np.nanmean(cs_s, axis=0),
        # augsynth's overall: the mean over units of each unit's post-window mean.
        "multisynth": float(np.mean([np.nanmean(v.tau)
                                     for v in pp.per_unit.values()])),
        "leads": int(cs_s.shape[1]),
    }


def _sdid(d: pd.DataFrame) -> Dict[str, float]:
    """One balanced block per cohort, as ``synthdid`` requires."""
    from mlsynth import SDID

    per, num, den = {}, 0.0, 0
    donors = sorted(set(d.fips) - set(CLEAN))
    for g, (_, cells) in sorted(REF_SDID.items()):
        coh = [f for f, y in CLEAN.items() if y == g]
        blk = d[d["fips"].isin(coh + donors)].copy()
        blk["trt"] = (blk["fips"].isin(coh) & (blk["year"] >= g)).astype(int)
        att = float(SDID({**_base(blk), "B": 0}).fit().effects.att)
        per[g] = att
        num += att * cells
        den += cells
    return {"per_cohort": per, "aggregate": num / den}


def _gsynth(d: pd.DataFrame) -> Dict[str, float]:
    from mlsynth import GSYNTH

    cfg = {**_base(d), "inference": False, "force": "two-way"}
    fixed = GSYNTH({**cfg, "r": 1}).fit()
    cved = GSYNTH({**cfg, "r_max": 5}).fit()
    return {"at_r1": float(fixed.effects.att),
            "r_selected": int(cved.design.r_selected)}


def run() -> dict:
    d = _panel()
    pp = _ppscm(d)
    sd = _sdid(d)
    gs = _gsynth(d)
    return {
        "n_states": int(d.fips.nunique()),
        "n_treated": len(CLEAN),
        "n_donors": int(d.fips.nunique()) - len(CLEAN),
        "n_years": int(d.year.nunique()),
        "leads": pp["leads"],
        "cs_simple_diff": abs(pp["cs_simple"] - REF["cs_simple"]),
        "cs_event_max_diff": float(np.max(np.abs(pp["cs_event"] - REF_CS_EVENT))),
        "multisynth_diff": abs(pp["multisynth"] - REF["multisynth"]),
        "sdid_aggregate_diff": abs(sd["aggregate"] - REF["sdid_aggregate"]),
        "sdid_cohort_max_diff": max(abs(sd["per_cohort"][g] - v)
                                    for g, (v, _) in REF_SDID.items()),
        "gsynth_at_r1_diff": abs(gs["at_r1"] - REF["gsynth_ife"]),
        # Recorded, not asserted to agree: the selection rules differ (#483).
        "gsynth_r_selected": gs["r_selected"],
        "gsynth_r_published": REF_GSYNTH_R,
    }


def comparison() -> dict:
    d = _panel()
    pp = _ppscm(d)
    sd = _sdid(d)
    gs = _gsynth(d)

    rows: List[Dict[str, Any]] = [
        {"quantity": "Callaway-Sant'Anna ATT (aggte simple)",
         "mlsynth": pp["cs_simple"], "reference": REF["cs_simple"]},
        {"quantity": "multisynth overall ATT",
         "mlsynth": pp["multisynth"], "reference": REF["multisynth"]},
        {"quantity": "SDID aggregate ATT",
         "mlsynth": sd["aggregate"], "reference": REF["sdid_aggregate"]},
        {"quantity": "gsynth IFE ATT (r fixed at 1)",
         "mlsynth": gs["at_r1"], "reference": REF["gsynth_ife"]},
    ]
    for h, (mine, ref) in enumerate(zip(pp["cs_event"], REF_CS_EVENT)):
        rows.append({"quantity": f"CS event study, e = {h}",
                     "mlsynth": float(mine), "reference": float(ref)})
    for g, (ref, _) in sorted(REF_SDID.items()):
        rows.append({"quantity": f"SDID cohort {g}",
                     "mlsynth": sd["per_cohort"][g], "reference": ref})
    rows.append({"quantity": "GSYNTH factor number selected",
                 "mlsynth": gs["r_selected"], "reference": REF_GSYNTH_R})

    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PPSCM / SDID / GSYNTH",
                         "config": {"n_leads": N_LEADS, "r": 1}},
        "reference": {"impl": "Ronczewski (2026) replication package, "
                              "Results/csv/",
                      "version": "augsynth 0.2.0, did 2.3.0, gsynth 1.4.0, "
                                 "synthdid 0.0.9"},
    }


# Tolerances are set by what each comparison is. The two exact rows get double
# precision, because the estimators are the same object and the only residual
# is arithmetic. The two quadratic programs get the scale their solvers stop at,
# measured and not guessed: multisynth at 8.2e-08 and SDID at 6.1e-05 on the
# aggregate, 2.3e-04 on the single-treated-unit Oregon cohort. The published
# multisynth event-study CSV is rounded to six decimals, so it is not compared
# here; the overall is full precision and is.
#
# gsynth_r_selected is pinned at 4 and gsynth_r_published at 1: they disagree,
# and the case records the disagreement instead of hiding it in a tolerance.
# Should mlsynth ever adopt fect's cross-validation, this is the metric that
# moves and it should move deliberately.
EXPECTED = {
    "cs_simple_diff": (0.0, 1e-14),
    "cs_event_max_diff": (0.0, 1e-14),
    "multisynth_diff": (0.0, 1e-06),
    "sdid_aggregate_diff": (0.0, 1e-03),
    "sdid_cohort_max_diff": (0.0, 1e-03),
    "gsynth_at_r1_diff": (0.0, 1e-14),
    "gsynth_r_selected": (4, 0),
    "gsynth_r_published": (1, 0),
    # Shape, pinned so a change in the source file or the balance rule surfaces
    # here instead of re-defining the sample the numbers describe.
    "n_states": (35, 0),
    "n_treated": (5, 0),
    "n_donors": (30, 0),
    "n_years": (20, 0),
    "leads": (6, 0),
}
