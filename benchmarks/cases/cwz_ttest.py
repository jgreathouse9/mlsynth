"""CWZ debiased SC t-test (arXiv:1812.10820): Table 5 carbon-tax replication.

Path A. Chernozhukov, Wuthrich & Zhu (2025), Table 5(a): the debiased SC
*t*-test (t-DISCo, K=3) of the Swedish carbon-tax effect on CO2 emissions per
capita (Andersson 2019 data: 15 countries, 1960-2005; Sweden treated 1990,
T0=30, T1=16). Per their footnote 24 the weights use *all* past outcomes as
predictors, i.e. the outcome-only SC. The paper reports ATT = -0.27 with a 90%
CI of [-0.41, -0.14]; mlsynth's ``VanillaSC(inference="ttest")`` reproduces it.

We also pin the debiased ATT on the two other canonical SC datasets validated
during development: the Basque Country (Abadie-Gardeazabal) and California
Proposition 99 (the 38-control-state ADH pool), both outcome-only, K=3.

Reference: the authors' R package ``scinference`` (``ttest.R::sc.cf``); the
Python port lives in ``mlsynth/utils/inferutils.debiased_sc_ttest``.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference import reference_value

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")


def _need(name: str) -> str:
    p = os.path.abspath(os.path.join(_BASE, name))
    if not os.path.exists(p):
        raise BenchmarkSkipped(f"{name} not available")
    return p


def _ttest(df, outcome, unitid, covs=None, win=None):
    from mlsynth import VanillaSC

    cfg = {"df": df, "outcome": outcome, "treat": "treated", "unitid": unitid,
           "time": "year", "backend": "outcome-only", "inference": "ttest",
           "ttest_K": 3, "alpha": 0.1, "display_graphs": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC(cfg).fit()
    return res.inference


def run() -> dict:
    # Table 5(a): Swedish carbon tax (outcome-only SC, K=3, alpha=0.1).
    ct = pd.read_stata(_need("carbontax_data.dta"))
    ct["treated"] = ((ct.country == "Sweden") & (ct.year >= 1990)).astype(int)
    inf = _ttest(ct, "CO2_transport_capita", "country")
    out = {
        "carbontax_att": float(inf.details["att_debiased"]),
        "carbontax_ci_lower": float(inf.ci_lower),
        "carbontax_ci_upper": float(inf.ci_upper),
    }

    # Basque Country (Abadie-Gardeazabal), national aggregate dropped.
    b = pd.read_csv(_need("basque_jasa.csv"))
    b = b[b.regionname != "Spain (Espana)"].copy()
    b["treated"] = ((b.regionname == "Basque Country (Pais Vasco)")
                    & (b.year >= 1975)).astype(int)
    out["basque_att"] = float(_ttest(b, "gdpcap", "regionname").details["att_debiased"])

    # California Proposition 99 (38-control-state ADH pool).
    p99 = pd.read_csv(_need("augmented_cali_long.csv"))
    p99["treated"] = p99["Proposition 99"].astype(int)
    out["prop99_att"] = float(_ttest(p99, "cigsale", "state").details["att_debiased"])
    return out


def comparison() -> dict:
    """mlsynth ``VanillaSC(inference="ttest")`` vs CWZ Table 5(a), the Swedish
    carbon-tax debiased-SC t-test: ATT and the 90% CI bounds, side by side.

    The reference side is a live ``scinference`` run captured in
    ``benchmarks/reference/cwz_ttest/`` (the authors' ``sc.cf`` t-test, version
    pinned), not transcribed -- laid against a fresh outcome-only ``VanillaSC``
    fit on the Andersson (2019) carbon-tax panel.
    """
    ct = pd.read_stata(_need("carbontax_data.dta"))
    ct["treated"] = ((ct.country == "Sweden") & (ct.year >= 1990)).astype(int)
    inf = _ttest(ct, "CO2_transport_capita", "country")
    att = float(inf.details["att_debiased"])
    lo, hi = float(inf.ci_lower), float(inf.ci_upper)
    rows = [
        {"quantity": "ATT", "mlsynth": round(att, 6),
         "reference": round(reference_value("cwz_ttest", "carbontax_att"), 6)},
        {"quantity": "CI_lower_90%", "mlsynth": round(lo, 6),
         "reference": round(reference_value("cwz_ttest", "carbontax_ci_lower"), 6)},
        {"quantity": "CI_upper_90%", "mlsynth": round(hi, 6),
         "reference": round(reference_value("cwz_ttest", "carbontax_ci_upper"), 6)},
    ]
    cfg = {"outcome": "CO2_transport_capita", "treat": "treated", "unitid": "country",
           "time": "year", "backend": "outcome-only", "inference": "ttest",
           "ttest_K": 3, "alpha": 0.1}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC", "config": cfg},
        "reference": {"impl": "R package scinference (sc.cf t-test, live run, captured)",
                      "version": "scinference 0.0.0.9000"},
    }


# Outcome-only SC is a deterministic convex program. Targets are pinned from a
# live scinference run captured in benchmarks/reference/cwz_ttest/ (not
# transcribed): mlsynth's ported t-test reproduces the package to ~1e-3.
_cw = lambda k: reference_value("cwz_ttest", k)
EXPECTED = {
    "carbontax_att": (_cw("carbontax_att"), 0.005),
    "carbontax_ci_lower": (_cw("carbontax_ci_lower"), 0.005),
    "carbontax_ci_upper": (_cw("carbontax_ci_upper"), 0.005),
    "basque_att": (_cw("basque_att"), 0.005),
    "prop99_att": (_cw("prop99_att"), 0.05),
}
