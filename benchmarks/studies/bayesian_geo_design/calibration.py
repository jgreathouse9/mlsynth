"""Arm 1: do the credible intervals cover?

Every location in Meta's GeoLift pre-test panel takes a turn as a placebo
treated unit. Nothing happened to any of them, so a nominal 90% interval should
contain zero about 90% of the time. An engine whose interval is narrower and
still covers has earned the narrowness; one that is narrower and under-covers is
over-confident, and the MDE built on it is optimistic by the same factor.

    python calibration.py [results/calibration.json]

Measured on the 40-location panel, post window 76..90:

    engine / interval                coverage   mean width
    augsynth (conformal)                89.7%          716
    sdid (placebo)                      92.5%         1323
    bscm (horseshoe, AR shock)          75.0%          315
    mvbbsc as shipped (iid shock)       65.0%          299
    mvbbsc + AR shock                   87.5%          418
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.geox_helpers.engines import resolve_engine

from . import engines as bayes

DATA = "basedata/geolift_market_data.csv"
PRE, ALPHA = 75, 0.10
ARMS = (("augsynth", dict(inference="conformal", conformal_type="iid", ns=400)),
        ("sdid", dict(inference="placebo", n_draws=200)),
        ("bscm", dict(autocorr=True)),
        ("mvbbsc_iid", dict(autocorr=False)),
        ("mvbbsc_ar", dict(autocorr=True)))


def _engine(name):
    if name.startswith("mvbbsc"):
        return bayes.MVBBSC
    if name == "bscm":
        return bayes.BSCM
    return resolve_engine(name)


def run(data_path: str = DATA) -> dict:
    wide = (pd.read_csv(data_path)
            .pivot(index="date", columns="location", values="Y").sort_index())
    n_periods = wide.shape[0]
    out = {}
    for name, ikw in ARMS:
        eng = _engine(name)
        cov, width, atts = [], [], []
        for loc in wide.columns:
            y = wide[loc].to_numpy(dtype=float)
            donors = wide.drop(columns=[loc]).to_numpy(dtype=float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fkw = ({"augment": "ridge", "fixed_effects": True}
                           if name == "augsynth" else {})
                    fit = eng.fit_once(y, donors, PRE, PRE, n_periods - 1, 1, **fkw)
                    _, det = eng.point_inference(fit, y, donors, PRE, PRE,
                                                 n_periods - 1, alpha=ALPHA, **ikw)
                lo, hi = det["ci_lower"], det["ci_upper"]
                if lo is None or hi is None or not np.isfinite([lo, hi]).all():
                    continue
                cov.append(bool(lo <= 0.0 <= hi))
                width.append(float(hi - lo))
                atts.append(float(eng.att(fit, y, PRE, n_periods - 1)))
            except Exception:
                continue
        out[name] = {"n": len(cov), "coverage": float(np.mean(cov)),
                     "mean_width": float(np.mean(width)),
                     "att_sd_across_units": float(np.std(atts, ddof=1))}
        print("%-14s n=%2d coverage %5.1f%%  mean width %6.0f"
              % (name, len(cov), 100 * np.mean(cov), np.mean(width)), flush=True)
    return {"nominal": 1 - ALPHA, "pre_periods": PRE, "arms": out}


if __name__ == "__main__":
    res = run()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=2)
