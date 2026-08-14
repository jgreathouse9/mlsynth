"""CWZ conformal test inversion (JASA 2021): the paper's own application.

Cross-validation, scenario 3. Chernozhukov, Wuthrich & Zhu (2021), JASA
116(536), 1849-1864, Section 5: Rhode Island's 2003 decriminalization of indoor
prostitution and log female gonorrhea incidence per 100,000 (Cunningham & Shah
2018), 50 control states, T0 = 19 (1985-2003), T1 = 6 (2004-2009), alpha = 0.1,
L1 statistic. The panel ships in their replication package and is vendored at
``basedata/logfemrate.txt``.

The reference side is a live run of the authors' own R, captured in
``benchmarks/reference/cwz_conformal/``: the ``scinference`` package at v1.0.0
(``567c688``), which is the version their later replication package pins, with
the JASA online supplement's ``functions_conformal_final.R`` re-derived beside
it so the packaged and published forms are checked against each other on the
same panel. They agree to the last digit on every deterministic quantity.

What is pinned, and why each is a different kind of check:

* ``p_mb`` -- the joint-null p-value under moving-block permutations, their
  default. Deterministic on both sides: the reference set is the T cyclic
  shifts, enumerated, so this is an exact match or a defect.
* ``lb_YYYY`` / ``ub_YYYY`` -- the six pointwise intervals, swept on the
  application's own grid ``seq(-5, 2, 0.01)`` passed through
  ``conformal_grid``. Sharing the grid is what makes these value-for-value: an
  inversion returns grid points, so a case that let each side pick its own grid
  would be comparing resolutions.
* ``placebo_mb_T1_*`` -- the specification tests of their Section 5, which hold
  out the last one, two and three pre-treatment periods and test them as if
  they were the post-period. No effect exists there, so these check the
  procedure's calibration on this panel before its answer on the real
  post-period is read.
* ``p_iid`` -- the same null under i.i.d. permutations, which resamples. The
  only stochastic quantity here, and its tolerance is a Monte-Carlo budget
  rather than a regression band.

mlsynth's side is ``VanillaSC(inference="conformal")`` throughout, read through
``res.inference``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference import reference_value

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")

T0, T1 = 19, 6
FIRST_YEAR = 1985
TREAT_YEAR = 2004
ALPHA = 0.1
# app_cunninghamshah_final.R:135 -- the grid the paper's figures are drawn on.
GRID = np.round(np.arange(-5.0, 2.0 + 1e-9, 0.01), 10)
# scinference's default draw count for the iid scheme.
N_PERM = 5000


def _panel() -> pd.DataFrame:
    """The gonorrhea panel in long form: column 0 is Rhode Island, 1-50 controls."""
    path = os.path.abspath(os.path.join(_BASE, "logfemrate.txt"))
    if not os.path.exists(path):
        raise BenchmarkSkipped("logfemrate.txt not available")
    Y = pd.read_csv(path, sep="\t").to_numpy(dtype=float)
    n_periods, n_units = Y.shape
    units = ["Rhode Island"] + [f"donor{j:02d}" for j in range(1, n_units)]
    return pd.DataFrame({
        "state": np.repeat(units, n_periods),
        "year": np.tile(np.arange(FIRST_YEAR, FIRST_YEAR + n_periods), n_units),
        "logfemrate": Y.T.ravel(),
    })


def _fit(df: pd.DataFrame, treat_year: int, **kw):
    from mlsynth import VanillaSC

    d = df.copy()
    d["treated"] = ((d.state == "Rhode Island") & (d.year >= treat_year)).astype(int)
    cfg = {"df": d, "outcome": "logfemrate", "treat": "treated", "unitid": "state",
           "time": "year", "backend": "outcome-only", "inference": "conformal",
           "alpha": ALPHA, "display_graphs": False, **kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


def _placebo_p(df: pd.DataFrame, hold_out: int) -> float:
    """Their specification test: treat the last ``hold_out`` pre-periods as post.

    The panel is truncated at the true treatment date first, so the window is
    entirely untreated and the null of no effect is true by construction.
    """
    pre = df[df.year < TREAT_YEAR]
    return float(
        _fit(pre, TREAT_YEAR - hold_out, conformal_type="block")
        .inference.details["joint_p_value"]
    )


def run() -> dict:
    df = _panel()

    res = _fit(df, TREAT_YEAR, conformal_type="block", conformal_grid=GRID)
    d = res.inference.details
    lo = np.asarray(d["pi_lower"], dtype=float)
    hi = np.asarray(d["pi_upper"], dtype=float)
    out = {"p_mb": float(d["joint_p_value"])}
    for k, year in enumerate(range(TREAT_YEAR, TREAT_YEAR + T1)):
        out[f"lb_{year}"] = float(lo[k])
        out[f"ub_{year}"] = float(hi[k])

    # The iid scheme at the reference's own draw count. Seeded, so the case is
    # deterministic; the tolerance covers the distance between two independent
    # 5000-draw estimates of the same p-value, not a difference in method.
    out["p_iid"] = float(
        _fit(df, TREAT_YEAR, conformal_type="iid", conformal_n_perm=N_PERM,
             conformal_finite_sample=True, seed=12345)
        .inference.details["joint_p_value"]
    )

    for t in (1, 2, 3):
        out[f"placebo_mb_T1_{t}"] = _placebo_p(df, t)
    return out


def comparison() -> dict:
    """mlsynth ``VanillaSC(inference="conformal")`` against ``scinference``.

    The paper's own application, quantity by quantity: the joint-null p-value
    under both permutation schemes, the six pointwise intervals on the paper's
    grid, and the three placebo specification tests.
    """
    got = run()
    rows = [{"quantity": "p_value (moving block)",
             "mlsynth": round(got["p_mb"], 6),
             "reference": round(reference_value("cwz_conformal", "p_mb"), 6)}]
    for year in range(TREAT_YEAR, TREAT_YEAR + T1):
        for side, label in (("lb", "lower"), ("ub", "upper")):
            rows.append({
                "quantity": f"CI_{label}_90%_{year}",
                "mlsynth": round(got[f"{side}_{year}"], 6),
                "reference": round(reference_value("cwz_conformal", f"{side}_{year}"), 6),
            })
    rows.append({"quantity": "p_value (iid, 5000 draws)",
                 "mlsynth": round(got["p_iid"], 6),
                 "reference": round(reference_value("cwz_conformal", "p_iid"), 6)})
    for t in (1, 2, 3):
        rows.append({
            "quantity": f"placebo p_value (T1={t})",
            "mlsynth": round(got[f"placebo_mb_T1_{t}"], 6),
            "reference": round(reference_value("cwz_conformal", f"placebo_mb_T1_{t}"), 6),
        })
    cfg = {"outcome": "logfemrate", "treat": "treated", "unitid": "state",
           "time": "year", "backend": "outcome-only", "inference": "conformal",
           "conformal_type": "block", "conformal_grid": "seq(-5, 2, 0.01)",
           "alpha": 0.1}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC", "config": cfg},
        "reference": {"impl": "R package scinference (conformal, live run, captured), "
                              "cross-checked against the JASA supplement's own functions",
                      "version": "scinference v1.0.0 (567c688)"},
    }


_r = lambda k: reference_value("cwz_conformal", k)

# Tolerances, in three tiers, because three different things are being claimed.
#
# The moving-block p-value and the six interval endpoints are deterministic on
# both sides -- the reference set is enumerated, the grid is shared, and the
# simplex QP agrees with limSolve's lsei(type=2) to 1e-8 on this panel. They are
# measured to reproduce exactly, so the tolerance is a floating-point band and
# nothing more; 1e-9 on the p-value, and half a grid step (0.005) on the
# endpoints, which is the finest a shared grid of spacing 0.01 can distinguish.
#
# The placebo tests are the same construction on a shorter window, so also
# exact.
#
# The iid p-value is the one Monte-Carlo quantity. Both sides draw 5000
# permutations from their own RNG, so the difference is the distance between two
# independent estimates of a p-value near 0.023: the standard error of one is
# sqrt(p(1-p)/5000) = 0.0021, and the difference of two independent estimates has
# 0.003. The band is three of those, wide enough that a re-seed does not flip it
# and narrow enough that a switch of permutation scheme (0.04 for the block
# scheme) or of p-value convention (a shift of 1/5001) would not fit inside it.
EXPECTED = {
    "p_mb": (_r("p_mb"), 1e-9),
    "lb_2004": (_r("lb_2004"), 0.005),
    "ub_2004": (_r("ub_2004"), 0.005),
    "lb_2005": (_r("lb_2005"), 0.005),
    "ub_2005": (_r("ub_2005"), 0.005),
    "lb_2006": (_r("lb_2006"), 0.005),
    "ub_2006": (_r("ub_2006"), 0.005),
    "lb_2007": (_r("lb_2007"), 0.005),
    "ub_2007": (_r("ub_2007"), 0.005),
    "lb_2008": (_r("lb_2008"), 0.005),
    "ub_2008": (_r("ub_2008"), 0.005),
    "lb_2009": (_r("lb_2009"), 0.005),
    "ub_2009": (_r("ub_2009"), 0.005),
    "p_iid": (_r("p_iid"), 0.009),
    "placebo_mb_T1_1": (_r("placebo_mb_T1_1"), 1e-9),
    "placebo_mb_T1_2": (_r("placebo_mb_T1_2"), 1e-9),
    "placebo_mb_T1_3": (_r("placebo_mb_T1_3"), 1e-9),
}
