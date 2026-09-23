"""CWZ debiased SC t-test: Table 3's Monte Carlo, live against the authors' code.

Path B, scenario 3. The application-calibrated simulation study behind Table 3
of Chernozhukov, Wüthrich & Zhu's t-test paper, run from the authors' own JPE
replication package: ``calibration_dgps.R`` for the calibration and
``common_functions.R::sim_one_sample`` for the draw and the estimators. The
reference side is captured in ``benchmarks/reference/cwz_ttest_mc/``.

This is the live counterpart to ``cwz_mc``, which reimplements the design in
Python and tracks the published cells. A published cell is rounded to two
decimals, so agreement with one is agreement to the printing precision and
cannot separate a correct port from one off by a percent. Here both sides run
and the comparison is at full precision.

Two claims, separated, as in ``cwz_conformal_mc``:

The estimator is identical. The reference dumps its first ten panels for DGP 1
and DGP 8 with the ATT and standard error it computed on each; mlsynth is handed
those exact panels and has to return those exact numbers. The debiased t-test is
a deterministic function of the panel -- :math:`K` fold refits and a
self-normalized statistic, no resampling anywhere -- so this comparison carries
no Monte-Carlo error at all and its tolerance is solver noise. DGP 1 is the
correctly specified case and DGP 8 the heterogeneous-trend stress case the paper
singles out, so the pair brackets what the estimator has to handle.

The design is faithful. mlsynth draws its own panels from its own port of the
design and has to reproduce the reference's coverage, interval length and bias
across all nine DGPs up to Monte-Carlo error. The calibration itself is read
from the reference's dump instead of re-derived: R's ``ar()`` selects its order
by AIC and solves Yule-Walker, and reimplementing that in Python would put a
second estimator between the two sides for no gain, since the calibration is an
input to the design and not the thing under test.

The geometry the paper reports comes through. Coverage is near nominal where the
theory covers the design (DGP 1 and 2), mildly short under a common trend with
one deviating donor (DGP 6), and short by a wide margin under heterogeneous
trends (DGP 8), which is outside the theory and the paper's stress case. DGP 3
and DGP 4 return identical cells, which is a property of the design and not a
transcription slip: they differ only by a linear trend common to every donor,
and a weight vector summing to one absorbs a common trend exactly, in the fit
and in the gap alike.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.cwz_common import (
    TTEST_DGPS, load_reference_draws, load_ttest_calibration, oracle_ttest,
    simulate_ttest_panel,
)
from benchmarks.reference import reference_value

_REF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reference", "cwz_ttest_mc"))

ALPHA = 0.1
K = 3
T0, T1 = 30, 16
N_REPS = 300           # matches the reference; the paper runs 5000
SEED = 20260814
DUMP_DGPS = (1, 8)


def _calibration() -> dict:
    if not os.path.exists(os.path.join(_REF, "calibration_units.tsv")):
        raise BenchmarkSkipped("cwz_ttest_mc calibration dump not available")
    return load_ttest_calibration(
        _REF,
        rho_u=reference_value("cwz_ttest_mc", "calib_rho_u"),
        var_u=reference_value("cwz_ttest_mc", "calib_var_u"),
    )


def _ttest(y: np.ndarray, Y0: np.ndarray) -> tuple:
    """``VanillaSC(inference="ttest")`` on one panel, through the public API."""
    from mlsynth import VanillaSC

    n_periods, n_donors = Y0.shape
    units = ["treated"] + [f"donor{j:02d}" for j in range(n_donors)]
    values = np.column_stack([y, Y0])
    df = pd.DataFrame({
        "unit": np.repeat(units, n_periods),
        "t": np.tile(np.arange(n_periods), n_donors + 1),
        "yv": values.T.ravel(),
    })
    df["treated"] = ((df.unit == "treated") & (df.t >= T0)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({
            "df": df, "outcome": "yv", "treat": "treated", "unitid": "unit",
            "time": "t", "backend": "outcome-only", "inference": "ttest",
            "ttest_K": K, "alpha": ALPHA, "display_graphs": False,
        }).fit()
    d = res.inference.details
    return float(d["att_debiased"]), float(d["se"])


def _cells(dgp: int, calib: dict) -> dict:
    """Coverage, mean length and mean bias over ``N_REPS`` of mlsynth's own draws."""
    from scipy.stats import t as student_t

    crit = float(student_t.ppf(1.0 - ALPHA / 2.0, df=K - 1))
    rng = np.random.default_rng(SEED + dgp)
    covered = np.empty(N_REPS)
    length = np.empty(N_REPS)
    bias = np.empty(N_REPS)
    covered_oracle = np.empty(N_REPS)
    for r in range(N_REPS):
        y, Y0, w_true = simulate_ttest_panel(
            dgp, calib, pre_periods=T0, post_periods=T1, rng=rng)
        att, se = _ttest(y, Y0)
        covered[r] = abs(att / se) <= crit
        length[r] = 2.0 * crit * se
        bias[r] = att
        att_o, se_o = oracle_ttest(y, Y0, T0, T1, w_true, K)
        covered_oracle[r] = abs(att_o / se_o) <= crit
    return {
        f"cov_sc_dgp{dgp}": float(covered.mean()),
        f"leng_sc_dgp{dgp}": float(length.mean()),
        f"bias_sc_dgp{dgp}": float(bias.mean()),
        f"cov_oracle_dgp{dgp}": float(covered_oracle.mean()),
    }


def run() -> dict:
    out: dict = {}

    # The seam: the reference's own panels, its own ATT and standard error.
    for dgp in DUMP_DGPS:
        path = os.path.join(_REF, f"draws_dgp{dgp}.tsv")
        if not os.path.exists(path):
            raise BenchmarkSkipped(f"draws_dgp{dgp}.tsv not available")
        for i, (y, Y0) in enumerate(load_reference_draws(path), start=1):
            att, se = _ttest(y, Y0)
            out[f"draw_dgp{dgp}_{i:02d}_att"] = att
            out[f"draw_dgp{dgp}_{i:02d}_se"] = se

    # The design: mlsynth's own draws, compared as Table 3's cells.
    calib = _calibration()
    for dgp in TTEST_DGPS:
        out.update(_cells(dgp, calib))
    return out


def comparison() -> dict:
    """mlsynth's debiased t-test against ``scinference``'s, on the authors' design.

    Seed-matched panels where both sides see identical data and must agree
    exactly, and the nine Table 3 coverage cells where each side draws its own.
    """
    got = run()
    rows = []
    for dgp in DUMP_DGPS:
        for i in (1, 5, 10):
            for what in ("att", "se"):
                key = f"draw_dgp{dgp}_{i:02d}_{what}"
                rows.append({"quantity": f"{what.upper()}, DGP {dgp}, draw {i}",
                             "mlsynth": round(got[key], 6),
                             "reference": round(reference_value("cwz_ttest_mc", key), 6)})
    for dgp in TTEST_DGPS:
        key = f"cov_sc_dgp{dgp}"
        rows.append({"quantity": f"coverage, DGP {dgp}",
                     "mlsynth": round(got[key], 4),
                     "reference": round(reference_value("cwz_ttest_mc", key), 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC",
                         "config": {"inference": "ttest", "ttest_K": K,
                                    "alpha": ALPHA, "backend": "outcome-only"}},
        "reference": {"impl": "R package scinference (ttest) driven by the authors' "
                              "own calibrated simulation design",
                      "version": "scinference v1.0.0 (567c688)"},
    }


_r = lambda k: reference_value("cwz_ttest_mc", k)

# Three tolerances, for three different quantities.
#
# The per-draw ATT and standard error are deterministic on both sides: K fold
# refits of a simplex SC and a self-normalized statistic, with no resampling
# anywhere. The only inexact step is the simplex solve, where mlsynth's QP and
# limSolve's lsei(type=1) agree to ~1e-7 on these panels, and the fold structure
# multiplies that by a small constant. 1e-5 absolute is a floating-point band on
# quantities of order 1e-2 to 1e-1, which is roughly a part in a thousand: far
# inside any real difference in fold construction, block length or debiasing
# term, all of which would move these by whole percent.
#
# Coverage is a proportion over 300 draws per side. The difference of two
# independent rates has standard error sqrt(2 p (1-p) / 300), largest at p = 0.5
# where it is 0.041, and DGP 8 sits near enough to that (0.63 against 0.56) for
# the worst case to be the one to size against: 0.12 is three of those. That
# still leaves the geometry the case exists to check intact, because the gap
# between DGP 1's near-nominal 0.88 and DGP 8's 0.63 is 0.25, twice the band. A
# port that lost the undercoverage would fail.
#
# Length and bias are averages of heavy-tailed quantities -- the misspecified
# DGPs 3 to 5 put an interval length in the hundreds -- so their bands are
# relative: 25 percent of the reference value for length, and for bias an
# absolute band scaled to the DGP, since a mean bias near zero has no relative
# scale to speak of.
_LENG_TOL = {dgp: max(0.25 * abs(_r(f"leng_sc_dgp{dgp}")), 0.05) for dgp in TTEST_DGPS}
_BIAS_TOL = {dgp: max(0.25 * abs(_r(f"bias_sc_dgp{dgp}")), 0.05) for dgp in TTEST_DGPS}

EXPECTED = {
    **{f"draw_dgp{dgp}_{i:02d}_{what}": (_r(f"draw_dgp{dgp}_{i:02d}_{what}"), 1e-5)
       for dgp in DUMP_DGPS for i in range(1, 11) for what in ("att", "se")},
    **{f"cov_sc_dgp{dgp}": (_r(f"cov_sc_dgp{dgp}"), 0.12) for dgp in TTEST_DGPS},
    **{f"cov_oracle_dgp{dgp}": (_r(f"cov_oracle_dgp{dgp}"), 0.12) for dgp in TTEST_DGPS},
    **{f"leng_sc_dgp{dgp}": (_r(f"leng_sc_dgp{dgp}"), _LENG_TOL[dgp]) for dgp in TTEST_DGPS},
    **{f"bias_sc_dgp{dgp}": (_r(f"bias_sc_dgp{dgp}"), _BIAS_TOL[dgp]) for dgp in TTEST_DGPS},
}
