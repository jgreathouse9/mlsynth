"""Path B: de Brabander et al. (2025) Monte Carlo -- Tables 9 to 12.

Tables 1 and 7 (see ``brabander_brexit_table1`` and
``brabander_brexit_insample``) say what the seven estimators claim about Brexit
and which of them tracks a known-zero effect best. Neither can say *why* one
does better, because on real data the truth is unobserved. Section 5 answers
that by simulating: calibrate a factor model to the Brexit panel, where the
true effect is zero by construction, and decompose each estimator's error into
the part traceable to covariates, the part traceable to the common factor, and
idiosyncratic noise.

The paper's finding is about the common-factor part. Demeaning removes it:
as the factor strength ``gamma`` rises, SC, MASC and ASCM pick up a bias that
grows roughly linearly, while DSC and SDID barely move. That is the mechanism
behind their recommendation, and it is what this case asserts.

Two independent checks, because a Monte Carlo aggregate on its own cannot
distinguish a wrong DGP from a wrong estimator:

1. Per-replication cross-validation. ``benchmarks/reference/brabander_mc/``
   holds 48 outcome matrices drawn by the authors' DGP in R together with
   ``synthdid``'s own error for each. mlsynth re-runs those same matrices and
   must reproduce each error value-for-value. This pins the estimators
   independently of any DGP question, and pins the simulator port independently
   of any estimator question.
2. Aggregate geometry. mlsynth's own simulator
   (:mod:`mlsynth.utils.sdid_helpers.simulate`) then reproduces the shape of
   Table 9 -- the bias slope in ``gamma``, and the ordering of methods by it.

    bias, gamma = 0 -> 1.5     paper      mlsynth     ratio to SC
    SC                        +0.362      +0.425       paper 1.0 / mlsynth 1.0
    DSC                       +0.073      +0.090       paper 5.0 / mlsynth 4.7
    SDID                      +0.068      +0.092       paper 5.3 / mlsynth 4.6

The ratio is the assertion, not the level and not the slope. At this case's
``M`` the Monte Carlo standard error on a bias *level* is about 0.08 -- larger
than the levels themselves -- and even the slopes still move by ~0.05 between
M=100 and M=200. The ratio is stable across both, because numerator and
denominator share the same draws, and it is what the paper's argument actually
rests on: demeaning cuts the factor-driven bias by a factor of roughly five.
Holding the draws common across ``gamma`` is what makes the comparison paired.

What is not covered, and why:

* ``SC(B) all`` -- the covariate-matching column. It needs ``Synth``'s nested V
  search per replication, which is the one expensive piece of this design.
* MASC and ASCM are in the aggregate check but not the cross-validation. The
  authors' ``masc`` package requires Gurobi, and ``augsynth`` 0.2.0 warns that
  its ridge tuning is unstable when the unit and pre-period counts are close --
  24 and 24 here -- and then errors out, so there is no reference to capture.
  mlsynth's ridge ASCM is cross-validated against a live augsynth run in
  ``ascm_kansas``, on a panel where augsynth does run.
* ``sigma = 0.25`` appears in the cross-validation but not the aggregate sweep;
  it rescales the noise without changing the geometry, and the runtime is
  better spent on ``gamma``.

Provenance
----------
* Paper: de Brabander, Juodis & Miyazato Szini (2025), Econometric Reviews,
  Section 5 and Tables 9-12. DGP: their equation (33); decomposition: (34).
* Authors' code: ``Simulations/simulations_no covariates.R``. The calibration
  window, the covariate set, the factor path, the loading construction, the
  noise scale, ``T0 = 24``, ``N0 = 23`` and the settings grid are read from it.
* Reference: ``benchmarks/reference/brabander_mc/reference.R`` (synthdid
  0.0.9), regenerable; the case skips its cross-validation half if absent.
* DGP port: :mod:`mlsynth.utils.sdid_helpers.simulate`, whose divergences from
  the paper's prose are pinned in ``mlsynth/tests/test_sdid_simulation.py``.
"""
from __future__ import annotations

import glob
import os
import warnings

import numpy as np

_REF = os.path.join(os.path.dirname(__file__), "..", "reference",
                    "brabander_mc")
_PANEL = os.path.join(os.path.dirname(__file__), "..", "reference",
                      "brabander_sdid_match", "brexit_panel.csv")

GAMMAS = (0.0, 0.25, 1.0, 1.5)
SIGMA = 1.0
M = 100                      # paper: 1000. See the tolerance note below.
SEED = 20250731
CROSSVAL_METHODS = ("SC", "DSC", "SDID")
ALL_METHODS = ("SC", "DSC", "SDID", "MASC", "ASCM")

_BASE = dict(outcome="y", treat="treat", unitid="unit", time="t",
             display_graphs=False)


def _long(Y):
    import pandas as pd

    n_periods, n_units = Y.shape
    return pd.DataFrame(
        [(f"u{j}", p + 1, Y[p, j], int(j == 0 and p == n_periods - 1))
         for p in range(n_periods) for j in range(n_units)],
        columns=["unit", "t", "y", "treat"])


def _fit_all(df, methods):
    """Each method's error at the final period; the true effect is zero."""
    from mlsynth import MASC, SDID, TSSC, VanillaSC

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "SC" in methods:
            out["SC"] = TSSC({"df": df, **_BASE, "method": "SC",
                              "inference": False}).fit()
        if "DSC" in methods:
            out["DSC"] = TSSC({"df": df, **_BASE, "method": "MSCa",
                               "inference": False}).fit()
        if "SDID" in methods:
            out["SDID"] = SDID({"df": df, **_BASE, "vce": "noinference",
                                "zeta": 0.0, "intercept_adjust": True}).fit()
        if "MASC" in methods:
            # One fold, per the Table 9 note ("reduced to one due to
            # computational costs"); m in [1, 10] as the paper sets.
            out["MASC"] = MASC({"df": df, **_BASE,
                                "m_grid": list(range(1, 11)),
                                "min_preperiods": 23}).fit()
        if "ASCM" in methods:
            out["ASCM"] = VanillaSC({"df": df, **_BASE, "augment": "ridge",
                                     "inference": False}).fit()
    return out


def _error(res):
    return float(np.asarray(res.time_series.estimated_gap, dtype=float)[-1])


def _crossvalidate() -> dict:
    """mlsynth against synthdid on the R-drawn panels, replication by replication."""
    import pandas as pd

    from benchmarks.compare import BenchmarkSkipped

    tau_path = os.path.join(_REF, "tau.csv")
    files = sorted(glob.glob(os.path.join(_REF, "outcome_*.csv")))
    if not os.path.exists(tau_path) or not files:
        raise BenchmarkSkipped(
            "brabander_mc reference capture absent; regenerate with "
            "benchmarks/reference/brabander_mc/reference.R")
    reference = pd.read_csv(tau_path)

    worst = {m: 0.0 for m in CROSSVAL_METHODS}
    n = 0
    for path in files:
        stem = os.path.basename(path)[len("outcome_"):-len(".csv")]
        setting, seed = stem.rsplit("_S", 1)
        Y = pd.read_csv(path).to_numpy(dtype=float)
        fits = _fit_all(_long(Y), CROSSVAL_METHODS)
        rows = reference[(reference.setting == setting)
                         & (reference.seed == int(seed))]
        for _, row in rows.iterrows():
            worst[row.method] = max(worst[row.method],
                                    abs(_error(fits[row.method]) - row.tau))
        n += 1
    return {"n_replications": float(n),
            **{f"crossval_max_dev_{m.lower()}": v for m, v in worst.items()}}


def _sweep() -> dict:
    """The aggregate Monte Carlo: bias and RMSE by gamma, on mlsynth's own DGP."""
    import pandas as pd

    from benchmarks.compare import BenchmarkSkipped
    from mlsynth.utils.sdid_helpers.simulate import (calibrate_brabander_dgp,
                                                     simulate_brabander)

    if not os.path.exists(_PANEL):  # pragma: no cover - the panel is committed
        raise BenchmarkSkipped(f"missing {_PANEL}")
    calibration = calibrate_brabander_dgp(pd.read_csv(_PANEL))

    stats = {}
    for gamma in GAMMAS:
        # Same seed at every gamma: the loadings and noise are shared, so the
        # comparison across gamma is paired and the slope is not swamped by the
        # level noise that dominates at this M.
        rng = np.random.default_rng(SEED)
        errors = {m: [] for m in ALL_METHODS}
        for _ in range(M):
            rep = simulate_brabander(calibration, gamma=gamma, sigma=SIGMA,
                                     rng=rng)
            for name, res in _fit_all(_long(rep.outcome), ALL_METHODS).items():
                errors[name].append(_error(res))
        stats[gamma] = {m: (float(np.mean(e)),
                            float(np.sqrt(np.mean(np.square(e)))))
                        for m, e in
                        ((m, np.asarray(errors[m])) for m in ALL_METHODS)}
    return stats


def run() -> dict:
    got = _crossvalidate()
    stats = _sweep()

    for gamma in GAMMAS:
        for m in ALL_METHODS:
            bias, rmse = stats[gamma][m]
            got[f"bias_{m.lower()}_g{gamma}"] = bias
            got[f"rmse_{m.lower()}_g{gamma}"] = rmse

    # The paper's mechanism: bias grows with the factor strength for the
    # estimators that do not demean, and barely moves for the two that do.
    slopes = {m: stats[1.5][m][0] - stats[0.0][m][0] for m in ALL_METHODS}
    for m, s in slopes.items():
        got[f"bias_slope_{m.lower()}"] = s
    # The paper-facing quantity is the ratio, not the slopes themselves. A
    # slope level still carries visible Monte Carlo noise at this M -- SC's
    # moves from 0.42 to 0.38 between M=100 and M=200 -- but the ratio is
    # stable across both (4.7) because the numerator and denominator share the
    # same draws, and it is what "demeaning removes the factor bias" means
    # quantitatively. Published: 0.362 / 0.073 = 5.0 and 0.362 / 0.068 = 5.3.
    got["bias_slope_ratio_sc_over_dsc"] = slopes["SC"] / slopes["DSC"]
    got["bias_slope_ratio_sc_over_sdid"] = slopes["SC"] / slopes["SDID"]
    got["demeaning_cuts_the_factor_bias"] = float(
        max(slopes["DSC"], slopes["SDID"])
        < 0.5 * min(slopes["SC"], slopes["MASC"], slopes["ASCM"]))
    got["bias_grows_monotonically_in_gamma"] = float(all(
        all(stats[a][m][0] < stats[b][m][0]
            for a, b in zip(GAMMAS, GAMMAS[1:]))
        for m in ("SC", "DSC", "SDID")))
    # Total RMSE is dominated by the noise, so it hardly moves with gamma --
    # the paper's Table 9 shows this and it is why bias, not RMSE, separates
    # the methods.
    got["rmse_is_flat_in_gamma"] = float(max(
        abs(stats[1.5][m][1] - stats[0.0][m][1]) for m in ("SC", "DSC", "SDID")
    ) < 0.10)
    return got


# Deterministic: the sweep is seeded per gamma (SEED) and the cross-validation
# reads frozen matrices, so re-runs reproduce every cell exactly.
#
# Two different kinds of tolerance here, for two different claims.
#
# The cross-validation cells are centered on the measured agreement with
# synthdid and gated at 1e-2. That band is the documented exact-solve versus
# ``min.decrease`` gap (``sdid_helpers/weights.py``); it is an order of
# magnitude below the effects being measured (biases of 0.1 to 0.4), so it
# separates "same estimator" from "different estimator" without demanding that
# mlsynth reproduce an early-stopping heuristic.
#
# The sweep cells are centered on mlsynth's measured values as regression
# guards, at 1e-3. They are reproducible to machine precision given the seed;
# the band only covers the rounding of the transcribed centers, and is still
# two orders below any change that would matter. They are NOT claims about the paper's
# cells: at M=100 the Monte Carlo standard error on a bias level is about 0.08,
# larger than the levels themselves, and even the slopes still move by ~0.05
# between M=100 and M=200.
#
# The paper-facing cells are the two slope ratios, gated at 1.5. Those are
# stable across M (4.7 at both 100 and 200 draws) because numerator and
# denominator share the same draws, and the band is wide enough to absorb the
# remaining Monte Carlo noise while excluding the only alternative that
# matters -- a ratio near 1, which would mean demeaning does nothing. Published
# values are 5.0 and 5.3; the booleans carry the qualitative orderings.
_XV, _MC, _RATIO = 1e-2, 1e-3, 1.5
EXPECTED = {
    "n_replications": (48.0, 0.0),
    "crossval_max_dev_sc": (0.004997, _XV),
    "crossval_max_dev_dsc": (0.001058, _XV),
    "crossval_max_dev_sdid": (0.002473, _XV),
    "bias_slope_sc": (0.42474, _MC),
    "bias_slope_dsc": (0.08961, _MC),
    "bias_slope_sdid": (0.09201, _MC),
    "bias_slope_ratio_sc_over_dsc": (4.74, _RATIO),
    "bias_slope_ratio_sc_over_sdid": (4.62, _RATIO),
    "demeaning_cuts_the_factor_bias": (1.0, 0.0),
    "bias_grows_monotonically_in_gamma": (1.0, 0.0),
    "rmse_is_flat_in_gamma": (1.0, 0.0),
}
