"""Differential benchmark: GEOX's readout against the SDID estimator it wraps.

GEOX is a design harness -- candidate nomination, backtest windows, effect
injection, power, MDE, ranking, constraints -- wrapped around a scoring engine.
With ``engine="sdid"`` the engine is mlsynth's synthetic difference-in-
differences, reached through ``mlsynth.utils.sdid_helpers.weights``, so the two
estimators are not independent implementations. What is unverified is the
harness *around* the fit: how GEOX aggregates the treated region, builds the
donor pool, indexes the pre/post split, and packages the readout. Any of those
can perturb the estimate without touching the estimator.

This case fixes the design question and asks the harness to disappear. Force one
market as the treated region (``treatment_size=1`` with ``to_be_treated``), hand
GEOX the real post-period, and compare its realized readout to
``SDID(...).fit()`` on the identical panel. Agreement to solver noise means the
harness is a no-op on the estimate; any drift is the harness, since the engine
is the same code.

The panel is the Abadie-Diamond-Hainmueller Proposition 99 smoking panel
(``basedata/smoking_data.csv``: 39 states x 31 years, 1970-2000, California
treated from 1989), which is also what ``sdid_prop99`` cross-validates against
the authors' ``synthdid`` R package. That gives the equivalence a second use:
whatever external validation SDID carries on this panel, GEOX inherits, and the
transitivity row measures it directly instead of asserting it.

Path: differential cross-validation, mlsynth against mlsynth. Not Path A or B --
no paper pairs GeoLift's market-selection loop with synthetic DiD, so there is
no published number for the composition. What can be checked is that composing
does not change the component, which is what this measures.

Four things are pinned, and one of them is a guard.

Equality. The realized ATT and every donor weight, against ``SDID(...).fit()``.

Generality. Six (treated unit, post split) pairs, so the match is not a property
of California in 1989.

Invariance. The readout across seven design-knob settings. The design search
chooses the region; once the region is fixed the readout must be a function of
the panel, the region and the split alone. A knob that moved it would mean the
search leaks into the answer.

Contrast. ``engine="augsynth"`` on the same inputs, which must *not* match. A
harness that collapsed every engine onto one number would pass every row above
while testing one estimator repeatedly.

Two quantities are reported that do not agree, because they are not the same
object. SDID's reported ``counterfactual`` is ``Y0 @ omega``, carrying no level
term, so it sits a constant below GEOX's and its ``pre_rmse`` is computed
against that level-free path. The constant is pinned instead of the paths: the
offset's standard deviation across the panel is a reported quantity, and zero
means the two differ by a level and nothing else. GEOX's path is the one for
which ``mean(observed - counterfactual)`` over the post window reproduces the
ATT both estimators report.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_REF = load_reference("sdid_prop99")["values"]

# (treated market, first post period). California 1989 is the real experiment;
# the rest are placebo assignments on the same panel, chosen to spread over the
# window (1980-1995) and over market size, so the match is not a property of one
# unit or one split.
_CASES = [
    ("California", 1989),
    ("Texas", 1989),
    ("Utah", 1985),
    ("Montana", 1995),
    ("Nevada", 1980),
    ("Pennsylvania", 1992),
]

# Knob settings the readout must be invariant to. Each drives the design search
# differently -- window length, how many backtests, the placebo draws, the
# injected grid, the reporting scale, the detection level -- and none of them is
# an input to the realized fit.
_KNOBS = [
    {},
    {"durations": [4]},
    {"durations": [6, 8]},
    {"n_backtests": 4},
    {"seed": 42, "n_draws": 30},
    {"effect_sizes": [-0.9, 0.0, 0.9]},
    {"how": "sum"},          # one treated market: sum and mean coincide
    {"alpha": 0.05},
]

_GRID = [-0.6, -0.3, 0.0, 0.3, 0.6]


def _panel(state: str, first_post: int) -> pd.DataFrame:
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["treat"] = ((df["state"] == state) & (df["year"] >= first_post)).astype(int)
    df["post"] = (df["year"] >= first_post).astype(int)
    return df


def _sdid(df: pd.DataFrame):
    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SDID({"df": df[["state", "year", "cigsale", "treat"]],
                     "outcome": "cigsale", "treat": "treat", "unitid": "state",
                     "time": "year", "display_graphs": False}).fit()


def _geox(df: pd.DataFrame, state: str, **over):
    from mlsynth import GEOX
    from mlsynth.config_models import GEOXConfig

    kwargs = dict(
        df=df[["state", "year", "cigsale", "post"]], unitid="state",
        time="year", outcome="cigsale", post_col="post",
        treatment_size=1, to_be_treated=[state], durations=[6],
        effect_sizes=_GRID, n_backtests=2, n_draws=10, seed=0, n_jobs=1,
        n_validation_backtests=0)
    kwargs.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = GEOX(GEOXConfig(**kwargs)).fit()
    if design.report is None:      # pragma: no cover - the grid always clears
        raise AssertionError(
            f"no design realized for {state}: the effect grid did not clear "
            "power_threshold, so there is nothing to compare.")
    return design


def run() -> dict:
    # --- California 1989: the real experiment, every accessor ---------------
    df = _panel("California", 1989)
    sdid, design = _sdid(df), _geox(df, "California")
    report = design.report

    att_diff = abs(float(report.att) - float(sdid.att))
    sw, gw = sdid.donor_weights, report.donor_weights
    weight_diff = (max(abs(sw[k] - gw[k]) for k in sw)
                   if set(sw) == set(gw) else float("inf"))

    # The two reported counterfactuals differ by SDID's level term, which its
    # own path omits. Zero spread means a level and nothing else.
    offset = (np.asarray(report.counterfactual, dtype=float).ravel()
              - np.asarray(sdid.counterfactual, dtype=float).ravel())
    offset_spread = float(np.std(offset))

    # GEOX's path is self-consistent: the post-window mean gap is the ATT.
    obs = np.asarray(report.time_series.observed_outcome, dtype=float).ravel()
    cf = np.asarray(report.counterfactual, dtype=float).ravel()
    n_pre = int((df["post"] == 0).sum() / df["state"].nunique())
    path_att_diff = abs(float(np.mean((obs - cf)[n_pre:])) - float(report.att))

    # --- generality: six treated units and post splits ---------------------
    worst = 0.0
    for state, first_post in _CASES:
        d = _panel(state, first_post)
        worst = max(worst, abs(float(_geox(d, state).report.att)
                               - float(_sdid(d).att)))

    # --- invariance: the readout does not move with the design knobs -------
    atts = [float(_geox(df, "California", **knob).report.att) for knob in _KNOBS]
    knob_spread = float(max(atts) - min(atts))

    # --- contrast: a different engine must give a different answer ---------
    aug = float(_geox(df, "California", engine="augsynth",
                      inference="placebo").report.att)
    engine_gap = abs(aug - float(sdid.att))

    return {
        "att_diff_vs_sdid": att_diff,
        "donor_weight_max_diff": weight_diff,
        "generality_max_att_diff": worst,
        "knob_invariance_spread": knob_spread,
        "counterfactual_offset_spread": offset_spread,
        "geox_path_att_consistency": path_att_diff,
        "geox_att_vs_synthdid_R": abs(float(report.att) - float(_REF["sdid_att"])),
        "augsynth_engine_gap": engine_gap,
    }


def comparison() -> dict:
    """GEOX's realized readout beside the SDID estimator, on Prop 99."""
    df = _panel("California", 1989)
    sdid, report = _sdid(df), _geox(df, "California").report
    aug = _geox(df, "California", engine="augsynth", inference="placebo").report
    return {
        "rows": [
            {"quantity": "ATT (California, 1989-2000)",
             "mlsynth": round(float(report.att), 9),
             "reference": round(float(sdid.att), 9)},
            {"quantity": "donor weights, max |difference|",
             "mlsynth": round(
                 max(abs(sdid.donor_weights[k] - report.donor_weights[k])
                     for k in sdid.donor_weights), 12),
             "reference": 0.0},
            {"quantity": "ATT under engine='augsynth' (contrast)",
             "mlsynth": round(float(aug.att), 9),
             "reference": round(float(sdid.att), 9)},
        ],
        "mlsynth_call": {
            "estimator": "GEOX",
            "config": {"post_col": "post", "treatment_size": 1,
                       "to_be_treated": ["California"], "engine": "sdid",
                       "n_backtests": 2, "n_draws": 10, "seed": 0}},
        "reference": {"impl": "mlsynth SDID (the engine GEOX wraps)",
                      "version": "same tree; differential, not external"},
    }


# Tolerances.
#
# The four agreement rows compare two calls into the same weight programs on the
# same arrays, so they are exactly zero on this tree, not merely close. 1e-9
# records that the claim is bit-level agreement while leaving room for a BLAS
# that reorders a reduction; it is four orders tighter than any real regression
# in aggregation, donor-pool construction or pre/post indexing could hide under,
# since those shift the ATT by whole packs.
#
# `geox_att_vs_synthdid_R` is inherited from `sdid_prop99` -- the same 1.6e-3
# unit-weight ridge (zeta) solver gap against the authors' R, carried through an
# equality that is exact -- so it takes that case's 0.02.
#
# `augsynth_engine_gap` is a guard, not a match: it fails if the two engines
# converge, which would make every row above vacuous. 0.71 packs is the observed
# separation and 0.35 is half of it, so the row fails only if the gap halves --
# loose enough that ordinary augsynth solver drift does not trip it, tight
# enough that a collapse to a single estimator does.
EXPECTED = {
    "att_diff_vs_sdid": (0.0, 1e-9),
    "donor_weight_max_diff": (0.0, 1e-9),
    "generality_max_att_diff": (0.0, 1e-9),
    "knob_invariance_spread": (0.0, 1e-9),
    "counterfactual_offset_spread": (0.0, 1e-9),
    "geox_path_att_consistency": (0.0, 1e-9),
    "geox_att_vs_synthdid_R": (0.0, 0.02),
    "augsynth_engine_gap": (0.7135, 0.35),
}
