"""Cross-validation: mlsynth VanillaSC ``inference="lto"`` vs the authors' own code.

Cross-validation against the canonical implementation. mlsynth's Leave-Two-Out
(LTO) refined placebo test (:func:`mlsynth.utils.vanillasc_helpers.lto.lto_placebo_test`,
the ``VanillaSC(inference="lto")`` path) is checked, value for value, against the
authors' own replication of Sudijono & Lei (2023/2025), *"Inference for Synthetic
Controls via Refined Placebo Tests"* (arXiv:2401.07152) --
`tsudijon/LeaveTwoOutSCI <https://github.com/tsudijon/LeaveTwoOutSCI>`_ -- across
*all three* of the paper's empirical applications: California Proposition 99,
West German reunification, and the Basque Country.

The LTO test replaces the ordinary placebo/permutation test (whose p-value lives
on the coarse grid ``{1/N, ..., 1}`` and has zero size for ``alpha < 1/N``) with
an ``O(N^2)`` construction: for every unordered pair ``{i, j}`` of control units,
the synthetic control is refit for the treated unit and for each of ``i, j`` on
the donor pool with ``i`` and ``j`` removed; the treated unit "wins" the triple
when its post/pre error ratio exceeds ``max`` of the two donors', and the naive
LTO p-value is the fraction of pairs it does not win.

Shared outcome-only SC (isolates the inference machinery)
---------------------------------------------------------
The synthetic control is outcome-only on both sides -- a simplex least-squares
fit of the pre-treatment outcome path -- so the comparison isolates the LTO
inference machinery (the pair loop, the rank statistic, the p-value) rather than
the SC solver, which mlsynth already cross-validates in ``vanillasc_prop99``,
``synth_prop99`` and ``mscmt_basque``. The p-value is a rank statistic (monotone
in the error ratio), so the MSE-ratio vs RMSPE-ratio convention is immaterial.

Value for value (three applications)
------------------------------------
Both sides run the identical deterministic pair loop, so mlsynth reproduces the
authors' naive LTO p-value exactly on every panel:

  ===============  ==========  =======  =====  ================
  Application      p-value     losses   pairs  treated RMSPE-r
  ===============  ==========  =======  =====  ================
  Prop 99 (CA)     0.10384     73       703    ~12.44
  W. Germany       0.00833     1        120    ~30.4
  Basque Country   0.69167     83       120    ~12.5
  ===============  ==========  =======  =====  ================

Reference (live captured run)
-----------------------------
``benchmarks/reference/lto_refined_placebo/reference.R`` reproduces the authors'
LTO pair loop with an independent LowRankQP simplex-SC solve on the three in-repo
panels; the p-values, loss counts and treated ratios are captured under
``benchmarks/reference/lto_refined_placebo/`` and pinned here via
:func:`reference_value`. The upstream repo is MIT-licensed ((c) 2023 Tim
Sudijono); the method is reproduced on the in-repo public data rather than
vendored. Regenerate with
``python benchmarks/reference/generate.py lto_refined_placebo``.

Provenance
----------
* ``basedata/smoking_data.csv`` -- ADH (2010) Prop 99 (39 states, 1970-2000;
  California treated 1989, ``T0=19``).
* ``basedata/german_reunification.csv`` -- West German reunification (17 OECD
  countries, 1960-2003; West Germany treated 1990, ``T0=30``).
* ``basedata/basque_jasa.csv`` -- Abadie-Gardeazabal Basque (Spain aggregate
  dropped; Basque Country treated 1971, ``T0=16``).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from benchmarks.reference import reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"

# (prefix, file, outcome, unitid, treated, treat_year, drop_units)
_PANELS = [
    ("ca", "smoking_data.csv", "cigsale", "state", "year", "California", 1989, [], 0.05),
    ("ger", "german_reunification.csv", "gdp", "country", "year", "West Germany", 1990, [], 0.05),
    ("bq", "basque_jasa.csv", "gdpcap", "regionname", "year",
     "Basque Country (Pais Vasco)", 1971, ["Spain (Espana)"], 0.10),
]


def _mlsynth_lto(fname, outcome, unitid, time, treated, treat_year, drop, alpha):
    from mlsynth import VanillaSC

    df = pd.read_csv(_BASE / fname)
    if drop:
        df = df[~df[unitid].isin(drop)].copy()
    df["treat"] = ((df[unitid] == treated) & (df[time] >= treat_year)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({
            "df": df, "outcome": outcome, "treat": "treat",
            "unitid": unitid, "time": time,
            "inference": "lto", "alpha": alpha, "display_graphs": False,
        }).fit()
    inf = res.inference
    return (float(inf.p_value), int(inf.details["treated_losses"]),
            int(inf.details["n_pairs"]), float(inf.details["treated_rmspe_ratio"]))


def _all():
    out = {}
    for prefix, fname, outcome, unitid, time, treated, ty, drop, alpha in _PANELS:
        out[prefix] = _mlsynth_lto(fname, outcome, unitid, time, treated, ty, drop, alpha)
    return out


def run() -> dict:
    res = _all()
    out = {}
    for pfx in ("ca", "ger", "bq"):
        p, losses, pairs, ratio = res[pfx]
        ref_p = reference_value("lto_refined_placebo", f"{pfx}_p_value")
        ref_losses = reference_value("lto_refined_placebo", f"{pfx}_treated_losses")
        ref_pairs = reference_value("lto_refined_placebo", f"{pfx}_n_pairs")
        ref_ratio = reference_value("lto_refined_placebo", f"{pfx}_treated_rmspe_ratio")
        out[f"{pfx}_p_abs_diff_vs_authors"] = float(abs(p - ref_p))
        out[f"{pfx}_losses_match"] = 1.0 if losses == int(round(ref_losses)) else 0.0
        out[f"{pfx}_n_pairs_match"] = 1.0 if pairs == int(round(ref_pairs)) else 0.0
        out[f"{pfx}_ratio_abs_diff_vs_authors"] = float(abs(ratio - ref_ratio))
    return out


def comparison() -> dict:
    """mlsynth LTO vs ``tsudijon/LeaveTwoOutSCI`` across all three applications."""
    res = _all()
    labels = {"ca": "Prop99", "ger": "W.Germany", "bq": "Basque"}
    rows = []
    for pfx in ("ca", "ger", "bq"):
        p, losses, pairs, ratio = res[pfx]
        rows.append({"quantity": f"{labels[pfx]}: naive LTO p-value",
                     "mlsynth": round(p, 6),
                     "reference": round(reference_value("lto_refined_placebo", f"{pfx}_p_value"), 6)})
        rows.append({"quantity": f"{labels[pfx]}: treated losses / pairs",
                     "mlsynth": f"{losses}/{pairs}",
                     "reference": f"{int(reference_value('lto_refined_placebo', f'{pfx}_treated_losses'))}/"
                                  f"{int(reference_value('lto_refined_placebo', f'{pfx}_n_pairs'))}"})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC (inference='lto')",
                         "config": {"inference": "lto",
                                    "panels": ["Prop99", "W.Germany", "Basque"]}},
        "reference": {"impl": "tsudijon/LeaveTwoOutSCI LTO pair loop "
                              "(outcome-only SC via LowRankQP), live run, captured",
                      "version": "Sudijono & Lei (2023/2025) arXiv:2401.07152, "
                                 "on in-repo panels"},
    }


# Both sides run the identical deterministic O(N^2) pair loop on the shared
# outcome-only simplex SC, so mlsynth reproduces the authors' naive LTO p-value
# to the digit on all three panels (CA 0.10384, W.Germany 0.00833, Basque
# 0.69167); losses/pairs match exactly and the treated RMSPE ratio agrees to
# solver tolerance (the ratio does not affect the rank-based p-value). Targets
# are pinned from the live captured R run via reference_value.
EXPECTED = {
    "ca_p_abs_diff_vs_authors": (0.0, 1e-6),
    "ca_losses_match": (1.0, 0.0),
    "ca_n_pairs_match": (1.0, 0.0),
    "ca_ratio_abs_diff_vs_authors": (0.0, 1e-2),
    "ger_p_abs_diff_vs_authors": (0.0, 1e-6),
    "ger_losses_match": (1.0, 0.0),
    "ger_n_pairs_match": (1.0, 0.0),
    "ger_ratio_abs_diff_vs_authors": (0.0, 1e-1),
    "bq_p_abs_diff_vs_authors": (0.0, 1e-6),
    "bq_losses_match": (1.0, 0.0),
    "bq_n_pairs_match": (1.0, 0.0),
    "bq_ratio_abs_diff_vs_authors": (0.0, 1e-2),
}
