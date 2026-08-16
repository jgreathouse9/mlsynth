"""COMPSC cross-validation: mlsynth vs the author's replication R (Pennsylvania).

Cross-validates ``COMPSC(geometry="alr", baseline="renewables")`` against the
replication script for Boussim, O. (2026), *"Compositional Synthetic Controls"*
(arXiv:2607.16991) -- ``quadprog::solve.QP`` on the stacked ALR log-odds, the
Aitchison distance, and the 42-donor placebo permutation, run live and captured
under ``benchmarks/reference/compsc_pennsylvania_r/``.

The companion case ``compsc_pennsylvania`` pins the same fit against the
*published tables*. This one pins it against the code that produced them, which
is a different question: the tables are rounded to three decimals, so they
cannot separate a genuine disagreement from a printing convention. Agreement
here is value-for-value:

  =============================  ==================
  quantity                       max abs difference
  =============================  ==================
  all ten donor weights          3e-9
  Aitchison pre-RMSPE            1e-10
  share pre-RMSPE (pp)           2e-9
  Table 2, 20 years x 5 columns  1e-7
  placebo RMSPE ratio            1e-8
  placebo p-value                1e-11
  =============================  ==================

That is solver tolerance -- mlsynth's active-set QP against ``quadprog``'s --
across a chain that includes 42 separate placebo QPs, and it holds without any
tuning of either side.

The tenth donor. Boussim's Table 1 lists nine weights, and
``compsc_pennsylvania`` recorded mlsynth's Arkansas weight of 0.000924 as an
immaterial difference from ``quadprog``, which was assumed to drive it to zero.
Running the author's script shows ``quadprog`` returns 0.000924 for Arkansas as
well, agreeing with mlsynth to 6e-10, and reports ten active donors. The
divergence is between the paper's printed table and the author's own code, not
between the code and mlsynth, and the earlier attribution was a guess made
without a reference to run.

Data. The script runs against ``basedata/pa_aeps_generation.csv`` -- the panel
mlsynth reads -- so both sides see identical inputs. It derives its own donor
pool (structural zeros plus OH, WV and VT), which on the pre-filtered panel
leaves the same 42 donors it derives from the author's full 51-state file. Run
against that file instead, the reference moves by 1e-7 on the weights and 7e-6
on the effects, so the two panels are interchangeable here at the precision this
case pins.

That stability is not automatic and is the reason it is measured rather than
assumed: the same class of check on ``Synth::synth`` in ``spillsynth_iterative_
germany`` moves the estimate by 149 USD under a smaller perturbation. A
deterministic active-set QP on a well-conditioned log-odds design behaves; a
nested V search need not.
"""
from __future__ import annotations

import warnings

from benchmarks.reference import load_reference

_BUNDLE = "compsc_pennsylvania_r"
_CATEGORIES = ("gas", "fossil", "renewables")


def _fit():
    from benchmarks.cases.compsc_pennsylvania import _fit as _paper_fit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _paper_fit("alr", "renewables")


def _reference():
    ref = load_reference(_BUNDLE)
    return ref["values"], ref["weights"]


def run() -> dict:
    from benchmarks.cases.compsc_pennsylvania import _years

    vals, wref = _reference()
    res = _fit()
    years = _years(res)
    k = {c: i for i, c in enumerate(_CATEGORIES)}
    wml = {s: float(w) for s, w in res.donor_weights.items() if w > 0}

    labels = set(wml) | set(wref)
    weight_diff = max(abs(wml.get(s, 0.0) - wref.get(s, 0.0)) for s in labels)

    effects_diff = 0.0
    for i, year in enumerate(years):
        if year <= 2003:
            continue
        mine = {
            "att_gas": 100 * res.share_effects[i, k["gas"]],
            "att_fos": 100 * res.share_effects[i, k["fossil"]],
            "att_ren": 100 * res.share_effects[i, k["renewables"]],
            "lrte_gf": res.log_ratio_effects[i, k["gas"], k["fossil"]],
            "lrte_gr": res.log_ratio_effects[i, k["gas"], k["renewables"]],
            "delta_A": res.aitchison_distance[i],
        }
        for name, value in mine.items():
            effects_diff = max(
                effects_diff, abs(float(value) - vals[f"{name}_{year}"]))

    pb = res.placebo
    return {
        # weights, value for value against the author's quadprog
        "n_active_donors": float(len(wml)),
        "n_active_donors_ref": vals["n_active_donors"],
        "weights_max_diff": weight_diff,
        "arkansas_weight": wml.get("AR", 0.0),
        "arkansas_weight_ref": wref.get("AR", 0.0),
        # pre-treatment fit
        "aitchison_rmspe_pre_diff": abs(
            float(res.pre_treatment_aitchison_rmspe) - vals["aitchison_rmspe_pre"]),
        "share_rmspe_pre_diff": abs(
            float(100 * res.pre_treatment_share_rmspe) - vals["share_rmspe_pre_pp"]),
        # every post-period effect the script reports
        "effects_max_diff": effects_diff,
        # placebo inference, 42 donor refits
        "placebo_ratio_diff": abs(
            float(pb.treated_ratio) - vals["rmspe_ratio_pa"]),
        "placebo_n_retained_diff": abs(
            float(pb.n_retained) - vals["placebo_n_retained"]),
        "placebo_p_value_diff": abs(float(pb.p_value) - vals["placebo_p_value"]),
    }


def comparison() -> dict:
    """mlsynth against the author's replication R, quantity by quantity."""
    from benchmarks.cases.compsc_pennsylvania import _years

    vals, wref = _reference()
    res = _fit()
    years = _years(res)
    k = {c: i for i, c in enumerate(_CATEGORIES)}
    wml = {s: float(w) for s, w in res.donor_weights.items() if w > 0}

    rows = [{"quantity": f"weight[{s}]",
             "mlsynth": round(wml.get(s, 0.0), 9),
             "reference": round(wref.get(s, 0.0), 9)}
            for s in sorted(set(wml) | set(wref), key=lambda s: -wref.get(s, 0.0))]
    rows += [
        {"quantity": "Aitchison RMSPE (pre)",
         "mlsynth": round(float(res.pre_treatment_aitchison_rmspe), 9),
         "reference": round(vals["aitchison_rmspe_pre"], 9)},
        {"quantity": "share RMSPE (pre, pp)",
         "mlsynth": round(float(100 * res.pre_treatment_share_rmspe), 9),
         "reference": round(vals["share_rmspe_pre_pp"], 9)},
    ]
    for year in (2010, 2016, 2022, 2023):
        i = years.index(year)
        rows.append({"quantity": f"ATT_gas[{year}] (pp)",
                     "mlsynth": round(float(100 * res.share_effects[i, k["gas"]]), 9),
                     "reference": round(vals[f"att_gas_{year}"], 9)})
        rows.append({"quantity": f"LRTE_gas|fos[{year}]",
                     "mlsynth": round(
                         float(res.log_ratio_effects[i, k["gas"], k["fossil"]]), 9),
                     "reference": round(vals[f"lrte_gf_{year}"], 9)})
    rows += [
        {"quantity": "placebo RMSPE ratio (PA)",
         "mlsynth": round(float(res.placebo.treated_ratio), 9),
         "reference": round(vals["rmspe_ratio_pa"], 9)},
        {"quantity": "placebo units retained",
         "mlsynth": float(res.placebo.n_retained),
         "reference": vals["placebo_n_retained"]},
        {"quantity": "placebo p-value",
         "mlsynth": round(float(res.placebo.p_value), 9),
         "reference": round(vals["placebo_p_value"], 9)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "COMPSC",
            "config": {"geometry": "alr", "baseline": "renewables",
                       "outcomes": list(_CATEGORIES), "target": "gas"},
        },
        "reference": {
            "impl": "Boussim (2026) csc_replication.R (quadprog::solve.QP on the "
                    "stacked ALR log-odds), run live on basedata/"
                    "pa_aeps_generation.csv",
            "version": "arXiv:2607.16991 replication script",
        },
    }


# Deterministic on both sides: an active-set QP with no RNG here, quadprog's
# dual method there, on a committed panel. Every ``*_diff`` cell is a distance
# to the captured reference run and is pinned at zero with a tolerance an order
# of magnitude above the observed residual, so a real divergence fails while
# BLAS-level noise does not. ``arkansas_weight`` is pinned against the
# reference's own value, not the paper's table, which omits it.
EXPECTED = {
    "n_active_donors": (10.0, 0.5),
    "n_active_donors_ref": (10.0, 0.5),
    "weights_max_diff": (0.0, 1e-7),
    "arkansas_weight": (0.000924, 1e-5),
    "arkansas_weight_ref": (0.000924, 1e-5),
    "aitchison_rmspe_pre_diff": (0.0, 1e-8),
    "share_rmspe_pre_diff": (0.0, 1e-7),
    "effects_max_diff": (0.0, 1e-5),
    "placebo_ratio_diff": (0.0, 1e-6),
    "placebo_n_retained_diff": (0.0, 0.5),
    "placebo_p_value_diff": (0.0, 1e-9),
}
