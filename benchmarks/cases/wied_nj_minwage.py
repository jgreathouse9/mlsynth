"""Path A: Wied (2026), New Jersey's 1992 minimum wage, via DRSC.

Reproduces the paper's empirical application on the author's own estimation
sample: Table 1's synthetic-control weights and Table 2's conditional
distributional treatment effects at five covariate values.

What is pinned, and why at these tolerances, follows from what the replication
measured rather than from taste.

The *deterministic* quantities -- the donor weights, the integrated effect
``f_hat``, the supremum statistics ``Tn`` -- involve no simulation and no
eigendecomposition. They reproduce the paper's printed precision exactly, so
they are pinned at that precision.

The *simulated* quantities -- critical values and therefore p-values -- are not
reproducible across LAPACK builds even with the seed fixed. Simulating the
limiting Gaussian process needs a matrix square root of the estimated kernel,
taken from ``numpy.linalg.eigh``, whose eigenvector signs are
implementation-defined; a different build gives a different factor and hence a
different realisation from identical draws. The distribution is unchanged, but
individual quantiles move by 0.2-0.5 percent. So p-values are pinned loosely,
and the qualitative pattern they carry -- which points reject on the corridor --
is pinned as a separate boolean row that a small numerical drift cannot flip.

One further property is asserted because it is load-bearing and invisible in the
published tables. The Gram matrix has a condition number near 1e5, and the
closed-form weight solve amplifies relative input error by roughly 1e6. A
float32 round-trip of the same sample moves the largest donor weight by ~0.04
(Florida 0.515 -> 0.505) while leaving the estimand almost untouched; random
noise of the same magnitude, which does not partially cancel the way rounding
does, moves weights by 0.12-0.17. ``weights_need_f64`` records the round-trip
gap, which is already 40x the tolerance the weights otherwise reproduce at. It is the reason the vendored sample is float64 and the reason
a future contributor should not "optimise" it.

Reference values: the paper's Tables 1 and 2. See
``benchmarks/reference/wied_nj_minwage/provenance.txt``. No R or Stata in CI.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parents[1] / "reference" / "wied_nj_minwage"

# Table 1: five largest synthetic control weights, by FIPS code.
PAPER_W = {12: 0.515, 36: 0.479, 46: -0.255, 32: 0.199, 29: -0.196}
# Table 2: f_hat x 1e3, Tn, p (full support), p (MW corridor).
PAPER_T2 = {
    "x0":  (0.58, 26.7, 0.294, 0.064),
    "x10": (1.71, 73.0, 0.054, 0.012),
    "xhl": (1.55, 37.1, 0.686, 0.150),
    "xlh": (0.61, 28.6, 0.750, 0.060),
    "x90": (0.21, 27.6, 0.824, 0.846),
}
EVAL = (("x0", 12, 10), ("x10", 10, 2), ("xhl", 16, 2),
        ("xlh", 10, 37), ("x90", 16, 37))
MW_LO, MW_HI = np.log(4.25), np.log(5.10)
COVS = ["e_std", "x_std", "x_std_sq"]


def _panel():
    d = pd.read_parquet(_DIR / "nj_estimation_sample.parquet")
    d["STATEFIP"] = d["STATEFIP"].astype(int)
    d["t"] = d["t"].astype(int)
    d["treat"] = ((d["STATEFIP"] == 34) & (d["t"] == 4)).astype(int)
    d["x_std_sq"] = d["x_std"] ** 2
    meta = json.loads((_DIR / "nj_meta.json").read_text())
    return d, meta


def _points(meta):
    pts = {}
    for name, ed, ex in EVAL:
        e = (ed - meta["me"]) / meta["se"]
        x = (ex - meta["mx"]) / meta["sx"]
        pts[name] = {"e_std": e, "x_std": x, "x_std_sq": x ** 2}
    return pts


def _fit(df, points):
    from mlsynth import DRSC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DRSC({
            "df": df, "outcome": "logwage", "treat": "treat",
            "unitid": "STATEFIP", "time": "t", "covariates": COVS,
            "evaluation_points": points, "n_grid": 38,
            "focus_region": (MW_LO, MW_HI), "display_graphs": False,
        }).fit()


def run() -> dict:
    d, meta = _panel()
    points = _points(meta)
    res = _fit(d, points)
    out: dict = {}

    # --- the sample is the paper's -------------------------------------
    out["n_total"] = float(len(d))
    out["n_donors"] = float(len(res.weights.donor_weights))
    out["m_active"] = float(len(res.outcome_grid))

    # --- structural diagnostics ----------------------------------------
    out["gram_condition_number"] = float(res.gram_condition_number)
    out["n_negative_weights"] = float(res.n_negative_weights)
    out["weights_sum_to_one"] = float(
        sum(res.weights.donor_weights.values()))

    # --- Table 1: deterministic, pinned tight --------------------------
    w = res.weights.donor_weights
    out["table1_max_weight_diff"] = float(max(
        abs(w[str(f)] - target) for f, target in PAPER_W.items()))

    # --- Table 2 -------------------------------------------------------
    f_diff, tn_diff, p_diff = 0.0, 0.0, 0.0
    for name, (pf, ptn, pp, ppc) in PAPER_T2.items():
        e = res.conditional_effects[name]
        f_diff = max(f_diff, abs(e.f_hat * 1e3 - pf))
        tn_diff = max(tn_diff, abs(e.t_stat - ptn))
        p_diff = max(p_diff, abs(e.p_value - pp),
                     abs(e.p_value_focused - ppc))
    out["table2_max_f_diff"] = f_diff
    out["table2_max_Tn_diff"] = tn_diff
    out["table2_max_p_diff"] = p_diff

    # The qualitative finding, as a boolean a p-value wobble cannot flip:
    # the corridor test rejects for low-skill young workers and for nobody
    # whose wages sit far above the new floor.
    ce = res.conditional_effects
    out["corridor_pattern_holds"] = float(
        ce["x10"].p_value_focused < 0.05 < ce["x90"].p_value_focused)
    out["x10_is_the_largest_effect"] = float(
        max(ce, key=lambda k: ce[k].f_hat) == "x10")
    out["x10_lower_bound_positive"] = float(ce["x10"].lower_bound > 0.0)

    # --- precision is load-bearing --------------------------------------
    d32 = d.copy()
    for c in ["logwage"] + COVS:
        d32[c] = d32[c].astype("float32").astype("float64")
    r32 = _fit(d32, points)
    w32 = r32.weights.donor_weights
    out["weights_need_f64"] = float(max(
        abs(w32[k] - w[k]) for k in w))
    out["estimand_survives_f32"] = float(max(
        abs(r32.conditional_effects[n].f_hat - ce[n].f_hat) * 1e3
        for n in ce))
    return out


#: Distances from the paper, so each row reads as "how far apart are we".
EXPECTED = {
    "n_total": (381953.0, 0.5),
    "n_donors": (42.0, 0.5),
    "m_active": (32.0, 0.5),

    "gram_condition_number": (9.64e4, 5e3),
    "n_negative_weights": (20.0, 0.5),
    "weights_sum_to_one": (1.0, 1e-9),

    # Deterministic: no simulation, no eigendecomposition.
    "table1_max_weight_diff": (0.0, 0.001),
    "table2_max_f_diff": (0.0, 0.01),
    "table2_max_Tn_diff": (0.0, 0.1),

    # Simulated: LAPACK eigenvector signs move critical values 0.2-0.5%,
    # which propagates to p-values. Loose on purpose; the pattern below is
    # what actually guards the finding.
    "table2_max_p_diff": (0.0, 0.05),
    "corridor_pattern_holds": (1.0, 0.5),
    "x10_is_the_largest_effect": (1.0, 0.5),
    "x10_lower_bound_positive": (1.0, 0.5),

    # A float32 round-trip moves the weights ~0.04 -- 40x the 0.001 they
    # otherwise reproduce at -- and the estimand by ~0.13e-3. Both halves
    # matter: the first says "keep float64", the second says "the published
    # effects are not hostage to it". The band is clear of zero, so a version
    # that silently stopped caring about precision would fail here.
    "weights_need_f64": (0.043, 0.03),
    "estimand_survives_f32": (0.0, 0.25),
}
