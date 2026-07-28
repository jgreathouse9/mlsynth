"""VanillaSC cross-validation vs Synth and tidysynth, including placebo inference.

Cross-validation (scenario 3, full reference packages). mlsynth's classic
synthetic control has a Path-A case already (``vanillasc_prop99``, which checks
the donor weights and pre-treatment fit against Abadie-Diamond-Hainmueller
(2010) Table 2). What had no external check at all was the *inference*: the
in-space placebo test that ranks the treated unit's post/pre RMSPE ratio among
the donors. This case supplies it, on the canonical Proposition 99 panel,
against the two R packages a user would otherwise reach for.

Two references, because they answer different questions.

Reference A -- Synth with a uniform ``custom.v``. Fixing V collapses Synth's
inner problem to exactly the V-free quadratic program mlsynth solves with
``backend="outcome-only"``, so the comparison isolates solver quality and the
placebo machinery from the predictor-weight search. That search is not
identified and is famously unstable across implementations, so leaving it on
would confound everything.

Reference B -- tidysynth run as its vignette intends: the full ADH predictor
spec with the V search on, and its own ``generate_placebos`` /
``grab_significance`` pipeline. This is the practical question: does mlsynth's
placebo inference reach the same conclusion as the package next door?

What is *asserted* versus merely *recorded*
-------------------------------------------

The gates below assert agreement only where two correct implementations of the
same estimand genuinely must agree: the placebo rank, the p-value, the
per-unit ordering of the placebo distribution, and the donor weights against
Abadie's published table.

Three quantities are deliberately **recorded but not gated**, because on them
mlsynth and the reference disagree and mlsynth is the one that is right --
turning "we are better" into a pass condition would let a genuine regression
slip through (a bug that lowered our objective for the wrong reason would still
pass). They are reported so a change shows up in the benchmark output:

* ``synth_minus_mlsynth_pre_ssr`` -- both minimise the same pre-period sum of
  squares under Reference A. mlsynth reaches 52.13 against Synth's 55.70, i.e.
  a 6.4 percent lower value of the very objective Synth is optimising. Synth's
  solution is also non-sparse, smearing ~1e-4 weights across roughly sixteen
  extra donors, which is characteristic of its interior-point ``ipop`` solver.
* ``synth_ipop_failures`` -- ``ipop`` aborts outright on Nebraska ("system is
  computationally singular"), so Synth silently fits 38 of the 39 placebos and
  the p-value's denominator changes underneath it. mlsynth fits all 39.
* ``adh_weight_maxdiff_{mlsynth,tidysynth}`` -- distance from ADH (2010)
  Table 2. Both are close; mlsynth is closer (0.005 vs 0.017).

A unit trap worth naming
------------------------

tidysynth's ``mspe_ratio`` is ``post_mspe / pre_mspe`` -- the SQUARED scale.
mlsynth reports the RMSPE ratio, its square root. Compared raw they look an
order of magnitude apart (120.49 vs 11.36); reconciled, they are
:math:`\\sqrt{120.49} = 10.98` against 11.36, a 3.5 percent gap driven by the
small weight differences. The gate compares the reconciled quantities.

Reference bundle: ``benchmarks/reference/vanillasc_xval_references/``, captured
by ``benchmarks/R/sc_references.R`` (install the packages with
``benchmarks/R/install_sc_references.sh``; CRAN is unreachable from some
sandboxes, so it builds from the read-only ``github.com/cran`` mirrors). R does
not run in CI: the frozen capture is used by default and the live R reference is
re-run when ``SC_REFERENCES_LIVE=1``.

Not included: ``SyntheticControlMethods`` (the Python package) was evaluated as
a third reference and rejected. Version 1.1.17 does not install under current
setuptools, passes a 2-D ``x0`` to ``scipy.optimize.minimize`` (rejected since
SciPy 1.11), and after that is shimmed it feeds NaN into cvxpy and returns a
null fit. A reference that needs several undocumented patches to run is not a
usable oracle -- any disagreement could be the patches rather than mlsynth.
"""

from __future__ import annotations

import json
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PANEL = os.path.join(_ROOT, "basedata", "smoking_data.csv")
_AUG = os.path.join(_ROOT, "basedata", "augmented_cali_long.csv")
_BUNDLE = os.path.join(_ROOT, "benchmarks", "reference",
                       "vanillasc_xval_references", "reference.json")
_RSCRIPT = os.path.join(_ROOT, "benchmarks", "R", "sc_references.R")

_TREATED = "California"
_PRE = list(range(1970, 1989))
_POST = list(range(1989, 2001))
_LAGS = (1975, 1980, 1988)

# ADH (2010) Table 2 synthetic-California donor weights.
_ADH = {"Utah": 0.334, "Nevada": 0.234, "Montana": 0.199,
        "Colorado": 0.164, "Connecticut": 0.069}


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #
def _reference() -> dict:
    """The captured R bundle, or a live re-run when SC_REFERENCES_LIVE=1."""
    if os.environ.get("SC_REFERENCES_LIVE") == "1":
        import tempfile
        if not _has_rscript():
            raise BenchmarkSkipped("SC_REFERENCES_LIVE=1 but Rscript is absent")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ref.json")
            subprocess.run(["Rscript", _RSCRIPT, out], check=True,
                           capture_output=True)
            with open(out) as fh:
                return json.load(fh)
    if not os.path.exists(_BUNDLE):  # pragma: no cover - bundle ships with repo
        raise BenchmarkSkipped(f"reference bundle missing: {_BUNDLE}")
    with open(_BUNDLE) as fh:
        return json.load(fh)


def _has_rscript() -> bool:
    from shutil import which
    return which("Rscript") is not None


# --------------------------------------------------------------------------- #
# mlsynth fits
# --------------------------------------------------------------------------- #
def _panel() -> pd.DataFrame:
    return pd.read_csv(_PANEL)


def _outcome_only_gap(df: pd.DataFrame, treated: str) -> np.ndarray:
    """Gap path from the V-free fit, treating ``treated`` as the treated unit."""
    from mlsynth import VanillaSC

    d = df.copy()
    d["treat"] = ((d.state == treated) & (d.year >= 1989)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treat", "unitid": "state",
            "time": "year", "backend": "outcome-only", "inference": False,
            "display_graphs": False,
        }).fit()
    return np.asarray(res.time_series.estimated_gap, dtype=float)


def _rmspe_ratio(gap: np.ndarray) -> float:
    n = len(_PRE)
    return float(np.sqrt((gap[n:] ** 2).mean()) / np.sqrt((gap[:n] ** 2).mean()))


def _adh_spec_fit():
    """mlsynth under the full ADH predictor spec, with placebo inference."""
    from mlsynth import VanillaSC

    d = pd.read_csv(_AUG)
    lags = {f"cig{lag}": d["state"].map(
                d[d.year == lag].set_index("state")["cigsale"])
            for lag in _LAGS}
    lags["treated"] = ((d.state == _TREATED) & (d.year >= 1989)).astype(int)
    d = pd.concat([d, pd.DataFrame(lags)], axis=1)   # one insert, not five
    covs = ["loginc", "p_cig", "pct15-24", "pc_beer",
            "cig1975", "cig1980", "cig1988"]
    windows = {"loginc": (1980, 1988), "p_cig": (1980, 1988),
               "pct15-24": (1980, 1988), "pc_beer": (1984, 1988),
               "cig1975": (1975, 1975), "cig1980": (1980, 1980),
               "cig1988": (1988, 1988)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treated",
            "unitid": "state", "time": "year", "covariates": covs,
            "covariate_windows": windows, "backend": "mscmt",
            "canonical_v": "min.loss.w", "seed": 0, "inference": "placebo",
            "display_graphs": False,
        }).fit()


def _maxdiff(weights: dict, ref: dict) -> float:
    return max(abs(float(weights.get(k, 0.0)) - v) for k, v in ref.items())


# --------------------------------------------------------------------------- #
def run() -> dict:
    ref = _reference()
    A = ref["reference_A_synth_uniform_v"]
    B = ref["reference_B_tidysynth_adh"]
    df = _panel()
    units = sorted(df.state.unique())

    # --- Reference A: the V-free program, every unit as pseudo-treated ----
    gaps = {u: _outcome_only_gap(df, u) for u in units}
    ratios = {u: _rmspe_ratio(g) for u, g in gaps.items()}
    ml_ratio = ratios[_TREATED]
    ml_rank = sum(r >= ml_ratio for r in ratios.values())
    ml_ssr = float((gaps[_TREATED][: len(_PRE)] ** 2).sum())

    ref_ratios = A["rmspe_ratios"]
    common = [u for u in units if u in ref_ratios]
    corr = float(np.corrcoef([ratios[u] for u in common],
                             [ref_ratios[u] for u in common])[0, 1])

    # --- Reference B: the ADH spec, mlsynth's own placebo inference -------
    adh = _adh_spec_fit()
    adh_w = {str(k): float(v) for k, v in adh.weights.donor_weights.items()}
    adh_details = adh.inference.details or {}
    ml_adh_rank = int(adh_details.get("rank", -1))
    ml_adh_ratio = float(adh_details.get("treated_rmspe_ratio", np.nan))

    return {
        # --- gated: agreement where two correct implementations must agree
        "A_rank_mlsynth": float(ml_rank),
        "A_rank_synth": float(A["rank"]),
        "A_rank_agreement": float(ml_rank == A["rank"]),
        "A_ratio_rank_correlation": corr,
        "B_rank_mlsynth": float(ml_adh_rank),
        "B_rank_tidysynth": float(B["rank"]),
        "B_pvalue_mlsynth": float(adh.inference.p_value),
        "B_pvalue_tidysynth": float(B["p_value"]),
        "B_pvalue_absdiff": abs(float(adh.inference.p_value) - float(B["p_value"])),
        # reconciled across the squared/root scale convention
        "B_ratio_mlsynth_root": ml_adh_ratio,
        "B_ratio_tidysynth_root": float(B["california_rmspe_ratio_root_scale"]),
        "B_ratio_reldiff": abs(ml_adh_ratio - B["california_rmspe_ratio_root_scale"])
                           / B["california_rmspe_ratio_root_scale"],
        "adh_weight_maxdiff_mlsynth": _maxdiff(adh_w, _ADH),

        # --- recorded, NOT gated (see module docstring)
        "adh_weight_maxdiff_tidysynth": _maxdiff(B["california_weights"], _ADH),
        "A_pre_ssr_mlsynth": ml_ssr,
        "A_pre_ssr_synth": float(A["california_pre_ssr"]),
        "synth_minus_mlsynth_pre_ssr": float(A["california_pre_ssr"]) - ml_ssr,
        "A_units_fitted_mlsynth": float(len(ratios)),
        "A_units_fitted_synth": float(A["n_units_fitted"]),
        "synth_ipop_failures": float(len(A["ipop_failures"])),
    }


def comparison() -> dict:
    """Side-by-side mlsynth vs the two R references."""
    got = run()
    rows = [
        {"quantity": "A: placebo rank (V-free)",
         "mlsynth": got["A_rank_mlsynth"], "reference": got["A_rank_synth"]},
        {"quantity": "A: pre-period SSR (same objective)",
         "mlsynth": got["A_pre_ssr_mlsynth"], "reference": got["A_pre_ssr_synth"]},
        {"quantity": "A: units fitted",
         "mlsynth": got["A_units_fitted_mlsynth"],
         "reference": got["A_units_fitted_synth"]},
        {"quantity": "B: placebo rank (ADH spec)",
         "mlsynth": got["B_rank_mlsynth"], "reference": got["B_rank_tidysynth"]},
        {"quantity": "B: placebo p-value",
         "mlsynth": got["B_pvalue_mlsynth"], "reference": got["B_pvalue_tidysynth"]},
        {"quantity": "B: RMSPE ratio (root scale)",
         "mlsynth": got["B_ratio_mlsynth_root"],
         "reference": got["B_ratio_tidysynth_root"]},
        {"quantity": "max |weight - ADH Table 2|",
         "mlsynth": got["adh_weight_maxdiff_mlsynth"],
         "reference": got["adh_weight_maxdiff_tidysynth"]},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC",
                         "config": {"backend": "outcome-only / mscmt",
                                    "inference": "placebo"}},
        "reference": {"impl": "Synth (uniform custom.v) + tidysynth (ADH spec)",
                      "version": "Synth 1.1.10, tidysynth 0.2.0"},
    }


# Gate. Only the agreement quantities are asserted; the mlsynth-is-better and
# reference-fragility numbers above are recorded in run()'s output but carry no
# EXPECTED entry, so they are visible without being targets.
EXPECTED = {
    # A -- the V-free program: same rank, same ordering of the placebo pool
    "A_rank_mlsynth": (3, 0.5),
    "A_rank_synth": (3, 0.5),
    "A_rank_agreement": (1.0, 0.5),
    "A_ratio_rank_correlation": (1.0, 0.02),
    # B -- the ADH spec: the practical comparison, exact agreement on inference
    "B_rank_mlsynth": (1, 0.5),
    "B_rank_tidysynth": (1, 0.5),
    "B_pvalue_absdiff": (0.0, 1e-9),
    "B_pvalue_mlsynth": (1 / 39, 1e-9),
    # reconciled across tidysynth's squared-scale convention
    "B_ratio_reldiff": (0.0, 0.06),
    # and mlsynth still lands on Abadie's published weights
    "adh_weight_maxdiff_mlsynth": (0.0, 0.01),
}
