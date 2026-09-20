r"""Botosaru & Ferman (2019): a synthetic control can match outcomes and miss covariates.

Path A (scenario 3 -- full replication archive). Reproduces the empirical content
of Botosaru, I. and Ferman, B., "On the role of covariates in the synthetic
control method", Econometrics Journal 22(2) (2019), 117-130, on the German
reunification panel of Abadie, Diamond & Hainmueller (2015).

Abadie, Diamond & Hainmueller (2010) show that a synthetic control matching the
treated unit's pre-treatment outcomes closely will, under a linear factor model,
also match its covariates. Botosaru & Ferman show that the implication is weaker
than it looks: a close match on pre-treatment outcomes bounds the covariate
discrepancy only through terms that can be large, so a synthetic control fit on
outcomes alone can reproduce the outcome path and still miss the covariates.

Their replication package makes the point on West Germany. It fits a synthetic
control using all 31 pre-treatment GDP lags and no covariate at all, then reports
what that control's covariate means turn out to be. The outcome match is close to
exact and two of the covariates are not:

=========== ============== ================== ==========
predictor   West Germany   synthetic control  gap
=========== ============== ================== ==========
GDP/capita       15808.9            15812.5      +0.02%
trade openness      56.78              56.41      -0.6%
inflation            2.59               4.99      +92%
industry share      34.54              33.48      -3.1%
schooling           55.50              51.20      -7.7%
investment rate     27.02              25.23      -6.6%
=========== ============== ================== ==========

The synthetic control is within 0.02% on the outcome being matched and off by a
factor of nearly two on inflation. That contrast is the paper's empirical point,
and it is what this case pins.

What Synth's ``customV(1, ..., 1)`` means
-----------------------------------------

The do-file's second specification passes all 31 GDP lags as predictors with
``customV(1 1 ... 1)``, which reads as an unweighted fit on the pre-treatment
outcomes. It is not one. Both Stata's ``synth`` and R's ``Synth`` divide every
predictor row by its cross-unit standard deviation before applying ``V``
(``divisor <- sqrt(apply(big.dataframe, 1, var))`` inside ``Synth::synth``), so a
flat ``V`` on the standardized predictors is a diagonal weighting by
:math:`1 / \mathrm{var}_k` on the raw scale. Cross-country GDP variance grows
with the level of GDP, so later pre-treatment years are weighted down and the
specification is not the plain outcome-only convex fit.

The two differ substantially on this panel. The standardized fit spreads weight
over all 16 donors and reaches a pre-treatment RMSPE of 84.2; the unweighted fit
puts weight on 6 donors and reaches 72.3. Both are reported here, and the
paper's conclusion survives either way -- the standardized fit's inflation mean
is 4.83 against West Germany's 2.59, the unweighted fit's is 4.99 -- so the
scaling changes the numbers and not the finding.

mlsynth's ``VanillaSC`` solves the unweighted program, so that is the variant it
is held to.

What is checked
---------------

1. ``VanillaSC`` is the plain simplex QP. Its donor weights agree with an
   independent ``quadprog::solve.QP`` solve of
   :math:`\min_w \|y_1 - Y_0 w\|^2` on the simplex, set up in the reference
   script without going through ``Synth`` at all.
2. Column 1 of the paper's Table 1 is ADH's published predictor table for West
   Germany, and is reproduced from the panel.
3. Column 4 -- the covariate means of the no-covariate synthetic control -- is
   reproduced, and the outcome-versus-covariate contrast above is asserted as a
   metric of its own, so the finding fails the case if it stops holding.
4. The reference harness is itself validated: run at ADH's own predictors and
   ADH's own ``V``, ``Synth`` returns the published synthetic Germany (Austria
   0.42, USA 0.22, Japan 0.16, Switzerland 0.11, Netherlands 0.09), which is how
   we know the ``dataprep`` specification matches the do-file's.

Provenance
----------

* Paper: Botosaru & Ferman (2019), Econometrics Journal 22(2), 117-130.
* Archive: the paper's replication package -- ``repgermany.dta`` and
  "Do-file - Covariates in SC method.do". The do-file's own comment records that
  columns 2 and 3 of Table 1 are taken from ADH (2015) and not recomputed, so
  columns 1 and 4 are what the package produces and what this case covers.
* Data: ``basedata/repgermany.dta``, which the repo already ships and which is
  identical to the ``repgermany.dta`` in the archive. R has no Stata reader
  here, so the case writes the panel to a temporary CSV for the reference
  script and passes it the path; nothing is duplicated in the repo.
* Reference: ``benchmarks/R/botosaru_ferman_covariates.R``, run live. Skips when
  R, ``Synth`` or ``quadprog`` is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_R_SCRIPT = _ROOT / "benchmarks" / "R" / "botosaru_ferman_covariates.R"
_DATA = _ROOT / "basedata" / "repgermany.dta"

TREATED = "West Germany"
T0 = 1990                    # the do-file's trperiod(1991)

# The do-file's predictor windows. Schooling is observed every five years, so
# 1980-1985 is two observations; the investment rate is a single 1980 value.
WINDOWS = {"gdp": (1981, 1990), "trade": (1981, 1990), "infrate": (1981, 1990),
           "industry": (1981, 1990), "schooling": (1980, 1985),
           "invest80": (1980, 1980)}


def _load_panel():
    """The ADH panel the archive ships, as the repo already stores it."""
    panel = pd.read_stata(_DATA)
    panel["year"] = panel.year.astype(int)
    panel["country"] = panel.country.astype(str)
    return panel


def _reference(panel):
    """Run the R script on a temporary CSV of the panel and parse ``key=value``."""
    from benchmarks.compare import BenchmarkSkipped

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH")
    if not _DATA.exists():
        raise BenchmarkSkipped(f"missing {_DATA}")
    probe = subprocess.run(
        [rscript, "-e",
         'q(status = if (all(sapply(c("Synth","quadprog"), requireNamespace, '
         'quietly = TRUE))) 0 else 1)'],
        capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R packages Synth/quadprog unavailable")
    with tempfile.TemporaryDirectory(prefix="bf_ref_") as tmp:
        csv_path = Path(tmp) / "repgermany.csv"
        panel.to_csv(csv_path, index=False)
        proc = subprocess.run([rscript, str(_R_SCRIPT), str(csv_path)],
                              capture_output=True, text=True, cwd=str(_ROOT),
                              timeout=1800)
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line and not line.startswith(" "):
            k, _, v = line.partition("=")
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    if "rmspe_pre_flat" not in out:
        raise BenchmarkSkipped(
            f"reference produced no flat-QP solve: {proc.stderr.strip()[-300:]}")
    return out


def _mlsynth_weights(panel):
    """``VanillaSC`` on GDP alone, treatment in 1991: donor weights and fit."""
    from mlsynth.estimators.vanillasc import VanillaSC

    df = panel[["country", "year", "gdp"]].copy()
    df["treat"] = ((df.country == TREATED) & (df.year > T0)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC(dict(df=df, outcome="gdp", treat="treat",
                             unitid="country", time="year",
                             display_graphs=False)).fit()
    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    obs = panel[panel.country == TREATED].sort_values("year").gdp.values.astype(float)
    n_pre = T0 - int(panel.year.min()) + 1
    rmspe = float(np.sqrt(np.mean((obs[:n_pre] - cf[:n_pre]) ** 2)))
    return w, rmspe


def _predictor_means(panel, weights):
    """Table 1: West Germany's predictor means, and the synthetic control's."""
    col1, col4 = {}, {}
    for v, (a, b) in WINDOWS.items():
        m = panel[(panel.year >= a) & (panel.year <= b)].groupby("country")[v].mean()
        col1[v] = float(m[TREATED])
        col4[v] = float(sum(weights.get(c, 0.0) * m[c]
                            for c in m.index if c != TREATED))
    return col1, col4


def run() -> dict:
    panel = _load_panel()
    ref = _reference(panel)

    w, rmspe = _mlsynth_weights(panel)
    col1, col4 = _predictor_means(panel, w)

    # 1. VanillaSC against the independent quadprog solve of the same program.
    flat = {k[len("w_flat_"):]: v for k, v in ref.items() if k.startswith("w_flat_")}
    key = lambda c: "".join(ch for ch in c if ch.isalpha())
    gap = max(abs(w.get(c, 0.0) - flat.get(key(c), 0.0))
              for c in panel.country.unique() if c != TREATED)

    out = {
        "weight_max_gap_vs_quadprog": float(gap),
        "rmspe_gap_vs_quadprog": abs(rmspe - ref["rmspe_pre_flat"]),
        "n_donors_unweighted": float(sum(v > 1e-6 for v in w.values())),
        # 2. Table 1 column 1 -- ADH's published predictor table.
        "col1_gdp": round(col1["gdp"], 1),
        "col1_trade": round(col1["trade"], 2),
        "col1_infrate": round(col1["infrate"], 3),
        "col1_industry": round(col1["industry"], 2),
        "col1_schooling": round(col1["schooling"], 1),
        "col1_invest80": round(col1["invest80"], 2),
        # 3. Table 1 column 4 -- the no-covariate synthetic control.
        "col4_gdp": round(col4["gdp"], 1),
        "col4_trade": round(col4["trade"], 2),
        "col4_infrate": round(col4["infrate"], 3),
        "col4_industry": round(col4["industry"], 2),
        "col4_schooling": round(col4["schooling"], 1),
        "col4_invest80": round(col4["invest80"], 2),
    }
    # The paper's finding: near-exact on the matched outcome, far off on a
    # covariate that was never matched.
    out["gdp_rel_error"] = round(abs(col4["gdp"] - col1["gdp"]) / col1["gdp"], 5)
    out["infrate_rel_error"] = round(
        abs(col4["infrate"] - col1["infrate"]) / col1["infrate"], 3)
    out["outcome_matched_covariate_missed"] = float(
        out["gdp_rel_error"] < 0.001 and out["infrate_rel_error"] > 0.5)

    # 4. The standardization finding, and the ADH-spec check on the harness.
    out["n_donors_standardized"] = ref["w_nocov_nonzero"]
    out["rmspe_standardized"] = round(ref["rmspe_pre_nocov"], 2)
    out["rmspe_unweighted"] = round(rmspe, 2)
    out["standardized_fits_worse"] = float(
        ref["rmspe_pre_nocov"] > rmspe)
    out["adh_w_austria"] = round(ref["w_adh_Austria"], 3)
    out["adh_w_usa"] = round(ref["w_adh_USA"], 3)
    out["adh_w_japan"] = round(ref["w_adh_Japan"], 3)
    out["adh_w_switzerland"] = round(ref["w_adh_Switzerland"], 3)
    out["adh_w_netherlands"] = round(ref["w_adh_Netherlands"], 3)
    return out


# Deterministic on both sides: one convex QP per specification, no sampling.
# The tolerances are solver tolerance, except where a published value is quoted.
EXPECTED = {
    # VanillaSC is the plain simplex QP, to an independent solver's precision.
    "weight_max_gap_vs_quadprog": (0.0, 1e-4),
    "rmspe_gap_vs_quadprog": (0.0, 1e-2),
    "n_donors_unweighted": (6.0, 0.0),
    # Table 1 column 1 is ADH (2015)'s published predictor table for West
    # Germany: 15808.9, 56.8, 2.6, 34.5, 55.5, 27.0.
    "col1_gdp": (15808.9, 0.1),
    "col1_trade": (56.78, 0.02),
    "col1_infrate": (2.595, 0.01),
    "col1_industry": (34.54, 0.02),
    "col1_schooling": (55.5, 0.1),
    "col1_invest80": (27.02, 0.02),
    # Column 4, the no-covariate synthetic control's own means.
    "col4_gdp": (15812.5, 1.0),
    "col4_trade": (56.41, 0.05),
    "col4_infrate": (4.990, 0.02),
    "col4_industry": (33.48, 0.05),
    "col4_schooling": (51.2, 0.2),
    "col4_invest80": (25.23, 0.05),
    # The finding.
    "gdp_rel_error": (0.00023, 0.0005),
    "infrate_rel_error": (0.923, 0.05),
    "outcome_matched_covariate_missed": (1.0, 0.0),
    # Synth's standardization is not a flat weighting of the pre-period.
    "n_donors_standardized": (16.0, 0.0),
    "rmspe_standardized": (84.21, 0.5),
    "rmspe_unweighted": (72.30, 0.5),
    "standardized_fits_worse": (1.0, 0.0),
    # The harness reproduces ADH's published synthetic Germany at ADH's own V,
    # which is what validates the dataprep specification against the do-file.
    "adh_w_austria": (0.417, 0.015),      # ADH 2015 report 0.42
    "adh_w_usa": (0.219, 0.015),          # 0.22
    "adh_w_japan": (0.155, 0.015),        # 0.16
    "adh_w_switzerland": (0.111, 0.015),  # 0.11
    "adh_w_netherlands": (0.090, 0.015),  # 0.09
}
